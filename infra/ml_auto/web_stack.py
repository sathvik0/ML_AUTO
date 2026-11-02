"""The application: VPC, ECS Fargate (Streamlit), an HTTPS Application Load
Balancer that signs every user in with Amazon Cognito, WAF, the secrets the
app reads, and its data tier — Aurora PostgreSQL Serverless v2 for projects,
runs and the model registry, and an encrypted S3 bucket for datasets and
model files. See docs/architecture.md for why each piece is here.

Context (cdk.json or -c):
  domainName        app host name, e.g. ml.example.com          (required)
  hostedZoneName    Route 53 zone that holds it, e.g. example.com (required)
  cognitoDomain     prefix for the hosted sign-in page            (required)
  natGateways       1 (default) or 2 for a NAT per AZ
  interfaceEndpoints  true to add ECR/Secrets/Logs endpoints (cuts NAT data,
                    adds ~$7/month per endpoint per AZ)
  minTasks / maxTasks  service size bounds (1 / 4)
  dbMaxCapacity     Aurora Serverless v2 ceiling in ACUs (4)
"""
from pathlib import Path

from aws_cdk import (CfnOutput, Duration, RemovalPolicy, Stack, aws_certificatemanager as acm,
                     aws_cognito as cognito, aws_ec2 as ec2, aws_ecr_assets as ecr_assets,
                     aws_ecs as ecs, aws_elasticloadbalancingv2 as elbv2,
                     aws_events as events, aws_events_targets as events_targets,
                     aws_elasticloadbalancingv2_actions as elb_actions,
                     aws_elasticloadbalancingv2_targets as elb_targets, aws_kms as kms,
                     aws_lambda as lambda_, aws_logs as logs, aws_rds as rds,
                     aws_route53 as route53,
                     aws_route53_targets as route53_targets, aws_s3 as s3,
                     aws_secretsmanager as secretsmanager, aws_wafv2 as wafv2)
from constructs import Construct

APP_DIR = Path(__file__).resolve().parents[2]
ROLES = ("admin", "analyst", "viewer")
PORT = 8501
API_PORT = 8000


class WebStack(Stack):
    def __init__(self, scope: Construct, cid: str, **kwargs) -> None:
        super().__init__(scope, cid, **kwargs)
        ctx = self.node.try_get_context
        domain = ctx("domainName")
        zone_name = ctx("hostedZoneName")
        cognito_prefix = ctx("cognitoDomain")
        missing = [n for n, v in (("domainName", domain),
                                  ("hostedZoneName", zone_name),
                                  ("cognitoDomain", cognito_prefix)) if not v]
        if missing:
            raise ValueError("Set context: " + ", ".join(missing)
                             + " (see infra/README.md)")
        base_url = f"https://{domain}"

        # ── network ──────────────────────────────────────────────────────
        vpc = ec2.Vpc(
            self, "Vpc", max_azs=2,
            nat_gateways=int(ctx("natGateways") or 1),
            subnet_configuration=[
                ec2.SubnetConfiguration(name="public",
                                        subnet_type=ec2.SubnetType.PUBLIC),
                ec2.SubnetConfiguration(
                    name="app",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS)])
        # S3 traffic (image layers, later datasets and models) skips the NAT.
        vpc.add_gateway_endpoint("S3", service=ec2.GatewayVpcEndpointAwsService.S3)
        if str(ctx("interfaceEndpoints")).lower() == "true":
            for name, svc in (
                    ("EcrApi", ec2.InterfaceVpcEndpointAwsService.ECR),
                    ("EcrDocker", ec2.InterfaceVpcEndpointAwsService.ECR_DOCKER),
                    ("Secrets", ec2.InterfaceVpcEndpointAwsService.SECRETS_MANAGER),
                    ("Logs", ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS)):
                vpc.add_interface_endpoint(name, service=svc)

        # ── identity: Cognito ────────────────────────────────────────────
        pool = cognito.UserPool(
            self, "Users",
            # Invite-only: an admin adds people (or a federated IdP does).
            self_sign_up_enabled=False,
            sign_in_aliases=cognito.SignInAliases(email=True),
            standard_attributes=cognito.StandardAttributes(
                email=cognito.StandardAttribute(required=True, mutable=True),
                fullname=cognito.StandardAttribute(required=False,
                                                   mutable=True)),
            mfa=cognito.Mfa.OPTIONAL,
            mfa_second_factor=cognito.MfaSecondFactor(sms=False, otp=True),
            password_policy=cognito.PasswordPolicy(
                min_length=12, require_digits=True, require_lowercase=True,
                require_uppercase=True, require_symbols=False),
            account_recovery=cognito.AccountRecovery.EMAIL_ONLY,
            removal_policy=RemovalPolicy.RETAIN)
        for i, role in enumerate(ROLES):
            cognito.CfnUserPoolGroup(
                self, f"Group{role.title()}", user_pool_id=pool.user_pool_id,
                group_name=role, precedence=i,
                description=f"ML Automation {role}")
        pool_domain = pool.add_domain("SignIn", cognito_domain=cognito.CognitoDomainOptions(
            domain_prefix=cognito_prefix))
        client = pool.add_client(
            "Alb", generate_secret=True,
            o_auth=cognito.OAuthSettings(
                flows=cognito.OAuthFlows(authorization_code_grant=True),
                scopes=[cognito.OAuthScope.OPENID, cognito.OAuthScope.EMAIL,
                        cognito.OAuthScope.PROFILE],
                callback_urls=[f"{base_url}/oauth2/idpresponse"],
                logout_urls=[f"{base_url}/"]),
            supported_identity_providers=[
                cognito.UserPoolClientIdentityProvider.COGNITO],
            access_token_validity=Duration.hours(1),
            id_token_validity=Duration.hours(1),
            refresh_token_validity=Duration.days(1))

        # ── secrets the app reads (values set after deploy, never here) ──
        app_keys = secretsmanager.Secret(
            self, "AppKeys", secret_name="ml-auto/app-keys",
            description="ANTHROPIC_API_KEY and OPENAI_API_KEY for ML "
                        "Automation. Set the values after deploying.",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                secret_string_template=(
                    '{"ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": ""}'),
                generate_string_key="unused"))

        # ── data: S3 for files, Aurora for records ───────────────────────
        data_key = kms.Key(self, "DataKey", enable_key_rotation=True,
                           alias="alias/ml-auto-data",
                           description="ML Automation datasets and models",
                           removal_policy=RemovalPolicy.RETAIN)
        data_bucket = s3.Bucket(
            self, "Data", encryption=s3.BucketEncryption.KMS,
            encryption_key=data_key, bucket_key_enabled=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True, versioned=True,
            object_ownership=s3.ObjectOwnership.BUCKET_OWNER_ENFORCED,
            lifecycle_rules=[s3.LifecycleRule(
                noncurrent_version_expiration=Duration.days(90))],
            removal_policy=RemovalPolicy.RETAIN)
        database = rds.DatabaseCluster(
            self, "Database",
            engine=rds.DatabaseClusterEngine.aurora_postgres(
                version=rds.AuroraPostgresEngineVersion.VER_16_9),
            credentials=rds.Credentials.from_generated_secret("mlauto"),
            default_database_name="mlauto",
            writer=rds.ClusterInstance.serverless_v2("writer"),
            # Pauses when idle: an internal tool is often quiet for hours.
            serverless_v2_min_capacity=0,
            serverless_v2_max_capacity=float(ctx("dbMaxCapacity") or 4),
            vpc=vpc, vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            storage_encrypted=True,
            backup=rds.BackupProps(retention=Duration.days(7)),
            deletion_protection=True,
            removal_policy=RemovalPolicy.SNAPSHOT)

        # ── load balancer ────────────────────────────────────────────────
        zone = route53.HostedZone.from_lookup(self, "Zone",
                                              domain_name=zone_name)
        cert = acm.Certificate(self, "Cert", domain_name=domain,
                               validation=acm.CertificateValidation.from_dns(zone))
        alb_logs = s3.Bucket(
            self, "AlbLogs", encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            lifecycle_rules=[s3.LifecycleRule(expiration=Duration.days(90))],
            removal_policy=RemovalPolicy.RETAIN)
        alb = elbv2.ApplicationLoadBalancer(
            self, "Alb", vpc=vpc, internet_facing=True,
            # Streamlit holds a WebSocket open while a user reads; the 60 s
            # default would drop them.
            idle_timeout=Duration.seconds(3600),
            drop_invalid_header_fields=True)
        alb.log_access_logs(alb_logs, prefix="alb")
        alb.add_redirect(source_port=80, target_port=443)
        route53.ARecord(self, "Alias", zone=zone, record_name=domain,
                        target=route53.RecordTarget.from_alias(
                            route53_targets.LoadBalancerTarget(alb)))

        # ── the web service ──────────────────────────────────────────────
        cluster = ecs.Cluster(self, "Cluster", vpc=vpc,
                              container_insights_v2=ecs.ContainerInsights.ENABLED)
        task = ecs.FargateTaskDefinition(
            self, "WebTask", cpu=2048, memory_limit_mib=8192,
            runtime_platform=ecs.RuntimePlatform(
                cpu_architecture=ecs.CpuArchitecture.X86_64,
                operating_system_family=ecs.OperatingSystemFamily.LINUX))
        log_group = logs.LogGroup(self, "WebLogs",
                                  retention=logs.RetentionDays.ONE_MONTH,
                                  removal_policy=RemovalPolicy.DESTROY)
        image = ecr_assets.DockerImageAsset(
            self, "WebImage", directory=str(APP_DIR),
            platform=ecr_assets.Platform.LINUX_AMD64)
        # What every service reading projects needs: storage and the
        # database (credentials from the cluster's own secret, by field).
        data_env = {"AWS_REGION": self.region,
                    "STORAGE_BUCKET": data_bucket.bucket_name}
        db_secrets = {
            env: ecs.Secret.from_secrets_manager(database.secret, field)
            for env, field in (("DB_HOST", "host"), ("DB_PORT", "port"),
                               ("DB_USER", "username"),
                               ("DB_PASSWORD", "password"),
                               ("DB_NAME", "dbname"))}
        task.add_container(
            "web", image=ecs.ContainerImage.from_docker_image_asset(image),
            port_mappings=[ecs.PortMapping(container_port=PORT)],
            logging=ecs.LogDrivers.aws_logs(stream_prefix="web",
                                            log_group=log_group),
            stop_timeout=Duration.seconds(120),
            environment={
                **data_env,
                "AUTH_MODE": "alb",
                "ALB_ARN": alb.load_balancer_arn,
                "COGNITO_USER_POOL_ID": pool.user_pool_id,
                "COGNITO_CLIENT_ID": client.user_pool_client_id,
                "SIGN_OUT_URL": f"{base_url}/logout",
                "APP_URL": base_url,
                "DEFAULT_ROLE": "viewer",
            },
            secrets={
                "ANTHROPIC_API_KEY": ecs.Secret.from_secrets_manager(
                    app_keys, "ANTHROPIC_API_KEY"),
                "OPENAI_API_KEY": ecs.Secret.from_secrets_manager(
                    app_keys, "OPENAI_API_KEY"),
                **db_secrets,
            },
            health_check=ecs.HealthCheck(
                command=["CMD-SHELL",
                         f"curl -fsS http://localhost:{PORT}/_stcore/health"
                         " || exit 1"],
                interval=Duration.seconds(30), timeout=Duration.seconds(5),
                start_period=Duration.seconds(60), retries=3))
        service = ecs.FargateService(
            self, "Web", cluster=cluster, task_definition=task,
            desired_count=int(ctx("minTasks") or 1),
            min_healthy_percent=100, max_healthy_percent=200,
            health_check_grace_period=Duration.seconds(90),
            circuit_breaker=ecs.DeploymentCircuitBreaker(rollback=True),
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS))
        data_bucket.grant_read_write(task.task_role)
        data_bucket.grant_delete(task.task_role)
        database.connections.allow_default_port_from(
            service, "the web service")
        target = elbv2.ApplicationTargetGroup(
            self, "WebTargets", vpc=vpc, port=PORT,
            protocol=elbv2.ApplicationProtocol.HTTP,
            protocol_version=elbv2.ApplicationProtocolVersion.HTTP1,
            targets=[service.load_balancer_target(container_name="web",
                                                  container_port=PORT)],
            health_check=elbv2.HealthCheck(path="/_stcore/health",
                                           healthy_http_codes="200"),
            # Each user's work lives in one task's memory (until item 2).
            stickiness_cookie_duration=Duration.days(1),
            deregistration_delay=Duration.seconds(300))
        scaling = service.auto_scale_task_count(
            min_capacity=int(ctx("minTasks") or 1),
            max_capacity=int(ctx("maxTasks") or 4))
        # Out freely, in rarely: every scale-in ends someone's session.
        scaling.scale_on_memory_utilization(
            "Memory", target_utilization_percent=60,
            scale_in_cooldown=Duration.minutes(30),
            scale_out_cooldown=Duration.minutes(2))

        # ── the model API: same image, its own service on /api/* ────────
        api_task = ecs.FargateTaskDefinition(
            self, "ApiTask", cpu=1024, memory_limit_mib=2048,
            runtime_platform=ecs.RuntimePlatform(
                cpu_architecture=ecs.CpuArchitecture.X86_64,
                operating_system_family=ecs.OperatingSystemFamily.LINUX))
        api_task.add_container(
            "api", image=ecs.ContainerImage.from_docker_image_asset(image),
            command=["uvicorn", "api.main:app", "--host", "0.0.0.0",
                     "--port", str(API_PORT), "--proxy-headers",
                     "--forwarded-allow-ips", "*"],
            port_mappings=[ecs.PortMapping(container_port=API_PORT)],
            logging=ecs.LogDrivers.aws_logs(stream_prefix="api",
                                            log_group=log_group),
            environment=data_env, secrets=db_secrets,
            health_check=ecs.HealthCheck(
                command=["CMD-SHELL",
                         f"curl -fsS http://localhost:{API_PORT}/api/health"
                         " || exit 1"],
                interval=Duration.seconds(30), timeout=Duration.seconds(5),
                start_period=Duration.seconds(30), retries=3))
        api_service = ecs.FargateService(
            self, "Api", cluster=cluster, task_definition=api_task,
            desired_count=1, min_healthy_percent=100,
            max_healthy_percent=200,
            health_check_grace_period=Duration.seconds(60),
            circuit_breaker=ecs.DeploymentCircuitBreaker(rollback=True),
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS))
        data_bucket.grant_read(api_task.task_role)
        database.connections.allow_default_port_from(
            api_service, "the model API")
        api_target = elbv2.ApplicationTargetGroup(
            self, "ApiTargets", vpc=vpc, port=API_PORT,
            protocol=elbv2.ApplicationProtocol.HTTP,
            targets=[api_service.load_balancer_target(
                container_name="api", container_port=API_PORT)],
            health_check=elbv2.HealthCheck(path="/api/health",
                                           healthy_http_codes="200"),
            deregistration_delay=Duration.seconds(30))
        api_scaling = api_service.auto_scale_task_count(
            min_capacity=1, max_capacity=int(ctx("maxApiTasks") or 6))
        api_scaling.scale_on_cpu_utilization(
            "Cpu", target_utilization_percent=60,
            scale_out_cooldown=Duration.minutes(1))

        # ── batch scoring: a scheduled task every 15 minutes ────────────
        # The deployment's own Snowflake account, for batch jobs only — it
        # must never pre-fill the web app's form with a service password.
        snowflake_service = secretsmanager.Secret(
            self, "SnowflakeService",
            secret_name="ml-auto/snowflake-service",
            description="Snowflake account batch jobs read and write with. "
                        "Set the values after deploying.",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                secret_string_template=(
                    '{"account": "", "user": "", "role": "", '
                    '"warehouse": "", "database": "", "schema": ""}'),
                generate_string_key="password"))
        batch_task = ecs.FargateTaskDefinition(
            self, "BatchTask", cpu=2048, memory_limit_mib=8192,
            runtime_platform=ecs.RuntimePlatform(
                cpu_architecture=ecs.CpuArchitecture.X86_64,
                operating_system_family=ecs.OperatingSystemFamily.LINUX))
        batch_task.add_container(
            "batch", image=ecs.ContainerImage.from_docker_image_asset(image),
            command=["python", "-m", "worker.batch", "--due"],
            logging=ecs.LogDrivers.aws_logs(stream_prefix="batch",
                                            log_group=log_group),
            environment=data_env,
            secrets={**db_secrets, **{
                f"SNOWFLAKE_{field.upper()}": ecs.Secret
                .from_secrets_manager(snowflake_service, field)
                for field in ("account", "user", "password", "role",
                              "warehouse", "database", "schema")}})
        data_bucket.grant_read_write(batch_task.task_role)
        batch_sg = ec2.SecurityGroup(self, "BatchSg", vpc=vpc,
                                     description="ML Automation batch jobs")
        database.connections.allow_default_port_from(batch_sg,
                                                     "batch jobs")
        events.Rule(
            self, "BatchSchedule",
            description="Run due ML Automation batch jobs",
            schedule=events.Schedule.rate(Duration.minutes(15)),
            targets=[events_targets.EcsTask(
                cluster=cluster, task_definition=batch_task,
                launch_type=ecs.LaunchType.FARGATE,
                subnet_selection=ec2.SubnetSelection(
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
                security_groups=[batch_sg])])

        # ── sign-out: only the load balancer can clear its own cookie ────
        logout = lambda_.Function(
            self, "Logout", runtime=lambda_.Runtime.PYTHON_3_12,
            handler="index.handler", timeout=Duration.seconds(5),
            code=lambda_.Code.from_asset(str(Path(__file__).parent
                                             / "logout")),
            environment={
                "LOGOUT_URL": (f"https://{cognito_prefix}.auth.{self.region}"
                               f".amazoncognito.com/logout?client_id="
                               f"{client.user_pool_client_id}&logout_uri="
                               f"{base_url}/")})
        logout_targets = elbv2.ApplicationTargetGroup(
            self, "LogoutTarget",
            targets=[elb_targets.LambdaTarget(logout)])
        logout_targets.set_attribute("lambda.multi_value_headers.enabled",
                                     "true")

        # ── listener: sign in, then the app ──────────────────────────────
        https = alb.add_listener(
            "Https", port=443, certificates=[cert],
            ssl_policy=elbv2.SslPolicy.RECOMMENDED_TLS,
            default_action=elb_actions.AuthenticateCognitoAction(
                user_pool=pool, user_pool_client=client,
                user_pool_domain=pool_domain,
                scope="openid email profile",
                session_timeout=Duration.hours(8),
                next=elbv2.ListenerAction.forward([target])))
        https.add_action(
            "SignOut", priority=1,
            conditions=[elbv2.ListenerCondition.path_patterns(["/logout"])],
            action=elbv2.ListenerAction.forward([logout_targets]))
        # Machines call the API with project keys; no browser sign-in.
        https.add_action(
            "ModelApi", priority=2,
            conditions=[elbv2.ListenerCondition.path_patterns(["/api/*"])],
            action=elbv2.ListenerAction.forward([api_target]))

        # ── WAF ──────────────────────────────────────────────────────────
        def managed(name, priority, overrides=()):
            return wafv2.CfnWebACL.RuleProperty(
                name=name, priority=priority,
                override_action=wafv2.CfnWebACL.OverrideActionProperty(
                    none={}),
                statement=wafv2.CfnWebACL.StatementProperty(
                    managed_rule_group_statement=wafv2.CfnWebACL
                    .ManagedRuleGroupStatementProperty(
                        vendor_name="AWS", name=name,
                        rule_action_overrides=[
                            wafv2.CfnWebACL.RuleActionOverrideProperty(
                                name=o, action_to_use=wafv2.CfnWebACL
                                .RuleActionProperty(count={}))
                            for o in overrides])),
                visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                    cloud_watch_metrics_enabled=True, metric_name=name,
                    sampled_requests_enabled=True))

        acl = wafv2.CfnWebACL(
            self, "Waf", scope="REGIONAL",
            default_action=wafv2.CfnWebACL.DefaultActionProperty(allow={}),
            visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                cloud_watch_metrics_enabled=True, metric_name="ml-auto",
                sampled_requests_enabled=True),
            rules=[
                # File uploads are request bodies far above the common rule
                # set's 8 KB limit: counted, not blocked.
                managed("AWSManagedRulesCommonRuleSet", 1,
                        ("SizeRestrictions_BODY",)),
                managed("AWSManagedRulesKnownBadInputsRuleSet", 2),
                managed("AWSManagedRulesAmazonIpReputationList", 3),
                wafv2.CfnWebACL.RuleProperty(
                    name="RateLimit", priority=4,
                    action=wafv2.CfnWebACL.RuleActionProperty(block={}),
                    statement=wafv2.CfnWebACL.StatementProperty(
                        rate_based_statement=wafv2.CfnWebACL
                        .RateBasedStatementProperty(
                            limit=3000, aggregate_key_type="IP")),
                    visibility_config=wafv2.CfnWebACL
                    .VisibilityConfigProperty(
                        cloud_watch_metrics_enabled=True,
                        metric_name="RateLimit",
                        sampled_requests_enabled=True)),
            ])
        wafv2.CfnWebACLAssociation(self, "WafAlb",
                                   resource_arn=alb.load_balancer_arn,
                                   web_acl_arn=acl.attr_arn)

        CfnOutput(self, "Url", value=base_url)
        CfnOutput(self, "UserPoolId", value=pool.user_pool_id)
        CfnOutput(self, "AppKeysSecret", value=app_keys.secret_name)
        CfnOutput(self, "LogGroup", value=log_group.log_group_name)
        CfnOutput(self, "DataBucket", value=data_bucket.bucket_name)
        CfnOutput(self, "DatabaseEndpoint",
                  value=database.cluster_endpoint.hostname)
