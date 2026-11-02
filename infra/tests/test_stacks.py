"""Both stacks synthesise offline (the Route 53 lookup answered from
context), with the settings the app depends on."""
import sys
import tempfile
from pathlib import Path

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Match, Template

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_auto.deploy_stack import DeployStack  # noqa: E402
from ml_auto.web_stack import WebStack  # noqa: E402

ACCOUNT, REGION = "123456789012", "us-east-1"


@pytest.fixture(scope="module")
def stacks():
    app = cdk.App(outdir=tempfile.mkdtemp(), context={
        "domainName": "ml.example.com", "hostedZoneName": "example.com",
        "cognitoDomain": "ml-auto-test",
        f"hosted-zone:account={ACCOUNT}:domainName=example.com:"
        f"region={REGION}": {"Id": "/hostedzone/Z123",
                             "Name": "example.com."}})
    env = cdk.Environment(account=ACCOUNT, region=REGION)
    web = WebStack(app, "MlAutoWeb", env=env)
    dep = DeployStack(app, "MlAutoDeploy", env=env, repo="sathvik0/ML_AUTO")
    return Template.from_stack(web), Template.from_stack(dep)


def test_every_request_signs_in(stacks):
    web, _ = stacks
    web.has_resource_properties("AWS::ElasticLoadBalancingV2::Listener", {
        "Port": 443, "DefaultActions": Match.array_with([
            Match.object_like({"Type": "authenticate-cognito"})])})
    web.has_resource_properties("AWS::ElasticLoadBalancingV2::ListenerRule", {
        "Conditions": [{"Field": "path-pattern",
                        "PathPatternConfig": {"Values": ["/logout"]}}]})


def test_streamlit_sessions_survive(stacks):
    web, _ = stacks
    web.has_resource_properties(
        "AWS::ElasticLoadBalancingV2::LoadBalancer",
        {"LoadBalancerAttributes": Match.array_with([
            {"Key": "idle_timeout.timeout_seconds", "Value": "3600"}])})
    web.has_resource_properties("AWS::ElasticLoadBalancingV2::TargetGroup", {
        "HealthCheckPath": "/_stcore/health",
        "TargetGroupAttributes": Match.array_with([
            {"Key": "stickiness.enabled", "Value": "true"}])})


def test_invite_only_with_three_roles(stacks):
    web, _ = stacks
    web.resource_count_is("AWS::Cognito::UserPoolGroup", 3)
    web.has_resource_properties("AWS::Cognito::UserPool", {
        "AdminCreateUserConfig": {"AllowAdminCreateUserOnly": True}})


def test_uploads_not_blocked_by_waf(stacks):
    web, _ = stacks
    web.has_resource_properties("AWS::WAFv2::WebACL", {
        "Rules": Match.array_with([Match.object_like({
            "Name": "AWSManagedRulesCommonRuleSet",
            "Statement": {"ManagedRuleGroupStatement": Match.object_like({
                "RuleActionOverrides": [{
                    "Name": "SizeRestrictions_BODY",
                    "ActionToUse": {"Count": {}}}]})}})])})


def test_container_gets_identity_settings_and_secrets(stacks):
    web, _ = stacks
    task = next(r for r in web.to_json()["Resources"].values()
                if r["Type"] == "AWS::ECS::TaskDefinition")
    c = task["Properties"]["ContainerDefinitions"][0]
    env = {e["Name"] for e in c["Environment"]}
    assert {"AUTH_MODE", "ALB_ARN", "AWS_REGION", "COGNITO_USER_POOL_ID",
            "COGNITO_CLIENT_ID", "SIGN_OUT_URL"} <= env
    assert {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"} <= {
        s["Name"] for s in c["Secrets"]}


def test_deploy_role_only_from_main(stacks):
    _, dep = stacks
    role = next(r for r in dep.to_json()["Resources"].values()
                if r["Type"] == "AWS::IAM::Role"
                and r["Properties"].get("RoleName") == "ml-auto-github-deploy")
    cond = role["Properties"]["AssumeRolePolicyDocument"]["Statement"][0][
        "Condition"]
    assert cond["StringLike"]["token.actions.githubusercontent.com:sub"] == \
        "repo:sathvik0/ML_AUTO:ref:refs/heads/main"


def test_database_is_serverless_encrypted_and_protected(stacks):
    web, _ = stacks
    web.has_resource_properties("AWS::RDS::DBCluster", {
        "Engine": "aurora-postgresql",
        "StorageEncrypted": True,
        "DeletionProtection": True,
        "BackupRetentionPeriod": 7,
        "ServerlessV2ScalingConfiguration": {"MinCapacity": 0,
                                             "MaxCapacity": 4}})
    web.has_resource("AWS::RDS::DBCluster", {
        "DeletionPolicy": "Snapshot"})


def test_data_bucket_private_versioned_kms(stacks):
    web, _ = stacks
    web.has_resource_properties("AWS::S3::Bucket", {
        "VersioningConfiguration": {"Status": "Enabled"},
        "PublicAccessBlockConfiguration": {
            "BlockPublicAcls": True, "BlockPublicPolicy": True,
            "IgnorePublicAcls": True, "RestrictPublicBuckets": True},
        "BucketEncryption": {"ServerSideEncryptionConfiguration": [
            Match.object_like({"ServerSideEncryptionByDefault":
                               Match.object_like({"SSEAlgorithm":
                                                  "aws:kms"})})]}})
    web.has_resource_properties("AWS::KMS::Key", {"EnableKeyRotation": True})


def test_container_gets_database_as_secrets(stacks):
    web, _ = stacks
    task = next(r for r in web.to_json()["Resources"].values()
                if r["Type"] == "AWS::ECS::TaskDefinition")
    c = task["Properties"]["ContainerDefinitions"][0]
    secrets = {s["Name"] for s in c["Secrets"]}
    assert {"DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD",
            "DB_NAME"} <= secrets
    env = {e["Name"] for e in c["Environment"]}
    assert "STORAGE_BUCKET" in env and "DB_PASSWORD" not in env


def test_only_the_app_reaches_the_database(stacks):
    web, _ = stacks
    # The port is a reference to the cluster's own, so match by purpose.
    ingress = [r["Properties"] for r in web.to_json()["Resources"].values()
               if r["Type"] == "AWS::EC2::SecurityGroupIngress"
               and r["Properties"].get("Description") == "the web service"]
    assert ingress and all("SourceSecurityGroupId" in i for i in ingress)
    assert not any(i.get("CidrIp") for i in ingress)


def _containers(web):
    return {c["Name"]: c for r in web.to_json()["Resources"].values()
            if r["Type"] == "AWS::ECS::TaskDefinition"
            for c in r["Properties"]["ContainerDefinitions"]}


def test_api_path_skips_browser_sign_in(stacks):
    web, _ = stacks
    rules = [r["Properties"] for r in web.to_json()["Resources"].values()
             if r["Type"] == "AWS::ElasticLoadBalancingV2::ListenerRule"]
    api = next(r for r in rules if r["Conditions"][0]["PathPatternConfig"]
               ["Values"] == ["/api/*"])
    assert [a["Type"] for a in api["Actions"]] == ["forward"]
    assert api["Priority"] == 2


def test_api_and_batch_containers(stacks):
    web, _ = stacks
    c = _containers(web)
    assert c["api"]["Command"][:2] == ["uvicorn", "api.main:app"]
    assert c["batch"]["Command"] == ["python", "-m", "worker.batch", "--due"]
    web.has_resource_properties("AWS::Events::Rule", {
        "ScheduleExpression": "rate(15 minutes)",
        "Targets": [Match.object_like({"EcsParameters": Match.object_like(
            {"LaunchType": "FARGATE"})})]})


def test_snowflake_service_account_only_in_batch(stacks):
    web, _ = stacks
    c = _containers(web)

    def names(container):
        return {s["Name"] for s in container.get("Secrets", [])} | {
            e["Name"] for e in container.get("Environment", [])}
    assert "SNOWFLAKE_PASSWORD" in names(c["batch"])
    assert not any(n.startswith("SNOWFLAKE_") for n in names(c["web"]))
    assert not any(n.startswith("SNOWFLAKE_") for n in names(c["api"]))


def test_api_cannot_write_data(stacks):
    web, _ = stacks
    res = web.to_json()["Resources"]
    api_role = next(k for k, r in res.items() if r["Type"] == "AWS::IAM::Role"
                    and k.startswith("ApiTaskTaskRole"))
    actions = set()
    for r in res.values():
        if r["Type"] == "AWS::IAM::Policy" and any(
                ref.get("Ref") == api_role for ref in r["Properties"]["Roles"]):
            for st_ in r["Properties"]["PolicyDocument"]["Statement"]:
                a = st_["Action"]
                actions |= set(a if isinstance(a, list) else [a])
    assert any(a.startswith("s3:GetObject") for a in actions)
    assert not any(a.startswith(("s3:PutObject", "s3:DeleteObject"))
                   for a in actions)


def _policy_actions(web, role_prefix):
    res = web.to_json()["Resources"]
    roles = [k for k, r in res.items() if r["Type"] == "AWS::IAM::Role"
             and k.startswith(role_prefix)]
    out = set()
    for r in res.values():
        if r["Type"] == "AWS::IAM::Policy" and any(
                ref.get("Ref") in roles for ref in r["Properties"]["Roles"]):
            for st_ in r["Properties"]["PolicyDocument"]["Statement"]:
                a = st_["Action"]
                out |= set(a if isinstance(a, list) else [a])
    return out


def test_model_alerts_and_ops_alarms_are_separate_topics(stacks):
    web, _ = stacks
    names = {r["Properties"]["TopicName"]
             for r in web.to_json()["Resources"].values()
             if r["Type"] == "AWS::SNS::Topic"}
    assert names == {"ml-auto-model-alerts", "ml-auto-ops-alarms"}
    c = _containers(web)
    for name in ("web", "batch"):
        assert "ALERTS_TOPIC_ARN" in {e["Name"] for e in
                                      c[name]["Environment"]}


def test_only_web_and_batch_publish_alerts(stacks):
    web, _ = stacks
    assert "sns:Publish" in _policy_actions(web, "WebTaskTaskRole")
    assert "sns:Publish" in _policy_actions(web, "BatchTaskTaskRole")
    assert "sns:Publish" not in _policy_actions(web, "ApiTaskTaskRole")


def test_platform_alarms_notify_ops(stacks):
    web, _ = stacks
    # The platform alarms (autoscaling adds its own, which act on scaling
    # policies rather than notify anyone).
    alarms = [r["Properties"] for r in web.to_json()["Resources"].values()
              if r["Type"] == "AWS::CloudWatch::Alarm"
              and str(r["Properties"].get("AlarmDescription", ""))
              .startswith("ML Automation:")]
    assert len(alarms) == 6
    ops = next(k for k, r in web.to_json()["Resources"].items()
               if r["Type"] == "AWS::SNS::Topic"
               and r["Properties"]["TopicName"] == "ml-auto-ops-alarms")
    for a in alarms:
        assert a["AlarmActions"] == [{"Ref": ops}]
        assert a["OKActions"] == [{"Ref": ops}]


def test_jobs_queue_encrypted_with_dead_letters(stacks):
    web, _ = stacks
    queues = [r["Properties"] for r in web.to_json()["Resources"].values()
              if r["Type"] == "AWS::SQS::Queue"]
    assert len(queues) == 2
    assert all(q.get("SqsManagedSseEnabled") for q in queues)
    main = next(q for q in queues if "RedrivePolicy" in q)
    assert main["RedrivePolicy"]["maxReceiveCount"] == 5


def test_workers_scale_from_zero_on_queue_depth(stacks):
    web, _ = stacks
    c = _containers(web)
    assert c["worker"]["Command"] == ["python", "-m", "worker.trainer"]
    res = web.to_json()["Resources"]
    worker = next(r["Properties"] for k, r in res.items()
                  if r["Type"] == "AWS::ECS::Service"
                  and k.startswith("Worker"))
    assert worker["DesiredCount"] == 0
    web.has_resource_properties("AWS::ApplicationAutoScaling::ScalableTarget",
                                {"MinCapacity": 0, "MaxCapacity": 4})
    web.has_resource_properties("AWS::ApplicationAutoScaling::ScalingPolicy",
                                {"PolicyType": "StepScaling"})
    alarms = [r["Properties"] for r in res.values()
              if r["Type"] == "AWS::CloudWatch::Alarm"
              and any(m.get("Expression") == "visible + inflight"
                      for m in r["Properties"].get("Metrics", []))]
    assert alarms, "the scaling alarms read queued + in-flight jobs"


def test_worker_may_protect_its_task(stacks):
    web, _ = stacks
    assert "ecs:UpdateTaskProtection" in _policy_actions(web,
                                                         "WorkerTaskTaskRole")


def test_web_sends_jobs_but_does_not_consume(stacks):
    web, _ = stacks
    acts = _policy_actions(web, "WebTaskTaskRole")
    assert "sqs:SendMessage" in acts and "sqs:ReceiveMessage" not in acts
    assert "sqs:ReceiveMessage" in _policy_actions(web, "WorkerTaskTaskRole")
