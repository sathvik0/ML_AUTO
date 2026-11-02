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
