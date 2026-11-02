"""The role GitHub Actions deploys with, through OIDC: short-lived
credentials, no stored AWS keys. It may only assume the roles `cdk
bootstrap` created, and only from this repository's main branch (or a
manual run of the deploy workflow on it)."""
from aws_cdk import CfnOutput, Stack, aws_iam as iam
from constructs import Construct

GITHUB = "token.actions.githubusercontent.com"


class DeployStack(Stack):
    def __init__(self, scope: Construct, cid: str, *, repo: str,
                 **kwargs) -> None:
        super().__init__(scope, cid, **kwargs)
        provider = iam.OpenIdConnectProvider(
            self, "GitHub", url=f"https://{GITHUB}",
            client_ids=["sts.amazonaws.com"])
        role = iam.Role(
            self, "DeployRole", role_name="ml-auto-github-deploy",
            assumed_by=iam.WebIdentityPrincipal(
                provider.open_id_connect_provider_arn,
                conditions={
                    "StringEquals": {f"{GITHUB}:aud": "sts.amazonaws.com"},
                    "StringLike": {f"{GITHUB}:sub":
                                   f"repo:{repo}:ref:refs/heads/main"}}),
            description="GitHub Actions deploys ML Automation with CDK")
        role.add_to_policy(iam.PolicyStatement(
            actions=["sts:AssumeRole"],
            resources=[f"arn:aws:iam::{self.account}:role/cdk-*"]))
        CfnOutput(self, "DeployRoleArn", value=role.role_arn)
