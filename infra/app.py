#!/usr/bin/env python3
"""CDK app: `MlAutoWeb` (the application) and `MlAutoDeploy` (the role
GitHub Actions deploys with). See infra/README.md."""
import os

import aws_cdk as cdk

from ml_auto.deploy_stack import DeployStack
from ml_auto.web_stack import WebStack

app = cdk.App()
env = cdk.Environment(account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
                      region=os.environ.get("CDK_DEFAULT_REGION",
                                            "us-east-1"))
WebStack(app, "MlAutoWeb", env=env,
         description="ML Automation: web tier (ECS, ALB + Cognito, WAF)")
DeployStack(app, "MlAutoDeploy", env=env,
            repo=app.node.try_get_context("githubRepo")
            or "sathvik0/ML_AUTO",
            description="ML Automation: GitHub Actions deploy role")
cdk.Tags.of(app).add("app", "ml-auto")
app.synth()
