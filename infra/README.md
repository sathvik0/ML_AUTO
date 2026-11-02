# Deploying ML Automation to AWS

The design and its reasons are in [docs/architecture.md](../docs/architecture.md).
This is the runbook. Two stacks:

- **MlAutoWeb** — VPC, ECS Fargate (the app), HTTPS ALB with Cognito sign-in
  on every request, WAF, Secrets Manager, logs.
- **MlAutoDeploy** — the role GitHub Actions deploys with (OIDC; no AWS
  keys stored anywhere).

## Before you start

- An AWS account, and the AWS CLI signed in as an administrator for the
  one-time setup (`aws configure sso` or `aws login`).
- A domain in **Route 53** (the app needs HTTPS for sign-in), e.g. the app
  at `ml.example.com` in the hosted zone `example.com`.
- Node.js 22 and Python 3.12. Docker running (the image is built at deploy).

```bash
cd infra
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt     # macOS/Linux: .venv/bin/pip
```

## One-time setup

```bash
# 1. Prepare the account for CDK (once per account and region)
npx aws-cdk@2 bootstrap aws://ACCOUNT_ID/us-east-1

# 2. The GitHub deploy role
npx aws-cdk@2 deploy MlAutoDeploy -c domainName=x -c hostedZoneName=x -c cognitoDomain=x
#    → note the DeployRoleArn output

# 3. The application (first deploy from your machine; later ones from GitHub)
npx aws-cdk@2 deploy MlAutoWeb \
  -c domainName=ml.example.com \
  -c hostedZoneName=example.com \
  -c cognitoDomain=ml-auto-yourcompany      # globally unique prefix
```

The first deploy takes ~15 minutes (certificate validation, NAT, image
build). Outputs: `Url`, `UserPoolId`, `AppKeysSecret`, `LogGroup`.

### Set the API keys

```bash
aws secretsmanager put-secret-value --secret-id ml-auto/app-keys \
  --secret-string '{"ANTHROPIC_API_KEY":"sk-ant-...","OPENAI_API_KEY":""}'
aws ecs update-service --cluster <cluster> --service <service> --force-new-deployment
```

The tasks read the secret at start, hence the redeploy.

### Snowflake for batch jobs (optional)

Scheduled jobs read and write Snowflake with the deployment's own service
account (never a user's sign-in, and never shown in the app):

```bash
aws secretsmanager put-secret-value --secret-id ml-auto/snowflake-service \
  --secret-string '{"account":"xy12345.us-east-1","user":"SVC_ML","password":"...",
                    "role":"ML_BATCH","warehouse":"WH_XS","database":"ANALYTICS","schema":"ML"}'
```

Give that role read access to the source schemas and create/insert on the
schema scores are written to — nothing more.

### Add the first admin

Sign-up is invite-only. Cognito emails each new user a temporary password.

```bash
POOL=<UserPoolId output>
aws cognito-idp admin-create-user --user-pool-id $POOL \
  --username you@example.com \
  --user-attributes Name=email,Value=you@example.com Name=email_verified,Value=true Name=name,Value="Your Name"
aws cognito-idp admin-add-user-to-group --user-pool-id $POOL \
  --username you@example.com --group-name admin
```

Roles are the groups `admin`, `analyst`, `viewer`; a user in none is a
viewer. What each may do is in `utils/auth.py`.

### Deploy from GitHub from now on

In the repository's **Settings → Secrets and variables → Actions →
Variables**, add `AWS_DEPLOY_ROLE_ARN` (from step 2), `AWS_REGION`,
`DOMAIN_NAME`, `HOSTED_ZONE_NAME`, `COGNITO_DOMAIN`. Create an environment
named `production` (add required reviewers to gate deploys). Then run
**Actions → Deploy → Run workflow** on `main`.

## Single sign-on with your company's identity provider

Cognito federates, so users sign in with Okta, Entra ID (Azure AD) or Google
Workspace and the app is unchanged. In the Cognito console: **User pool →
Sign-in experience → Federated identity provider sign-in → Add** (SAML or
OIDC), map the IdP's email and name, and map an IdP group claim to the
`custom:` attribute or use a pre-token-generation Lambda to place users in
`admin` / `analyst` / `viewer`. Then add the provider to the app client's
identity providers. (This will move into the stack once the IdP is chosen.)

## The model API and batch jobs

- `https://<host>/api/health` → `{"status":"ok"}` (no sign-in on `/api/*`;
  callers use project API keys created on the project page).
- `https://<host>/api/docs` — interactive reference.
- Batch jobs run from the `BatchSchedule` rule every 15 minutes; their logs
  are in the same log group under the `batch/` prefix.

## Checks after a deploy

1. `curl https://ml.example.com/_stcore/health` → redirected to sign-in (the
   ALB protects every path, health included, from outside).
2. Sign in → the app shows your name and role at the top right.
3. **Sign out** → you land on the sign-in page; Back does not get you in.
4. As a `viewer`: Train, Feature engineering, Use own model and Snowflake
   are unavailable with an explanation.
5. Leave a page open for 10 minutes, then use it: still connected (idle
   timeout and stickiness are right).

## Costs

See docs/architecture.md. `cdk destroy MlAutoWeb` removes the stack; the
user pool and the ALB log bucket are retained on purpose (users and audit
history) and must be deleted by hand.
