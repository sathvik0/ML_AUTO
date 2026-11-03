# Deployment plan: ML Automation on AWS

Prepared 2026-11-03 for the stack on `main` after PR #7 (roadmap items 1–6).
Nothing here has been created yet; every step that costs money is marked
and waits for your go-ahead.

## 1. What gets created

One CloudFormation stack (`MlAutoWeb`) plus a small one for GitHub deploys
(`MlAutoDeploy`, free):

| Piece | Size as built | Runs |
| --- | --- | --- |
| Web app (Streamlit) | Fargate 2 vCPU / 8 GB | always, 1–4 tasks |
| Model API (FastAPI) | Fargate 1 vCPU / 2 GB | always, 1–6 tasks |
| Training workers | Fargate 4 vCPU / 16 GB | only while jobs wait or run (0–4) |
| Batch + monitoring runner | Fargate 2 vCPU / 8 GB | every 15 min, ~1–2 min each |
| Database | Aurora PostgreSQL Serverless v2, 0–4 ACU | see §3 |
| Load balancer + WAF | 1 ALB, 3 AWS managed rule groups + rate limit | always |
| Network | VPC across 2 AZs, 1 NAT gateway | always |
| Storage | S3 (KMS, versioned), ALB logs bucket | always |
| Identity | Cognito user pool (invite-only) | always |
| Also | 3 secrets, 1 KMS key, 2 SQS queues, 2 SNS topics, 8 alarms, logs, 1 Lambda | |

## 2. Monthly cost

US East (N. Virginia) list prices as I know them; prices change, so confirm
in the [AWS Pricing Calculator](https://calculator.aws/) before deciding.
"Light use" means a handful of users in working hours, ~40 worker-hours of
training and a few GB of data a month.

| Item | As built today | After the §3 fixes |
| --- | ---: | ---: |
| Web app, 24/7 | $85 | $43 (1 vCPU / 4 GB) |
| Model API, 24/7 | $36 | $36 |
| Training workers (~40 h) | $9 | $3 (Fargate Spot) |
| Batch / monitoring runner | $8 (every 15 min) | $1 (hourly, Spot) |
| Aurora Serverless v2 | $46 (never pauses: 0.5 ACU × 730 h) | $15–25 (pauses when idle) |
| Load balancer | $22 | $22 |
| NAT gateway + data | $34 | $34 |
| Public IPv4 addresses (ALB ×2, NAT) | $11 | $11 |
| WAF | $10 | $10 |
| CloudWatch (logs, alarms, Container Insights) | $15 | $5 (Insights off) |
| Secrets Manager, KMS | $3 | $3 |
| S3, ECR images | $2 | $2 |
| Route 53 hosted zone | $0.50 | $0.50 |
| Cognito (under 10,000 monthly users), SQS, SNS, EventBridge, ACM | $0 | $0 |
| **Total** | **≈ $280 / month** | **≈ $185–195 / month** |

Not included:
- **Claude API usage** (your Anthropic bill, not AWS): roughly $0.05–0.10
  per assistant turn or feature-ideas request on `claude-opus-5-5`; $10–50
  a month at light use.
- **A domain**, if you need to buy one (~$15/year for a .com in Route 53).
- **Heavier use**: training is the variable — each worker-hour is $0.23
  ($0.07 on Spot); every extra always-on web task adds $43–85.
- **A second environment** (staging): roughly doubles the fixed part;
  see §5.

Ways to go lower later: a 1-year Compute Savings Plan (~20–50% off the
always-on tasks), Graviton/ARM tasks (~20% cheaper; needs an ARM image
build), or turning the API service off until it has callers.

## 3. Fix before deploying — done on branch `fix/pre-deploy-costs`

1. **Aurora never pauses.** The web and API services run around the clock
   and keep a pool of open database connections, and the runner connects
   every 15 minutes — so the "scales to zero" setting saves nothing.
   Fix: release idle connections in the app (`NullPool`, or a short pool
   recycle) and run the scheduler hourly unless a team needs 15 minutes.
2. **Training workers and the runner on Fargate Spot.** Jobs already
   survive a lost worker (the claim/heartbeat/retry design of item 5), so
   Spot's interruptions cost a restart, not a result — at ~70% off.
3. **Task sizes from settings**, not hard-coded: a pilot web task of
   1 vCPU / 4 GB, raised by changing one value.
4. **Container Insights optional** (off by default for a pilot).
5. **A budget alert in the stack** (AWS Budgets → email at 80% and 100% of
   a monthly amount you choose).
6. **An image retention rule** so old builds don't pile up in ECR.

## 4. Decisions only you can make

| # | Decision | My recommendation |
| --- | --- | --- |
| 1 | AWS account | A new account just for this app (under AWS Organizations if your company has one). Not a personal account with other things in it. |
| 2 | Region | `ap-south-1` (Mumbai) if your users and data are in India — lower latency, and data stays in-country; `us-east-1` otherwise. Costs differ by a few percent. |
| 3 | Domain | A subdomain you control, e.g. `ml.yourcompany.com`, delegated to a Route 53 hosted zone; or buy one in Route 53. |
| 4 | Sign-in page prefix | e.g. `mlauto-yourcompany` (must be globally unique). |
| 5 | People | The first admin's email; who gets model alerts; who gets ops alarms. |
| 6 | Monthly budget | e.g. $300, for the budget alert. |
| 7 | Company single sign-on | Later is fine: Okta / Entra ID / Google can be connected after launch. |
| 8 | Optional services | Your Anthropic key (AI features); a Snowflake service account (scheduled jobs). |

## 5. Step-by-step

Times are hands-on time. 💲 marks the step that starts billing.

**Phase 0 — account (you, ~1 hour)**
1. Create the AWS account; turn on MFA for the root user and lock it away.
2. Create an administrator in IAM Identity Center and sign in as that user
   from now on (`aws configure sso` on this machine).
3. Set up the domain (§4.3) and note the hosted zone name.

**Phase 1 — the §3 fixes (done; you review and merge the PR)**

**Phase 2 — prepare the account (from this machine; no Docker needed)**
4. `cd infra` → `python -m venv .venv` → install `requirements.txt`.
5. `npx aws-cdk@2 bootstrap aws://ACCOUNT/REGION` — sets up CDK in the
   account (an S3 bucket and ECR repository; cents a month).
6. `npx aws-cdk@2 deploy MlAutoDeploy …` — creates the GitHub deploy role
   (free). Note its `DeployRoleArn` output.

**Phase 3 — connect GitHub (you, 10 minutes)**
7. Repository → Settings → Environments → create `production`, with
   yourself as required reviewer.
8. Repository variables: `AWS_DEPLOY_ROLE_ARN`, `AWS_REGION`, `DOMAIN_NAME`,
   `HOSTED_ZONE_NAME`, `COGNITO_DOMAIN`.

**Phase 4 — 💲 first deploy (GitHub Actions, ~20–30 minutes)**
9. Actions → Deploy → Run workflow on `main`; approve it. GitHub builds the
   Docker image (this machine's Docker isn't needed) and creates the
   stack. Billing starts here.

**Phase 5 — switch it on (~30 minutes)**
10. Put the Anthropic key (and optionally OpenAI) in `ml-auto/app-keys`;
    optionally the Snowflake service account in
    `ml-auto/snowflake-service`; restart the services.
11. Create the first admin in Cognito and add them to `admin`.
12. Confirm the alert-email subscriptions (SNS sends a confirmation link).
13. Run the checks in §6.

**Phase 6 — before real company data**
14. Connect company single sign-on (Cognito federation).
15. A staging environment, in its own account, deployed before production.
16. Give the app a database role without schema rights, and export the
    audit log to S3 Object Lock (see infra/README.md → Audit trail).
17. Decide backups beyond Aurora's 7 days and S3 versioning (cross-region
    copy if the data warrants it).
18. A security review / penetration test, and your data-classification
    sign-off for what may be uploaded and sent to the Claude API.

## 6. Checks after the first deploy

| Check | Expect |
| --- | --- |
| Open the URL | Cognito sign-in page; after it, your name and role at the top |
| Sign out | Back to sign-in; Back button doesn't get you in |
| View as a viewer user | Train, Feature engineering, Use own model, Snowflake refused with a reason |
| Sample data → train in the background | Worker starts (1–2 min from zero), progress, Open results |
| Register → request production → second owner approves | Version in production |
| Create an API key → `curl` the predict endpoint | 200 with predictions; `/api/docs` loads |
| Batch job, monitor (Check now) | Scores to download; drift result and an alert email |
| Admin → Audit log → Verify chain | Chain intact |
| Leave a page idle 15 minutes, then use it | Still connected |
| CloudWatch alarms | All `OK` |

## 7. Undo

`npx aws-cdk@2 destroy MlAutoWeb` removes the stack. Kept on purpose, and
deleted by hand if you want them gone: the Cognito user pool (users), the
data and ALB-log buckets, the KMS key (scheduled deletion, 7–30 days) and a
final Aurora snapshot. Kept, these cost under $5 a month.
