# Architecture: hosting ML Automation on AWS

Status: **accepted** (2026-11-02). This is the target the roadmap builds
towards; each roadmap item below says which part of it it delivers.

## Context

The app is a **stateful Streamlit server**. Each user's work — the loaded
file, preparation steps, trained models, the assistant's conversation —
lives in one server process's memory for as long as the browser tab is open,
and the browser talks to it over a long-lived **WebSocket**. Training runs
inside that process today.

That shapes everything:

1. A user must keep reaching the **same container** (sticky sessions), and
   the load balancer must not drop an idle socket.
2. Work must **survive** a refresh, a deploy or a crashed task — so it has to
   move out of process memory into a database and object storage.
3. Heavy work (training, batch scoring, drift runs) must **leave the web
   process**, or one user's 20-minute tuning run starves everyone else.
4. Enterprise buyers need **SSO, roles, an audit trail and a model API**.

## Decision

AWS, defined in code with the **AWS CDK (Python)**, deployed by **GitHub
Actions** through OIDC (no long-lived AWS keys anywhere).

```
                 Route 53 ─▶ ACM certificate
                                │
 Users ──HTTPS──▶ Application Load Balancer ── WAF (managed rules, rate limit)
                  │  • authenticate-cognito on every request
                  │  • sticky sessions, idle timeout 3600 s
                  │  • /logout ─▶ Lambda (clears the ALB session, Cognito logout)
                  │  • /api/*  ─▶ Model API service (item 3)
                  ▼
        ┌─────────────── VPC, 2 AZs, private subnets ───────────────┐
        │                                                           │
        │  ECS Fargate: web (Streamlit)      ECS Fargate: api        │
        │     │  reads user + groups from       (FastAPI, item 3)    │
        │     │  ALB-signed OIDC headers                             │
        │     │                                                      │
        │     ├──▶ SQS jobs queue ──▶ ECS Fargate: worker (item 5)   │
        │     │                       autoscales on queue depth      │
        │     │                                                      │
        │     ├──▶ Aurora PostgreSQL Serverless v2 (item 2)          │
        │     │      projects, runs, model registry, audit log       │
        │     └──▶ S3 (KMS) datasets, models, reports (item 2)       │
        │                                                           │
        │  EventBridge Scheduler ─▶ worker: batch scoring,           │
        │      scheduled drift checks (items 3–4) ─▶ SNS ─▶ email/Slack│
        └───────────────────────────────────────────────────────────┘
          Secrets Manager (Anthropic, OpenAI, Snowflake) · CloudWatch
          logs/metrics/alarms · ECR images · VPC endpoints for S3,
          ECR, Secrets Manager, Logs · NAT for Snowflake/Anthropic/OpenAI
```

Identity: an **Amazon Cognito user pool** with three groups — `admin`,
`analyst`, `viewer`. Cognito federates to the customer's IdP (Okta, Azure AD
/ Entra ID, Google Workspace) over SAML or OIDC, so the app sees one identity
format whatever the customer uses. The ALB authenticates every request
before it reaches a container; the app verifies the ALB's signed headers
itself (defence in depth) and reads the user's groups from the Cognito access
token the ALB forwards.

### Why these choices

| Choice | Over | Because |
| --- | --- | --- |
| ECS Fargate | EKS | Same container model, no cluster to run; EKS pays off only with many services and a platform team. |
| ECS Fargate | App Runner, Lambda | Streamlit needs long-lived WebSockets and sticky sessions; Lambda cannot hold a socket, App Runner's session affinity and timeouts do not fit. |
| ALB + Cognito auth | Auth inside the app | No unauthenticated request ever reaches Streamlit; SSO federation for free; the app only reads verified headers. |
| Aurora PostgreSQL Serverless v2 | DynamoDB | Projects, runs, permissions and audit are relational and queried many ways; Serverless v2 scales down when idle. |
| S3 | EFS | Datasets and models are write-once objects; S3 is cheaper, versioned, and lifecycle-managed. |
| SQS + Fargate worker | Step Functions, AWS Batch | Jobs are single steps today; a queue plus an autoscaled worker is the least machinery. Batch is the upgrade path for GPU or very long jobs. |
| CDK (Python) | Terraform, CloudFormation | Same language as the app; typed constructs for ECS/ALB/Cognito; synthesises to CloudFormation. |
| GitHub Actions + OIDC | Stored AWS keys | Short-lived credentials scoped to one role; nothing to leak. |

### Service settings that matter

| Where | Setting | Value | Why |
| --- | --- | --- | --- |
| Target group | Stickiness | `lb_cookie`, 1 day | Session state is in one task's memory (until item 2 persists it). |
| Target group | Health check | `/_stcore/health` | Fast; `/` returns the page shell. |
| Target group | Deregistration delay | 300 s | Draining a task ends live sessions. |
| Listener | Idle timeout | 3600 s | The 60 s default drops users who are reading. |
| Service | Min healthy | 100% | Never remove capacity mid-session. |
| Task | Size | 1 vCPU / 4 GB to start (`webCpu` / `webMemory`) | Memory is the limit (~0.5 GB per 300k-row session). |
| Autoscaling | Target | 60% memory | Scale out freely, in rarely. |

## Roadmap mapping

| # | Item | Delivers |
| --- | --- | --- |
| 1 | SSO and roles | Container image, CDK stack (VPC, ECS, ALB, Cognito, WAF, secrets, logout), app-side identity and permissions. |
| 2 | Persistence | Aurora PostgreSQL Serverless v2 (scales to zero, encrypted, 7-day backups) and a KMS-encrypted, versioned S3 bucket: projects, autosaved datasets/steps/choices/runs, a model registry with versions, per-project sharing (owner/editor/viewer). Schema managed by Alembic. |
| 3 | Deployment | Stages (none → staging → production → archived) with four-eyes approval for production; a FastAPI model API service on `/api/*` (project API keys, hashed; 1 vCPU / 2 GB, CPU autoscaling); batch scoring jobs (project dataset, upload or Snowflake → storage or Snowflake, cron schedules) run by a scheduled Fargate Spot task (hourly by default) with its own Snowflake service account. |
| 4 | Monitoring | Monitors check a production model against newer data on a schedule (run by the same scheduled task): drift on its own features and, when outcomes are present, accuracy against its test score → ok / warning / alert. Alerts go to an SNS topic (email; Slack or Teams through AWS Chatbot) and an optional per-monitor Slack/Teams webhook (allow-listed hosts only). Optional retraining registers a **staging** version — production still needs approval — and never stacks more than one waiting. Platform alarms (5xx, unhealthy tasks, API latency) go to a separate ops topic. |
| 5 | Scale | Training runs as a background job: the web app records it (the database is the source of truth) and sends its id to an encrypted SQS queue with a dead-letter queue; a worker service (4 vCPU / 16 GB, Fargate Spot) scales from zero on queued + in-flight jobs, claims each job atomically, reports progress (also its heartbeat and how it learns of a cancel), and protects its task from scale-in while it works. Jobs interrupted by a lost worker restart automatically (up to 3 attempts); failed or cancelled jobs can be retried. Locally a thread pool stands in for the queue. |
| 6 | Governance | An append-only, SHA-256 hash-chained audit trail in Aurora, written in the same transaction as each action (a PostgreSQL trigger refuses UPDATE/DELETE; `prev_hash` is unique so the chain cannot fork; history survives project deletion; also logged for a SIEM). Fairness checks per group of sensitive attributes on out-of-sample predictions (four-fifths rule, TPR/FPR gaps; MAE ratio and bias gap for regression), with owner acceptance and a reason; a per-project policy can require a passing or accepted check before production. Admins read the whole log and verify the chain; owners see their project's activity. |

## Cost (us-east-1, idle-to-light use, indicative)

≈ $185–195/month for a pilot; the breakdown is in
[deployment-plan.md](deployment-plan.md). Aurora only pauses when nothing
holds a connection: the app opens one per unit of work (no pool), and the
runner wakes it hourly. VPC endpoints cut NAT data charges for S3 and ECR.
