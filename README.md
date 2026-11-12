# Machine Learning

A Streamlit app for training, comparing, using and monitoring
machine-learning models on any tabular file (.csv / .xlsx / .xls) or a
Snowflake table.

## Flow

A sidebar moves between the areas of the app: **Home** (your projects,
approvals waiting, monitors to check, recent runs), **Data**, the current
**Analysis**, the open project's sections (Runs, Models, Monitoring & batch,
Access & activity) and, for admins, the Audit log. The top bar says where
you are, whether the work is saved, and who is signed in.

Inside an analysis, a row of tabs shows the steps in order, underlines the
one you are on, ticks the ones before it and greys out any that cannot be
opened yet (Results before training). Click a step to go to it.

1. **Data** — upload a file, load a Snowflake table or query, or try the
   sample (below).
2. **Explore & prepare** (EDA) — overview, columns, distributions, gaps,
   correlations, target and outliers; preparation steps (drop duplicates or
   missing rows, keep a range or values, log / square-root transforms,
   binning) that are replayed in the exported script.
3. **Feature engineering** (AI-assisted)
   - **AI feature ideas**: pick the target; Claude reads the columns and how
     strongly each already relates to it, and proposes new features as
     scripts. Each is run here and scored by its mutual information with the
     target, ranked against the existing columns; add it, or edit it first.
   - **Describe a column** in plain language ("claims paid per premium
     dollar"); Claude writes the pandas script, which you edit, preview and
     stage. Voice input is available when an OpenAI key is set.
   - Every script is checked against an allow-list (no imports, loops, file
     access or attributes outside a fixed pandas/numpy vocabulary) before it
     runs. Nothing is added until you confirm; confirmed columns become
     preparation steps, so exported scripts and newly scored files rebuild
     them.
4. **Configure & train**
   - **Assistant**: describe the objective in plain words ("the retention
     team can call 10% of customers; find the ones most likely to leave").
     It recommends the task, target, models and their settings, the metric
     to judge them by, validation, class balancing, preparation and
     encoding, with its reasons — and applies them with one click. Every
     value is checked against the page's options first.
   - Choose the task (classification, regression, clustering), the target,
     the features and the models; preparation (missing values, scaling,
     **categorical encoding: one-hot, ordinal or target encoding**, rare
     categories); split, cross-validation, class balancing and
     hyperparameter search. Leakage checks flag columns that predict the
     target almost on their own.
5. **Results** — leaderboard against a feature-blind baseline, metrics,
   ROC / PR / gains curves, confusion matrix, residuals, feature importance,
   SHAP and partial dependence, cluster profiles; score a new file; write
   predictions to Snowflake.

Beside the steps:

- **Use own model** — upload a model (this app's file, scikit-learn,
  XGBoost or LightGBM) and either **evaluate** it against actual values, or
  **predict** every row of this file, an uploaded file or a Snowflake
  table, then download the predictions or write them to Snowflake. An
  evaluated model can be **registered** in a project like one trained
  here: the file is kept as uploaded, with the column mapping and column
  types used in the evaluation, and goes through stages, approvals, the
  model API, batch jobs and monitoring (monitoring does not retrain it).
- **Drift analysis** — compare the file loaded here with newer data
  (upload, Snowflake, or the bundled recent-customers sample):
  - *Covariate drift*: PSI per column with Stable / Moderate / Major bands,
    Kolmogorov–Smirnov and chi-square tests, missing-value changes, new
    categories, distributions side by side, and a drift classifier whose ROC
    AUC measures drift across all columns at once.
  - *Correlational drift*: the change in Spearman correlations between
    numeric columns, the pairs that moved most, and how each feature's
    relationship with the target has moved (concept drift).

Downloads: predictions and the scored file (CSV), the leaderboard, the
trained model, a model card, a PowerPoint summary, scoring code and a
standalone Python script that rebuilds every model, or everything as a zip;
drift reports as CSV.

## Sample data

`sample_data/insurance_churn.csv`: 10,000 synthetic insurance customers
(demographics, policy, payments, claims, service contacts, digital
engagement). One file serves all three tasks:

- **Classification** — `Churned` (Yes/No): did the customer leave at renewal?
  About 22% did; price rises, service calls, late payments and short tenure
  drive it.
- **Regression** — `Annual_Premium` (driven by policy type, coverage, age,
  region and claims) or `Customer_Lifetime_Value`.
- **Clustering** — the behaviour columns fall into four customer segments
  (loyal families, young digital, price-sensitive, high-claims) that are not
  in the file.

`sample_data/insurance_churn_recent.csv`: 5,000 newer customers that drift
on purpose (larger price rises, more young digital customers, more of the
book in the West, auto-debit to card, higher engagement, more missing credit
scores, and late payments newly tied to price rises) — for Drift analysis.

Regenerate both with `python tools/make_sample_data.py` (seeded, so the
files are identical).

## Keys and connections

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` (keep it
out of version control), or set environment variables.

- `ANTHROPIC_API_KEY` — AI feature ideas, script generation and the
  assistant (Claude, `claude-opus-5-5`; declined requests are retried on a
  fallback model automatically). Without it you can still write scripts by
  hand and configure by hand.
- `OPENAI_API_KEY` — optional microphone on Feature engineering.
- `[snowflake]` — optional; pre-fills the connection form (account, user,
  role, warehouse, database, schema; password, key-pair or browser SSO on
  the form). Needs `snowflake-connector-python`.

## Sign-in and roles

In production the app sits behind an AWS load balancer that signs every user
in with Amazon Cognito (single sign-on with Okta, Entra ID or Google
Workspace through Cognito federation). The app verifies the load balancer's
signed headers itself and reads the user's role from their Cognito groups:

| Role | Can |
| --- | --- |
| viewer | load data, explore, run drift analysis, view results |
| analyst | + change the data, feature engineering, train, AI features, use own models, Snowflake |
| admin | + system administration (grows with later roadmap items) |

Locally the app runs with `AUTH_MODE=dev` (the default): a development user,
with a **View as** switcher in the account menu to see each role's view.

## Projects

Work is saved in **projects**. Create one on Home (or save the
current work from the project menu at the top), and while it is open everything
saves itself: the loaded dataset, the preparation steps, the configure
choices and every training run. The project page lists the runs (open any
of them on the results page again) and the **model registry** — register a
model from the results page and each registration becomes a new numbered
version you can download. Owners share a project by email as viewer, editor
or owner; what a person may do is the lower of their project role and their
sign-in role.

Locally, projects live in `data/app.db` (SQLite) and files in
`data/objects`; on AWS in Aurora PostgreSQL and S3 (`DATABASE_URL` or
`DB_HOST`…, and `STORAGE_BUCKET`, set by the stack). The schema is managed
with Alembic (`migrations/`) and migrated when the app starts.

## Putting models to work

From a project's page:

- **Stages and approvals** — move a registered version to *staging* (or back,
  or *archive* it) directly; *production* needs a request and approval by a
  project owner other than the requester. Approving archives the model's
  previous production version. Every decision is kept in the stage history.
- **Model API** — owners create API keys (shown once; only a hash is
  stored). Other systems then call

  ```bash
  curl -X POST https://<host>/api/v1/models/<model>/predict \
    -H "Authorization: Bearer mlk_..." -H "Content-Type: application/json" \
    -d '{"rows": [{"Age": 41, "Region": "West"}]}'
  ```

  for the production version (`?stage=staging` or `?version=N` to choose),
  up to 10,000 rows a call; reference docs at `/api/docs`. A key reaches its
  own project's models only. Locally: `uvicorn api.main:app --port 8000`.

  Each row is checked against what the model was trained on:
  `ML_Warnings` names a value outside the training range, a category never
  seen, or a gap (null when nothing is off). `?explain=3` adds
  `ML_Reasons`, the three inputs that moved the prediction most against
  their typical training value (up to 1,000 rows a call). Every request is
  kept: the project's Models section lists them under **Recent
  predictions** (who called, which version, rows, warnings, time) with the
  rows and answers to download, and the response carries its `request_id`.
  Model files registered before these checks score as before, without
  them; register again to get them. Batch jobs get the warnings too, and
  keep their scored output as before.
- **Batch scoring** — jobs score the project's dataset, an uploaded file or
  a Snowflake query with a model's production (or staging) version, and
  write a downloadable file or a Snowflake table — on demand or on a
  schedule. `python -m worker.batch --due` runs whatever is due (on AWS a
  scheduled task runs it every 15 minutes).

## Training in the background

With a project open, **Train in the background** (Configure & train) queues
the run instead of training in the page: a live panel shows its progress,
**Cancel** stops it at the next model fit, and **Open results** opens it
when it is done — you can leave the page or close the tab meanwhile. The
project page lists recent training jobs (open, cancel, retry).

On AWS the job goes to an SQS queue and a worker service that scales from
zero (`python -m worker.trainer`); a worker that stops mid-run is noticed
and the job restarts on another, up to three times. Locally a background
thread runs it (`JOBS_THREADS`, 1 by default). **Train models** still
trains in the page, which suits quick runs and unsaved work.

## Monitoring models in production

A **monitor** (project page → Monitoring) checks a model's production
version against newer rows — the project's dataset, an uploaded file or a
Snowflake query — on demand or on a schedule:

- **Drift** on the model's own features, against the exact data it was
  trained on: columns with major or moderate PSI, the drift classifier's
  AUC, and the relationships that moved.
- **Accuracy**, when the newer rows include the actual outcome: the model's
  metric on them against its score at training.

Each check ends **ok**, **warning** or **alert** by the monitor's
thresholds, and the trend is charted over time. Alerts go to the
deployment's alert topic (email, or Slack/Teams via AWS Chatbot) and, if
set, the monitor's own Slack or Teams webhook. A monitor can also
**retrain on alert**: the model's winning settings on the training data
plus the newly labelled rows, registered as a *staging* version for review
— production still needs approval, and only one retrained version waits at
a time. Models trained elsewhere and registered from Use own model are not
retrained; their monitors compare new data with the file they were
evaluated on. `python -m worker.batch --due` runs due monitors with the batch
jobs.

## Governance

- **Audit trail** — every meaningful action (sign-in, project and sharing
  changes, datasets, runs, model registration, stage requests and
  approvals, API keys, batch jobs, monitors, background jobs, fairness
  checks) is recorded with who, when, what and on which project. Events are
  append-only and **hash-chained** (each covers the one before), written in
  the same transaction as the action; admins open **Audit log** from the
  account menu, filter, export and **verify the chain**. Project owners see
  their project's **Activity**.
- **Fairness** — on the project page, check a model version across groups
  of sensitive attributes (gender, age band, region…), on its out-of-sample
  predictions: selection-rate ratio (four-fifths rule), true- and
  false-positive-rate gaps for classifiers; error ratio and bias gap for
  regression. Groups' actual rates are shown beside the predictions, so a
  real difference can be told from model bias; an owner can accept a
  flagged report with a written reason.
- **Policy** — a project can require a passing (or accepted) fairness check
  before a version may be requested for production; approvers see the
  fairness result beside each request.

## Deploying to AWS

See [docs/architecture.md](docs/architecture.md) for the design and
[infra/README.md](infra/README.md) for the runbook (AWS CDK, Python).

## Tests

```bash
python -m pytest -q tests          # app
python -m pytest -q infra/tests    # infrastructure (needs infra/requirements.lock)
```

## Run

```bash
python -m venv .venv
.venv\Scripts\python -m pip install --require-hashes -r requirements-dev.lock
.venv\Scripts\python -m streamlit run app.py
```

(On macOS / Linux use `.venv/bin/python`.)

## Dependencies

`requirements.txt` (and `requirements-dev.txt`, `infra/requirements.txt`)
list what the app needs, as ranges. The `.lock` files beside them pin every
package, its dependencies included, to one version with its hashes, for
Linux, macOS and Windows alike. The Docker image, CI and the deploy
workflow install only from the locks, so they all get the same packages,
and CI fails when a lock no longer covers its requirements
(`python tools/check_locks.py`).

After changing a requirements file, or to take newer versions, regenerate
the locks with [uv](https://docs.astral.sh/uv/) and run the tests:

```bash
uv pip compile requirements.txt --universal --python-version 3.12 --generate-hashes -o requirements.lock
uv pip compile requirements.txt requirements-dev.txt --universal --python-version 3.12 --generate-hashes -o requirements-dev.lock
cd infra && uv pip compile requirements.txt --universal --python-version 3.12 --generate-hashes -o requirements.lock
```

## Layout

```
app.py                     entry point: home, data page and page routing
components/shell.py        sidebar and top bar around every page
components/home.py         home dashboard (logic/overview.py: its numbers)
components/nav.py          the steps of an analysis, as tabs
components/ml_eda.py       Explore & prepare
components/ml_features.py  Feature engineering (AI ideas, AI column builder)
components/voice_input.py  optional voice prompt for it
components/ml_config.py    Configure & train
components/assistant_panel.py  the configuration assistant
components/ml_results.py   Results
components/ml_evaluate.py  Use own model (evaluate or predict)
components/drift_page.py   Drift analysis
components/snowflake_panel.py  Snowflake connection, picker and writer
components/ml_summary.py   PowerPoint summary
logic/claude.py            the structured Claude call the AI features share
logic/derive.py            script generation, feature ideas, the sandbox
logic/ml_assistant.py      the assistant's context and turns
logic/ml_drift.py          covariate and correlational drift
logic/snowflake_io.py      Snowflake reads and writes
logic/ml.py                profiling, leakage checks, training
logic/ml_recipe.py         model / pipeline recipes (embedded in exported scripts)
logic/ml_prep.py           preparation steps
logic/ml_explore.py        EDA statistics
logic/ml_eval.py           metrics, explanations, exports
logic/ml_upload.py         loading, checking and predicting with an uploaded model
logic/schema.py            column-type detection
utils/                     logging, error guard, file loading, secrets, scrolling
tools/make_sample_data.py  generates the sample datasets
```
