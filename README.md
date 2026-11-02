# Machine Learning

A Streamlit app for training, comparing, using and monitoring
machine-learning models on any tabular file (.csv / .xlsx / .xls) or a
Snowflake table.

## Flow

A process bar at the top of every page shows the steps in order, highlights
the one you are on, ticks the ones before it and greys out any that cannot
be opened yet (Results before training). Click a step to go to it.

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
  table, then download the predictions or write them to Snowflake.
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

## Deploying to AWS

See [docs/architecture.md](docs/architecture.md) for the design and
[infra/README.md](infra/README.md) for the runbook (AWS CDK, Python).

## Tests

```bash
python -m pytest -q tests          # app
python -m pytest -q infra/tests    # infrastructure (needs infra/requirements.txt)
```

## Run

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
```

(On macOS / Linux use `.venv/bin/python`.)

## Layout

```
app.py                     entry point: start page and page routing
components/nav.py          process bar shown on every page
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
