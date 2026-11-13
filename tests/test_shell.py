"""The app shell: the home page's numbers, and the real pages rendering
inside the sidebar and top bar without an error."""
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from logic import deployment, ml, ml_eval, overview, projects
from utils import db

ANA = {"sub": "a", "email": "ana@example.com", "role": "analyst"}
BEN = {"sub": "b", "email": "ben@example.com", "role": "analyst"}
# Absolute: Streamlit versions differ on what a relative path is relative to.
APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL",
                       f"sqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(tmp_path / "objects"))
    monkeypatch.setenv("AUTH_MODE", "dev")
    monkeypatch.setenv("JOBS_MODE", "none")
    monkeypatch.delenv("STORAGE_BUCKET", raising=False)
    db.migrate()


def _trained():
    df = pd.read_csv("sample_data/insurance_churn.csv").head(600)
    config = {"task": "classification", "target": "Churned",
              "positive": "Yes", "target_rule": None,
              "numeric": ["Tenure_Years", "Service_Calls_12M"],
              "categorical": [],
              "prep": {"num_missing": "median", "cat_missing": "category",
                       "scale": "standard", "rare_share": 0.0,
                       "missing_flags": False, "log_target": False,
                       "pca": 0, "cat_encoding": "onehot"},
              "validation": {"test_size": 0.25, "folds": 0,
                             "stratify": True, "group": None},
              "seed": 1, "metric": None, "balance": False,
              "models": [{"key": "logistic", "params": ml.default_params(
                  "classification", "logistic"), "search": None}],
              "k_scan": None}
    run = ml.train(df, config, lambda f, t: None)
    run["prep_steps"], run["importance"], run["frame"] = [], {}, df
    return run


def test_summary_counts_only_what_the_user_can_see():
    assert overview.summary(ANA) == {"projects": [], "production": 0,
                                     "approvals": [], "monitors": [],
                                     "models": [], "runs": []}
    run = _trained()
    pid = projects.create_project(ANA, "Churn")["id"]
    rid = projects.save_run(ANA, pid, run)
    v = projects.register_model(
        ANA, pid, rid, ml_eval.model_bytes(run, ml.result(run, "logistic")),
        name="churn", model_key="logistic", task="classification",
        target="Churned")
    deployment.request_production(ANA, pid, v["id"])
    got = overview.summary(ANA)
    assert [p["name"] for p in got["projects"]] == ["Churn"]
    assert got["production"] == 0
    assert [(a["model"], a["to_stage"], a["project"])
            for a in got["approvals"]] == [("churn", "production", "Churn")]
    assert [r["best_model"] for r in got["runs"]] == ["logistic"]
    assert got["models"] == []              # registered, not yet staged
    deployment.set_stage(ANA, pid, v["id"], "staging")
    assert [(m["name"], m["stage"], m["monitor"])
            for m in overview.summary(ANA)["models"]] == [
        ("churn", "staging", None)]
    # Someone the project is not shared with sees none of it.
    assert overview.summary(BEN)["approvals"] == []
    assert overview.summary(BEN)["runs"] == []


def _labels(at):
    return [b.label for b in at.sidebar.button]


def test_pages_render_inside_the_shell():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert _labels(at)[:3] == ["Home", "Data", "Analysis"]
    assert "Audit log" in _labels(at)             # dev sign-in is an admin
    assert [m.label for m in at.metric] == [
        "Projects", "Models in production", "Approvals waiting",
        "Monitors to check"]
    analysis = next(b for b in at.sidebar.button if b.label == "Analysis")
    assert analysis.disabled                      # nothing loaded yet

    # The data page, then the sample: lands on Explore & prepare.
    next(b for b in at.sidebar.button if b.label == "Data").click().run()
    assert not at.exception
    assert any(m.value == "## Data" for m in at.markdown)
    next(b for b in at.button if b.label == "Try sample data").click().run()
    assert not at.exception
    assert at.session_state["page"] == "ml_eda"
    at.run()
    assert not at.exception
    steps = [b.label for b in at.button if (b.key or "").startswith(
        "ml_nav_")]
    assert steps[:2] == ["Data", "2. Explore & prepare"]

    # Home again keeps the analysis one click away.
    next(b for b in at.sidebar.button if b.label == "Home").click().run()
    assert not at.exception
    analysis = next(b for b in at.sidebar.button if b.label == "Analysis")
    assert not analysis.disabled
    analysis.click().run()
    assert at.session_state["page"] == "ml_eda"


def test_project_sections_from_the_sidebar():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.text_input(key="proj_landing_name").set_value("Shell test").run()
    next(b for b in at.button if b.key == "proj_landing_create").click().run()
    assert not at.exception
    assert _labels(at)[3:7] == ["Runs", "Models", "Monitoring & batch",
                                "Access & activity"]
    for label in ("Runs", "Models", "Monitoring & batch",
                  "Access & activity"):
        next(b for b in at.sidebar.button if b.label == label).click().run()
        assert not at.exception, label
        assert at.session_state["page"] == "ml_project"
        assert any(m.value == f"## {label}" for m in at.markdown), label


def test_table_escapes_text_and_keeps_chips():
    from components import shell
    assert "chip-good" in shell.status("production")
    assert "chip-bad" in shell.status("alert")
    assert ">OK<" in shell.status("ok")
    assert "&lt;b&gt;" in shell.status("<b>")
    at = AppTest.from_function(_table_page).run()
    html = at.markdown[0].value
    assert "&lt;script&gt;" in html and "<script>" not in html
    assert "chip chip-info" in html


def _table_page():
    from components import shell
    shell.table([("Name", "n"), ("Stage", "s")],
                [{"n": "<script>x</script>", "s": shell.status("staging")}])


def _results_page():
    from components import ml_results
    ml_results.render()


def test_results_page_cards_and_leaderboard():
    run = _trained()
    at = AppTest.from_function(_results_page, default_timeout=120)
    at.session_state["ml_run"] = run
    at.session_state["ml_df"] = run["frame"]
    at.run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert labels[0] == "Best model" and "Baseline (no model)" in labels
    assert at.metric[0].value == "Logistic regression"
    board = next(m.value for m in at.markdown if "ent-table" in m.value)
    assert "chip-good'>Best" in board and "chip-idle'>Baseline" in board


def _configure_page():
    from components import ml_config
    ml_config.render()


def test_configure_page_plan_panel():
    df = pd.read_csv("sample_data/insurance_churn.csv").head(800)
    at = AppTest.from_function(_configure_page, default_timeout=120)
    at.session_state["ml_source"] = df
    at.session_state["ml_df"] = df
    at.session_state["ml_steps"] = []
    at.run()
    assert not at.exception
    plan = next(m.value for m in at.markdown if "plan-rows" in m.value)
    assert "<span>Task</span><b>Classification</b>" in plan
    assert "<span>Target</span><b>Churned (Yes)</b>" in plan
    models = next(m.value for m in at.markdown if "plan-models" in m.value)
    assert "Logistic regression" in models
    train = next(b for b in at.button if b.key == "ml_train")
    assert not train.disabled
    # Tuning shows beside each model in the plan.
    next(r for r in at.radio if r.key == "ml_w_tune_mode").set_value(
        "random").run()
    assert not at.exception
    models = next(m.value for m in at.markdown if "plan-models" in m.value)
    assert "random search" in models
    # Clustering has no target, validation or combine panels to trip on.
    next(r for r in at.radio if r.key == "ml_w_task").set_value(
        "clustering").run()
    assert not at.exception
    plan = next(m.value for m in at.markdown if "plan-rows" in m.value)
    assert "<b>Clustering</b>" in plan and "Target" not in plan


def _project_page():
    from components import project_panel, shell
    shell.render("ml_project")
    project_panel.render()


def test_project_sections_show_chips_for_real_records():
    """A project with a run, a staged model and a pending approval: each
    section renders, with its statuses as chips."""
    dev = {"sub": "dev", "email": "dev@localhost", "role": "admin",
           "name": "Local developer"}
    run = _trained()
    pid = projects.create_project(dev, "Chips")["id"]
    projects.share(dev, pid, "ana@example.com", "editor")
    rid = projects.save_run(dev, pid, run)
    v = projects.register_model(
        dev, pid, rid, ml_eval.model_bytes(run, ml.result(run, "logistic")),
        name="churn", model_key="logistic", task="classification",
        target="Churned", metric="roc_auc", score=0.7)
    deployment.set_stage(dev, pid, v["id"], "staging")
    deployment.request_production(dev, pid, v["id"])

    def page(section):
        at = AppTest.from_function(_project_page, default_timeout=120)
        at.session_state["proj_section"] = section
        at.session_state["proj"] = {"id": pid, "name": "Chips",
                                    "role": "owner"}
        at.run()
        assert not at.exception, section
        return " ".join(m.value for m in at.markdown)

    runs = page("runs")
    assert "ent-table" in runs and "logistic" in runs
    models = page("models")
    assert "chip-info'>Staging" in models          # registry and "Now:"
    assert "chip-warn'>Pending" in models          # stage history
    assert "0.700" in models
    page("operations")                             # nothing scheduled yet
    access = page("access")
    assert "chip-good'>Owner" in access and "chip-info'>Editor" in access


def _explore_page():
    from components import ml_eda
    ml_eda.render()


def _features_page():
    from components import ml_features
    ml_features.render()


def _loaded(page):
    df = pd.read_csv("sample_data/insurance_churn.csv").head(800)
    at = AppTest.from_function(page, default_timeout=120)
    at.session_state["ml_source"] = df
    at.session_state["ml_df"] = df
    at.session_state["ml_steps"] = []
    return at.run()


def test_explore_page_cards_and_column_plan():
    at = _loaded(_explore_page)
    assert not at.exception
    assert [m.label for m in at.metric][:2] == ["Rows", "Columns"]
    plan = next(m.value for m in at.markdown if "ent-table" in m.value
                and "Customer_ID" in m.value)
    assert "chip-good'>Used" in plan
    assert "chip-idle'>Identifier" in plan            # Customer_ID is off
    assert any(b.key == "ml_e_continue" for b in at.button)


def test_feature_engineering_page_sections():
    at = _loaded(_features_page)
    assert not at.exception
    heads = [m.value for m in at.markdown if m.value.startswith("#### ")]
    for h in ("#### Build a column", "#### Columns", "#### Your data",
              "#### Explore the columns"):
        assert h in heads, h
    confirm = next(b for b in at.button if (b.key or "").endswith(
        "confirm"))
    assert confirm.disabled                         # nothing waiting yet


def test_configure_page_time_split():
    df = pd.read_csv("sample_data/insurance_churn.csv").head(800)
    df["Renewal_Date"] = (pd.Timestamp("2024-01-01") + pd.to_timedelta(
        range(len(df)), unit="D")).strftime("%Y-%m-%d")
    at = AppTest.from_function(_configure_page, default_timeout=120)
    at.session_state["ml_source"] = df
    at.session_state["ml_df"] = df
    at.session_state["ml_steps"] = []
    at.run()
    next(r for r in at.radio if r.key == "ml_w_split").set_value(
        "time").run()
    assert not at.exception
    assert next(s for s in at.selectbox if s.key == "ml_w_time_col").value \
        == "Renewal_Date"                       # the date column first
    assert next(s for s in at.selectbox if s.key == "ml_w_group").disabled
    plan = next(m.value for m in at.markdown if "plan-rows" in m.value)
    assert "latest as test" in plan and "walk-forward folds" in plan
    assert "<b>Renewal_Date</b>" in plan
    assert not next(b for b in at.button if b.key == "ml_train").disabled


def test_register_an_uploaded_model_from_results():
    import io
    import joblib
    from sklearn.linear_model import LogisticRegression
    from logic import ml_upload
    df = pd.read_csv("sample_data/insurance_churn.csv").head(600)
    cols = ["Tenure_Years", "Service_Calls_12M"]
    model = LogisticRegression().fit(df[cols], df["Churned"])
    buf = io.BytesIO()
    joblib.dump(model, buf)
    info, _ = ml_upload.read(buf.getvalue(), "outside.joblib")
    run, why = ml_upload.evaluate(df, info, {c: c for c in cols},
                                  "classification", "Churned", "Yes",
                                  prof=ml.profile(df))
    assert why is None
    dev = {"sub": "dev", "email": "dev@localhost", "role": "admin"}
    pid = projects.create_project(dev, "Outside model")["id"]
    at = AppTest.from_function(_results_page, default_timeout=120)
    at.session_state["ml_run"] = run
    at.session_state["ml_df"] = df
    at.session_state["proj"] = {"id": pid, "name": "Outside model",
                                "role": "owner"}
    at.run()
    assert not at.exception
    assert any("Registered as the file you uploaded" in c.value
               for c in at.caption)
    reg = next(b for b in at.button if (b.key or "").startswith("proj_reg_"))
    assert not reg.disabled
    reg.click().run()
    assert not at.exception
    got = projects.list_models(dev, pid)
    assert [(m["name"], m["model_key"]) for m in got] == [
        ("churned_model", "uploaded")]
