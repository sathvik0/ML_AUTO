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
    assert "&lt;b&gt;" in shell.status("<b>")
    at = AppTest.from_function(_table_page).run()
    html = at.markdown[0].value
    assert "&lt;script&gt;" in html and "<script>" not in html
    assert "chip chip-info" in html


def _table_page():
    from components import shell
    shell.table([("Name", "n"), ("Stage", "s")],
                [{"n": "<script>x</script>", "s": shell.status("staging")}])
