"""Runs a team can keep track of: names and notes, two runs side by side,
and cloning a run's settings to change one thing and train again."""
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from streamlit.testing.v1 import AppTest

from logic import audit, ml, projects, run_compare
from logic.projects import AccessDenied
from utils import db

DEV = {"sub": "dev", "email": "dev@localhost", "role": "admin"}
VIC = {"sub": "v", "email": "vic@example.com", "role": "viewer"}
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL",
                       f"sqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(tmp_path / "objects"))
    monkeypatch.setenv("AUTH_MODE", "dev")
    monkeypatch.setenv("JOBS_MODE", "none")
    monkeypatch.delenv("STORAGE_BUCKET", raising=False)
    db.migrate()


@pytest.fixture(scope="module")
def df():
    return pd.read_csv("sample_data/insurance_churn.csv").head(1500)


def config(numeric, models=("logistic",), folds=3, widgets=None, C=1.0):
    out = {"task": "classification", "target": "Churned", "positive": "Yes",
           "target_rule": None, "numeric": list(numeric),
           "categorical": ["Region"],
           "prep": {"num_missing": "median", "cat_missing": "category",
                    "scale": "standard", "rare_share": 0.0,
                    "missing_flags": False, "log_target": False, "pca": 0,
                    "cat_encoding": "onehot"},
           "validation": {"test_size": 0.25, "folds": folds,
                          "stratify": True, "group": None},
           "seed": 1, "metric": "roc_auc", "balance": False,
           "models": [{"key": k, "params": {**ml.default_params(
               "classification", k), **({"C": C} if k == "logistic"
                                        else {})}, "search": None}
               for k in models], "k_scan": None}
    if widgets is not None:
        out["widgets"] = widgets
    return out


def _train(df, cfg, steps=()):
    run = ml.train(df, cfg, lambda f, t: None)
    run["prep_steps"], run["importance"], run["frame"] = list(steps), {}, df
    return run


def test_compare_lists_only_what_differs(df):
    a = _train(df, config(["Tenure_Years", "Service_Calls_12M"]))
    b = _train(df, config(["Tenure_Years", "Late_Payments_12M"],
                          models=("logistic", "tree"), C=0.1),
               steps=[{"kind": "drop_duplicates"}])
    got = run_compare.compare(a, b)
    assert got["features"] == {"only_a": ["Service_Calls_12M"],
                               "only_b": ["Late_Payments_12M"], "shared": 2}
    assert got["steps"]["only_b"] == [
        "Drop rows that repeat an earlier row in every column"]
    settings = {k: (x, y) for k, x, y in got["settings"]}
    assert settings["Logistic regression: C"] == ("1", "0.1")
    assert settings["Decision tree: max_depth"][0] == "—"   # only in B
    assert "Test share" not in settings                     # the same
    summary = {k: (x, y) for k, x, y in got["summary"]}
    assert summary["Models compared"] == (1, 2)
    models = {m: (x, y) for m, x, y in got["models"]}
    assert models["Decision tree"][0] is None and models["Decision tree"][1]
    assert got["metric"] == "ROC AUC"


def test_names_and_notes_saved_and_audited(df):
    pid = projects.create_project(DEV, "Runs")["id"]
    projects.share(DEV, pid, VIC["email"], "viewer")
    rid = projects.save_run(DEV, pid, _train(df, config(["Tenure_Years"])))
    assert projects.list_runs(DEV, pid)[0]["name"] == ""
    projects.describe_run(DEV, pid, rid, "  Tenure only ",
                          "Checks how far tenure goes alone.")
    r = projects.list_runs(DEV, pid)[0]
    assert (r["name"], r["note"]) == ("Tenure only",
                                      "Checks how far tenure goes alone.")
    with pytest.raises(AccessDenied):
        projects.describe_run(VIC, pid, rid, "x", "")
    with pytest.raises(AccessDenied, match="up to"):
        projects.describe_run(DEV, pid, rid, "x" * 201, "")
    e = audit.events(DEV, pid, action="run.describe")[0]
    assert e["details"]["name"] == "Tenure only"


def test_existing_runs_get_empty_names_on_upgrade(tmp_path):
    """A database at the previous revision keeps its runs."""
    from alembic import command
    from alembic.config import Config
    url = f"sqlite:///{(tmp_path / 'old.db').as_posix()}"
    cfg = Config()
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.attributes["database_url"] = url
    command.upgrade(cfg, "0005")
    eng = create_engine(url)
    with eng.begin() as c:
        c.execute(text("INSERT INTO projects (id, name, description, "
                       "owner_email, archived, settings, created_at, "
                       "updated_at) VALUES ('p', 'P', '', 'a@x', 0, '{}', "
                       "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"))
        c.execute(text("INSERT INTO runs (id, project_id, task, leaderboard, "
                       "external, object_key, sklearn, created_by, "
                       "created_at) VALUES ('r', 'p', 'classification', "
                       "'[]', 0, 'k', '', 'a@x', CURRENT_TIMESTAMP)"))
    command.upgrade(cfg, "head")
    with eng.connect() as c:
        assert c.execute(text("SELECT name, note FROM runs")).one() == \
            ("", "")


def _project_page():
    from components import project_panel, shell
    shell.render("ml_project")
    project_panel.render()


def _page(pid, df=None):
    at = AppTest.from_function(_project_page, default_timeout=120)
    at.session_state["proj"] = {"id": pid, "name": "Runs", "role": "owner"}
    at.session_state["proj_section"] = "runs"
    if df is not None:
        at.session_state["ml_source"] = df
        at.session_state["ml_df"] = df
    return at.run()


def test_clone_opens_configure_with_the_run_settings(df):
    pid = projects.create_project(DEV, "Runs")["id"]
    did = projects.save_dataset(DEV, pid, df, "churn.csv", "upload")
    widgets = {"ml_w_task": "classification", "ml_w_target": "Churned",
               "ml_w_folds": 3}
    steps = [{"kind": "drop_duplicates"}]
    rid = projects.save_run(DEV, pid, _train(
        df, config(["Tenure_Years"], widgets=widgets), steps), did)
    projects.describe_run(DEV, pid, rid, "Tenure", "")
    at = _page(pid)
    assert not at.exception
    table = next(m.value for m in at.markdown if "ent-table" in m.value)
    assert "<td class=''>Tenure</td>" in table
    next(b for b in at.button if b.key == "proj_run_clone").click().run()
    assert not at.exception
    ss = at.session_state
    assert ss["page"] == "ml_config"
    assert ss["ml_saved"] == widgets
    assert ss["ml_steps"] == steps and ss["proj_dataset_id"] == did
    assert len(ss["ml_source"]) == len(df)


def test_old_runs_cannot_be_cloned_but_can_be_compared(df):
    pid = projects.create_project(DEV, "Runs")["id"]
    did = projects.save_dataset(DEV, pid, df, "churn.csv", "upload")
    projects.save_run(DEV, pid, _train(df, config(["Tenure_Years"])), did)
    projects.save_run(DEV, pid, _train(
        df, config(["Tenure_Years", "Service_Calls_12M"])), did)
    at = _page(pid)
    next(b for b in at.button if b.key == "proj_run_clone").click().run()
    assert any("does not keep its configure choices" in w.value
               for w in at.warning)
    assert at.session_state["page"] != "ml_config" \
        if "page" in at.session_state else True
    text_ = " ".join(m.value for m in at.markdown) + " ".join(
        c.value for c in at.caption)
    assert "Settings that differ" in text_
    assert "only in B: Service_Calls_12M" in text_ or \
        "only in A: Service_Calls_12M" in text_


def _configure_page():
    from components import ml_config
    ml_config.render()


def test_training_on_the_page_keeps_its_choices(df):
    small = df.head(500)
    at = AppTest.from_function(_configure_page, default_timeout=180)
    at.session_state["ml_source"] = small
    at.session_state["ml_df"] = small
    at.session_state["ml_steps"] = []
    at.run()
    next(b for b in at.button if b.key == "ml_train").click().run()
    assert not at.exception
    widgets = at.session_state["ml_run"]["config"]["widgets"]
    assert widgets["ml_w_task"] == "classification"
    assert all(k.startswith("ml_w_") for k in widgets)
