"""Monitoring what the model API actually scores: the logged requests as a
monitor's rows, outcomes joined when they arrive, and the log kept only as
long as the project says."""
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from logic import (audit, deployment, ml, ml_eval, monitoring,
                   prediction_log, projects, serving)
from logic.projects import AccessDenied
from utils import db, storage
from worker import batch

ANA = {"sub": "a", "email": "ana@example.com", "role": "analyst"}
OLA = {"sub": "o", "email": "ola@example.com", "role": "analyst"}
BEN = {"sub": "b", "email": "ben@example.com", "role": "analyst"}
NUMERIC = ["Tenure_Years", "Service_Calls_12M", "Late_Payments_12M", "Age"]


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL",
                       f"sqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(tmp_path / "objects"))
    monkeypatch.delenv("STORAGE_BUCKET", raising=False)
    monkeypatch.delenv("ALERTS_TOPIC_ARN", raising=False)
    serving._CACHE.clear()
    db.migrate()


@pytest.fixture(scope="module")
def data():
    return pd.read_csv("sample_data/insurance_churn.csv")


@pytest.fixture(scope="module")
def run(data):
    df = data.head(3000)
    config = {"task": "classification", "target": "Churned",
              "positive": "Yes", "target_rule": None, "numeric": NUMERIC,
              "categorical": ["Region"],
              "prep": {"num_missing": "median", "cat_missing": "category",
                       "scale": "standard", "rare_share": 0.0,
                       "missing_flags": False, "log_target": False,
                       "pca": 0, "cat_encoding": "onehot"},
              "validation": {"test_size": 0.25, "folds": 0,
                             "stratify": True, "group": None},
              "seed": 1, "metric": "roc_auc", "balance": False,
              "models": [{"key": "logistic", "params": ml.default_params(
                  "classification", "logistic"), "search": None}],
              "k_scan": None}
    out = ml.train(df, config, lambda f, t: None)
    out["prep_steps"], out["importance"], out["frame"] = [], {}, df
    return out


@pytest.fixture
def live(run):
    pid = projects.create_project(ANA, "Live")["id"]
    projects.share(ANA, pid, OLA["email"], "owner")
    rid = projects.save_run(ANA, pid, run)
    res = ml.result(run, "logistic")
    v = projects.register_model(
        ANA, pid, rid, ml_eval.model_bytes(run, res), name="churn",
        model_key="logistic", task="classification", target="Churned",
        metric="roc_auc", score=res["test"]["roc_auc"])
    tid = deployment.request_production(ANA, pid, v["id"])
    deployment.approve(OLA, pid, tid)
    key, _ = deployment.create_key(ANA, pid, "crm")
    from api.main import app
    with TestClient(app) as c:
        def send(rows):
            body = [{k: (None if pd.isna(x) else x) for k, x in r.items()}
                    for r in rows.to_dict("records")]
            got = c.post("/api/v1/models/churn/predict", json={"rows": body},
                         headers={"Authorization": f"Bearer {key}"})
            assert got.status_code == 200, got.text
        yield {"pid": pid, "send": send}


def _requests(data, start, n, shift=0):
    rows = data.iloc[start:start + n][["Customer_ID", "Region"] + NUMERIC]
    rows = rows.copy()
    rows["Age"] = rows["Age"] + shift
    return rows


def _monitor(pid, source):
    return monitoring.create_monitor(ANA, pid, name="Live",
                                     model_name="churn", source=source)


def test_monitor_reads_the_logged_requests(live, data):
    live["send"](_requests(data, 4000, 300))
    live["send"](_requests(data, 4300, 300))
    mid = _monitor(live["pid"], {"type": "logged", "days": 30})
    got = monitoring.run_now(ANA, live["pid"], mid)
    assert got["status"] in ("ok", "warning"), got
    runs = monitoring.monitor_runs(ANA, live["pid"], mid)
    summary = runs[0]["summary"]
    assert summary["rows"] == 600 and summary["performance"] is None


def test_drift_in_what_the_api_scores_is_caught(live, data):
    live["send"](_requests(data, 4000, 500, shift=25))
    mid = _monitor(live["pid"], {"type": "logged", "days": 30})
    got = monitoring.run_now(ANA, live["pid"], mid)
    assert got["status"] in ("warning", "alert")
    assert "Age" in got["message"]


def test_outcomes_join_gives_live_accuracy(live, data):
    live["send"](_requests(data, 4000, 400))
    live["send"](_requests(data, 4000, 100))     # the same customers again
    arrived = data.iloc[4000:4300][["Customer_ID", "Churned"]]
    did = projects.save_dataset(ANA, live["pid"], arrived, "outcomes.csv",
                                "upload")
    mid = _monitor(live["pid"], {
        "type": "logged", "days": 30,
        "outcomes": {"source": {"type": "dataset", "dataset_id": did},
                     "id": "Customer_ID"}})
    got = monitoring.run_now(ANA, live["pid"], mid)
    perf = monitoring.monitor_runs(ANA, live["pid"], mid)[0]["summary"][
        "performance"]
    assert got["status"] != "failed", got
    assert perf["labelled"] == 300              # one row per customer
    assert perf["metric"] == "roc_auc" and 0.5 < perf["current"] < 1


def test_no_requests_is_a_clear_failure(live):
    mid = _monitor(live["pid"], {"type": "logged", "days": 7})
    got = monitoring.run_now(ANA, live["pid"], mid)
    assert got["status"] == "failed"
    assert "No requests to churn were logged in the last 7 days" in \
        got["message"]


def test_outcomes_need_an_id(live):
    with pytest.raises(AccessDenied, match="ID column"):
        _monitor(live["pid"], {"type": "logged", "days": 30, "outcomes": {
            "source": {"type": "dataset", "dataset_id": "x"}, "id": ""}})


def _age(pid, days):
    """Pretend every logged request is `days` old."""
    with db.session() as s:
        for r in s.query(db.PredictionLog).filter_by(project_id=pid):
            r.at = db.now() - timedelta(days=days)
        s.commit()


def test_window_leaves_out_older_requests(live, data):
    live["send"](_requests(data, 4000, 50))
    _age(live["pid"], 40)
    assert prediction_log.window(live["pid"], "churn", 30).empty
    assert len(prediction_log.window(live["pid"], "churn", 90)) == 50


def test_retention_purges_old_requests(live, data):
    pid = live["pid"]
    assert prediction_log.retention(pid) == prediction_log.DEFAULT_DAYS
    live["send"](_requests(data, 4000, 20))
    key = prediction_log.recent(ANA, pid)[0]["id"]
    with db.session() as s:
        obj = s.get(db.PredictionLog, key).object_key
    with pytest.raises(AccessDenied):
        prediction_log.set_retention(BEN, pid, 30)      # not an owner
    with pytest.raises(AccessDenied, match="offered"):
        prediction_log.set_retention(ANA, pid, 12)
    prediction_log.set_retention(ANA, pid, 30)
    _age(pid, 20)
    assert prediction_log.purge() == 0                   # still inside
    _age(pid, 31)
    assert prediction_log.purge() == 1
    assert prediction_log.recent(ANA, pid) == []
    with pytest.raises(storage.StorageError):
        storage.get(obj)
    e = audit.events(ANA, pid, action="project.retention")[0]
    assert e["details"] == {"prediction_log_days": 30}


def test_keep_for_good(live, data):
    prediction_log.set_retention(ANA, live["pid"], None)
    live["send"](_requests(data, 4000, 10))
    _age(live["pid"], 4000)
    assert prediction_log.purge() == 0


def test_the_runner_purges(monkeypatch):
    called = []
    monkeypatch.setattr(prediction_log, "purge", lambda: called.append(1))
    monkeypatch.setattr(deployment, "run_due", lambda: [])
    monkeypatch.setattr(monitoring, "run_due", lambda: [])
    assert batch.main(["--due"]) == 0 and called == [1]


def test_window_rows_match_what_was_sent(live, data):
    sent = _requests(data, 4000, 30)
    live["send"](sent)
    got = prediction_log.window(live["pid"], "churn", 30)
    assert list(got["Customer_ID"]) == list(sent["Customer_ID"])
    assert not [c for c in got.columns if c.startswith("ML_")]
    assert np.isclose(got["Age"].astype(float).mean(), sent["Age"].mean())


def _project_page():
    from components import project_panel, shell
    shell.render("ml_project")
    project_panel.render()


def test_create_a_monitor_on_the_log_from_the_page(live, data, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("AUTH_MODE", "dev")
    monkeypatch.setenv("DEV_USER_EMAIL", ANA["email"])
    did = projects.save_dataset(ANA, live["pid"],
                                data.iloc[4000:4100][["Customer_ID",
                                                      "Churned"]],
                                "outcomes.csv", "upload")
    projects.save_workspace(ANA, live["pid"], [], {})
    at = AppTest.from_function(_project_page, default_timeout=120)
    at.session_state["proj"] = {"id": live["pid"], "name": "Live",
                                "role": "owner"}
    at.session_state["proj_section"] = "operations"
    at.run()
    assert not at.exception
    at.text_input(key="mon_name").set_value("API drift")
    at.checkbox(key="mon_join").check().run()
    at.text_input(key="mon_join_id").set_value("Customer_ID").run()
    assert not at.exception
    next(b for b in at.button if b.key == "mon_go").click().run()
    assert not at.exception
    got = monitoring.list_monitors(ANA, live["pid"])
    assert len(got) == 1
    src = got[0]["source"]
    assert src["type"] == "logged" and src["days"] == 30
    assert src["outcomes"] == {"source": {"type": "dataset",
                                          "dataset_id": did},
                               "id": "Customer_ID"}
    card = " ".join(m.value for m in at.markdown)
    assert "API requests, last 30 days, outcomes joined on Customer_ID" \
        in card
