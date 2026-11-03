"""Model monitoring: drift and accuracy checks against the real sample
data, alerts, retraining into staging, and the schedule."""
from datetime import timedelta

import pandas as pd
import pytest

from logic import deployment, ml, ml_eval, monitoring, projects
from logic.projects import AccessDenied
from utils import alerts, db

ANA = {"sub": "a", "email": "ana@example.com", "role": "analyst"}
OLA = {"sub": "o", "email": "ola@example.com", "role": "analyst"}
BEN = {"sub": "b", "email": "ben@example.com", "role": "analyst"}
VIC = {"sub": "v", "email": "vic@example.com", "role": "viewer"}
FEATURES = ["Age", "Tenure_Years", "Premium_Change_Pct", "Late_Payments_12M",
            "Service_Calls_12M", "Digital_Engagement", "Policies_Held"]


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL",
                       f"sqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(tmp_path / "objects"))
    monkeypatch.delenv("STORAGE_BUCKET", raising=False)
    monkeypatch.delenv("ALERTS_TOPIC_ARN", raising=False)
    db.migrate()


@pytest.fixture(scope="module")
def base():
    return pd.read_csv("sample_data/insurance_churn.csv")


@pytest.fixture(scope="module")
def recent():
    return pd.read_csv("sample_data/insurance_churn_recent.csv")


@pytest.fixture(scope="module")
def trained(base):
    train = base.iloc[:6000]
    config = {"task": "classification", "target": "Churned",
              "positive": "Yes", "target_rule": None,
              "numeric": FEATURES, "categorical": ["Payment_Method"],
              "prep": {"num_missing": "median", "cat_missing": "category",
                       "scale": "standard", "rare_share": 0.0,
                       "missing_flags": False, "log_target": False,
                       "pca": 0, "cat_encoding": "onehot"},
              "validation": {"test_size": 0.2, "folds": 0,
                             "stratify": True, "group": None},
              "seed": 1, "metric": None, "balance": False,
              "models": [{"key": "logistic", "params":
                          ml.default_params("classification", "logistic"),
                          "search": None}], "k_scan": None}
    run = ml.train(train, config, lambda f, t: None)
    run["prep_steps"], run["importance"] = [], {}
    run["frame"] = train
    return run


@pytest.fixture
def live(trained):
    """churn v1 in production in Ana's project (Ola co-owns, Ben edits)."""
    pid = projects.create_project(ANA, "Monitored")["id"]
    projects.share(ANA, pid, OLA["email"], "owner")
    projects.share(ANA, pid, BEN["email"], "editor")
    projects.share(ANA, pid, VIC["email"], "viewer")
    rid = projects.save_run(ANA, pid, trained)
    res = ml.result(trained, "logistic")
    v = projects.register_model(
        ANA, pid, rid, ml_eval.model_bytes(trained, res), name="churn",
        model_key="logistic", task="classification", target="Churned",
        metric=trained["metric"], score=float(res["test"][trained["metric"]]))
    tid = deployment.request_production(ANA, pid, v["id"])
    deployment.approve(OLA, pid, tid)
    return {"pid": pid, "vid": v["id"]}


def _source(pid, frame):
    return deployment.save_input(ANA, pid, frame, "rows.csv")


class Outbox(list):
    def __call__(self, channel, subject, message, webhook):
        self.append((channel, subject, message, webhook))
        return channel


def test_drifted_rows_raise_an_alert(live, recent):
    out = Outbox()
    mid = monitoring.create_monitor(
        BEN, live["pid"], name="weekly", model_name="churn",
        source=_source(live["pid"], recent),
        webhook="https://hooks.slack.com/services/T0/B0/x")
    got = monitoring.run_now(BEN, live["pid"], mid, sender=out)
    assert got["status"] == "alert" and got["alerted"]
    run = monitoring.monitor_runs(ANA, live["pid"], mid)[0]
    s = run["summary"]
    assert "Premium_Change_Pct" in s["major"]
    assert s["drift_auc"] > 0.75
    assert s["performance"]["labelled"] == len(recent)
    assert [c for c, *_ in out] == ["sns", "webhook"]
    assert "churn v1" in out[0][2] and "ALERT" in out[0][2]


def test_same_population_is_ok(live, base):
    unseen = base.iloc[6000:]                  # never trained on
    mid = monitoring.create_monitor(
        BEN, live["pid"], name="control", model_name="churn",
        source=_source(live["pid"], unseen))
    got = monitoring.run_now(BEN, live["pid"], mid, sender=Outbox())
    assert got["status"] == "ok", got["message"]
    s = monitoring.monitor_runs(ANA, live["pid"], mid)[0]["summary"]
    assert s["major"] == [] and s["drift_auc"] < 0.6
    perf = s["performance"]
    assert abs(perf["current"] - perf["baseline"]) < 0.05


def test_alert_retrains_into_staging(live, recent):
    mid = monitoring.create_monitor(
        BEN, live["pid"], name="auto", model_name="churn",
        source=_source(live["pid"], recent), retrain="staging")
    out = Outbox()
    got = monitoring.run_now(BEN, live["pid"], mid, sender=out)
    assert got["retrained"] == 2
    stages = {m["version"]: m["stage"]
              for m in projects.list_models(ANA, live["pid"])}
    assert stages == {1: "production", 2: "staging"}   # production intact
    assert "v2 is in staging" in out[0][2]
    runs = projects.list_runs(ANA, live["pid"])
    assert runs[0]["created_by"] == monitoring.SERVICE["email"]


def test_no_production_version(live, recent):
    other = projects.create_project(ANA, "Empty")["id"]
    mid = monitoring.create_monitor(
        ANA, other, name="x", model_name="churn",
        source=deployment.save_input(ANA, other, recent, "r.csv"))
    got = monitoring.run_now(ANA, other, mid, sender=Outbox())
    assert got["status"] == "failed" and "no production version" in \
        got["message"]


@pytest.mark.parametrize("url,ok", [
    ("https://hooks.slack.com/services/T/B/x", True),
    ("https://acme.webhook.office.com/webhookb2/x", True),
    ("http://hooks.slack.com/services/x", False),
    ("https://169.254.169.254/latest/meta-data", False),
    ("https://internal.example.com/hook", False),
    ("https://hooks.slack.com.evil.com/x", False),
])
def test_webhook_rules(url, ok):
    assert (alerts.webhook_problem(url) is None) == ok


def test_bad_webhook_refused_and_hidden(live, recent):
    with pytest.raises(AccessDenied):
        monitoring.create_monitor(
            BEN, live["pid"], name="x", model_name="churn",
            source=_source(live["pid"], recent),
            webhook="https://10.0.0.5/hook")
    monitoring.create_monitor(
        BEN, live["pid"], name="y", model_name="churn",
        source=_source(live["pid"], recent),
        webhook="https://hooks.slack.com/services/T/B/secret")
    assert "secret" in monitoring.list_monitors(ANA, live["pid"])[0][
        "webhook"]
    assert monitoring.list_monitors(BEN, live["pid"])[0]["webhook"] == "set"


def test_viewer_cannot_create(live, recent):
    with pytest.raises(AccessDenied):
        monitoring.create_monitor(
            VIC, live["pid"], name="x", model_name="churn",
            source={"type": "dataset", "dataset_id": "nope"})


def test_schedule_runs_once_when_due(live, recent):
    mid = monitoring.create_monitor(
        BEN, live["pid"], name="daily", model_name="churn",
        source=_source(live["pid"], recent.head(500)), schedule="0 6 * * *")
    m = monitoring.list_monitors(ANA, live["pid"])[0]
    later = deployment._aware(m["next_run_at"]) + timedelta(minutes=1)
    assert monitoring.run_due() == []
    assert [r["status"] for r in monitoring.run_due(later)] == ["alert"]
    assert monitoring.run_due(later) == []
    assert len(monitoring.monitor_runs(ANA, live["pid"], mid)) == 1


def test_sns_publish(monkeypatch):
    sent = {}

    class FakeSns:
        def publish(self, **kw):
            sent.update(kw)
    import boto3
    monkeypatch.setattr(boto3, "client", lambda name: FakeSns())
    monkeypatch.setenv("ALERTS_TOPIC_ARN", "arn:aws:sns:us-east-1:1:alerts")
    assert alerts.send("Subject", "Body") == ["sns"]
    assert sent["TopicArn"].endswith(":alerts") and sent["Message"] == "Body"


def test_alert_failure_does_not_raise():
    def broken(*a):
        raise RuntimeError("down")
    assert alerts.send("s", "m", webhook="https://hooks.slack.com/x",
                       sender=broken) == []


def test_repeated_alerts_do_not_stack_retrains(live, recent):
    mid = monitoring.create_monitor(
        BEN, live["pid"], name="daily", model_name="churn",
        source=_source(live["pid"], recent), retrain="staging")
    first = monitoring.run_now(BEN, live["pid"], mid, sender=Outbox())
    second = monitoring.run_now(BEN, live["pid"], mid, sender=Outbox())
    assert (first["retrained"], second["retrained"]) == (2, None)
    assert "already waiting in staging" in second["message"]
    assert [m["version"] for m in projects.list_models(ANA, live["pid"])] \
        == [2, 1]
