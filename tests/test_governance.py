"""Governance: the tamper-evident audit trail, fairness checks and the
fairness-before-production policy — on the real sample data."""
import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from logic import audit, deployment, fairness, ml, ml_eval, projects
from logic.projects import AccessDenied
from utils import db

ANA = {"sub": "a", "email": "ana@example.com", "role": "analyst"}
OLA = {"sub": "o", "email": "ola@example.com", "role": "analyst"}
BEN = {"sub": "b", "email": "ben@example.com", "role": "analyst"}
ADA = {"sub": "d", "email": "ada@example.com", "role": "admin"}


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL",
                       f"sqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(tmp_path / "objects"))
    monkeypatch.delenv("STORAGE_BUCKET", raising=False)
    db.migrate()


@pytest.fixture(scope="module")
def trained():
    df = pd.read_csv("sample_data/insurance_churn.csv").head(4000)
    config = {"task": "classification", "target": "Churned",
              "positive": "Yes", "target_rule": None,
              "numeric": ["Tenure_Years", "Premium_Change_Pct",
                          "Service_Calls_12M", "Late_Payments_12M"],
              "categorical": ["Payment_Method"],
              "prep": {"num_missing": "median", "cat_missing": "category",
                       "scale": "standard", "rare_share": 0.0,
                       "missing_flags": False, "log_target": False,
                       "pca": 0, "cat_encoding": "onehot"},
              "validation": {"test_size": 0.25, "folds": 4,
                             "stratify": True, "group": None},
              "seed": 3, "metric": None, "balance": False,
              "models": [{"key": "logistic", "params":
                          ml.default_params("classification", "logistic"),
                          "search": None}], "k_scan": None}
    run = ml.train(df, config, lambda f, t: None)
    run["prep_steps"], run["importance"], run["frame"] = [], {}, df
    return run


@pytest.fixture
def project(trained):
    pid = projects.create_project(ANA, "Governed")["id"]
    projects.share(ANA, pid, OLA["email"], "owner")
    projects.share(ANA, pid, BEN["email"], "editor")
    rid = projects.save_run(ANA, pid, trained)
    v = projects.register_model(
        ANA, pid, rid, ml_eval.model_bytes(trained, ml.result(trained,
                                                              "logistic")),
        name="churn", model_key="logistic", task="classification",
        target="Churned")
    return {"pid": pid, "vid": v["id"]}


# ── the audit trail ─────────────────────────────────────────────────────
def test_actions_leave_events_in_order(project):
    got = [e["action"] for e in reversed(audit.events(ANA, project["pid"]))]
    assert got == ["project.create", "project.share", "project.share",
                   "run.save", "model.register"]
    share = audit.events(ANA, project["pid"], action="project.share")[0]
    assert share["details"] == {"role": "editor"}
    assert share["actor"] == "ana@example.com"


def test_chain_verifies_and_detects_tampering(project):
    assert audit.verify() == {"ok": True, "events": 5, "first_bad": None}
    with db.engine().begin() as c:          # someone edits an event
        c.execute(text("UPDATE audit_events SET actor = 'mallory' "
                       "WHERE seq = 2"))
    assert audit.verify() == {"ok": False, "events": 1, "first_bad": 2}


def test_deleting_an_event_is_detected(project):
    with db.engine().begin() as c:
        c.execute(text("DELETE FROM audit_events WHERE seq = 3"))
    got = audit.verify()
    assert not got["ok"] and got["first_bad"] == 4


def test_refused_action_leaves_no_event(project):
    before = len(audit.events(ANA, project["pid"]))
    with pytest.raises(AccessDenied):
        projects.share(BEN, project["pid"], "x@example.com", "viewer")
    assert len(audit.events(ANA, project["pid"])) == before


def test_chain_cannot_fork(project):
    with db.session() as s:
        last = s.execute(text("SELECT prev_hash FROM audit_events "
                              "ORDER BY seq DESC LIMIT 1")).scalar()
        s.add(db.AuditEvent(actor="x", project_id=None, action="forged",
                            prev_hash=last, hash="f" * 64, details={}))
        with pytest.raises(IntegrityError):
            s.flush()


def test_who_may_read_the_log(project):
    with pytest.raises(AccessDenied):
        audit.events(BEN, project["pid"])        # editors may not
    with pytest.raises(AccessDenied):
        audit.events(ANA)                        # the whole log: admins
    assert len(audit.events(ADA)) == 5


def test_history_outlives_the_project(project):
    projects.delete(ANA, project["pid"])
    acts = [e["action"] for e in audit.events(ADA)]
    assert acts[0] == "project.delete" and "project.create" in acts
    assert audit.verify()["ok"]


def test_system_actors(project, monkeypatch):
    audit.record_now("system:batch", "batch.run", project["pid"],
                     "batch_job", "j1", status="succeeded")
    e = audit.events(ADA, action="batch.run")[0]
    assert (e["actor"], e["role"]) == ("system:batch", "system")


# ── fairness ────────────────────────────────────────────────────────────
def test_gender_passes_payment_method_flags(project):
    rid, report = fairness.check(BEN, project["pid"], project["vid"],
                                 ["Gender", "Payment_Method"])
    g = report["attributes"]["Gender"]
    assert g["status"] == "pass", g["flags"]
    assert g["metrics"]["disparate_impact"] > 0.8
    pm = report["attributes"]["Payment_Method"]
    assert pm["status"] == "flag" and "disparate impact" in pm["flags"][0]
    rates = {r["group"]: r["actual_rate"] for r in pm["groups"]}
    assert rates["Auto-debit"] < rates["Cheque"]   # real differences shown
    assert report["status"] == "flag"


def test_numeric_attribute_banded_and_out_of_sample_only(project, trained):
    _, report = fairness.check(BEN, project["pid"], project["vid"], ["Age"])
    groups = report["attributes"]["Age"]["groups"]
    assert len(groups) == 4 and all("–" in g["group"] for g in groups)
    assert report["rows"] == 4000           # 4-fold OOF + test: every row
    assert sum(g["rows"] for g in groups) == report["rows"]


def test_small_groups_not_judged():
    rows = [{"group": "a", "rows": 500, "selection_rate": .5, "tpr": .5,
             "fpr": .1, "accuracy": .8},
            {"group": "b", "rows": 12, "selection_rate": .01, "tpr": .1,
             "fpr": .0, "accuracy": .5}]
    status, _, flags = fairness._assess_classification(rows)
    assert (status, flags) == ("insufficient", [])


def test_policy_blocks_production_until_checked(project):
    fairness.set_policy(ANA, project["pid"], True)
    with pytest.raises(AccessDenied, match="requires a fairness check"):
        deployment.request_production(BEN, project["pid"], project["vid"])
    rid, _ = fairness.check(BEN, project["pid"], project["vid"],
                            ["Payment_Method"])
    with pytest.raises(AccessDenied, match="flagged"):
        deployment.request_production(BEN, project["pid"], project["vid"])
    with pytest.raises(AccessDenied):                 # editors can't accept
        fairness.accept(BEN, project["pid"], rid, "x" * 30)
    with pytest.raises(AccessDenied, match="Explain why"):
        fairness.accept(ANA, project["pid"], rid, "fine")
    fairness.accept(ANA, project["pid"], rid,
                    "Auto-debit customers genuinely churn less; the gap "
                    "follows the actual rates, not the model.")
    deployment.request_production(BEN, project["pid"], project["vid"])
    acts = [e["action"] for e in audit.events(ANA, project["pid"])][:4]
    assert acts == ["model.request_production", "fairness.accept",
                    "fairness.check", "project.policy"]


def test_passing_report_satisfies_policy(project):
    fairness.set_policy(ANA, project["pid"], True)
    fairness.check(BEN, project["pid"], project["vid"], ["Gender"])
    deployment.request_production(BEN, project["pid"], project["vid"])


def test_suggested_attributes():
    cols = ["Customer_ID", "Age", "Gender", "Region", "Annual_Premium",
            "Marital_Status"]
    assert fairness.suggest(cols) == ["Age", "Gender", "Region",
                                      "Marital_Status"]
