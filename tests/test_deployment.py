"""Stages and approvals, API keys and the model API, and batch scoring —
on a throwaway database and local storage."""
from datetime import timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from logic import deployment, ml, ml_eval, projects
from logic.projects import AccessDenied
from utils import db

ANA = {"sub": "a", "email": "ana@example.com", "role": "analyst"}   # owner
BEN = {"sub": "b", "email": "ben@example.com", "role": "analyst"}   # editor
OLA = {"sub": "o", "email": "ola@example.com", "role": "analyst"}   # owner 2
VIC = {"sub": "v", "email": "vic@example.com", "role": "viewer"}


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL",
                       f"sqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(tmp_path / "objects"))
    monkeypatch.delenv("STORAGE_BUCKET", raising=False)
    db.migrate()


@pytest.fixture(scope="module")
def sample():
    return pd.read_csv("sample_data/insurance_churn.csv").head(1500)


@pytest.fixture(scope="module")
def trained(sample):
    config = {"task": "classification", "target": "Churned",
              "positive": "Yes", "target_rule": None,
              "numeric": ["Age", "Tenure_Years", "Service_Calls_12M"],
              "categorical": ["Region"],
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
    run = ml.train(sample, config, lambda f, t: None)
    run["prep_steps"], run["importance"] = [], {}
    return run


@pytest.fixture
def setup(trained, sample):
    """A project owned by Ana and Ola, Ben an editor, Vic a viewer, with
    two registered versions of "churn" and the sample as its dataset."""
    p = projects.create_project(ANA, "Deploy")
    pid = p["id"]
    projects.share(ANA, pid, BEN["email"], "editor")
    projects.share(ANA, pid, OLA["email"], "owner")
    projects.share(ANA, pid, VIC["email"], "viewer")
    rid = projects.save_run(ANA, pid, trained)
    data = ml_eval.model_bytes(trained, ml.result(trained, "logistic"))
    v1 = projects.register_model(ANA, pid, rid, data, name="churn",
                                 model_key="logistic", task="classification")
    v2 = projects.register_model(ANA, pid, rid, data, name="churn",
                                 model_key="logistic", task="classification")
    did = projects.save_dataset(ANA, pid, sample, "churn.csv", "upload")
    return {"pid": pid, "v1": v1["id"], "v2": v2["id"], "did": did}


def stage_of(pid, vid):
    return next(m["stage"] for m in projects.list_models(ANA, pid)
                if m["id"] == vid)


def promote(pid, vid, requester=BEN, approver=ANA):
    tid = deployment.request_production(requester, pid, vid)
    deployment.approve(approver, pid, tid)


# ── stages ──────────────────────────────────────────────────────────────
def test_editor_moves_to_staging(setup):
    deployment.set_stage(BEN, setup["pid"], setup["v1"], "staging")
    assert stage_of(setup["pid"], setup["v1"]) == "staging"


def test_production_needs_a_request(setup):
    with pytest.raises(AccessDenied):
        deployment.set_stage(ANA, setup["pid"], setup["v1"], "production")


def test_requester_cannot_approve_own_request(setup):
    tid = deployment.request_production(ANA, setup["pid"], setup["v1"])
    with pytest.raises(AccessDenied):
        deployment.approve(ANA, setup["pid"], tid)
    deployment.approve(OLA, setup["pid"], tid)        # a second owner
    assert stage_of(setup["pid"], setup["v1"]) == "production"


def test_editor_cannot_approve(setup):
    tid = deployment.request_production(ANA, setup["pid"], setup["v1"])
    with pytest.raises(AccessDenied):
        deployment.approve(BEN, setup["pid"], tid)


def test_viewer_cannot_request(setup):
    with pytest.raises(AccessDenied):
        deployment.request_production(VIC, setup["pid"], setup["v1"])


def test_new_production_archives_the_old(setup):
    promote(setup["pid"], setup["v1"])
    promote(setup["pid"], setup["v2"])
    assert (stage_of(setup["pid"], setup["v1"]),
            stage_of(setup["pid"], setup["v2"])) == ("archived",
                                                     "production")


def test_reject_and_cancel(setup):
    tid = deployment.request_production(BEN, setup["pid"], setup["v1"])
    deployment.reject(ANA, setup["pid"], tid, "not yet")
    assert stage_of(setup["pid"], setup["v1"]) == "none"
    tid = deployment.request_production(BEN, setup["pid"], setup["v1"])
    with pytest.raises(AccessDenied):
        deployment.cancel(ANA, setup["pid"], tid)      # not the requester
    deployment.cancel(BEN, setup["pid"], tid)
    log = deployment.transitions(ANA, setup["pid"])
    assert [t["status"] for t in log] == ["cancelled", "rejected"]


def test_one_pending_request_per_version(setup):
    deployment.request_production(BEN, setup["pid"], setup["v1"])
    with pytest.raises(AccessDenied):
        deployment.request_production(BEN, setup["pid"], setup["v1"])


# ── API keys and the API ────────────────────────────────────────────────
def test_only_owners_make_keys(setup):
    with pytest.raises(AccessDenied):
        deployment.create_key(BEN, setup["pid"], "ci")
    key, _ = deployment.create_key(ANA, setup["pid"], "ci")
    assert key.startswith("mlk_")
    assert deployment.authenticate(key) == setup["pid"]
    listed = deployment.list_keys(ANA, setup["pid"])[0]
    assert key not in str(listed) and listed["prefix"] == key[:10]


def test_revoked_key_rejected(setup):
    key, kid = deployment.create_key(ANA, setup["pid"], "old")
    deployment.revoke_key(ANA, setup["pid"], kid)
    assert deployment.authenticate(key) is None
    assert deployment.authenticate("mlk_made_up") is None


@pytest.fixture
def client():
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_api_needs_a_key(setup, client):
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/v1/models").status_code == 401
    assert client.get("/api/v1/models", headers={
        "Authorization": "Bearer mlk_nope"}).status_code == 401


def test_api_predicts_like_the_model(setup, client, trained, sample):
    promote(setup["pid"], setup["v1"])
    key, _ = deployment.create_key(ANA, setup["pid"], "ci")
    rows = sample.drop(columns=["Churned"]).head(25)
    body = {"rows": rows.astype(object).where(rows.notna(), None)
            .to_dict("records")}
    r = client.post("/api/v1/models/churn/predict", json=body,
                    headers={"X-API-Key": key})
    assert r.status_code == 200, r.text
    got = r.json()
    assert (got["version"], got["stage"], got["rows"]) == (1, "production",
                                                            25)
    pipe = ml.result(trained, "logistic")["pipeline"]
    want = ml.predict("classification", pipe,
                      ml_eval.features_frame(trained, rows),
                      trained["classes"])[0]
    assert got["predictions"] == list(want)
    assert {"ML_Probability_Yes", "ML_Probability_No"} <= set(
        got["outputs"][0])


def test_api_version_stage_and_errors(setup, client, sample):
    key, _ = deployment.create_key(ANA, setup["pid"], "ci")
    h = {"Authorization": f"Bearer {key}"}
    rows = {"rows": sample.drop(columns=["Churned"]).head(3).astype(object)
            .where(sample.head(3).notna(), None).to_dict("records")}
    assert client.post("/api/v1/models/churn/predict", json=rows,
                       headers=h).status_code == 404   # no production yet
    assert client.post("/api/v1/models/churn/predict?version=2", json=rows,
                       headers=h).json()["version"] == 2
    bad = {"rows": [{"Age": 30}]}
    r = client.post("/api/v1/models/churn/predict?version=1", json=bad,
                    headers=h)
    assert r.status_code == 422 and "Missing columns" in r.json()["detail"]
    many = {"rows": [{"Age": 1}] * 10_001}
    assert client.post("/api/v1/models/churn/predict?version=1", json=many,
                       headers=h).status_code == 413


def test_key_cannot_reach_another_project(setup, client, trained):
    other = projects.create_project(BEN, "Ben's")
    key, _ = deployment.create_key(BEN, other["id"], "mine")
    promote(setup["pid"], setup["v1"])
    r = client.post("/api/v1/models/churn/predict",
                    json={"rows": [{"Age": 1}]},
                    headers={"X-API-Key": key})
    assert r.status_code == 404                     # not visible at all
    models = client.get("/api/v1/models", headers={"X-API-Key": key})
    assert models.json() == {"models": []}


# ── batch scoring ───────────────────────────────────────────────────────
def test_batch_job_scores_a_dataset(setup):
    promote(setup["pid"], setup["v1"])
    jid = deployment.create_job(
        BEN, setup["pid"], name="nightly", model_name="churn",
        source={"type": "dataset", "dataset_id": setup["did"]},
        destination={"type": "storage"})
    got = deployment.run_now(BEN, setup["pid"], jid)
    assert got["status"] == "succeeded" and got["rows"] == 1500
    run = deployment.job_runs(ANA, setup["pid"], jid)[0]
    out = pd.read_csv(pd.io.common.BytesIO(
        deployment.run_output(ANA, setup["pid"], run["id"])))
    assert "ML_Prediction" in out.columns and len(out) == 1500


def test_batch_job_without_production_fails_cleanly(setup):
    jid = deployment.create_job(
        BEN, setup["pid"], name="early", model_name="churn",
        source={"type": "dataset", "dataset_id": setup["did"]},
        destination={"type": "storage"})
    got = deployment.run_now(BEN, setup["pid"], jid)
    assert got["status"] == "failed" and "No production version" in \
        got["message"]
    assert deployment.list_jobs(ANA, setup["pid"])[0]["last_status"] == \
        "failed"


def test_uploaded_file_source(setup, sample):
    promote(setup["pid"], setup["v1"])
    src = deployment.save_input(BEN, setup["pid"], sample.head(40),
                                "new.csv")
    jid = deployment.create_job(BEN, setup["pid"], name="file",
                                model_name="churn", source=src,
                                destination={"type": "storage"})
    assert deployment.run_now(BEN, setup["pid"], jid)["rows"] == 40


def test_snowflake_source_and_destination(setup, sample, monkeypatch):
    from logic import snowflake_io as sf
    promote(setup["pid"], setup["v1"])
    written = {}
    monkeypatch.setattr(sf, "run_query",
                        lambda conn, sql, d=None, s=None: (sample.head(7),
                                                           None))
    monkeypatch.setattr(sf, "write_table",
                        lambda conn, df, d, s, t, mode="": written.update(
                            rows=len(df), table=t) or (len(df), None))
    jid = deployment.create_job(
        BEN, setup["pid"], name="sf", model_name="churn",
        source={"type": "snowflake", "query": "SELECT * FROM CUSTOMERS"},
        destination={"type": "snowflake", "database": "DB", "schema": "ML",
                     "table": "CHURN_SCORES"})
    got = deployment.run_now(BEN, setup["pid"], jid,
                             snowflake=lambda: object())
    assert got["status"] == "succeeded"
    assert written == {"rows": 7, "table": "CHURN_SCORES"}


def test_schedule_runs_when_due_and_only_once(setup, monkeypatch):
    promote(setup["pid"], setup["v1"])
    jid = deployment.create_job(
        BEN, setup["pid"], name="hourly", model_name="churn",
        source={"type": "dataset", "dataset_id": setup["did"]},
        destination={"type": "storage"}, schedule="0 * * * *")
    job = deployment.list_jobs(ANA, setup["pid"])[0]
    assert job["next_run_at"] is not None
    assert deployment.run_due() == []                 # not due yet
    later = deployment._aware(job["next_run_at"]) + timedelta(minutes=1)
    assert [r["status"] for r in deployment.run_due(later)] == ["succeeded"]
    assert deployment.run_due(later) == []            # claimed, not re-run
    assert len(deployment.job_runs(ANA, setup["pid"], jid)) == 1


def test_bad_schedule_refused(setup):
    with pytest.raises(AccessDenied):
        deployment.create_job(
            BEN, setup["pid"], name="x", model_name="churn",
            source={"type": "dataset", "dataset_id": setup["did"]},
            destination={"type": "storage"}, schedule="every tuesday")


def test_runner_cli(setup):
    from worker import batch
    assert batch.main(["--due"]) == 0
