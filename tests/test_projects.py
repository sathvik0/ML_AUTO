"""Projects: access rules, sharing, datasets, runs and the model registry,
on a throwaway SQLite database and local storage."""
import pandas as pd
import pytest

from logic import ml, ml_eval, projects
from logic.projects import AccessDenied
from utils import db, storage

ANA = {"sub": "a", "email": "ana@example.com", "name": "Ana",
       "role": "analyst"}
BEN = {"sub": "b", "email": "ben@example.com", "name": "Ben",
       "role": "analyst"}
VIC = {"sub": "v", "email": "vic@example.com", "name": "Vic",
       "role": "viewer"}
ADA = {"sub": "d", "email": "ada@example.com", "name": "Ada",
       "role": "admin"}


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    url = f"sqlite:///{(tmp_path / 'app.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(tmp_path / "objects"))
    monkeypatch.delenv("STORAGE_BUCKET", raising=False)
    db.migrate()


@pytest.fixture
def sample():
    return pd.read_csv("sample_data/insurance_churn.csv").head(1500)


def test_owner_creates_and_lists():
    p = projects.create_project(ANA, "Churn")
    assert projects.list_projects(ANA)[0]["name"] == "Churn"
    assert projects.get_project(ANA, p["id"])["role"] == "owner"


def test_viewer_cannot_create():
    with pytest.raises(AccessDenied):
        projects.create_project(VIC, "Nope")


def test_non_member_cannot_see():
    p = projects.create_project(ANA, "Private")
    assert projects.list_projects(BEN) == []
    with pytest.raises(AccessDenied):
        projects.get_project(BEN, p["id"])
    with pytest.raises(AccessDenied):
        projects.workspace(BEN, p["id"])


def test_admin_sees_everything():
    p = projects.create_project(ANA, "Private")
    assert [x["id"] for x in projects.list_projects(ADA)] == [p["id"]]
    assert projects.get_project(ADA, p["id"])["role"] == "owner"


def test_sharing_roles():
    p = projects.create_project(ANA, "Shared")
    projects.share(ANA, p["id"], "Ben@Example.com", "viewer")
    assert projects.get_project(BEN, p["id"])["role"] == "viewer"
    with pytest.raises(AccessDenied):
        projects.save_workspace(BEN, p["id"], [], {})
    projects.share(ANA, p["id"], "ben@example.com", "editor")
    projects.save_workspace(BEN, p["id"], [{"kind": "drop_duplicates"}], {})
    assert projects.workspace(ANA, p["id"])["steps"] == [
        {"kind": "drop_duplicates"}]
    with pytest.raises(AccessDenied):          # editors do not share
        projects.share(BEN, p["id"], "vic@example.com", "viewer")


def test_global_viewer_is_capped_at_viewer():
    p = projects.create_project(ANA, "Shared")
    projects.share(ANA, p["id"], VIC["email"], "editor")
    assert projects.get_project(VIC, p["id"])["role"] == "viewer"
    with pytest.raises(AccessDenied):
        projects.save_workspace(VIC, p["id"], [], {})


def test_always_one_owner():
    p = projects.create_project(ANA, "Solo")
    with pytest.raises(AccessDenied):
        projects.unshare(ANA, p["id"], ANA["email"])
    with pytest.raises(AccessDenied):
        projects.share(ANA, p["id"], ANA["email"], "viewer")
    projects.share(ANA, p["id"], BEN["email"], "owner")
    projects.unshare(ANA, p["id"], ANA["email"])     # Ben still owns it
    assert projects.list_projects(ANA) == []


def test_dataset_round_trip(sample):
    p = projects.create_project(ANA, "Data")
    did = projects.save_dataset(ANA, p["id"], sample, "churn.csv", "upload")
    back, name = projects.load_dataset(ANA, p["id"], did)
    assert name == "churn.csv"
    pd.testing.assert_frame_equal(back, sample, check_dtype=False)
    ws = projects.workspace(ANA, p["id"])
    assert ws["dataset_id"] == did and ws["steps"] == []


def test_dataset_from_another_project_refused(sample):
    a = projects.create_project(ANA, "A")
    b = projects.create_project(BEN, "B")
    did = projects.save_dataset(BEN, b["id"], sample, "b.csv", "upload")
    with pytest.raises(AccessDenied):
        projects.load_dataset(ANA, a["id"], did)


def _train(df):
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
    run = ml.train(df, config, lambda f, t: None)
    run["prep_steps"], run["importance"] = [], {}
    return run


def test_run_saved_and_reloaded(sample):
    p = projects.create_project(ANA, "Runs")
    run = _train(sample)
    rid = projects.save_run(ANA, p["id"], run)
    listed = projects.list_runs(ANA, p["id"])
    assert listed[0]["id"] == rid and listed[0]["best_model"] == "logistic"
    assert listed[0]["score"] == pytest.approx(
        ml.result(run, "logistic")["test"][run["metric"]])
    back = projects.load_run(ANA, p["id"], rid)
    res = ml.result(back, "logistic")
    X = ml_eval.features_frame(back, sample.head(5))
    assert list(res["pipeline"].predict(X)) == list(
        ml.result(run, "logistic")["pipeline"].predict(X))


def test_model_versions(sample):
    p = projects.create_project(ANA, "Registry")
    run = _train(sample)
    rid = projects.save_run(ANA, p["id"], run)
    res = ml.result(run, "logistic")
    data = ml_eval.model_bytes(run, res)
    v1 = projects.register_model(ANA, p["id"], rid, data, name="churn",
                                 model_key="logistic", task="classification")
    v2 = projects.register_model(ANA, p["id"], rid, data, name="churn",
                                 model_key="logistic", task="classification")
    assert (v1["version"], v2["version"]) == (1, 2)
    assert projects.model_file(ANA, p["id"], v2["id"]) == data
    assert [m["version"] for m in projects.list_models(ANA, p["id"])] == [2,
                                                                         1]


def test_delete_removes_files(sample, tmp_path):
    p = projects.create_project(ANA, "Gone")
    projects.save_dataset(ANA, p["id"], sample, "x.csv", "upload")
    assert (tmp_path / "objects" / "projects" / p["id"]).exists()
    projects.delete(ANA, p["id"])
    assert not (tmp_path / "objects" / "projects" / p["id"]).exists()
    assert projects.list_projects(ANA) == []


def test_storage_refuses_paths_outside():
    with pytest.raises(storage.StorageError):
        storage.put("../../escape.txt", b"x")


def test_postgres_keeps_no_idle_connections(monkeypatch):
    # Idle pooled connections would stop Aurora Serverless from pausing.
    pg = "postgresql+psycopg://u:p@db.invalid/x"
    db._engine.cache_clear()
    assert type(db.engine(pg).pool).__name__ == "NullPool"
    monkeypatch.setenv("DB_POOL", "queue")
    db._engine.cache_clear()
    assert type(db.engine(pg).pool).__name__ == "QueuePool"
    db._engine.cache_clear()
