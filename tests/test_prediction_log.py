"""The model API in production: inputs checked against what the model was
trained on, reasons behind each prediction, and every request kept."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from logic import (deployment, inputs, ml, ml_eval, ml_upload,
                   prediction_log, projects, serving)
from logic.projects import AccessDenied
from utils import db, storage

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
    serving._CACHE.clear()
    db.migrate()


@pytest.fixture(scope="module")
def df():
    return pd.read_csv("sample_data/insurance_churn.csv").head(2000)


def _run(df, task="classification"):
    config = {"task": task,
              "target": "Churned" if task == "classification"
              else "Annual_Premium",
              "positive": "Yes" if task == "classification" else None,
              "target_rule": None, "numeric": NUMERIC,
              "categorical": ["Region"],
              "prep": {"num_missing": "median", "cat_missing": "category",
                       "scale": "standard", "rare_share": 0.0,
                       "missing_flags": False, "log_target": False,
                       "pca": 0, "cat_encoding": "onehot"},
              "validation": {"test_size": 0.25, "folds": 0,
                             "stratify": task == "classification",
                             "group": None},
              "seed": 1, "metric": None, "balance": False,
              "models": [{"key": "logistic" if task == "classification"
                          else "ridge",
                          "params": ml.default_params(
                              task, "logistic" if task == "classification"
                              else "ridge"), "search": None}],
              "k_scan": None}
    run = ml.train(df, config, lambda f, t: None)
    run["prep_steps"], run["importance"], run["frame"] = [], {}, df
    return run


@pytest.fixture(scope="module")
def run(df):
    return _run(df)


def _info(run, key="logistic"):
    info, why = ml_upload.read(ml_eval.model_bytes(run, ml.result(run, key)),
                               "m.joblib")
    assert why is None
    return info


ODD = pd.DataFrame([
    {"Tenure_Years": 0.5, "Service_Calls_12M": 20, "Late_Payments_12M": 6,
     "Age": 400, "Region": "Mars"},
    {"Tenure_Years": 12, "Service_Calls_12M": 0, "Late_Payments_12M": 0,
     "Age": 50, "Region": "West"},
    {"Tenure_Years": None, "Service_Calls_12M": 1, "Late_Payments_12M": 0,
     "Age": 30, "Region": None}])


def test_profile_comes_from_training_rows_only(run, df):
    prof = _info(run)["inputs"]
    train = df.iloc[[i for i, s in enumerate(run["split"])
                     if s.startswith("train")]]
    assert prof["Age"]["max"] == float(train["Age"].max())
    assert prof["Age"]["typical"] == float(train["Age"].median())
    assert set(prof["Region"]["values"]) == set(train["Region"].astype(str))


def test_warnings_name_each_problem(run):
    out = serving.score(_info(run), ODD)
    w = list(out["ML_Warnings"])
    assert "Age 400 is above the training range" in w[0]
    assert "Service_Calls_12M 20 is above" in w[0]
    assert "Region 'Mars' was not seen in training" in w[0]
    assert w[1] is None
    assert "Tenure_Years is missing" in w[2] and "Region is missing" in w[2]


def test_too_many_problems_are_counted():
    prof = inputs.profile(pd.DataFrame({f"x{i}": [0.0, 1.0]
                                        for i in range(8)}),
                          {f"x{i}": "numeric" for i in range(8)})
    row = pd.DataFrame([{f"x{i}": 5.0 for i in range(8)}])
    assert inputs.warnings(row, prof)[0].endswith("and 3 more")


def test_reasons_point_at_the_inputs_that_moved_it(run):
    out = serving.score(_info(run), ODD.head(2), explain=3)
    first = out["ML_Reasons"].iloc[0]
    assert first.startswith("Service_Calls_12M = 20 (typical")
    assert first.count(";") == 2                     # three reasons
    # Effects are on the class of interest's probability: a loyal
    # customer's long tenure pushes it down.
    second = out["ML_Reasons"].iloc[1]
    assert second.startswith("Tenure_Years = 12 (typical ")
    assert second.split(";")[0].split(": ")[1].startswith("-")


def test_regression_reasons_are_in_the_target_units(df):
    reg = _run(df, "regression")
    out = serving.score(_info(reg, "ridge"), ODD.head(1), explain=2)
    assert out["ML_Reasons"].iloc[0].count("(typical") == 2


def test_reasons_need_a_new_file_and_few_rows(run, df):
    info = _info(run)
    with pytest.raises(serving.ScoringError, match="up to 1,000 rows"):
        serving.score(info, pd.concat([ODD] * 400, ignore_index=True),
                      explain=1)
    old = {**info, "inputs": None}               # registered before this
    out = serving.score(old, ODD.head(2))
    assert "ML_Warnings" not in out.columns
    with pytest.raises(serving.ScoringError, match="register it again"):
        serving.score(old, ODD.head(2), explain=1)


@pytest.fixture
def live(run):
    pid = projects.create_project(ANA, "Live")["id"]
    projects.share(ANA, pid, OLA["email"], "owner")
    rid = projects.save_run(ANA, pid, run)
    v = projects.register_model(
        ANA, pid, rid, ml_eval.model_bytes(run, ml.result(run, "logistic")),
        name="churn", model_key="logistic", task="classification")
    tid = deployment.request_production(ANA, pid, v["id"])
    deployment.approve(OLA, pid, tid)
    key, _ = deployment.create_key(ANA, pid, "crm")
    from api.main import app
    with TestClient(app) as c:
        yield {"pid": pid, "client": c,
               "h": {"Authorization": f"Bearer {key}"}}


def _rows():
    return [{k: (None if pd.isna(v) else v) for k, v in r.items()}
            for r in ODD.to_dict("records")]


def test_api_answers_with_warnings_reasons_and_a_request_id(live):
    got = live["client"].post("/api/v1/models/churn/predict?explain=2",
                              json={"rows": _rows()}, headers=live["h"])
    assert got.status_code == 200, got.text
    body = got.json()
    assert body["request_id"]
    first = body["outputs"][0]
    assert "Age 400" in first["ML_Warnings"]
    assert first["ML_Reasons"].count("(typical") == 2
    assert body["outputs"][1]["ML_Warnings"] is None


def test_every_request_is_kept_with_its_rows(live):
    c, h = live["client"], live["h"]
    c.post("/api/v1/models/churn/predict", json={"rows": _rows()}, headers=h)
    logs = prediction_log.recent(ANA, live["pid"])
    assert len(logs) == 1
    r = logs[0]
    assert (r["model"], r["version"], r["stage"], r["source"], r["caller"],
            r["rows"], r["flagged"]) == ("churn", 1, "production", "api",
                                         "key: crm", 3, 2)
    kept = prediction_log.rows(ANA, live["pid"], r["id"])
    assert list(kept["Age"]) == [400, 50, 30]
    assert {"ML_Prediction", "ML_Probability_Yes", "ML_Warnings"} <= set(
        kept.columns)
    with pytest.raises(AccessDenied):
        prediction_log.recent(BEN, live["pid"])


def test_a_failed_log_does_not_fail_the_answer(live, monkeypatch):
    def broken(key, data):
        raise OSError("storage is down")
    monkeypatch.setattr(storage, "put", broken)
    got = live["client"].post("/api/v1/models/churn/predict",
                              json={"rows": _rows()}, headers=live["h"])
    assert got.status_code == 200 and got.json()["request_id"] is None
