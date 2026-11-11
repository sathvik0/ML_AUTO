"""Models trained elsewhere: evaluated here, registered as the very file
uploaded, then scored by the API path, batch jobs and monitoring like a
model trained here."""
import io

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from logic import (deployment, ml, ml_eval, ml_upload, monitoring, projects,
                   serving)
from logic.monitoring import CheckError
from utils import db

ANA = {"sub": "a", "email": "ana@example.com", "role": "analyst"}
OLA = {"sub": "o", "email": "ola@example.com", "role": "analyst"}
# The notebook's own names: the file says Tenure_Years, the model tenure.
INPUTS = {"tenure": "Tenure_Years", "calls": "Service_Calls_12M",
          "late": "Late_Payments_12M", "region": "Region"}


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


@pytest.fixture(scope="module")
def notebook_model(df):
    """A model fitted in a notebook, on its own column names."""
    X = df[list(INPUTS.values())].rename(
        columns={v: k for k, v in INPUTS.items()})
    pipe = Pipeline([
        ("prep", ColumnTransformer([
            ("num", Pipeline([("fill", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]),
             ["tenure", "calls", "late"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["region"])])),
        ("model", LogisticRegression(max_iter=1000))])
    pipe.fit(X, df["Churned"])
    buf = io.BytesIO()
    joblib.dump(pipe, buf)
    return pipe, buf.getvalue()


def _evaluate(df, data):
    info, why = ml_upload.read(data, "churn_notebook.joblib")
    assert why is None
    columns = dict(INPUTS)          # mapped by hand, as on the page
    run, why = ml_upload.evaluate(df, info, columns, "classification",
                                  "Churned", "Yes", prof=ml.profile(df))
    assert why is None
    return run


def _register(df, data, threshold=None):
    run = _evaluate(df, data)
    pid = projects.create_project(ANA, "Outside")["id"]
    projects.share(ANA, pid, OLA["email"], "owner")
    rid = projects.save_run(ANA, pid, run)
    res = ml.result(run, ml_upload.KEY)
    if threshold is not None:
        res["threshold"] = threshold
    v = projects.register_model(
        ANA, pid, rid, ml_eval.model_bytes(run, res), name="churn_nb",
        model_key=ml_upload.KEY, task="classification", target="Churned",
        metric=run["metric"], score=res["test"][run["metric"]])
    return pid, v, run


def _direct(pipe, rows):
    X = rows[list(INPUTS.values())].rename(
        columns={v: k for k, v in INPUTS.items()})
    return pipe.predict(X), pipe.predict_proba(X)[:, list(
        pipe.classes_).index("Yes")]


def test_registered_file_scores_like_the_notebook(df, notebook_model):
    pipe, data = notebook_model
    pid, v, _ = _register(df, data)
    stored = projects.model_file(ANA, pid, v["id"])
    info, why = ml_upload.read(stored, "model.joblib")
    assert why is None and info["kind"] == "external"
    assert info["columns"] == INPUTS and info["inner_kind"] == "estimator"
    rows = df.tail(50).drop(columns=["Churned"])
    out = serving.score(info, rows)
    pred, proba = _direct(pipe, rows)
    assert (out["ML_Prediction"].to_numpy() == pred).all()
    assert np.allclose(out["ML_Probability_Yes"], proba)


def test_a_few_api_rows_keep_the_registered_types(df, notebook_model):
    """Two rows cannot show that Region is a category; the stored types
    say so."""
    _, data = notebook_model
    pid, v, _ = _register(df, data)
    info = deployment.model_info(projects.list_models(ANA, pid)[0]
                                 | {"object_key": _key(v)})
    rows = pd.DataFrame([{"Tenure_Years": 2.0, "Service_Calls_12M": 5,
                          "Late_Payments_12M": 3, "Region": "North"},
                         {"Tenure_Years": 9.0, "Service_Calls_12M": 0,
                          "Late_Payments_12M": 0, "Region": "West"}])
    out = serving.score(info, rows)
    assert list(out["ML_Prediction"]) and \
        out["ML_Probability_Yes"].between(0, 1).all()
    with pytest.raises(serving.ScoringError, match="Region"):
        serving.score(info, rows.drop(columns=["Region"]))


def _key(v):
    with db.session() as s:
        return s.get(db.ModelVersion, v["id"]).object_key


def test_cutoff_goes_with_the_registered_file(df, notebook_model):
    pipe, data = notebook_model
    pid, v, _ = _register(df, data, threshold=0.15)
    info, _ = ml_upload.read(projects.model_file(ANA, pid, v["id"]), "m")
    rows = df.tail(80).drop(columns=["Churned"])
    out = serving.score(info, rows)
    _, proba = _direct(pipe, rows)
    assert (out["ML_Prediction"] == np.where(proba >= 0.15, "Yes", "No")) \
        .all()


def test_approved_to_production_and_monitored(df, notebook_model):
    _, data = notebook_model
    pid, v, _ = _register(df, data)
    deployment.request_production(ANA, pid, v["id"])
    t = deployment.transitions(OLA, pid, pending_only=True)[0]
    deployment.approve(OLA, pid, t["id"])
    assert deployment.resolve_version(pid, "churn_nb")["version"] == 1
    did = projects.save_dataset(ANA, pid, df.tail(600), "newer.csv", "upload")
    mid = monitoring.create_monitor(
        ANA, pid, name="Watch", model_name="churn_nb",
        source={"type": "dataset", "dataset_id": did}, retrain="staging")
    got = monitoring.run_now(ANA, pid, mid)
    assert got["status"] in ("ok", "warning", "alert"), got


def test_never_retrained_here(df, notebook_model):
    _, data = notebook_model
    run = _evaluate(df, data)
    with pytest.raises(CheckError, match="retrain it where it was built"):
        monitoring._retrain("p", {"name": "x"}, run, df, "m")


def test_only_the_uploaded_model_and_only_with_its_file(df, notebook_model):
    _, data = notebook_model
    run = _evaluate(df, data)
    with pytest.raises(ValueError, match="Only the uploaded model"):
        ml_eval.model_bytes(run, ml.result(run, ml.BASELINE))
    run["upload"]["file"] = None           # an evaluation from before
    with pytest.raises(ValueError, match="upload it again"):
        ml_eval.model_bytes(run, ml.result(run, ml_upload.KEY))


def test_a_booster_by_position(df):
    """An XGBoost .json model knows no column names and answers in class
    codes; registered, it still answers in labels."""
    xgb = pytest.importorskip("xgboost")
    cols = ["Tenure_Years", "Service_Calls_12M", "Late_Payments_12M"]
    m = xgb.XGBClassifier(n_estimators=20, max_depth=3)
    m.fit(df[cols].to_numpy(), (df["Churned"] == "Yes").astype(int))
    data = m.get_booster().save_raw("json").decode().encode()
    info, why = ml_upload.read(bytes(data), "booster.json")
    assert why is None
    assert info["features"] is None and info["n_features"] == 3
    # As the page maps a model that reads by position.
    columns = {f"input {i + 1}": c for i, c in enumerate(cols)}
    run, why = ml_upload.evaluate(df, info, columns, "classification",
                                  "Churned", "Yes", prof=ml.profile(df),
                                  by_position=True)
    assert why is None
    bundle = ml_eval.model_bytes(run, ml.result(run, ml_upload.KEY))
    got, why = ml_upload.read(bundle, "model.joblib")
    assert why is None and got["inner_kind"] == "xgboost"
    out = serving.score(got, df.tail(30).drop(columns=["Churned"]))
    assert set(out["ML_Prediction"]) <= {"Yes", "No"}
