"""The decision cut-off: found on the out-of-fold scores, saved with the
model file, and used wherever that file scores rows."""
import numpy as np
import pandas as pd
import pytest

from logic import audit, ml, ml_eval, ml_upload, projects, serving
from logic.projects import AccessDenied
from utils import db

ANA = {"sub": "a", "email": "ana@example.com", "role": "analyst"}
VIC = {"sub": "v", "email": "vic@example.com", "role": "viewer"}


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL",
                       f"sqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(tmp_path / "objects"))
    monkeypatch.delenv("STORAGE_BUCKET", raising=False)
    db.migrate()


@pytest.fixture(scope="module")
def trained():
    df = pd.read_csv("sample_data/insurance_churn.csv").head(3000)
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
    return run, df


def _brute(hit, score, goal, miss=1.0, alarm=1.0):
    best, at = None, None
    for t in np.round(np.arange(0.01, 1.0, 0.01), 2):
        s = ml_eval.at_threshold(hit, score, t)
        p, n = s["tp"] + s["fn"], s["fp"] + s["tn"]
        value = {"f1": s["f1"] or 0.0,
                 "youden": s["tp"] / p - s["fp"] / n,
                 "cost": -(s["fn"] * miss + s["fp"] * alarm)}[goal]
        if best is None or value > best:
            best, at = value, t
    return at


@pytest.mark.parametrize("goal,miss", [("f1", 1), ("youden", 1),
                                       ("cost", 1), ("cost", 8)])
def test_finder_matches_trying_every_cutoff(goal, miss):
    rng = np.random.default_rng(1)
    hit = rng.random(4000) < 0.2
    score = np.clip(rng.normal(0.25 + 0.3 * hit, 0.15), 0, 1)
    assert ml_eval.best_threshold(hit, score, goal, miss, 1) == \
        _brute(hit, score, goal, miss)


def test_costlier_misses_lower_the_cutoff():
    rng = np.random.default_rng(2)
    hit = rng.random(4000) < 0.2
    score = np.clip(rng.normal(0.3 + 0.3 * hit, 0.15), 0, 1)
    assert ml_eval.best_threshold(hit, score, "cost", 10, 1) < \
        ml_eval.best_threshold(hit, score, "cost", 1, 10)


def test_without_a_cutoff_the_likelier_class_wins(trained):
    run, df = trained
    res = ml.result(run, "logistic")
    assert "threshold" not in res
    labels = ml_eval.predictions(run, res)
    assert (labels == res["pred"]).all()


def test_saved_cutoff_reaches_every_scorer(trained):
    run, df = trained
    res = dict(ml.result(run, "logistic"), threshold=0.2)
    j = run["classes"].index("Yes")
    p = res["proba"][:, j]
    expected = np.where(p >= 0.2, "Yes", "No")
    assert (ml_eval.predictions(run, res) == expected).all()
    assert (ml_eval.prediction_columns(run, res)["ML_Prediction"]
            == expected).all()
    new = df.tail(300).drop(columns=["Churned"])
    scored, _ = ml_eval.score_file(run, res, new)
    want = np.where(scored["ML_Probability"] >= 0.2, "Yes", "No")
    assert (scored["ML_Prediction"] == want).all()
    # The model file: the API, batch jobs and monitoring all score with it.
    info, why = ml_upload.read(ml_eval.model_bytes(run, res), "model.joblib")
    assert why is None and info["threshold"] == 0.2
    out = serving.score(info, new)
    assert (out["ML_Prediction"] == np.where(
        out["ML_Probability_Yes"] >= 0.2, "Yes", "No")).all()
    assert (out["ML_Prediction"] == "Yes").mean() > \
        (ml.predict("classification", res["pipeline"],
                    ml_eval.features_frame(run, new),
                    run["classes"])[0] == "Yes").mean()
    assert "at least 0.20 (chosen cut-off)" in ml_eval.model_card(run, res)


def test_old_model_files_score_as_before(trained):
    run, df = trained
    res = ml.result(run, "logistic")
    info, _ = ml_upload.read(ml_eval.model_bytes(run, res), "model.joblib")
    info.pop("threshold")             # a file saved before cut-offs existed
    out = serving.score(info, df.tail(100).drop(columns=["Churned"]))
    assert set(out["ML_Prediction"]) <= {"Yes", "No"}


def test_cutoff_saved_with_the_stored_run(trained):
    run, _ = trained
    pid = projects.create_project(ANA, "Cut-offs")["id"]
    projects.share(ANA, pid, VIC["email"], "viewer")
    rid = projects.save_run(ANA, pid, run)
    stored = projects.load_run(ANA, pid, rid)
    res = ml.result(stored, "logistic")
    res["threshold"] = 0.31
    with pytest.raises(AccessDenied):
        projects.set_threshold(VIC, pid, rid, stored, "logistic", 0.31)
    projects.set_threshold(ANA, pid, rid, stored, "logistic", 0.31)
    again = projects.load_run(ANA, pid, rid)
    assert ml.result(again, "logistic")["threshold"] == 0.31
    e = audit.events(ANA, pid, action="run.threshold")[0]
    assert e["details"] == {"model": "logistic", "threshold": 0.31}
