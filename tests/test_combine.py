"""Stacking the best models and calibrating probabilities: chosen by
cross-validation, saved as plain scikit-learn, and scored everywhere."""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from logic import ml, ml_eval, ml_upload, serving

NUMERIC = ["Tenure_Years", "Premium_Change_Pct", "Service_Calls_12M",
           "Late_Payments_12M"]


def config(task="classification", stack=0, calibration="none", folds=3,
           models=("logistic", "tree", "random_forest", "xgboost")):
    return {"task": task,
            "target": "Churned" if task == "classification"
            else "Annual_Premium",
            "positive": "Yes" if task == "classification" else None,
            "target_rule": None, "numeric": NUMERIC,
            "categorical": ["Payment_Method"],
            "prep": {"num_missing": "median", "cat_missing": "category",
                     "scale": "standard", "rare_share": 0.0,
                     "missing_flags": False, "log_target": False, "pca": 0,
                     "cat_encoding": "onehot"},
            "validation": {"test_size": 0.25, "folds": folds,
                           "stratify": task == "classification",
                           "group": None},
            "seed": 3, "metric": None, "balance": False, "stack": stack,
            "calibration": calibration,
            "models": [{"key": k, "params": ml.default_params(task, k),
                        "search": None} for k in models],
            "k_scan": None}


@pytest.fixture(scope="module")
def df():
    return pd.read_csv("sample_data/insurance_churn.csv").head(1500)


@pytest.fixture(scope="module")
def stacked(df):
    return ml.train(df, config(stack=2), lambda f, t: None)


def _cv(run, key):
    return ml._mean(ml.result(run, key)["cv"][run["metric"]])


def test_stack_holds_the_best_by_cross_validation(stacked):
    res = ml.result(stacked, ml.STACK)
    members = [m["model"] for m in res["params"]["members"]]
    others = [r for r in stacked["results"] if r["key"] not in
              (ml.BASELINE, ml.STACK, "xgboost") and not r.get("error")]
    want = sorted(others, key=lambda r: -_cv(stacked, r["key"]))[:2]
    assert members == [r["key"] for r in want]
    assert "xgboost" not in members            # learns codes, not labels
    assert res["label"].startswith("Stack: ")
    assert res["cv"] and res["test"]["roc_auc"] > 0.6
    assert "combined by a logistic regression" in \
        ml_eval.settings_text("classification", res)
    # It competes like any other model.
    assert stacked["best"] == max(
        (r["key"] for r in stacked["results"]
         if r["key"] != ml.BASELINE and not r.get("error")),
        key=lambda k: _cv(stacked, k))


def test_stack_model_file_loads_and_scores(stacked, df):
    run = dict(stacked, prep_steps=[])
    res = ml.result(run, ml.STACK)
    info, why = ml_upload.read(ml_eval.model_bytes(run, res), "model.joblib")
    assert why is None
    out = serving.score(info, df.tail(40).drop(columns=["Churned"]))
    assert set(out["ML_Prediction"]) <= {"Yes", "No"}
    assert out["ML_Probability_Yes"].between(0, 1).all()


def test_no_stack_without_folds(df):
    run = ml.train(df, config(stack=3, folds=0), lambda f, t: None)
    assert ml.STACK not in [r["key"] for r in run["results"]]


def test_regression_stack(df):
    run = ml.train(df, config("regression", stack=2,
                              models=("ridge", "tree")), lambda f, t: None)
    res = ml.result(run, ml.STACK)
    assert np.isfinite(res["test"]["r2"])


def test_calibration_wraps_and_keeps_labels(df):
    run = ml.train(df, config(calibration="sigmoid",
                              models=("logistic", "xgboost")),
                   lambda f, t: None)
    run["prep_steps"] = []
    for key in ("logistic", "xgboost"):
        res = ml.result(run, key)
        assert type(res["pipeline"][-1]).__name__ == "CalibratedClassifierCV"
        assert "calibrated (Platt scaling)" in \
            ml_eval.settings_text("classification", res)
        assert "Calibration" in ml_eval.shap_problem(run, res)
        info, why = ml_upload.read(ml_eval.model_bytes(run, res),
                                   "model.joblib")
        assert why is None
        out = serving.score(info, df.tail(40).drop(columns=["Churned"]))
        # XGBoost answers in codes even when calibrated: still labels here.
        assert set(out["ML_Prediction"]) <= {"Yes", "No"}


def test_calibration_tracks_observed_rates(df):
    """Calibrated, the predicted churn rate on the test rows matches the
    observed one more closely (on a model whose raw scores are off)."""
    def gap(run):
        res = ml.result(run, "tree")
        test = ml_eval.rows(run, "test")
        hit = ml_eval.truth(run, test) == "Yes"
        cal = ml_eval.calibration(hit, ml_eval.positive_score(run, res,
                                                              test))
        return float(np.average(np.abs(cal["predicted"] - cal["observed"]),
                                weights=cal["records"]))
    raw = ml.train(df, config(models=("tree",)), lambda f, t: None)
    cal = ml.train(df, config(calibration="isotonic", models=("tree",)),
                   lambda f, t: None)
    assert gap(cal) < gap(raw)


def test_exported_script_rebuilds_the_stack(stacked, df, tmp_path):
    run = dict(stacked, prep_steps=[])
    code = ml_eval.script(run, ml.result(run, ml.STACK))
    script, data = tmp_path / "train.py", tmp_path / "file.csv"
    script.write_text(code, encoding="utf-8")
    df.to_csv(data, index=False)
    got = subprocess.run([sys.executable, str(script), str(data)],
                         capture_output=True, text=True, cwd=tmp_path)
    assert got.returncode == 0, got.stderr[-2000:]
