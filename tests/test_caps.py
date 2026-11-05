"""Capping extreme numbers inside the pipeline: levels learnt from each
fit's training rows only, kept in the saved model, and replayed by the
exported script."""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from logic import ml, ml_eval, ml_recipe, ml_upload, serving

NUMERIC = ["Tenure_Years", "Annual_Premium", "Total_Claim_Amount",
           "Service_Calls_12M"]


def config(cap, task="classification", folds=3, models=("logistic",)):
    return {"task": task, "target": "Churned" if task != "clustering"
            else None, "positive": "Yes", "target_rule": None,
            "numeric": NUMERIC, "categorical": ["Region"],
            "prep": {"num_missing": "median", "cat_missing": "category",
                     "scale": "standard", "rare_share": 0.0,
                     "missing_flags": False, "log_target": False, "pca": 0,
                     "cat_encoding": "onehot", "cap": cap},
            "validation": {"test_size": 0.25, "folds": folds,
                           "stratify": True, "group": None},
            "seed": 4, "metric": None, "balance": False,
            "models": [{"key": k, "params": ml.default_params(task, k),
                        "search": None} for k in models],
            "k_scan": None}


@pytest.fixture(scope="module")
def df():
    return pd.read_csv("sample_data/insurance_churn.csv").head(2500)


def test_no_test_or_validation_row_sets_a_cap(df, monkeypatch):
    seen = []
    real = ml_recipe.with_caps

    def spy(recipe, X):
        seen.append(set(X.index))
        return real(recipe, X)
    monkeypatch.setattr(ml_recipe, "with_caps", spy)
    run = ml.train(df, config("pct"), lambda f, t: None)
    test = set(np.flatnonzero(run["split"] == "test"))
    train = set(np.flatnonzero(run["split"] == "train (out of fold)"))
    # Baseline and logistic regression: each fold, then the final fit.
    assert len(seen) == 2 * (3 + 1)
    assert all(not (rows & test) for rows in seen)
    assert sum(rows == train for rows in seen) == 2
    assert all(rows <= train for rows in seen)      # a fold: part of train


def test_levels_are_the_training_rows_percentiles(df):
    run = ml.train(df, config("pct"), lambda f, t: None)
    res = ml.result(run, "logistic")
    train = df.iloc[np.flatnonzero(run["split"] != "test")]
    want = [np.nanquantile(pd.to_numeric(train[c]), [0.01, 0.99])
            for c in NUMERIC]
    assert np.allclose(res["recipe"]["caps"]["low"], [w[0] for w in want])
    assert np.allclose(res["recipe"]["caps"]["high"], [w[1] for w in want])
    assert "1st and 99th percentiles" in ml_eval.model_card(run, res)


def test_extreme_values_score_as_the_cap(df):
    run = ml.train(df, config("iqr"), lambda f, t: None)
    res = ml.result(run, "logistic")
    caps = res["recipe"]["caps"]
    row = df.head(1).drop(columns=["Churned"]).copy()
    at_cap, beyond = row.copy(), row.copy()
    at_cap["Annual_Premium"] = caps["high"][1]
    beyond["Annual_Premium"] = caps["high"][1] * 100
    info, _ = ml_upload.read(ml_eval.model_bytes(run, res), "model.joblib")
    a = serving.score(info, at_cap)["ML_Probability_Yes"].iloc[0]
    b = serving.score(info, beyond)["ML_Probability_Yes"].iloc[0]
    assert a == pytest.approx(b)


def test_off_by_default_changes_nothing(df):
    run = ml.train(df, config("none"), lambda f, t: None)
    res = ml.result(run, "logistic")
    assert "caps" not in res["recipe"]
    num = res["pipeline"].named_steps["prep"].named_transformers_["num"]
    assert "cap" not in num.named_steps


def test_clustering_caps_on_all_rows(df):
    cfg = config("pct", task="clustering", models=("kmeans",))
    cfg["models"][0]["params"]["n_clusters"] = 3
    run = ml.train(df, cfg, lambda f, t: None)
    assert ml.result(run, "kmeans")["recipe"]["caps"]["high"]


def test_exported_script_learns_its_own_caps(df, tmp_path):
    run = ml.train(df, config("pct", folds=0), lambda f, t: None)
    run["prep_steps"] = []
    code = ml_eval.script(run, ml.result(run, "logistic"))
    assert "def with_caps" in code
    script, data = tmp_path / "train.py", tmp_path / "file.csv"
    script.write_text(code, encoding="utf-8")
    df.to_csv(data, index=False)
    got = subprocess.run([sys.executable, str(script), str(data)],
                         capture_output=True, text=True, cwd=tmp_path)
    assert got.returncode == 0, got.stderr[-2000:]
