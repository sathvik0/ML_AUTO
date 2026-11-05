"""A rare class: resampling only the rows being fitted (SMOTE-NC or
undersampling), and the exported script doing the same."""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from logic import ml, ml_eval, ml_recipe

NUMERIC = ["Age", "Tenure_Years", "Service_Calls_12M", "Late_Payments_12M",
           "Claim_Satisfaction"]
CATEGORICAL = ["Region", "Payment_Method"]


@pytest.fixture(scope="module")
def rare():
    """About 8% churners: most of the sample's Yes rows dropped."""
    df = pd.read_csv("sample_data/insurance_churn.csv").head(6000)
    yes = df.index[df["Churned"] == "Yes"]
    drop = np.random.default_rng(0).choice(yes, int(len(yes) * 0.7),
                                           replace=False)
    return df.drop(index=drop)


def config(resample, folds=3, models=("logistic",)):
    return {"task": "classification", "target": "Churned", "positive": "Yes",
            "target_rule": None, "numeric": NUMERIC,
            "categorical": CATEGORICAL,
            "prep": {"num_missing": "median", "cat_missing": "category",
                     "scale": "standard", "rare_share": 0.0,
                     "missing_flags": False, "log_target": False, "pca": 0,
                     "cat_encoding": "onehot"},
            "validation": {"test_size": 0.25, "folds": folds,
                           "stratify": True, "group": None},
            "seed": 5, "metric": "average_precision", "balance": False,
            "resample": resample,
            "models": [{"key": k, "params": ml.default_params(
                "classification", k), "search": None} for k in models],
            "k_scan": None}


def frame(df):
    X = ml_recipe.prepare_features(df, NUMERIC, CATEGORICAL)
    return X, ml_recipe.label_text(df["Churned"]).to_numpy()


def test_smallest_share(rare):
    share = ml.smallest_share(rare["Churned"])
    assert share < ml.IMBALANCED
    assert ml.smallest_share(pd.Series(["a", "a"])) is None


def test_smote_balances_with_plausible_rows(rare):
    X, y = frame(rare)
    recipe = {"task": "classification", "numeric": NUMERIC,
              "categorical": CATEGORICAL, "seed": 1, "resample": "smote"}
    Xo, yo = ml_recipe.resample(recipe, X, y)
    counts = pd.Series(yo).value_counts()
    assert counts["Yes"] == counts["No"] == (y == "No").sum()
    new = Xo.iloc[len(X):]
    yes = X[y == "Yes"]
    for c in NUMERIC:                    # between two real rows of the class
        assert new[c].min() >= yes[c].min() and new[c].max() <= yes[c].max()
    for c in CATEGORICAL:
        assert set(new[c].dropna()) <= set(yes[c].dropna())
    assert new["Claim_Satisfaction"].isna().any()      # gaps stay possible
    assert (Xo.dtypes == X.dtypes).all()
    again, _ = ml_recipe.resample(recipe, X, y)
    pd.testing.assert_frame_equal(Xo, again)           # seeded


def test_undersample_keeps_real_rows(rare):
    X, y = frame(rare)
    recipe = {"task": "classification", "numeric": NUMERIC,
              "categorical": CATEGORICAL, "seed": 1, "resample": "under"}
    Xo, yo = ml_recipe.resample(recipe, X, y)
    assert (pd.Series(yo).value_counts() == (y == "Yes").sum()).all()
    assert set(Xo.index) <= set(X.index)


def test_only_rows_being_fitted_are_resampled(rare, monkeypatch):
    seen = []
    real = ml_recipe.resample

    def spy(recipe, X, y):
        seen.append(set(X.index))
        return real(recipe, X, y)
    monkeypatch.setattr(ml_recipe, "resample", spy)
    run = ml.train(rare, config("smote"), lambda f, t: None)
    labels = rare.index.to_numpy()
    test = set(labels[run["split"] == "test"])
    train = set(labels[run["split"] == "train (out of fold)"])
    assert len(seen) == 2 * (3 + 1)        # baseline and logistic
    assert all(rows <= train and not rows & test for rows in seen)
    # Every score is still measured on real rows: the test set is untouched.
    assert len(run["split"]) == len(rare)


def test_resampling_lifts_recall_at_the_default_cutoff(rare):
    def recall(run):
        res = ml.result(run, "logistic")
        test = ml_eval.rows(run, "test")
        hit = ml_eval.truth(run, test) == "Yes"
        return (res["pred"][test][hit] == "Yes").mean()
    plain = ml.train(rare, config("none"), lambda f, t: None)
    smote = ml.train(rare, config("smote"), lambda f, t: None)
    assert recall(smote) > recall(plain) + 0.2
    card = ml_eval.model_card(smote, ml.result(smote, "logistic"))
    assert "oversampled with SMOTE" in card


def test_regression_ignores_the_setting(rare):
    cfg = config("smote")
    cfg.update(task="regression", target="Annual_Premium", positive=None)
    cfg["models"] = [{"key": "ridge", "params": ml.default_params(
        "regression", "ridge"), "search": None}]
    cfg["metric"] = None
    run = ml.train(rare, cfg, lambda f, t: None)
    assert ml.result(run, "ridge")["recipe"]["resample"] == "none"


def test_exported_script_resamples(rare, tmp_path):
    run = ml.train(rare, config("under", folds=0), lambda f, t: None)
    run["prep_steps"] = []
    code = ml_eval.script(run, ml.result(run, "logistic"))
    assert "def resample" in code
    script, data = tmp_path / "train.py", tmp_path / "file.csv"
    script.write_text(code, encoding="utf-8")
    rare.to_csv(data, index=False)
    got = subprocess.run([sys.executable, str(script), str(data)],
                         capture_output=True, text=True, cwd=tmp_path)
    assert got.returncode == 0, got.stderr[-2000:]
