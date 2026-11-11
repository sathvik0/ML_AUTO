"""Time-aware validation: the latest rows are the test, cross-validation
walks forward, and nothing on the training side is as late as a test row."""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from logic import ml, ml_eval, ml_recipe


@pytest.fixture(scope="module")
def df():
    out = pd.read_csv("sample_data/insurance_churn.csv").head(3000)
    rng = np.random.default_rng(0)
    out["Renewal_Date"] = (pd.Timestamp("2024-01-01") + pd.to_timedelta(
        rng.integers(0, 700, len(out)), unit="D")).strftime("%d/%m/%Y")
    out["Month"] = rng.integers(1, 25, len(out))
    return out


def config(time="Renewal_Date", folds=4, group=None, models=("logistic",),
           search=None):
    return {"task": "classification", "target": "Churned", "positive": "Yes",
            "target_rule": None,
            "numeric": ["Tenure_Years", "Service_Calls_12M",
                        "Late_Payments_12M"],
            "categorical": ["Region"],
            "prep": {"num_missing": "median", "cat_missing": "category",
                     "scale": "standard", "rare_share": 0.0,
                     "missing_flags": False, "log_target": False, "pca": 0,
                     "cat_encoding": "onehot"},
            "validation": {"test_size": 0.2, "folds": folds,
                           "stratify": False, "group": group, "time": time},
            "seed": 1, "metric": None, "balance": False,
            "models": [{"key": k, "params": ml.default_params(
                "classification", k), "search": search} for k in models],
            "k_scan": None}


def test_folds_walk_forward_and_never_split_a_time():
    times = np.repeat(np.arange(20.0), 5)[np.random.default_rng(1)
                                         .permutation(100)]
    folds = ml_recipe.time_folds(times, 4)
    assert len(folds) == 4
    for train, val in folds:
        assert times[train].max() < times[val].min()
        assert not set(times[train]) & set(times[val])
    starts = [times[val].min() for _, val in folds]
    assert starts == sorted(starts)
    seen = np.unique(np.concatenate([v for _, v in folds]))
    assert times[np.setdiff1d(np.arange(100), seen)].max() < min(starts)


def test_split_puts_ties_on_the_test_side():
    times = np.array([1, 2, 3, 3, 3, 4, 5, 5, 5, 5], dtype=float)
    recipe = {"validation": {"test_size": 0.3, "time": "t"}}
    train, test = ml_recipe.split(np.zeros(10), None, recipe, times)
    assert times[train].max() < times[test].min()
    assert sorted(times[test]) == [5, 5, 5, 5]   # the latest time, all of it


def test_training_respects_time(df):
    run = ml.train(df, config(), lambda f, t: None)
    d = pd.to_datetime(df["Renewal_Date"], format="%d/%m/%Y")
    s = pd.Series(run["split"])
    assert d[s.ne("test").to_numpy()].max() < d[s.eq("test").to_numpy()] \
        .min()
    ts = run["time_split"]
    assert ts["column"] == "Renewal_Date" and ts["unvalidated"] > 0
    assert run["base"]["validation"]["time_format"] == "%d/%m/%Y"
    assert s.eq("train (in sample)").sum() == ts["unvalidated"]
    res = ml.result(run, "logistic")
    assert len(res["cv"]["roc_auc"]) == 4
    assert not np.isnan(res["proba"]).any()     # every row has a score
    card = ml_eval.model_card(run, res)
    assert "Split by time on Renewal_Date" in card


def test_a_number_can_be_the_time(df):
    run = ml.train(df, config(time="Month", folds=3), lambda f, t: None)
    s = pd.Series(run["split"])
    assert df["Month"][s.ne("test").to_numpy()].max() < \
        df["Month"][s.eq("test").to_numpy()].min()


def test_tuning_searches_on_walk_forward_folds(df):
    search = {"mode": "grid", "space": {"C": [0.1, 1.0]}, "n_iter": 2}
    run = ml.train(df, config(search=search), lambda f, t: None)
    res = ml.result(run, "logistic")
    assert res["tuned"] and len(res["search"]) == 2


def test_validation_refuses_what_cannot_split(df):
    gaps = df.copy()
    gaps.loc[gaps.index[:5], "Renewal_Date"] = None
    assert any("have no Renewal_Date" in e
               for e in ml.validate(gaps, config()))
    assert any("not both" in e
               for e in ml.validate(df, config(group="Region")))
    few = df.assign(Renewal_Date="01/01/2025")
    assert any("distinct value" in e for e in ml.validate(few, config()))
    assert ml.validate(df, config()) == []


def test_exported_scripts_split_by_time(df, tmp_path):
    data = tmp_path / "file.csv"
    df.to_csv(data, index=False)
    run = ml.train(df, config(folds=3), lambda f, t: None)
    run["prep_steps"] = []
    search = {"mode": "random", "space": {"C": [0.1, 1.0]}, "n_iter": 2}
    for name, code in (
            ("train.py", ml_eval.script(run, ml.result(run, "logistic"))),
            ("configured.py", ml_eval.config_script(
                df, config(folds=3, search=search), []))):
        if name == "configured.py":
            assert "'time_format': '%d/%m/%Y'" in code
        script = tmp_path / name
        script.write_text(code, encoding="utf-8")
        got = subprocess.run([sys.executable, str(script), str(data)],
                             capture_output=True, text=True, cwd=tmp_path)
        assert got.returncode == 0, (name, got.stderr[-2000:])
