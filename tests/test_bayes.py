"""Bayesian search with Optuna: scored by the same cross-validation, stops
weak settings early, honours a time limit, and falls back to random search
where Optuna is missing."""
import subprocess
import sys

import pandas as pd
import pytest

from logic import ml, ml_eval

pytest.importorskip("optuna")


def config(space, n_iter=12, seconds=0, mode="bayes", folds=3):
    return {"task": "classification", "target": "Churned", "positive": "Yes",
            "target_rule": None,
            "numeric": ["Tenure_Years", "Premium_Change_Pct",
                        "Service_Calls_12M", "Late_Payments_12M"],
            "categorical": ["Payment_Method"],
            "prep": {"num_missing": "median", "cat_missing": "category",
                     "scale": "standard", "rare_share": 0.0,
                     "missing_flags": False, "log_target": False, "pca": 0,
                     "cat_encoding": "onehot"},
            "validation": {"test_size": 0.25, "folds": folds,
                           "stratify": True, "group": None},
            "seed": 3, "metric": None, "balance": False,
            "models": [{"key": "tree", "params": ml.default_params(
                "classification", "tree"),
                "search": {"mode": mode, "space": space, "n_iter": n_iter,
                           "seconds": seconds}}],
            "k_scan": None}


SPACE = {"max_depth": [2, 3, 4, 5, 6, 8, 10, None],
         "min_samples_leaf": [1, 5, 20, 50, 100]}


@pytest.fixture(scope="module")
def df():
    return pd.read_csv("sample_data/insurance_churn.csv").head(1500)


@pytest.fixture(scope="module")
def run(df):
    return ml.train(df, config(SPACE), lambda f, t: None)


def test_best_setting_wins_by_cross_validation(run):
    res = ml.result(run, "tree")
    table = res["search"]
    assert res["tuned"] and "stopped_early" in table
    done = table[~table["stopped_early"]]
    top = done.sort_values("cv_mean", ascending=False).iloc[0]
    assert res["params"]["min_samples_leaf"] == top["min_samples_leaf"]
    assert ml._mean(res["cv"][run["metric"]]) == pytest.approx(
        top["cv_mean"])
    for k, values in SPACE.items():
        assert res["params"][k] in values
    assert table.loc[table["stopped_early"], "cv_mean"].isna().all()
    assert "Bayesian search, Optuna" in ml_eval.model_card(run, res)


def test_seeded_runs_repeat(run, df):
    again = ml.train(df, config(SPACE), lambda f, t: None)
    assert ml.result(again, "tree")["params"] == \
        ml.result(run, "tree")["params"]


def test_time_limit_ends_the_search(df):
    wide = {"max_depth": list(range(2, 30)),
            "min_samples_leaf": list(range(1, 60))}
    run = ml.train(df, config(wide, n_iter=100, seconds=1),
                   lambda f, t: None)
    assert len(ml.result(run, "tree")["search"]) < 100


def test_counted_like_random_search():
    m = config(SPACE, n_iter=12)["models"][0]
    assert ml.search_combinations(m) == 12
    small = config({"max_depth": [2, 3]}, n_iter=12)["models"][0]
    assert ml.search_combinations(small) == 2


def test_falls_back_to_random_without_optuna(df, monkeypatch):
    real = ml.package_problem
    monkeypatch.setattr(ml, "package_problem", lambda name: (
        "not installed" if name == "optuna" else real(name)))
    run = ml.train(df, config(SPACE, n_iter=6), lambda f, t: None)
    table = ml.result(run, "tree")["search"]
    assert len(table) == 6 and "stopped_early" not in table


def test_exported_script_samples_at_random(df, tmp_path):
    # The configure page's script searches; scripts exported after training
    # retrain with the settings the app found.
    code = ml_eval.config_script(df, config(SPACE, n_iter=4), [])
    assert "'mode': 'bayes'" in code and "import optuna" not in code
    script, data = tmp_path / "train.py", tmp_path / "file.csv"
    script.write_text(code, encoding="utf-8")
    df.to_csv(data, index=False)
    got = subprocess.run([sys.executable, str(script), str(data)],
                         capture_output=True, text=True, cwd=tmp_path)
    assert got.returncode == 0, got.stderr[-2000:]
