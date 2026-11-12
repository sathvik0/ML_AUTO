"""How sure the results are: bootstrap intervals on the test rows, the
paired best-vs-runner-up check, and scores by segment."""
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from logic import ml, ml_stats


@pytest.fixture(scope="module")
def df():
    return pd.read_csv("sample_data/insurance_churn.csv").head(3000)


def config(task="classification", models=("logistic", "random_forest"),
           metric=None):
    target = "Churned" if task == "classification" else "Annual_Premium"
    return {"task": task, "target": target,
            "positive": "Yes" if task == "classification" else None,
            "target_rule": None,
            "numeric": ["Tenure_Years", "Service_Calls_12M",
                        "Late_Payments_12M", "Age"],
            "categorical": ["Region", "Sales_Channel"],
            "prep": {"num_missing": "median", "cat_missing": "category",
                     "scale": "standard", "rare_share": 0.0,
                     "missing_flags": False, "log_target": False, "pca": 0,
                     "cat_encoding": "onehot"},
            "validation": {"test_size": 0.25, "folds": 0,
                           "stratify": task == "classification",
                           "group": None},
            "seed": 1, "metric": metric, "balance": False,
            "models": [{"key": k, "params": ml.default_params(task, k),
                        "search": None} for k in models], "k_scan": None}


@pytest.fixture(scope="module")
def run(df):
    out = ml.train(df, config(), lambda f, t: None)
    out["prep_steps"], out["importance"], out["frame"] = [], {}, df
    return out


def test_intervals_hold_each_score_and_repeat(run):
    iv, pair = ml_stats.intervals(run)
    for key, v in iv.items():
        if key == ml.BASELINE:
            continue
        assert v["low"] < v["value"] < v["high"]
        assert v["high"] - v["low"] < 0.15          # 750 test rows
    again, _ = ml_stats.intervals(run)
    assert again == iv                               # seeded


def test_paired_check_names_the_runner_up(run):
    _, pair = ml_stats.intervals(run)
    others = [r["key"] for r in run["results"]
              if r["key"] not in (ml.BASELINE, run["best"])]
    assert pair["best"] == run["best"] and pair["other"] in others
    assert 0 <= pair["wins"] <= 1 and pair["low"] <= pair["high"]
    assert pair["resamples"] == ml_stats.RESAMPLES


def test_lower_is_better_metrics_keep_the_sign(df):
    run = ml.train(df, config("regression", ("ridge", "tree"), "rmse"),
                   lambda f, t: None)
    iv, pair = ml_stats.intervals(run)
    best = ml.result(run, run["best"])["test"]["rmse"]
    other = ml.result(run, pair["other"])["test"]["rmse"]
    assert best < other
    # A positive difference favours the best, on RMSE too.
    assert pair["wins"] > 0.5 and pair["high"] > 0


def test_one_model_has_no_pair(df):
    run = ml.train(df, config(models=("logistic",)), lambda f, t: None)
    iv, pair = ml_stats.intervals(run)
    assert pair is None and set(iv) == {ml.BASELINE, "logistic"}


def test_segments_cover_the_test_rows(run, df):
    res = ml.result(run, run["best"])
    seg = ml_stats.segments(run, res, df, "Sales_Channel")
    assert seg["rows"].sum() == run["n_test"]
    scored = seg.dropna(subset=["gap"])
    assert list(scored["gap"]) == sorted(scored["gap"])   # worst first
    bands = ml_stats.segments(run, res, df, "Age")
    assert len(bands) == 4 and all(" to " in s for s in bands["segment"])
    # A tiny segment is listed but not scored.
    tiny = df.assign(Flag=np.where(np.arange(len(df)) % 200 == 0, "rare",
                                   "common"))
    got = ml_stats.segments(run, res, tiny, "Flag").set_index("segment")
    assert got.loc["rare", "value"] is None or pd.isna(
        got.loc["rare", "value"])


def _results_page():
    from components import ml_results
    ml_results.render()


def test_results_page_shows_intervals_and_segments(run):
    at = AppTest.from_function(_results_page, default_timeout=180)
    at.session_state["ml_run"] = run
    at.session_state["ml_df"] = run["frame"]
    at.run()
    assert not at.exception
    board = next(m.value for m in at.markdown if "ent-table" in m.value)
    assert "ROC AUC 95% interval" in board and " to " in board
    sure = [c.value for c in at.caption if "resamples of the test rows" in
            c.value] + [w.value for w in at.warning
                        if "resamples of the test rows" in w.value]
    assert sure and "beat" in sure[0]
    seg = [m.value for m in at.markdown if "Vs all test rows" in m.value]
    assert seg and "Test rows" in seg[0]
