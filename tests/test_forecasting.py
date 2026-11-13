"""Forecasting: past values of a column per series, WAPE, a naive forecast
to beat, and a caution when past values meet a random split."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from logic import asof, ml, ml_eval, ml_prep, ml_stats

APP = str(Path(__file__).resolve().parents[1] / "app.py")
STEP = {"kind": "lags", "column": "Sales", "time": "Date", "group": "Store",
        "lags": [1, 7], "windows": [7], "time_format": "%Y-%m-%d"}
FEATURES = ["Sales_lag_1", "Sales_lag_7", "Sales_mean_7"]


@pytest.fixture(scope="module")
def sales():
    df = pd.read_csv("sample_data/store_sales.csv")
    return df[df["Store"].isin(["S01", "S03", "S07"])].reset_index(drop=True)


def config(time="Date"):
    return {"task": "regression", "target": "Sales", "positive": None,
            "target_rule": None, "numeric": list(FEATURES),
            "categorical": ["Store", "Promotion", "Holiday"],
            "prep": {"num_missing": "median", "cat_missing": "category",
                     "scale": "standard", "rare_share": 0.0,
                     "missing_flags": False, "log_target": False, "pca": 0,
                     "cat_encoding": "onehot"},
            "validation": {"test_size": 0.2, "folds": 0, "stratify": False,
                           "group": None, "time": time},
            "seed": 1, "metric": "wape", "balance": False,
            "models": [{"key": "ridge", "params": ml.default_params(
                "regression", "ridge"), "search": None}],
            "k_scan": None}


def test_past_values_per_series_in_time_order():
    df = pd.DataFrame({"t": [3, 1, 2, 1, 2, 3],
                       "g": ["a", "a", "a", "b", "b", "b"],
                       "y": [30.0, 10, 20, 100, 200, 300]})
    got = ml_prep.past_values(df, {"column": "y", "time": "t", "group": "g",
                                   "lags": [1], "windows": [2]})
    # Row 0 is a's third day: yesterday was 20; the two before average 15.
    assert np.allclose(got["y_lag_1"].fillna(-1), [20, -1, 10, -1, 100, 200])
    assert np.allclose(got["y_mean_2"].fillna(-1), [15, -1, 10, -1, 100, 150])
    # A row never sees its own value.
    assert not (got["y_lag_1"] == df["y"]).any()


def test_one_series_and_dates(sales):
    one = sales[sales["Store"] == "S01"].sample(frac=1, random_state=0)
    step = {**STEP, "group": None}
    out = ml_prep.apply_steps(one, [step])
    ordered = out.assign(d=pd.to_datetime(out["Date"])).sort_values("d")
    assert np.allclose(ordered["Sales_lag_1"].iloc[1:],
                       ordered["Sales"].iloc[:-1])
    assert "Past values of Sales by Date" in ml_prep.describe(STEP)


def test_wape():
    m = ml.regression_metrics(np.array([10.0, 0, 30]),
                              np.array([12.0, 1, 27]))
    assert m["wape"] == pytest.approx(6 / 40 * 100)
    assert ml.regression_metrics(np.zeros(3), np.ones(3))["wape"] is None
    assert ml_eval.fmt_metric("wape", 12.345) == "12.3%"


@pytest.fixture(scope="module")
def run(sales):
    df = ml_prep.apply_steps(sales, [STEP])
    out = ml.train(df, config(), lambda f, t: None, source=sales,
                   steps=[STEP])
    out.setdefault("prep_steps", [STEP])     # as the Configure page does
    out["df"] = df
    return out


def test_naive_forecasts_to_beat(run):
    got = ml_stats.naive(run, run["df"])
    assert [r["k"] for r in got] == [1, 7]
    for r in got:
        assert r["rows"] > 100
        assert r["skill"] == pytest.approx(1 - r["model"] / r["naive"])
        assert r["model"] < r["naive"]          # the model beats copying
    # Not a forecast: no time split, no comparison.
    assert ml_stats.naive({**run, "time_split": None}, run["df"]) == []
    assert ml_stats.naive(run) == []           # no file, nothing read


def test_caution_past_values_with_a_random_split(sales):
    df = ml_prep.apply_steps(sales, [STEP])
    random = asof.cautions(df, config(time=None), [STEP])
    assert any("Past values of Sales are features but the split is random"
               in c for c in random)
    assert not any("Past values" in c
                   for c in asof.cautions(df, config(), [STEP]))


def test_exported_scripts_carry_the_step(sales, tmp_path):
    code = ml_eval.prep_script([STEP])
    space = {"__name__": "prep"}
    exec(compile(code, "prep.py", "exec"), space)
    out = space["apply_steps"](sales, space["PREP_STEPS"])
    pd.testing.assert_frame_equal(out, ml_prep.apply_steps(sales, [STEP]))
    assert "SCORING[\"wape\"]" in ml_eval._SCRIPT_MAIN


def _results_page():
    from components import ml_results
    ml_results.render()


def test_results_page_shows_the_naive_comparison(run):
    at = AppTest.from_function(_results_page, default_timeout=180)
    at.session_state["ml_run"] = run
    at.session_state["ml_df"] = run["df"]
    at.run()
    assert not at.exception
    assert any(m.value == "**Against a naive forecast**" for m in at.markdown)
    table = next(m.value for m in at.markdown
                 if "7 rows earlier (a week, daily)" in m.value)
    assert "Better by" in table


def _explore_page():
    from components import ml_eda
    ml_eda.render()


def test_prepare_builder_adds_the_step(sales):
    at = AppTest.from_function(_explore_page, default_timeout=120)
    at.session_state["ml_source"] = sales
    at.session_state["ml_df"] = sales
    at.session_state["ml_steps"] = []
    at.run()
    next(s for s in at.selectbox if s.key == "ml_e_kind").set_value(
        "lags").run()
    next(s for s in at.selectbox if s.key == "ml_e_lag_col").set_value(
        "Sales").run()
    next(s for s in at.selectbox if s.key == "ml_e_lag_time").set_value(
        "Date").run()
    next(s for s in at.selectbox if s.key == "ml_e_lag_group").set_value(
        "Store").run()
    assert not at.exception
    next(b for b in at.button if b.key == "ml_e_add").click().run()
    assert not at.exception
    step = at.session_state["ml_steps"][-1]
    assert {k: step[k] for k in ("kind", "column", "time", "group", "lags",
                                 "windows")} == {
        "kind": "lags", "column": "Sales", "time": "Date", "group": "Store",
        "lags": [1, 7], "windows": [7]}
    assert "Sales_lag_7" in at.session_state["ml_df"].columns


def test_sales_sample_button(tmp_path, monkeypatch):
    from utils import db
    monkeypatch.setenv("DATABASE_URL",
                       f"sqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(tmp_path / "objects"))
    monkeypatch.setenv("AUTH_MODE", "dev")
    monkeypatch.setenv("JOBS_MODE", "none")
    monkeypatch.delenv("STORAGE_BUCKET", raising=False)
    db.migrate()
    at = AppTest.from_file(APP, default_timeout=120).run()
    next(b for b in at.sidebar.button if b.label == "Data").click().run()
    next(b for b in at.button if b.key == "sample_sales").click().run()
    assert not at.exception
    assert list(at.session_state["ml_source"].columns) == [
        "Date", "Store", "Region", "Promotion", "Holiday", "Sales"]
