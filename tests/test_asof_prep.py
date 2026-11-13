"""Nothing from the future: preparation values learnt from the training
rows only, and cautions about features recorded after the outcome."""
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from logic import asof, ml, ml_eval, ml_prep


@pytest.fixture(scope="module")
def df():
    out = pd.read_csv("sample_data/insurance_churn.csv").head(2000)
    out.loc[out.index[::7], "Annual_Premium"] = np.nan        # gaps to fill
    return out


def config(steps_cols=("Annual_Premium", "Tenure_Years"), task="classification"):
    return {"task": task, "target": "Churned", "positive": "Yes",
            "target_rule": None,
            "numeric": list(steps_cols) + ["Service_Calls_12M"],
            "categorical": ["Region"],
            "prep": {"num_missing": "median", "cat_missing": "category",
                     "scale": "standard", "rare_share": 0.0,
                     "missing_flags": False, "log_target": False, "pca": 0,
                     "cat_encoding": "onehot"},
            "validation": {"test_size": 0.3, "folds": 0, "stratify": True,
                           "group": None},
            "seed": 3, "metric": None, "balance": False,
            "models": [{"key": "logistic", "params": ml.default_params(
                "classification", "logistic"), "search": None}],
            "k_scan": None}


STEPS = [
    {"kind": "fill", "column": "Annual_Premium", "value": -1.0,
     "learn": {"fill": "median"}},
    {"kind": "cap", "column": "Tenure_Years", "low": None, "high": -1.0,
     "learn": {"cap": "pct", "low": None, "high": 0.9}},
    {"kind": "drop_duplicates"},
    {"kind": "bin", "column": "Service_Calls_12M", "edges": [0, 1, 2],
     "into": "Calls_band", "learn": {"bins": 4, "how": "quantile"}},
]


def test_relearn_each_kind():
    rows = pd.DataFrame({"x": [1.0, 2, 3, 4, 100], "c": list("aabbb")})
    cap = ml_prep.relearn({"kind": "cap", "column": "x", "low": None,
                           "high": 0, "learn": {"cap": "iqr", "k": 1.5}},
                          rows)
    q1, q3 = np.quantile(rows["x"], [0.25, 0.75])
    assert cap["high"] == pytest.approx(q3 + 1.5 * (q3 - q1))
    fill = ml_prep.relearn({"kind": "fill", "column": "c", "value": "z",
                            "learn": {"fill": "mode"}}, rows)
    assert fill["value"] == "b"
    frozen = {"kind": "fill", "column": "x", "value": 7.0}
    assert ml_prep.relearn(frozen, rows) is frozen


def test_refit_learns_from_training_rows_only(df):
    train = df.index[:1000]
    got = ml_prep.refit(df, STEPS, train,
                        {"Annual_Premium", "Tenure_Years", "Calls_band"})
    assert got[0]["value"] == float(df.loc[train, "Annual_Premium"].median())
    assert got[1]["high"] == pytest.approx(
        np.quantile(df.loc[train, "Tenure_Years"], 0.9))
    assert got[2] == STEPS[2]                         # a row step as it was
    assert len(got[3]["edges"]) >= 3
    # A step on a column that is not a feature keeps its values.
    kept = ml_prep.refit(df, STEPS, train, {"Tenure_Years"})
    assert kept[0]["value"] == -1.0


def test_training_relearns_and_keeps_those_steps(df):
    prepared = ml_prep.apply_steps(df, STEPS)
    cfg = config(("Annual_Premium", "Tenure_Years"))
    run = ml.train(prepared, cfg, lambda f, t: None, source=df, steps=STEPS)
    assert run["prep_learnt"]
    train = prepared.index[np.flatnonzero(
        np.char.startswith(np.asarray(run["split"], dtype=str), "train"))]
    learnt = run["prep_steps"]
    assert learnt[0]["value"] == float(
        df.loc[train, "Annual_Premium"].median())
    assert learnt[1]["high"] == pytest.approx(
        np.quantile(df.loc[train, "Tenure_Years"], 0.9))
    assert run["frame"]["Tenure_Years"].max() == pytest.approx(
        learnt[1]["high"])
    card = ml_eval.model_card(run, ml.result(run, "logistic"))
    assert "worked out from the training rows only" in card


def test_no_learnt_steps_no_change(df):
    plain = [{"kind": "drop_duplicates"}]
    run = ml.train(ml_prep.apply_steps(df, plain), config(), lambda f, t: None,
                   source=df, steps=plain)
    assert "prep_steps" not in run and "prep_learnt" not in run


def _with_dates(df):
    out = df.copy()
    rng = np.random.default_rng(0)
    base = pd.Timestamp("2024-01-01") + pd.to_timedelta(
        rng.integers(0, 600, len(out)), unit="D")
    out["Renewal_Date"] = base.strftime("%Y-%m-%d")
    out["Cancel_Date"] = (base + pd.to_timedelta(
        rng.integers(1, 90, len(out)), unit="D")).strftime("%Y-%m-%d")
    out["Signup_Date"] = (base - pd.to_timedelta(
        rng.integers(30, 900, len(out)), unit="D")).strftime("%Y-%m-%d")
    out["Cancellation_Reason"] = "price"
    for c in ("Cancel_Date", "Signup_Date"):
        out[f"{c}_month"] = pd.to_datetime(out[c]).dt.month
    return out


def test_cautions_late_dates_and_names(df):
    data = _with_dates(df)
    cfg = {"task": "classification", "target": "Churned",
           "numeric": ["Age", "Cancel_Date_month", "Signup_Date_month"],
           "categorical": ["Cancellation_Reason"],
           "validation": {"time": "Renewal_Date"}}
    got = asof.cautions(data, cfg)
    assert any(c.startswith("Cancel_Date is later than Renewal_Date in "
                            "100%") for c in got)
    assert not any(c.startswith("Signup_Date") for c in got)  # before: fine
    assert any(c.startswith("Cancellation_Reason: its name") for c in got)
    # Without a time column, only the names can be judged.
    no_time = {**cfg, "validation": {}}
    assert not any("later than" in c for c in asof.cautions(data, no_time))


def test_the_sample_raises_no_caution(df):
    cfg = config()
    cfg["numeric"] = [c for c in df.columns if c not in (
        "Customer_ID", "Churned") and pd.api.types.is_numeric_dtype(df[c])]
    cfg["categorical"] = [c for c in df.columns if c not in (
        "Customer_ID", "Churned") and not pd.api.types.is_numeric_dtype(
        df[c])]
    assert asof.cautions(df, cfg) == []


def _configure_page():
    from components import ml_config
    ml_config.render()


def test_configure_page_shows_the_caution(df):
    data = _with_dates(df).head(600)
    at = AppTest.from_function(_configure_page, default_timeout=120)
    at.session_state["ml_source"] = data
    at.session_state["ml_df"] = data
    at.session_state["ml_steps"] = []
    at.run()
    next(r for r in at.radio if r.key == "ml_w_split").set_value(
        "time").run()
    assert not at.exception
    texts = [w.value for w in at.warning]
    assert any("Cancel_Date_month: its name" in t for t in texts)
    assert any("Cancel_Date is later than Renewal_Date" in t for t in texts)
    # A one-value column is not a feature, so it is not judged.
    assert not any("Cancellation_Reason" in t for t in texts)
    # Cautions do not block training.
    assert not next(b for b in at.button if b.key == "ml_train").disabled
