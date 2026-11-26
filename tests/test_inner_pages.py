"""The inner pages on white cards: Drift analysis, Use own model and the
lower tabs of Results draw without an error, step by step."""
import io

import pandas as pd
import pytest
import streamlit
from streamlit.testing.v1 import AppTest

from logic import ml


@pytest.fixture(scope="module")
def df():
    return pd.read_csv("sample_data/insurance_churn.csv").head(1500)


def _drift_page():
    from components import drift_page
    drift_page.render()


def test_drift_page_cards(df):
    at = AppTest.from_function(_drift_page, default_timeout=180)
    at.session_state["ml_df"] = df
    at.run()
    assert any(m.value == "#### What to compare" for m in at.markdown)
    next(r for r in at.radio if r.key == "ml_d_source").set_value(
        "sample").run()
    next(b for b in at.button if b.key == "ml_d_go").click().run()
    assert not at.exception
    assert [m.label for m in at.metric] == [
        "Columns compared", "Major drift", "Moderate drift",
        "Drift classifier AUC"]
    assert any(m.value == "#### Downloads" for m in at.markdown)


def _own_model_page():
    from components import ml_evaluate
    ml_evaluate.render()


def test_own_model_page_steps(df, monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "dev")
    at = AppTest.from_function(_own_model_page, default_timeout=180)
    at.session_state["ml_df"] = df
    at.run()
    assert not at.exception
    heads = [m.value for m in at.markdown if m.value.startswith("#### ")]
    assert heads == ["#### 1. The model"]        # the rest wait for a file


def _results_page():
    from components import ml_results
    ml_results.render()


def test_results_tabs_render(df):
    num = ["Age", "Tenure_Years", "Service_Calls_12M"]
    cfg = {"task": "classification", "target": "Churned", "positive": "Yes",
           "target_rule": None, "numeric": num, "categorical": ["Region"],
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
    run = ml.train(df, cfg, lambda f, t: None)
    run["importance"] = {}
    at = AppTest.from_function(_results_page, default_timeout=180)
    at.session_state["ml_run"] = run
    at.session_state["ml_df"] = df
    at.run()
    assert not at.exception
    # The targeting tab's numbers are cards like the ones at the top.
    labels = [m.label for m in at.metric]
    assert {"Precision", "Recall", "F1", "Test rows flagged"} <= set(labels)


def test_own_model_page_evaluate_and_predict(df, monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "dev")
    from logic import ml_eval
    cfg = {"task": "classification", "target": "Churned", "positive": "Yes",
           "target_rule": None, "numeric": ["Age", "Tenure_Years"],
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
    run = ml.train(df, cfg, lambda f, t: None)
    data = ml_eval.model_bytes(run, ml.result(run, "logistic"))

    def chosen(*args, **kwargs):            # AppTest cannot upload a file
        f = io.BytesIO(data)
        f.name = "model.joblib"
        return f

    monkeypatch.setattr(streamlit, "file_uploader", chosen)
    at = AppTest.from_function(_own_model_page, default_timeout=180)
    at.session_state["ml_df"] = df
    at.run()
    assert not at.exception
    heads = [m.value for m in at.markdown if m.value.startswith("#### ")]
    assert heads == ["#### 1. The model", "#### 2. Its inputs",
                     "#### 3. Evaluate"]
    next(b for b in at.button if b.key == "ml_u_go").click().run()
    assert not at.exception
    assert at.session_state["page"] == "ml_results"
    next(r for r in at.radio if r.key == "ml_u_mode").set_value(
        "predict").run()
    assert not at.exception
    assert "#### 3. Predict" in [m.value for m in at.markdown]
    next(b for b in at.button if b.key == "ml_u_predict").click().run()
    assert not at.exception
    assert any(s.value.startswith("Predicted 1,500 rows") for s in
               at.success)
