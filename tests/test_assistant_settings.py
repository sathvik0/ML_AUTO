"""The Configure assistant knows every setting on the page: its prompt lists
them, its answer has a field for each, and Apply sets the page's widgets,
leaving out what does not fit."""
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from components import assistant_panel, ml_config
from logic import ml, ml_assistant, ml_prep

NEW = ("split", "time_column", "cap", "rare_share", "missing_flags",
       "feature_select", "log_target", "resample", "calibration", "stack")


def proposal(**over):
    base = {k: "" for k in ml_assistant._SCHEMA["properties"]["proposal"][
        "required"]}
    base["models"] = []
    return {**base, **over}


@pytest.fixture(scope="module")
def sales():
    df = pd.read_csv("sample_data/store_sales.csv").head(1200)
    return ml_prep.apply_steps(df, ml_prep.date_step(df, "Date")[:1])


def test_schema_and_prompt_cover_the_new_settings(sales):
    spec = ml_assistant._SCHEMA["properties"]["proposal"]
    for field in NEW:
        assert field in spec["properties"] and field in spec["required"]
    assert set(spec["properties"]) == set(spec["required"])     # strict
    text = ml_assistant.system_prompt(sales, ml.profile(sales),
                                      assistant_panel._options())
    assert "Split by time" in text and "bayes" in text
    assert "time_column (with split time): Date" in text
    for field in ("cap", "resample", "calibration", "stack"):
        assert f"- {field}" in text


def _assign(p, df):
    """`_assignments` reads the session only for what the page holds."""
    return assistant_panel._assignments(p, df, ml.profile(df))


def test_regression_settings_map_onto_widgets(sales):
    out, notes = _assign(proposal(
        task="regression", target="Sales", split="time", time_column="Date",
        cap="iqr", rare_share="0.01", missing_flags="yes",
        feature_select="20", log_target="yes", stack="3", tune_mode="bayes",
        num_missing="iterative", resample="smote", calibration="sigmoid"),
        sales)
    w = ml_config.PREFIX
    assert out[f"{w}split"] == "time" and out[f"{w}time_col"] == "Date"
    assert out[f"{w}cap"] == "iqr" and out[f"{w}rare_share"] == 0.01
    assert out[f"{w}missing_flags"] is True and out[f"{w}log_target"] is True
    assert out[f"{w}select"] == 20 and out[f"{w}stack"] == 3
    assert out[f"{w}tune_mode"] == "bayes"
    assert out[f"{w}num_missing"] == "iterative"
    # Classification-only settings are left out of a regression, and said.
    assert f"{w}resample" not in out and f"{w}calibration" not in out
    assert "resample does not apply to regression." in notes
    assert "calibration does not apply to regression." in notes


def test_values_that_do_not_fit_are_listed(sales):
    out, notes = _assign(proposal(
        task="regression", target="Sales", split="time",
        time_column="Store", stack="7", cap="always"), sales)
    w = ml_config.PREFIX
    assert f"{w}time_col" not in out and "Store cannot be the time " \
        "column." in notes
    assert f"{w}stack" not in out and "stack 7 is not an option here." in notes
    assert "cap always is not an option here." in notes


def test_rare_class_widgets_get_the_metric_and_balance():
    df = pd.read_csv("sample_data/insurance_churn.csv").head(1500)
    out, _ = _assign(proposal(task="classification", target="Churned",
                              metric="average_precision",
                              balance_classes="yes", resample="under",
                              calibration="isotonic"), df)
    w = ml_config.PREFIX
    assert out[f"{w}metric_binary"] == out[f"{w}metric_binary_rare"] == \
        "average_precision"
    assert out[f"{w}balance"] is out[f"{w}balance_rare"] is True
    assert out[f"{w}resample"] == "under"
    assert out[f"{w}calibration"] == "isotonic"


def _configure_page():
    from components import ml_config as page
    page.render()


def test_apply_on_the_page(sales, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-placeholder")
    monkeypatch.setenv("AUTH_MODE", "dev")
    at = AppTest.from_function(_configure_page, default_timeout=120)
    at.session_state["ml_source"] = sales
    at.session_state["ml_df"] = sales
    at.session_state["ml_steps"] = []
    at.session_state["ml_a_shown"] = [
        {"role": "user", "text": "Forecast daily sales per store."},
        {"role": "assistant", "text": "Split by time on Date.",
         "proposal": proposal(task="regression", target="Sales",
                              split="time", time_column="Date",
                              feature_select="10", cap="pct")}]
    at.run()
    assert not at.exception
    next(b for b in at.button if b.key == "ml_a_apply_1").click().run()
    assert not at.exception
    w = ml_config.PREFIX
    assert at.session_state[f"{w}task"] == "regression"
    assert next(r for r in at.radio if r.key == f"{w}split").value == "time"
    assert next(s for s in at.selectbox if s.key == f"{w}time_col"
                ).value == "Date"
    assert next(s for s in at.selectbox if s.key == f"{w}select").value == 10
    assert next(s for s in at.selectbox if s.key == f"{w}cap").value == "pct"
    chat = [m.value for m in at.markdown if "Keep the strongest columns: 10"
            in m.value]
    assert chat
