"""An iterative fill for numeric gaps, feature selection inside each fit,
and a data-quality score for the file."""
import io

import joblib
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from logic import ml, ml_eval, ml_explore, ml_recipe, ml_upload


@pytest.fixture(scope="module")
def df():
    out = pd.read_csv("sample_data/insurance_churn.csv").head(2000)
    out.loc[out.index[::5], "Annual_Premium"] = np.nan
    return out


def _columns(df):
    num = [c for c in df.columns if c not in ("Customer_ID", "Churned")
           and pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in df.columns if c not in ("Customer_ID", "Churned")
           and c not in num]
    return num, cat


def config(df, models=("logistic", "lightgbm"), **prep):
    num, cat = _columns(df)
    return {"task": "classification", "target": "Churned", "positive": "Yes",
            "target_rule": None, "numeric": num, "categorical": cat,
            "prep": {"num_missing": "median", "cat_missing": "category",
                     "scale": "standard", "rare_share": 0.0,
                     "missing_flags": False, "log_target": False, "pca": 0,
                     "cat_encoding": "onehot", **prep},
            "validation": {"test_size": 0.3, "folds": 3, "stratify": True,
                           "group": None},
            "seed": 3, "metric": None, "balance": False,
            "models": [{"key": k, "params": ml.default_params(
                "classification", k), "search": None} for k in models],
            "k_scan": None}


def test_iterative_fill_predicts_from_other_numbers():
    rng = np.random.default_rng(0)
    a = rng.normal(size=400)
    X = pd.DataFrame({"a": a, "b": 2 * a + rng.normal(0, 0.05, 400)})
    X.loc[::4, "b"] = np.nan
    rec = {"task": "regression", "model": "linear", "params": {},
           "numeric": ["a", "b"], "categorical": [], "seed": 0,
           "prep": {"num_missing": "iterative", "scale": "none"}}
    front = ml_recipe.build_pipeline(rec)[:-1].fit(X)
    filled = front.transform(X)[::4, 1]
    assert np.allclose(filled, 2 * a[::4], atol=0.2)     # not one median


def test_training_with_both_and_explaining(df):
    run = ml.train(df, config(df, num_missing="iterative", select=8,
                              missing_flags=True), lambda f, t: None)
    assert all(r["error"] is None for r in run["results"])
    res = ml.result(run, "logistic")
    assert list(res["pipeline"].named_steps) == ["prep", "select", "model"]
    names = ml_eval.feature_map(res["pipeline"], res["recipe"])
    assert len(names) == 8
    ex = ml_eval.explain(run, res, df, np.arange(10))
    assert set(ex["features"]) == set(names)
    card = ml_eval.model_card(run, res)
    assert "predicted from the other numbers (iterative)" in card
    assert "Feature selection: the 8 prepared columns" in card
    # Gradient boosting reads encoded categories once columns are selected.
    boost = ml.result(run, "lightgbm")["recipe"]
    assert not ml_recipe.native_categories(boost)


def test_more_than_there_are_keeps_all(df):
    run = ml.train(df, config(df, models=("logistic",), select=50),
                   lambda f, t: None)
    res = ml.result(run, "logistic")
    full = ml.train(df, config(df, models=("logistic",)), lambda f, t: None)
    assert len(ml_eval.feature_map(res["pipeline"], res["recipe"])) == len(
        ml_eval.feature_map(ml.result(full, "logistic")["pipeline"],
                            ml.result(full, "logistic")["recipe"]))


def test_saved_model_passes_the_upload_check(df):
    num, _ = _columns(df)
    rec = {"task": "classification", "model": "logistic", "params": {},
           "numeric": num, "categorical": [], "seed": 0,
           "prep": {"num_missing": "iterative", "scale": "standard",
                    "select": 5}}
    X = ml_recipe.prepare_features(df, num, [])
    pipe = ml_recipe.build_pipeline(rec).fit(X, df["Churned"])
    buf = io.BytesIO()
    joblib.dump(pipe, buf)
    back = ml_upload._guarded_load(buf.getvalue())
    assert np.array_equal(back.predict_proba(X), pipe.predict_proba(X))


def test_quality_score():
    clean = pd.DataFrame({"x": np.arange(100.0), "y": list("ab") * 50,
                          "z": np.arange(100) % 7,
                          "w": np.linspace(0, 1, 100)})   # rows differ
    q = ml_explore.quality(clean, ml.profile(clean))
    assert q["score"] == 100 and q["grade"] == "good"
    messy = clean.copy()
    messy.loc[:39, "x"] = np.nan                       # 40% of x: 10% cells
    messy["k"] = 1                                     # constant
    messy = pd.concat([messy, messy.head(10)], ignore_index=True)  # dups
    q = ml_explore.quality(messy, ml.profile(messy))
    by = {c["check"]: c for c in q["checks"]}
    assert by["unique"]["points"] < 20 and "10 duplicate rows" in \
        by["unique"]["finding"]
    assert "k" in by["columns"]["finding"]
    assert by["filled"]["points"] < 35
    assert q["score"] == round(sum(c["points"] for c in q["checks"]))
    assert q["grade"] in ("fair", "poor")


def _explore_page():
    from components import ml_eda
    ml_eda.render()


def test_explore_page_shows_the_score(df):
    at = AppTest.from_function(_explore_page, default_timeout=120)
    at.session_state["ml_source"] = df
    at.session_state["ml_df"] = df
    at.session_state["ml_steps"] = []
    at.run()
    assert not at.exception
    assert any(m.value.startswith("**Data quality: ") for m in at.markdown)
    assert any("Usable columns" in m.value for m in at.markdown)


def _configure_page():
    from components import ml_config
    ml_config.render()


def test_configure_page_offers_both(df):
    at = AppTest.from_function(_configure_page, default_timeout=120)
    at.session_state["ml_source"] = df
    at.session_state["ml_df"] = df
    at.session_state["ml_steps"] = []
    at.run()
    assert not at.exception
    sel = next(s for s in at.selectbox if s.key == "ml_w_select")
    assert sel.value == 0
    fill = next(s for s in at.selectbox if s.key == "ml_w_num_missing")
    fill.set_value("iterative").run()
    assert not at.exception
    assert any("iterative fill predicts" in c.value for c in at.caption)
