"""Date columns: found by the profile, turned into numbers by a recorded
step, and the same numbers on a file scored later and in the exported
script."""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from logic import ml, ml_eval, ml_prep


@pytest.fixture(scope="module")
def churn():
    df = pd.read_csv("sample_data/insurance_churn.csv").head(3000)
    rng = np.random.default_rng(0)
    start = pd.Timestamp("2021-01-01") + pd.to_timedelta(
        rng.integers(0, 1000, len(df)), unit="D")
    df["Policy_Start"] = start.strftime("%d/%m/%Y")      # day first
    return df


def test_profile_marks_dates_and_leaves_numbers_alone(churn):
    prof = ml.profile(churn).set_index("column")
    assert prof.loc["Policy_Start", "kind"] == "date"
    assert not prof.loc["Policy_Start", "use"]
    assert "Turn dates into numbers" in prof.loc["Policy_Start", "note"]
    assert prof.loc["Tenure_Years", "kind"] == "feature"
    assert not ml_prep.looks_like_date(pd.Series([20240331, 20240401]))
    assert not ml_prep.looks_like_date(pd.Series(["West", "East"]))


def test_step_is_frozen_and_reads_day_first(churn):
    step, problem = ml_prep.date_step(churn, "Policy_Start")
    assert problem is None
    assert step["format"] == "%d/%m/%Y"
    assert step["reference"] == pd.to_datetime(
        churn["Policy_Start"], format="%d/%m/%Y").min().strftime("%Y-%m-%d")
    assert "hour" not in step["parts"]          # no time of day recorded
    out = ml_prep.apply_step(churn, step)
    assert out["Policy_Start_days_since"].min() == 0
    assert set(out["Policy_Start_weekday"].unique()) == set(range(7))
    month = out["Policy_Start_month"]
    assert np.allclose(out["Policy_Start_month_sin"] ** 2
                       + out["Policy_Start_month_cos"] ** 2, 1)
    assert month.between(1, 12).all()
    prof = ml.profile(out).set_index("column")
    assert "used instead" in prof.loc["Policy_Start", "note"]
    assert prof.loc["Policy_Start_month", "use"]


def test_time_of_day_and_gaps():
    s = pd.DataFrame({"t": ["2024-03-30 22:15", "2024-03-31 06:00", None,
                            "not a date"]})
    step, _ = ml_prep.date_step(s, "t")
    assert "hour" in step["parts"]
    out = ml_prep.apply_step(s, step)
    assert out["t_hour"].tolist()[:2] == [22, 6]
    assert out["t_is_weekend"].tolist()[:2] == [1, 1]
    assert out.loc[2:, "t_year"].isna().all()   # gaps stay gaps


def test_new_file_scored_with_the_same_numbers(churn):
    step, _ = ml_prep.date_step(churn, "Policy_Start")
    parts = [f"Policy_Start_{p}" for p in ("month", "weekday",
                                           "days_since")]
    train = ml_prep.apply_steps(churn, [step])
    config = {"task": "classification", "target": "Churned",
              "positive": "Yes", "target_rule": None,
              "numeric": ["Tenure_Years", "Service_Calls_12M"] + parts,
              "categorical": ["Region"],
              "prep": {"num_missing": "median", "cat_missing": "category",
                       "scale": "standard", "rare_share": 0.0,
                       "missing_flags": False, "log_target": False,
                       "pca": 0, "cat_encoding": "onehot"},
              "validation": {"test_size": 0.2, "folds": 3, "stratify": True,
                             "group": None},
              "seed": 1, "metric": None, "balance": False,
              "models": [{"key": "logistic", "params": ml.default_params(
                  "classification", "logistic"), "search": None}],
              "k_scan": None}
    run = ml.train(train, config, lambda f, t: None)
    run["prep_steps"] = [step]
    res = ml.result(run, "logistic")
    # A later file arrives with the date as text and no feature columns;
    # the step fills them in, with the reference date fixed at training.
    new = churn.tail(50).drop(columns=["Churned"])
    scored, notes = ml_eval.score_file(run, res, new)
    assert scored is not None, notes
    assert scored["ML_Probability"].between(0, 1).all()
    again = ml_prep.apply_steps(new, [step], rows=False)
    assert (again["Policy_Start_days_since"]
            == train.tail(50)["Policy_Start_days_since"]).all()


def test_exported_script_makes_the_same_columns(churn, tmp_path):
    step, _ = ml_prep.date_step(churn, "Policy_Start")
    script = tmp_path / "prepare.py"
    script.write_text(ml_eval.prep_script([step]), encoding="utf-8")
    data = tmp_path / "file.csv"
    churn.head(200).to_csv(data, index=False)
    subprocess.run([sys.executable, str(script), str(data)],
                   check=True, capture_output=True, cwd=tmp_path)
    got = pd.read_csv(tmp_path / "file_prepared.csv")
    want = ml_prep.apply_steps(churn.head(200), [step])
    for part in step["parts"]:
        col = f"Policy_Start_{part}"
        assert np.allclose(got[col], want[col], equal_nan=True), col
