"""How sure the results are: bootstrap intervals for each model's score on
the test rows, a paired check of whether the best model really beats the
runner-up, and the score in each segment of a column.

All of it reads the predictions already made on the test rows; nothing is
trained again. A resample draws the test rows with replacement, the same
draw for every model, so two models are compared on identical rows.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score)

from logic import ml

RESAMPLES = 300
LEVEL = 0.95
CLEAR_WIN = 0.95          # a win in this share of resamples is "clear"
MIN_SEGMENT = 30          # fewer test rows than this: not scored
MAX_SEGMENTS = 15
AUTO_ROWS = 50_000        # above this many test rows, intervals wait for a click


def metric(task, key, y, pred, proba, classes, positive):
    """One metric from predictions already made, or None when it cannot be
    computed on these rows (an AUC where one class is missing)."""
    if task == "regression":
        return ml.regression_metrics(y, pred)[key]   # WAPE among them
    y = np.asarray(y, dtype=object)
    classes = list(classes)
    if key in ("roc_auc", "average_precision"):
        truth = y == positive
        score = proba[:, classes.index(positive)]
        if key == "roc_auc":
            return (float(roc_auc_score(truth, score))
                    if 0 < truth.sum() < len(truth) else None)
        return float(average_precision_score(truth, score)) \
            if truth.any() else None
    if key == "roc_auc_ovr":
        return (float(roc_auc_score(y, proba, multi_class="ovr",
                                    labels=classes))
                if set(y) == set(classes) else None)
    if key == "log_loss":
        return float(log_loss(y, proba, labels=classes))
    pred = np.asarray(pred, dtype=object)
    if key == "accuracy":
        return float(np.mean(pred == y))
    if key == "balanced_accuracy":
        return float(balanced_accuracy_score(y, pred))
    one = dict(pos_label=positive, zero_division=0)
    macro = dict(average="macro", zero_division=0, labels=classes)
    fn, kw = {"f1": (f1_score, one), "precision": (precision_score, one),
              "recall": (recall_score, one), "f1_macro": (f1_score, macro),
              "precision_macro": (precision_score, macro),
              "recall_macro": (recall_score, macro)}[key]
    return float(fn(y, pred, **kw))


def _test(run):
    pos = np.flatnonzero(np.asarray(run["split"]) == "test")
    return pos, run["y"].to_numpy()[pos]


def _score(run, res, rows, y):
    proba = res.get("proba")
    return metric(run["task"], run["metric"], y, res["pred"][rows],
                  None if proba is None else proba[rows], run["classes"],
                  run.get("positive"))


def intervals(run, resamples=RESAMPLES, seed=0):
    """{model key: {"value", "low", "high"}} on the run's selection metric,
    and the paired check of the best model against the runner-up:
    {"best", "other", "wins", "low", "high", "resamples"} (a positive
    difference favours the best), or None with fewer than two models."""
    pos, y = _test(run)
    models = [r for r in run["results"] if not r.get("error")]
    if not len(pos) or run["task"] == "clustering":
        return {}, None
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(pos), size=(resamples, len(pos)))
    up = ml.higher_is_better(run["family"], run["metric"])
    scores = {}
    for r in models:
        vals = [_score(run, r, pos[d], y[d]) for d in draws]
        scores[r["key"]] = np.array([np.nan if v is None else v
                                     for v in vals], dtype="float64")
    tail = (1 - LEVEL) / 2 * 100
    out = {}
    for r in models:
        s = scores[r["key"]]
        ok = s[np.isfinite(s)]
        out[r["key"]] = {
            "value": r["test"].get(run["metric"]),
            "low": float(np.percentile(ok, tail)) if len(ok) else None,
            "high": float(np.percentile(ok, 100 - tail)) if len(ok) else None}
    ranked = sorted((r for r in models if r["key"] != ml.BASELINE
                     and r["test"].get(run["metric"]) is not None),
                    key=lambda r: r["test"][run["metric"]], reverse=up)
    pair = None
    best = run.get("best")
    if best in scores and len(ranked) >= 2:
        other = next(r["key"] for r in ranked if r["key"] != best)
        diff = scores[best] - scores[other]
        if not up:
            diff = -diff
        diff = diff[np.isfinite(diff)]
        if len(diff):
            pair = {"best": best, "other": other,
                    "wins": float((diff > 0).mean()),
                    "low": float(np.percentile(diff, tail)),
                    "high": float(np.percentile(diff, 100 - tail)),
                    "resamples": int(len(diff))}
    return out, pair


def _segments(values, test_rows):
    """Each test row's segment: the value itself for a column with few
    values, the commonest values and "other" for more, quartile bands for
    a number with many values."""
    s = values.iloc[test_rows]
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.95 and num.nunique() > 12:
        bands = pd.qcut(num, 4, duplicates="drop")
        text = bands.map(lambda b: f"{b.left:,.4g} to {b.right:,.4g}")
        return text.astype(object).where(num.notna(), "(missing)") \
            .astype(str).to_numpy()
    text = s.astype(object).where(s.notna(), "(missing)").astype(str)
    keep = text.value_counts().index[:MAX_SEGMENTS]
    return text.where(text.isin(keep), "other").to_numpy()


def segments(run, res, frame, column):
    """The model's score in each segment of `column` on the test rows, with
    the rows and the actual rate (or mean) there; worst first."""
    pos, y = _test(run)
    seg = _segments(frame[column].reset_index(drop=True), pos)
    task, key = run["task"], run["metric"]
    up = ml.higher_is_better(run["family"], key)
    overall = res["test"].get(key)
    rows = []
    for g in pd.unique(seg):
        m = seg == g
        n = int(m.sum())
        value = (_score(run, res, pos[m], y[m]) if n >= MIN_SEGMENT
                 else None)
        if task == "classification" and run.get("positive"):
            actual = float(np.mean(y[m] == run["positive"]))
        elif task == "regression":
            actual = float(np.mean(np.asarray(y[m], dtype="float64")))
        else:
            actual = None
        gap = (None if value is None or overall is None else
               (value - overall) if up else (overall - value))
        rows.append({"segment": g, "rows": n, "actual": actual,
                     "value": value, "gap": gap})
    out = pd.DataFrame(rows)
    return out.sort_values("gap", na_position="last").reset_index(drop=True)


def naive(run, df=None):
    """For a forecast (a regression trained with a time split and past
    values of its target): each naive forecast, "the value k rows earlier",
    against the best model on the same test rows. [] otherwise. `df` is
    the prepared file the run trained on, when the run keeps no `frame`.

    [{"k", "rows", "naive", "model", "skill"}]: the run's metric for each,
    and how much of the naive forecast's error the model removes (for a
    lower-is-better metric; for R², the gain)."""
    frame = run["frame"] if run.get("frame") is not None else df
    if run["task"] != "regression" or not run.get("time_split") \
            or frame is None or not run.get("best"):
        return []
    target, key = run.get("target"), run["metric"]
    lags = sorted({int(k) for s in run.get("prep_steps") or []
                   if s["kind"] == "lags" and s["column"] == target
                   for k in s.get("lags") or []})
    pos, y = _test(run)
    best = ml.result(run, run["best"])
    up = ml.higher_is_better(run["family"], key)
    out = []
    for k in lags:
        col = f"{target}_lag_{k}"
        if col not in frame.columns:
            continue
        guess = pd.to_numeric(frame[col], errors="coerce") \
            .to_numpy(dtype="float64")[pos]
        ok = np.isfinite(guess) & np.isfinite(np.asarray(y, dtype="float64"))
        if ok.sum() < MIN_SEGMENT:
            continue
        yy = np.asarray(y, dtype="float64")[ok]
        n_val = ml.regression_metrics(yy, guess[ok])[key]
        m_val = ml.regression_metrics(yy, best["pred"][pos][ok])[key]
        if n_val is None or m_val is None:
            continue
        skill = ((m_val - n_val) if up else
                 (1 - m_val / n_val) if n_val else None)
        out.append({"k": k, "rows": int(ok.sum()), "naive": float(n_val),
                    "model": float(m_val), "skill": skill})
    return out
