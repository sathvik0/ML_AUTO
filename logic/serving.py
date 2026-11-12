"""Scoring with a registered model: one path for the model API and for
batch jobs, the same one the app's "Use own model" takes.

A registered version is this app's model file: the fitted pipeline, its
recipe and classes, and the preparation steps it was trained after. The
steps' column work is replayed on the incoming rows (row steps are not:
every row gets a prediction), then the model predicts.
"""
from collections import OrderedDict

import numpy as np
import pandas as pd

from logic import inputs, ml_prep, ml_upload

MAX_API_ROWS = 10_000
MAX_EXPLAIN_ROWS = 1_000  # reasons score each row once per input
MAX_REASONS = 5


class ScoringError(ValueError):
    """Rows the model cannot score, in words a caller can act on."""


_CACHE = OrderedDict()
CACHE_SIZE = 16


def load(version_id, fetch):
    """The model file for a version, read once: a version's file never
    changes, so it is cached by id. `fetch()` returns its bytes."""
    if version_id in _CACHE:
        _CACHE.move_to_end(version_id)
        return _CACHE[version_id]
    info, why = ml_upload.read(fetch(), "model.joblib")
    if why:
        raise ScoringError(why)
    if info["kind"] not in ("app", "external"):
        raise ScoringError("Not a model file from this app.")
    _CACHE[version_id] = info
    while len(_CACHE) > CACHE_SIZE:
        _CACHE.popitem(last=False)
    return info


def score(info, rows: pd.DataFrame, explain=0) -> pd.DataFrame:
    """`rows` with the model's columns added (ML_Prediction, probabilities
    per class, or ML_Cluster), and, for a model file that keeps its inputs'
    training profile, ML_Warnings (a value outside the training range, a
    category never seen, a gap) and with `explain` the top inputs behind
    each prediction (ML_Reasons)."""
    if rows.empty:
        raise ScoringError("No rows to score.")
    rows = rows.rename(columns=str)
    steps = info.get("prep_steps") or []
    if steps:
        try:
            rows = ml_prep.apply_steps(rows, steps, rows=False)
        except ml_prep.StepError as e:
            raise ScoringError(f"The model's preparation does not fit these "
                               f"rows. {e}") from None
    # Each model input from its file column (the same name for this app's
    # own models; as mapped at registration for one trained elsewhere).
    columns = info.get("columns") or {f: f for f in info["features"]}
    missing = [c for c in columns.values() if c not in rows.columns]
    if missing:
        raise ScoringError("Missing columns: " + ", ".join(missing[:20])
                           + (f" and {len(missing) - 20} more"
                              if len(missing) > 20 else ""))
    out, why = ml_upload.predict_only(
        rows, info, columns, info["task"],
        by_position=bool(info.get("by_position")))
    if why:
        raise ScoringError(why)
    names = info.get("cluster_names")
    if info["task"] == "clustering" and isinstance(names, dict) \
            and "ML_Cluster" in out.columns:
        out["ML_Cluster"] = out["ML_Cluster"].map(
            lambda c: names.get(c, names.get(str(c), c)))
    prof = {c: d for c, d in (info.get("inputs") or {}).items()
            if c in set(columns.values())}
    if prof:
        found = inputs.warnings(rows, prof)
        out["ML_Warnings"] = _text_column(found, out.index)
    if explain and info["task"] != "clustering":
        if not prof:
            raise ScoringError("This model file was registered before "
                               "reasons were kept; register it again to "
                               "get them.")
        if len(rows) > MAX_EXPLAIN_ROWS:
            raise ScoringError(f"Reasons are given for up to "
                               f"{MAX_EXPLAIN_ROWS:,} rows per request.")
        out["ML_Reasons"] = _text_column(
            _reasons(info, rows, columns, prof,
                     min(int(explain), MAX_REASONS)), out.index)
    return out


def _text_column(values, index):
    """Text per row, None where there is none. As objects: pandas 3 would
    otherwise read the list as strings and turn None into NaN."""
    return pd.Series([v or None for v in values], index=index, dtype=object)


def _target_score(info, frame, pred=None):
    """The number a reason moves: the class of interest's probability, the
    probability of the class predicted for the row, or the prediction."""
    if info["task"] == "regression":
        return frame["ML_Prediction"].to_numpy(dtype="float64")
    positive = info.get("positive")
    col = f"ML_Probability_{positive}"
    if positive and col in frame.columns:
        return frame[col].to_numpy(dtype="float64")
    cols = {c: f"ML_Probability_{c}" for c in pd.unique(pred)}
    if not all(c in frame.columns for c in cols.values()):
        raise ScoringError("This model gives no probabilities, so the "
                           "inputs behind a prediction cannot be weighed.")
    return np.array([frame[cols[p]].iat[i] for i, p in enumerate(pred)],
                    dtype="float64")


def _reasons(info, rows, columns, prof, top):
    """For each row, the inputs whose value moved the prediction most:
    each input is set, in turn, to its typical training value (median or
    commonest category), and the change in the score is its effect."""
    feats = [c for c in columns.values() if c in prof]
    n = len(rows)
    blocks = [rows] + [rows.assign(**{c: prof[c]["typical"]}) for c in feats]
    big = pd.concat(blocks, ignore_index=True)
    got, why = ml_upload.predict_only(
        big, info, columns, info["task"],
        by_position=bool(info.get("by_position")))
    if why:
        raise ScoringError(why)
    pred = None
    if info["task"] == "classification":
        # Each copy is weighed on the class predicted for the row itself.
        pred = np.tile(got["ML_Prediction"].to_numpy(dtype=object)[:n],
                       len(feats) + 1)
    s = _target_score(info, got, pred)
    base = s[:n]
    effects = np.stack([base - s[(j + 1) * n:(j + 2) * n]
                        for j in range(len(feats))], axis=1)
    texts = []
    for i in range(n):
        parts = []
        for j in np.argsort(-np.abs(effects[i]))[:top]:
            if not np.isfinite(effects[i, j]) or abs(effects[i, j]) < 1e-9:
                continue
            col = feats[j]
            value, typ = rows[col].iat[i], prof[col]["typical"]
            shown = (f"{value:,.6g}" if isinstance(value, (int, float,
                                                          np.number))
                     else str(value))
            typ_s = f"{typ:,.6g}" if isinstance(typ, float) else str(typ)
            parts.append(f"{col} = {shown} (typical {typ_s}): "
                         f"{effects[i, j]:+.3f}")
        texts.append("; ".join(parts) or None)
    return texts


def output_columns(frame):
    return [c for c in frame.columns if str(c).startswith("ML_")]
