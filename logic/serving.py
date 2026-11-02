"""Scoring with a registered model: one path for the model API and for
batch jobs, the same one the app's "Use own model" takes.

A registered version is this app's model file: the fitted pipeline, its
recipe and classes, and the preparation steps it was trained after. The
steps' column work is replayed on the incoming rows (row steps are not:
every row gets a prediction), then the model predicts.
"""
from collections import OrderedDict

import pandas as pd

from logic import ml_prep, ml_upload

MAX_API_ROWS = 10_000


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
    if info["kind"] != "app":
        raise ScoringError("Not a model file from this app.")
    _CACHE[version_id] = info
    while len(_CACHE) > CACHE_SIZE:
        _CACHE.popitem(last=False)
    return info


def score(info, rows: pd.DataFrame) -> pd.DataFrame:
    """`rows` with the model's columns added (ML_Prediction, probabilities
    per class, or ML_Cluster)."""
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
    missing = [f for f in info["features"] if f not in rows.columns]
    if missing:
        raise ScoringError("Missing columns: " + ", ".join(missing[:20])
                           + (f" and {len(missing) - 20} more"
                              if len(missing) > 20 else ""))
    out, why = ml_upload.predict_only(rows, info,
                                      {f: f for f in info["features"]},
                                      info["task"])
    if why:
        raise ScoringError(why)
    names = info.get("cluster_names")
    if info["task"] == "clustering" and isinstance(names, dict) \
            and "ML_Cluster" in out.columns:
        out["ML_Cluster"] = out["ML_Cluster"].map(
            lambda c: names.get(c, names.get(str(c), c)))
    return out


def output_columns(frame):
    return [c for c in frame.columns if str(c).startswith("ML_")]
