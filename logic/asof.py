"""Is every feature known when the prediction is made? A column recorded
after the outcome (a cancellation reason, a closing date, a claim filed
after renewal) makes a model look far better than it will be on new rows,
because the answer is in the data.

The leakage check (`ml.leakage`) catches a column that predicts the target
almost on its own. This catches what it cannot: columns whose dates fall
after the row's time, and columns whose names say they describe what came
after. Both are cautions, not errors: only the person who knows the data
can say when a column is recorded.
"""
import re

import numpy as np
import pandas as pd

from logic import ml_prep

LATE_SHARE = 0.01     # dates after the row's time in more rows than this
# Words in a column name that usually describe what happened afterwards.
AFTER_WORDS = ("after", "post", "cancel", "cancelled", "canceled",
               "cancellation", "terminated", "termination", "closed",
               "closure", "exit", "renewed", "outcome", "result",
               "resolution", "refund", "refunded", "chargeback",
               "writeoff", "collections", "churned", "lapsed", "reason")


def _words(name):
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name))
    return set(re.split(r"[^a-z0-9]+", spaced.lower())) - {""}


def _named(features, target):
    out = []
    for col in features:
        if col == target:
            continue
        hit = sorted(_words(col) & set(AFTER_WORDS))
        if hit:
            out.append(f"{col}: its name ('{hit[0]}') suggests it describes "
                       f"what happened after the outcome. If it is only "
                       f"known afterwards, it leaks the answer and the "
                       f"model will not work on new rows: leave it out.")
    return out


def _dates(df, col):
    s = df[col]
    if pd.api.types.is_numeric_dtype(s) and not \
            pd.api.types.is_bool_dtype(s):
        return None
    if not ml_prep.looks_like_date(s):
        return None
    d = ml_prep._dates(s, ml_prep.date_format(s))
    return d


def _later(df, features, time, time_format):
    """Date columns (used as such, or through their date features) whose
    values fall after the row's time."""
    if not time or time not in df.columns:
        return []
    t = df[time]
    if pd.api.types.is_numeric_dtype(t) and not \
            pd.api.types.is_bool_dtype(t):
        return []                      # a month index: no dates to compare
    t = ml_prep._dates(t, time_format or ml_prep.date_format(t))
    used = set(features)
    out = []
    for col in df.columns:
        if col == time or col not in used and not any(
                f.startswith(f"{col}_") for f in used):
            continue
        d = _dates(df, col)
        if d is None:
            continue
        both = d.notna() & t.notna()
        if not both.any():
            continue
        late = (d[both] > t[both]).to_numpy()
        share = float(np.mean(late))
        if share > LATE_SHARE:
            out.append(f"{col} is later than {time} in {share:.0%} of rows: "
                       f"a value recorded after the row's time can leak the "
                       f"outcome. Use only what was known on {time}, or "
                       f"leave {col} and its date features out.")
    return out


def cautions(df, config):
    """Plain sentences about features that may be recorded after the
    outcome; empty when nothing looks late."""
    if config.get("task") == "clustering":
        return []
    features = list(config.get("numeric") or []) + list(
        config.get("categorical") or [])
    val = config.get("validation") or {}
    return (_later(df, features, val.get("time"), val.get("time_format"))
            + _named(features, config.get("target")))
