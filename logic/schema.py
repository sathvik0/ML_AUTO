"""Column-type detection used by Machine Learning's profiling.

`detect_dtype` decides whether a column is numeric or categorical,
`_guess_role` spots identifiers from the column name, and `base_column` folds
a derived copy ("Revenue (norm)") back into the column it rescales.
"""
import numpy as np
import pandas as pd

NORM_SUFFIX = " (norm)"
# Deciles are a derived column like the normalized ones, but keep their own
# suffix so 1-10 rank buckets are never confused with a 0-1 rescale.
DECILE_SUFFIX = " (decile)"


def derived_suffixes():
    return (NORM_SUFFIX, DECILE_SUFFIX)


def base_column(field):
    """Strip a derived suffix to get the underlying raw column name."""
    for suf in derived_suffixes():
        if field.endswith(suf):
            return field[:-len(suf)]
    return field


# ── detection & defaults ────────────────────────────────────────────────

DTYPE_SAMPLE = 20_000
# Distinct values at which a column reads as a flag rather than a measure.
FLAG_LEVELS = 2


def _probe(series):
    """A bounded sample of the non-null values, for type detection only.

    Positional draws with replacement, not Series.sample: sample() draws
    WITHOUT replacement, which walks the whole column — measured at 1.08s
    across twelve columns at 3M rows, where this is 0.02s. A probe only has to
    answer "what sort of column is this", so repeats in it cost nothing.
    """
    if len(series) <= DTYPE_SAMPLE:
        return series.dropna()
    idx = np.random.default_rng(0).integers(0, len(series), DTYPE_SAMPLE)
    return series.iloc[idx].dropna()


def detect_dtype(series):
    # Deciding "is this numeric?" does not need every row. to_numeric over a
    # 15M-row text column cost 6.5s apiece, and a sample answers a 90%
    # threshold with overwhelming confidence.
    probe = _probe(series)
    if probe.empty:
        # Either the column is genuinely empty or the probe landed entirely on
        # nulls in a sparse one. Only now is a full scan worth it.
        non_null = series.dropna()
        if non_null.empty:
            return "categorical"
        probe = _probe(non_null)
    # Two distinct values is a flag wearing numeric clothes (0/1). Sampling can
    # only under-count distinct values, so the worst this does is call a
    # near-binary column a flag, which is the reading wanted anyway.
    if probe.nunique() == FLAG_LEVELS:
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    parsed = pd.to_numeric(probe, errors="coerce")
    return "numeric" if parsed.notna().mean() >= 0.9 else "categorical"


# A geography identifies a territory, not a record: without this,
# "Region_ID" matched the "_id" rule and was tagged as an identifier.
GEOGRAPHY_HINTS = ("territory", "area", "region", "district", "zone")


def _guess_role(name, dtype):
    low = name.lower()
    if any(k in low for k in GEOGRAPHY_HINTS):
        return "feature"
    if any(k in low for k in ("npi", "mdm", "_id", "id_")) or low == "id":
        return "id"
    return "feature"
