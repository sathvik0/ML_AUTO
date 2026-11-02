"""Small dataframe fixes shared by the tabular sections."""
import re

import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype


def dedupe_columns(df):
    """Make column names unique the way pandas' CSV reader does: TRX, TRX.1.

    `logic/schema.py` keys everything by column name, so a duplicate does not
    merely confuse a table — `df[name]` returns a *frame* and type detection
    dies on "The truth value of a Series is ambiguous", which tells a user
    nothing. The CSV reader already guarantees unique names; this is for the
    frames the app builds itself (a Snowflake read, mainly).
    """
    if not df.columns.duplicated().any():
        return df
    seen, out = {}, []
    for name in df.columns:
        if name in seen:
            seen[name] += 1
            out.append(f"{name}.{seen[name]}")
        else:
            seen[name] = 0
            out.append(name)
    df = df.copy(deep=False)
    df.columns = out
    return df


def label_index(index):
    """Render index labels as text.

    A margins row puts "Total" beside the dimension's own values. Where those
    are numeric — tiers stored as 1-4, a decile column — the reset column ends
    up holding ints and one str, Arrow refuses to serialize it, and Streamlit
    falls back to a coerced copy after logging a traceback. Index labels are
    display text, so casting them costs nothing.
    """
    if isinstance(index, pd.MultiIndex):
        return index.set_levels([lv.astype(str) for lv in index.levels])
    return index.astype(str)


_NUMERIC_TEXT = re.compile(r"^-?[\d,]+(\.\d+)?\s*%?$")
_BLANK = {"", "-", "—", "nan", "None"}


def _reads_as_number(series):
    """True for a numeric column, or a text one already formatted for display.

    Several tables format to strings before rendering ("1,234", "62.5%"); to
    the reader those are still numbers and belong under the same alignment.
    """
    if is_numeric_dtype(series):
        return True
    if series.dtype != object and not isinstance(series.dtype, pd.StringDtype):
        return False
    values = [str(v).strip() for v in series.dropna()]
    seen = [v for v in values if v not in _BLANK]
    return bool(seen) and all(_NUMERIC_TEXT.match(v) for v in seen)


def centre_numbers(data):
    """`column_config` centring every column that reads as numbers.

    Streamlit's grid aligns by dtype — numbers right, text left — and drops a
    Styler's `text-align`, so `column_config` is the only lever that works.
    Accepts a frame or a Styler.
    """
    frame = getattr(data, "data", data)
    if not isinstance(frame, pd.DataFrame):
        return None
    cfg = {c: st.column_config.Column(alignment="center")
           for c in frame.columns
           if isinstance(c, str) and _reads_as_number(frame[c])}
    return cfg or None
