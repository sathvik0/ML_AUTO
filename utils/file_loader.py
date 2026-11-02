"""Read an uploaded CSV / Excel file into a DataFrame — any columns allowed.

Type detection and column configuration happen later on the data-prep page
(see logic/schema.py); this just loads the raw table.

Both readers try a fast parser first and fall back to pandas' default, so an
uploaded file can never fail to load just because the fast path dislikes it.
Measured on a 300k-row file:

    CSV    C engine 245ms  ->  pyarrow   18ms   (13x)
    Excel  openpyxl 9.7s   ->  calamine  2.5s   (4x)

Both fast readers returned a frame identical to the default one (same shape,
dtypes and values) on that file and on BOM / empty-cell / mixed-type /
quoted-comma / ragged-row cases.
"""
import logging

import pandas as pd
import streamlit as st

from utils import log
from utils.frames import dedupe_columns

_log = log.get("file_loader")


def _read_csv(buf) -> pd.DataFrame:
    """pyarrow parses ~13x faster and agrees with the C engine everywhere we
    tested, with one exception: it leaves duplicate column names as-is where
    the C engine disambiguates them (TRX, TRX.1). schema.py keys columns by
    name, so duplicates would silently collide — fall back when they appear.
    """
    try:
        df = pd.read_csv(buf, engine="pyarrow")
        if not df.columns.duplicated().any():
            return _decode_bytes(df)
        reason = "duplicate column names"
    except Exception as e:
        reason = type(e).__name__
    # Correct either way; logged because it is ~13x slower on a big file.
    log.event(_log, "fast_reader_skipped", kind="csv", reason=reason)
    buf.seek(0)
    return pd.read_csv(buf)


def _decode_bytes(df: pd.DataFrame) -> pd.DataFrame:
    """Turn any `bytes` cell back into text.

    A file that is not UTF-8 — a cp1252 export out of Excel is the usual one —
    makes pyarrow type that column as binary and hand back ``b'Jos\\xe9'``,
    where pandas' own engines refuse the file outright. Bytes poison
    everything downstream: they render as ``b'Oncology'``, never match a
    filter and export that way. Decoding here keeps the file usable, which
    beats both. latin-1 is the fallback because it cannot fail and is what
    those exports are.

    pyarrow types a whole column binary or not at all, so the first non-null
    value decides it and an ordinary UTF-8 file pays nothing.
    """
    df.columns = [_text(c) for c in df.columns]
    for col in df.columns:
        if df[col].dtype != object:
            continue
        values = df[col].to_numpy()
        first = next((v for v in values if v is not None and v == v), None)
        if isinstance(first, bytes):
            df[col] = [_text(v) for v in values]
    return df


def _text(value):
    if not isinstance(value, bytes):
        return value
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError:
        return value.decode("latin-1")


def _read_excel(buf) -> pd.DataFrame:
    """calamine is ~4x faster than openpyxl. It is an optional dependency —
    without it the import fails and openpyxl handles the file as before.
    """
    try:
        return pd.read_excel(buf, engine="calamine")
    except Exception as e:
        # ~4x slower on a big file, and the likeliest cause is calamine not
        # being installed — worth seeing when an upload is slow.
        log.event(_log, "fast_reader_skipped", kind="excel",
                  reason=type(e).__name__)
        buf.seek(0)
        return pd.read_excel(buf)


@st.cache_data(max_entries=5, show_spinner=False)
def _read(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    buf = pd.io.common.BytesIO(file_bytes)
    frame = _read_csv(buf) if file_name.lower().endswith(".csv") else _read_excel(buf)
    # Both readers disambiguate duplicate headers today; this keeps that true
    # whichever engine ends up handling the file.
    return dedupe_columns(frame)


def read_path(path: str) -> pd.DataFrame:
    """Same fast readers for a file on disk (the bundled sample dataset)."""
    with open(path, "rb") as f:
        return _read(f.read(), path)


def load_file(uploaded_file):
    """Return a DataFrame (column names stripped) or None on failure."""
    name = str(uploaded_file.name)
    kind = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    try:
        with log.timer() as t:
            data = uploaded_file.read()
            df = _read(data, uploaded_file.name)
    except Exception as e:
        log.event(_log, "file_unreadable", level=logging.WARNING,
                  exc_info=True, kind=kind)
        st.error(f"Unable to read the file: {e}")
        return None
    # The size and shape, never the name: a file name can carry a client's.
    log.event(_log, "file_loaded", kind=kind, bytes=len(data), rows=len(df),
              columns=df.shape[1], ms=t.ms)
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty or df.shape[1] == 0:
        st.error("The file has no rows or no columns.")
        return None
    return df
