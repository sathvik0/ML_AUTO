"""What a model's inputs looked like when it was trained, kept in its model
file: each numeric input's range and median, each categorical input's
values and commonest one. Scoring checks new rows against it (a value
outside the range, a category never seen, a gap) and uses the typical
values to say which inputs pushed a prediction up or down.

Imports nothing from this app, so both model-file writers can use it.
"""
import numpy as np
import pandas as pd

MAX_VALUES = 500          # categories kept per input; beyond, "seen" is not checked
MAX_WARNINGS = 5          # per row, the rest counted


def profile(frame, kinds):
    """{column: description} for the columns in `kinds` ({column:
    "numeric" | "categorical"}), from the rows of `frame`."""
    out = {}
    for col, kind in kinds.items():
        if col not in frame.columns:
            continue
        s = frame[col]
        if kind == "numeric":
            v = pd.to_numeric(s, errors="coerce").astype("float64")
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            out[col] = {"kind": "numeric", "min": float(v.min()),
                        "max": float(v.max()), "typical": float(v.median())}
        else:
            text = s.dropna().astype(str)
            if not len(text):
                continue
            counts = text.value_counts()
            out[col] = {"kind": "categorical",
                        "typical": str(counts.index[0]),
                        "values": [str(x) for x in counts.index[:MAX_VALUES]],
                        "complete": len(counts) <= MAX_VALUES}
    return out


def _num(x):
    return f"{x:,.6g}"


def warnings(rows, prof):
    """One text per row ("" when nothing is off): values outside the
    training range, categories never seen, and gaps."""
    issues = [[] for _ in range(len(rows))]
    for col, d in prof.items():
        if col not in rows.columns:
            continue
        s = rows[col]
        if d["kind"] == "numeric":
            v = pd.to_numeric(s, errors="coerce").astype("float64").to_numpy()
            gap = ~np.isfinite(v)
            low = np.where(~gap & (v < d["min"]))[0]
            high = np.where(~gap & (v > d["max"]))[0]
            for i in low:
                issues[i].append(f"{col} {_num(v[i])} is below the training "
                                 f"range ({_num(d['min'])} to "
                                 f"{_num(d['max'])})")
            for i in high:
                issues[i].append(f"{col} {_num(v[i])} is above the training "
                                 f"range ({_num(d['min'])} to "
                                 f"{_num(d['max'])})")
        else:
            gap = s.isna().to_numpy()
            if d.get("complete"):
                seen = set(d["values"])
                text = s.astype(str).to_numpy()
                for i in np.where(~gap)[0]:
                    if text[i] not in seen:
                        issues[i].append(f"{col} '{text[i]}' was not seen "
                                         f"in training")
        for i in np.where(gap)[0]:
            issues[i].append(f"{col} is missing")
    out = []
    for found in issues:
        extra = len(found) - MAX_WARNINGS
        out.append("; ".join(found[:MAX_WARNINGS])
                   + (f"; and {extra} more" if extra > 0 else ""))
    return out
