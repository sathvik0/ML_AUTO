"""Preparing a file for Machine Learning: steps recorded as data.

Each step is a small dict whose numbers are frozen when it is added — a cap
at the 99th percentile is stored as the value that percentile had — so the
same steps rebuild the same file in the app, in the exported scripts, and on
a new file being scored. Nothing here is fitted again.

Like ml_recipe, this module is the exported scripts' code, verbatim (minus
this docstring and the one import below): it uses numpy, pandas and
`label_text` only.

Row steps (dropping duplicates, rows with gaps, rows outside a range or a
set of values) shape the file a model learns from. They are not applied to a
file being scored: every row in it gets a prediction, and the note says so.
"""
import numpy as np
import pandas as pd

from logic.ml_recipe import label_text

ROW_STEPS = ("drop_duplicates", "drop_missing", "keep_range", "keep_values")
TRANSFORMS = {"log1p": "log(1 + x)", "sqrt": "square root",
              "square": "square", "reciprocal": "1 / x"}
BLANK = "(blank)"


class StepError(ValueError):
    """A step that cannot run on this file, in words."""


def _numbers(series):
    v = pd.to_numeric(series, errors="coerce").astype("float64")
    return v.where(np.isfinite(v))


def _need(df, column, n):
    if column not in df.columns:
        raise StepError(f"Step {n}: {column} is not in this file.")


def _transform(values, how):
    with np.errstate(divide="ignore", invalid="ignore"):
        if how == "log1p":
            out = np.log1p(values)
        elif how == "sqrt":
            out = np.sqrt(values)
        elif how == "square":
            out = np.square(values)
        elif how == "reciprocal":
            out = 1.0 / values
        else:
            raise StepError(f"Unknown transform: {how}.")
    out = pd.Series(out, index=values.index, dtype="float64")
    return out.where(np.isfinite(out))


def _column_script(code, df):
    """A column written as a script on the Feature engineering page: it
    assigns `result` from `df`, `pd` and `np`. The app checks every script
    against an allow-list before running it; here it runs as written."""
    env = {"df": df.copy(deep=False), "pd": pd, "np": np}
    exec(code, env, env)
    value = env["result"]
    if isinstance(value, pd.Series):
        return value.reindex(df.index)
    return pd.Series(list(value), index=df.index)


def _dates(series, fmt):
    """A column as timestamps; what does not read as a date becomes a gap.
    Time zones are turned into UTC, then dropped, so every file agrees."""
    if not pd.api.types.is_datetime64_any_dtype(series):
        series = label_text(series)
        fmt = fmt or "mixed"
    else:
        fmt = None
    d = pd.to_datetime(series, errors="coerce", utc=True, format=fmt)
    return d.dt.tz_localize(None)


def date_parts(series, fmt, reference):
    """{part: numbers} for one date column; a gap stays a gap in every
    part. Months, weekdays and hours also come as a point on a circle (sine
    and cosine), so December sits next to January."""
    d = _dates(series, fmt)
    ok = d.notna()
    month, weekday, hour = d.dt.month, d.dt.weekday, d.dt.hour
    turn = 2 * np.pi
    parts = {
        "year": d.dt.year, "month": month, "day": d.dt.day,
        "weekday": weekday, "is_weekend": (weekday >= 5).astype("float64"),
        "hour": hour,
        "days_since": (d - pd.Timestamp(reference)).dt.total_seconds()
        / 86400,
        "month_sin": np.sin(turn * (month - 1) / 12),
        "month_cos": np.cos(turn * (month - 1) / 12),
        "weekday_sin": np.sin(turn * weekday / 7),
        "weekday_cos": np.cos(turn * weekday / 7),
        "hour_sin": np.sin(turn * hour / 24),
        "hour_cos": np.cos(turn * hour / 24)}
    return {k: pd.Series(v, index=series.index, dtype="float64").where(ok)
            for k, v in parts.items()}


def bin_labels(edges):
    """One label per bin; the first and last bins are open-ended, so every
    value in a later file lands in one of them."""
    inner = [float(e) for e in edges[1:-1]]
    if not inner:
        return ["all values"]
    labels = [f"below {inner[0]:,.6g}"]
    labels += [f"{a:,.6g} to below {b:,.6g}" for a, b in zip(inner, inner[1:])]
    labels.append(f"{inner[-1]:,.6g} or more")
    return labels


def apply_step(df, step, n=1, rows=True):
    """One step on a copy of `df`; the index is kept, so a row can always be
    traced back to the file it came from."""
    kind = step["kind"]
    if kind in ROW_STEPS and not rows:
        return df
    if kind == "drop_columns":
        missing = [c for c in step["columns"] if c not in df.columns]
        return df.drop(columns=[c for c in step["columns"]
                                if c not in missing])
    if kind == "drop_duplicates":
        return df[~df.duplicated(keep="first")]
    if kind == "drop_missing":
        for c in step["columns"]:
            _need(df, c, n)
        gap = pd.Series(False, index=df.index)
        for c in step["columns"]:
            s = df[c]
            gap |= (_numbers(s).isna() if pd.api.types.is_numeric_dtype(s)
                    else s.isna())
        return df[~gap]
    if kind == "script":
        out = df.copy()
        out[step["into"]] = _column_script(step["code"], df)
        return out
    column = step["column"]
    _need(df, column, n)
    if kind == "keep_range":
        v = _numbers(df[column])
        keep = v.isna()
        inside = pd.Series(True, index=df.index)
        if step.get("low") is not None:
            inside &= v >= step["low"]
        if step.get("high") is not None:
            inside &= v <= step["high"]
        return df[keep | inside]
    if kind == "keep_values":
        text = label_text(df[column]).fillna(BLANK)
        return df[text.isin(list(step["values"]))]
    out = df.copy()
    if kind == "cap":
        out[column] = _numbers(df[column]).clip(lower=step.get("low"),
                                                upper=step.get("high"))
    elif kind == "transform":
        out[step["into"]] = _transform(_numbers(df[column]), step["how"])
    elif kind == "bin":
        edges = [-np.inf] + [float(e) for e in step["edges"][1:-1]] + [np.inf]
        labels = bin_labels(edges)
        cut = pd.cut(_numbers(df[column]), edges, labels=labels, right=False)
        out[step["into"]] = cut.astype(object).where(cut.notna(), np.nan)
    elif kind == "date_parts":
        parts = date_parts(df[column], step.get("format"), step["reference"])
        for part in step["parts"]:
            out[f"{column}_{part}"] = parts[part]
    elif kind == "as_category":
        out[column] = label_text(df[column])
    elif kind == "as_number":
        out[column] = _numbers(df[column])
    elif kind == "fill":
        s = df[column]
        if pd.api.types.is_numeric_dtype(s) and \
                not pd.api.types.is_bool_dtype(s):
            try:
                value = float(step["value"])
            except (TypeError, ValueError):
                raise StepError(f"Step {n}: {column} holds numbers; "
                                f"{step['value']} is not one.") from None
            out[column] = _numbers(s).fillna(value)
        else:
            out[column] = s.astype(object).where(s.notna(),
                                                 str(step["value"]))
    else:
        raise StepError(f"Step {n}: unknown step {kind}.")
    return out


def apply_steps(df, steps, rows=True):
    """Every step in order. `rows=False` leaves out the steps that drop rows,
    for a file whose every row is to be scored."""
    out = df
    for n, step in enumerate(steps or [], 1):
        out = apply_step(out, step, n, rows)
    return out


# ── building steps (the app's side; not part of the exported scripts) ──
def _checked_script(code, df):
    """In the app, a column script runs only through the allow-list sandbox
    (`logic.derive`): steps also arrive inside uploaded model files."""
    from logic import derive
    try:
        return derive.run_script(code, df)
    except derive.ScriptError as e:
        raise StepError(f"Column script: {e}") from None


# Replaces the plain runner above for every step run in the app.
globals()["_column_script"] = _checked_script


def is_row_step(step):
    return step["kind"] in ROW_STEPS


def describe(step):
    """One line for the list of steps and for the script's comments."""
    kind = step["kind"]

    def num(v):
        return f"{v:,.6g}"

    if kind == "drop_columns":
        return "Drop " + ", ".join(step["columns"])
    if kind == "drop_duplicates":
        return "Drop rows that repeat an earlier row in every column"
    if kind == "drop_missing":
        return "Drop rows with a gap in " + ", ".join(step["columns"])
    c = step.get("column")
    if kind == "keep_range":
        lo, hi = step.get("low"), step.get("high")
        if lo is not None and hi is not None:
            where = f"between {num(lo)} and {num(hi)}"
        elif lo is not None:
            where = f"at least {num(lo)}"
        elif hi is not None:
            where = f"at most {num(hi)}"
        else:
            return f"Keep every row ({c} has no bound)"
        return f"Keep rows where {c} is {where}; rows with no {c} stay"
    if kind == "keep_values":
        values = list(step["values"])
        shown = ", ".join(map(str, values[:6]))
        if len(values) > 6:
            shown += f" and {len(values) - 6} more"
        return f"Keep rows where {c} is one of: {shown}"
    if kind == "cap":
        lo, hi = step.get("low"), step.get("high")
        parts = ([f"below {num(lo)} up to {num(lo)}"] if lo is not None
                 else []) + ([f"above {num(hi)} down to {num(hi)}"]
                             if hi is not None else [])
        return f"Cap {c}: " + " and ".join(parts)
    if kind == "transform":
        return (f"{step['into']} = {TRANSFORMS[step['how']]} of {c}"
                if step["into"] != c else
                f"Replace {c} with its {TRANSFORMS[step['how']]}")
    if kind == "bin":
        return (f"{step['into']}: {c} cut into "
                f"{len(step['edges']) - 1} bins at fixed values")
    if kind == "date_parts":
        return (f"Date features from {c}: "
                + ", ".join(f"{c}_{p}" for p in step["parts"])
                + f" (days since {step['reference']})")
    if kind == "as_category":
        return f"Read {c} as categories"
    if kind == "as_number":
        return f"Read {c} as numbers; text that is not a number becomes a gap"
    if kind == "fill":
        return f"Fill gaps in {c} with {step['value']}"
    if kind == "script":
        return f"New column {step['into']} from a script (Feature engineering)"
    return kind


def quantile(values, q):
    """A percentile of the numbers in `values`, gaps left out."""
    v = _numbers(pd.Series(values)).dropna()
    return float(np.quantile(v, q)) if len(v) else None


def iqr_fences(values, k=1.5):
    """(low, high): k interquartile ranges beyond the quartiles."""
    v = _numbers(pd.Series(values)).dropna()
    if not len(v):
        return None, None
    q1, q3 = np.quantile(v, [0.25, 0.75])
    return float(q1 - k * (q3 - q1)), float(q3 + k * (q3 - q1))


def bin_edges(values, bins, how="quantile"):
    """Frozen edges: equal-count (quantile) or equal-width. Tied quantiles
    collapse, so a column with many repeats gets fewer bins than asked."""
    v = _numbers(pd.Series(values)).dropna()
    if not len(v):
        return []
    if how == "quantile":
        edges = np.quantile(v, np.linspace(0, 1, int(bins) + 1))
    else:
        edges = np.linspace(v.min(), v.max(), int(bins) + 1)
    return [float(e) for e in np.unique(edges)]


def domain_problem(values, how):
    """Why a transform cannot run on these values, or None."""
    v = _numbers(pd.Series(values)).dropna()
    if how == "log1p" and (v <= -1).any():
        return (f"{int((v <= -1).sum()):,} values are -1 or below; "
                f"log(1 + x) needs every value above -1.")
    if how == "sqrt" and (v < 0).any():
        return (f"{int((v < 0).sum()):,} values are below zero; the square "
                f"root needs none.")
    if how == "reciprocal" and (v == 0).any():
        return (f"{int((v == 0).sum()):,} values are zero; 1 / x is not "
                f"defined there.")
    return None


def relearn(step, rows):
    """`step` with its learnt values worked out again from `rows` (a step
    that records no "learn" keeps its values). Values that cannot be
    worked out on these rows (a column with no numbers) are kept."""
    how = step.get("learn")
    if not how or step["column"] not in rows.columns:
        return step
    v = _numbers(rows[step["column"]]).dropna()
    out = dict(step)
    if step["kind"] == "cap" and len(v):
        if how["cap"] == "pct":
            out["low"] = quantile(v, how["low"]) \
                if how.get("low") is not None else None
            out["high"] = quantile(v, how["high"]) \
                if how.get("high") is not None else None
        else:
            out["low"], out["high"] = iqr_fences(v, how["k"])
    elif step["kind"] == "bin" and len(v):
        edges = bin_edges(v, how["bins"], how["how"])
        if len(edges) >= 3:
            out["edges"] = edges
    elif step["kind"] == "fill":
        if how["fill"] in ("median", "mean") and len(v):
            out["value"] = float(v.median() if how["fill"] == "median"
                                 else v.mean())
        elif how["fill"] == "mode":
            counts = rows[step["column"]].dropna().astype(str).value_counts()
            if len(counts):
                out["value"] = str(counts.index[0])
    return out


def refit(source, steps, train_labels, columns):
    """The steps with their learnt values worked out from the training rows
    only (`train_labels`, index labels of `source`), each on the file as the
    steps before it leave it. Only steps on `columns` (the features) are
    worked out again; one on the target, the time or a group column keeps
    its values, so the rows and the split stay as they were."""
    keep = set(train_labels)
    out, cur = [], source
    for n, step in enumerate(steps or [], 1):
        if step.get("learn") and (step.get("column") in columns
                                  or step.get("into") in columns):
            step = relearn(step, cur[cur.index.isin(keep)])
        out.append(step)
        cur = apply_step(cur, step, n)
    return out


def effect(df, step):
    """(rows, columns) after the step, without keeping the result."""
    out = apply_step(df, step)
    return len(out), out.shape[1]


DATE_SAMPLE = 500           # values read to decide a column is dates
DATE_SHARE = 0.9            # share of them that must read as dates
# A date written out: 2024-03-31, 31/03/2024, 3.31.24, 2024-03-31T09:30.
_DATE_TEXT = r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}([ T]\d{1,2}:\d{2}.*)?\s*$"


def _read_share(text, fmt):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = pd.to_datetime(text, errors="coerce", utc=True, format=fmt)
    return float(d.notna().mean())


def looks_like_date(series):
    """A datetime column, or text that reads as dates (numbers such as
    20240331 or a year on its own are left as numbers)."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series) or \
            pd.api.types.is_bool_dtype(series):
        return False
    text = series.dropna().astype(str).head(DATE_SAMPLE)
    if len(text) == 0:
        return False
    if text.str.match(_DATE_TEXT).mean() < DATE_SHARE:
        return False
    return _read_share(text, "mixed") >= DATE_SHARE


def date_format(series):
    """The one format the column is written in, or None when it mixes
    several (then each value is read on its own, which is slower). Day
    first wins only when it reads more values (31/03 settles it); when
    both read every value, month first is taken."""
    import warnings

    from pandas.tseries.api import guess_datetime_format
    if pd.api.types.is_datetime64_any_dtype(series):
        return None
    text = label_text(series).dropna().head(DATE_SAMPLE)
    found = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # "day first was asked for"
        for value in text.head(50):
            for dayfirst in (False, True):
                fmt = guess_datetime_format(value, dayfirst=dayfirst)
                if fmt and fmt not in found:
                    found.append(fmt)
    best, share = None, _read_share(text, "mixed")
    for fmt in found:
        s = _read_share(text, fmt)
        if s > share or (best is None and s >= share):
            best, share = fmt, s
    return best


def date_step(df, column):
    """(step, problem): a date column's parts, the reference date for
    "days since" fixed now (the earliest date), and the hour only when the
    column records a time of day."""
    fmt = date_format(df[column])
    d = _dates(df[column], fmt).dropna()
    if len(d) < 2:
        return None, f"{column} has too few values that read as dates."
    parts = ["year", "month", "day", "weekday", "is_weekend", "days_since",
             "month_sin", "month_cos", "weekday_sin", "weekday_cos"]
    if (d != d.dt.normalize()).any():
        parts += ["hour", "hour_sin", "hour_cos"]
    return {"kind": "date_parts", "column": column, "format": fmt,
            "reference": d.min().normalize().strftime("%Y-%m-%d"),
            "parts": parts}, None
