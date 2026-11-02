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


_column_script = _checked_script  # noqa: F811  (replaces the plain runner)


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


def effect(df, step):
    """(rows, columns) after the step, without keeping the result."""
    out = apply_step(df, step)
    return len(out), out.shape[1]
