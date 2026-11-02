"""Build a new column from the existing ones with an LLM-written script.

The user describes the column in plain language, Claude writes a short pandas
script, the user edits it, and the result becomes a real column (Feature
engineering page). Because that script is generated rather than typed by a
developer, it is checked against an AST allow-list and run with no builtins
beyond a safe handful — wide enough for real column work (comparisons,
np.where, .str, .fillna) and nothing else.

Contract: the script assigns `result`, one value per row.

    result = df['Total_Claim_Amount'] / df['Annual_Premium']

Deliberately NOT allowed: imports, def/class/lambda, loops and comprehensions
(a Python-level loop over 300k rows is also far too slow), attribute names
starting with "_" (the usual `__globals__` escape), and any attribute outside
`_ALLOWED_ATTRS` — which is an allow-list precisely because `pd` and `np` are
package roots: banning the file-touching names one by one still left
`pd.compat.os`, `np.savetxt` and `pd.io.common.get_handle` wide open.
"""
import ast

import numpy as np
import pandas as pd

from logic import claude

RESULT_VAR = "result"
PREVIEW_ROWS = 8

_ALLOWED_NODES = (
    ast.Module, ast.Expr, ast.Assign, ast.Name, ast.Load, ast.Store,
    ast.Constant, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call,
    ast.keyword, ast.Attribute, ast.Subscript, ast.Slice, ast.List, ast.Tuple,
    ast.Dict, ast.Set, ast.IfExp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not, ast.Invert,
    ast.And, ast.Or, ast.BitAnd, ast.BitOr, ast.BitXor,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
    ast.Is, ast.IsNot,
)

# Root objects the script may touch. Anything else must be a name it assigned.
_ROOT_NAMES = {"df", "pd", "np"}

# Attribute names that read/write the outside world or evaluate strings.
# Kept ahead of the allow-list below purely for the error message: these are
# the things a model actually reaches for, and naming them beats "unknown
# attribute".
_BANNED_ATTRS = {
    "eval", "query", "exec", "open", "system", "popen", "apply", "applymap",
    "pipe", "transform", "to_csv", "to_excel", "to_pickle", "to_sql",
    "to_json", "to_parquet", "to_hdf", "to_clipboard", "to_feather",
    "read_csv", "read_excel", "read_pickle", "read_sql", "read_json",
    "iterrows", "itertuples", "itemgetter", "constructor",
}

# Every attribute the script may name, on anything. This is an ALLOW-list, and
# it has to be: `pd` and `np` are package roots, so a ban-list leaves the whole
# module graph open. `pd.compat.os` reaches the real os module, `np.savetxt`
# writes any file, and `pd.io.common.get_handle` reads one — none of which look
# like filesystem access from a ban-list's point of view. Banning them one at a
# time only invites pd.core, pd.api, np.lib and whatever the next release adds.
#
# Add a name here when a legitimate script needs it, and add an escape attempt
# to the smoke suite when you do.
_ALLOWED_ATTRS = {
    # Series / DataFrame shape and access
    "columns", "dtype", "dtypes", "iloc", "index", "loc", "name", "shape",
    "size", "values",
    # arithmetic and comparison spelled as methods
    "add", "div", "divide", "eq", "floordiv", "ge", "gt", "le", "lt", "mod",
    "mul", "multiply", "ne", "pow", "radd", "rdiv", "rmul", "rsub", "sub",
    "subtract", "truediv",
    # the everyday column vocabulary
    "abs", "astype", "between", "bfill", "clip", "combine_first", "count",
    "cumsum", "diff", "dropna", "ffill", "fillna", "head", "idxmax", "idxmin",
    "isin", "isna", "isnull", "map", "mask", "max", "mean", "median", "min",
    "nlargest", "notna", "notnull", "nsmallest", "nunique", "quantile", "rank",
    "replace", "round", "shift", "sort_values", "std", "sum", "tail",
    "to_numpy", "unique", "value_counts", "var", "where",
    # .str accessor
    "str", "capitalize", "contains", "endswith", "extract", "find", "get",
    "isalpha", "isdigit", "isnumeric", "join", "len", "lower", "lstrip",
    "match", "pad", "rstrip", "slice", "split", "startswith", "strip", "title",
    "upper", "zfill",
    # .dt accessor
    "dt", "date", "day", "dayofweek", "dayofyear", "days", "days_in_month",
    "hour", "minute", "month", "quarter", "second", "total_seconds", "weekday",
    "year",
    # numpy
    "arange", "array", "ceil", "cumprod", "digitize", "exp", "float64",
    "floor", "inf", "int64", "isfinite", "isinf", "isnan", "log", "log1p",
    "log10", "log2", "maximum", "minimum", "nan", "nanmax", "nanmean",
    "nanmedian", "nanmin", "nansum", "ones", "percentile", "power", "select",
    "sign", "sqrt", "zeros",
    # pandas top level
    "Categorical", "NA", "NaT", "Series", "cut", "factorize", "qcut",
    "to_datetime", "to_numeric", "to_timedelta",
}

_SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "round": round, "len": len,
    "int": int, "float": float, "str": str, "bool": bool, "sum": sum,
    "sorted": sorted, "list": list, "dict": dict, "set": set, "tuple": tuple,
    "range": range, "zip": zip, "enumerate": enumerate, "True": True,
    "False": False, "None": None,
}


class ScriptError(ValueError):
    """Raised for a script that is unsafe or does not honour the contract."""


def validate_script(code: str) -> ast.Module:
    """Parse and allow-list check. Returns the AST, or raises ScriptError."""
    code = (code or "").strip()
    if not code:
        raise ScriptError("The script is empty.")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise ScriptError(f"Syntax error on line {e.lineno}: {e.msg}") from None

    assigned = set()
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ScriptError(
                f"{type(node).__name__} is not allowed here. Use vectorised "
                f"pandas/numpy on whole columns — no imports, functions, loops "
                f"or comprehensions.")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise ScriptError(f"Attribute {node.attr!r} is not allowed.")
            if node.attr in _BANNED_ATTRS:
                raise ScriptError(
                    f"{node.attr!r} is not allowed — it can reach outside the "
                    f"dataset or run code from a string.")
            if node.attr not in _ALLOWED_ATTRS:
                raise ScriptError(
                    f"{node.attr!r} is not on the list of allowed pandas and "
                    f"numpy operations. Rewrite the column using the usual "
                    f"ones (arithmetic, comparisons, np.where, .fillna, "
                    f".replace, .round, .clip, .astype, .str, .dt).")
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    assigned.add(t.id)
                elif isinstance(t, ast.Subscript):
                    raise ScriptError(
                        "Assign to a variable, not into the dataframe — "
                        f"finish with `{RESULT_VAR} = ...`.")

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) \
                and node.id not in _ROOT_NAMES and node.id not in assigned \
                and node.id not in _SAFE_BUILTINS:
            raise ScriptError(
                f"Unknown name {node.id!r}. Available: df, pd, np, and "
                f"variables the script assigns.")

    if RESULT_VAR not in assigned:
        raise ScriptError(f"The script must assign `{RESULT_VAR}` — that is "
                          f"the column that gets created.")
    return tree


def run_script(code: str, df: pd.DataFrame) -> pd.Series:
    """Validate, execute, and return the new column aligned to df.index."""
    tree = validate_script(code)
    # Shallow copy: a script that adds a column to `df` cannot touch the real
    # frame, and copying 300k rows of data is avoided.
    env = {"__builtins__": dict(_SAFE_BUILTINS), "df": df.copy(deep=False),
           "pd": pd, "np": np}
    try:
        exec(compile(tree, "<derived column>", "exec"), env, env)
    except Exception as e:
        raise ScriptError(f"{type(e).__name__}: {e}") from None

    value = env.get(RESULT_VAR)
    if isinstance(value, pd.DataFrame):
        raise ScriptError(f"`{RESULT_VAR}` is a table, not a single column.")
    if isinstance(value, pd.Series):
        # reindex, not assign: sort_values / nlargest scramble the order and
        # still align correctly by label. But it also silently fills a SHORT
        # result with NaN, so `df['Age'].head(3)` became a column that was
        # 99% empty with nothing said. Coverage is the test, not length —
        # a script may legitimately produce NaN for a row it cannot score.
        if not value.index.equals(df.index):
            missing = len(df.index.difference(value.index))
            if missing:
                raise ScriptError(
                    f"`{RESULT_VAR}` covers {len(df) - missing:,} of the "
                    f"{len(df):,} rows — it needs one value per row.")
            if value.index.has_duplicates:
                raise ScriptError(f"`{RESULT_VAR}` has more than one value "
                                  f"for some rows.")
        out = value.reindex(df.index)
    elif isinstance(value, (list, tuple, np.ndarray)):
        if len(value) != len(df):
            raise ScriptError(f"`{RESULT_VAR}` has {len(value):,} values but "
                              f"the dataset has {len(df):,} rows.")
        out = pd.Series(list(value), index=df.index)
    elif np.isscalar(value) or value is None:
        raise ScriptError(f"`{RESULT_VAR}` is a single value — it must vary "
                          f"per row. Build it from a df column.")
    else:
        raise ScriptError(f"`{RESULT_VAR}` is a {type(value).__name__}, which "
                          f"cannot become a column.")
    return out


_GEN_SYSTEM = """You write one short pandas script that derives a single new column (a feature for machine learning) from a tabular dataset.

Rules for the script:
- Assign the column to `result`. The last statement should be `result = ...`.
- `df`, `pd` and `np` are available. Nothing else — no imports.
- Vectorised operations only. No def, lambda, loops, comprehensions, .apply, .eval or .query. The dataset can be 300,000 rows, so a Python-level loop is both blocked and far too slow.
- Reference columns exactly as they appear in the column list, including any odd spelling.
- Guard division: `df['a'] / df['b'].replace(0, np.nan)` then `.fillna(0)`.
- Keep it to a few lines and add a brief comment only where the intent is not obvious.
- Never write files or read anything outside `df`.

Name the column in Title_Snake_Case, short and descriptive, and do not reuse an existing column name."""

_GEN_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string",
                 "description": "New column name, Title_Snake_Case."},
        "code": {"type": "string",
                 "description": "The pandas script, ending in `result = ...`."},
        "explanation": {"type": "string",
                        "description": "One sentence on what it computes."},
    },
    "required": ["name", "code", "explanation"],
    "additionalProperties": False,
}


def column_context(df: pd.DataFrame, limit: int = 60) -> str:
    """Column names, types and example values for the prompt."""
    from logic import schema
    lines = []
    for col in list(df.columns)[:limit]:
        s = df[col]
        if schema.detect_dtype(s) == "numeric":
            num = pd.to_numeric(s, errors="coerce")
            lines.append(f"- {col} (numeric, {num.min():g} to {num.max():g})")
        else:
            vals = s.dropna().astype(str).unique()[:6]
            more = "…" if s.nunique(dropna=True) > 6 else ""
            lines.append(f"- {col} (text: {', '.join(vals)}{more})")
    if len(df.columns) > limit:
        lines.append(f"- …and {len(df.columns) - limit} more columns")
    return "\n".join(lines)


def generate_script(prompt: str, df: pd.DataFrame, *, api_key=None,
                    client=None, model=claude.MODEL,
                    existing_note: str = "") -> dict:
    """Ask Claude for {name, code, explanation}. Never executes anything."""
    user = (f"Columns available:\n{column_context(df)}\n\n"
            f"{existing_note}Create this column: {prompt}")
    try:
        got = claude.ask(_GEN_SYSTEM, [{"role": "user", "content": user}],
                         _GEN_SCHEMA, api_key=api_key, client=client,
                         model=model)
    except claude.AIError as e:
        raise ScriptError(str(e)) from None
    return {"name": got.get("name") or "New_Column",
            "code": (got.get("code") or "").strip(),
            "explanation": got.get("explanation", "")}


def friendly_error(e) -> str:
    """A failed call in words the page can show."""
    if isinstance(e, ScriptError):
        return str(e)
    return claude.friendly_error(e)


# ── feature ideas ───────────────────────────────────────────────────────
SIGNAL_ROWS = 20_000      # rows used to measure a column's signal

_SUGGEST_SYSTEM = _GEN_SYSTEM.replace(
    "You write one short pandas script that derives a single new column",
    "You are a feature engineer. You propose new columns, each a short "
    "pandas script that derives a single new column") + """

Propose features that are likely to help a model predict the target: ratios and rates that normalise one quantity by another, interactions between drivers, flags at meaningful thresholds, counts or sums of related columns, and transforms of skewed quantities. Use domain sense about what the columns mean. Never use the target column itself, or any column computed from it, in a script: that would leak the answer. Do not propose a column that merely copies or rescales an existing one."""

_SUGGEST_SCHEMA = {
    "type": "object",
    "properties": {
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string",
                             "description": "New column name, "
                                            "Title_Snake_Case."},
                    "code": {"type": "string",
                             "description": "The pandas script, ending in "
                                            "`result = ...`."},
                    "rationale": {"type": "string",
                                  "description": "One or two sentences: "
                                                 "why it should help predict "
                                                 "the target."},
                },
                "required": ["name", "code", "rationale"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["suggestions"],
    "additionalProperties": False,
}


def _signal_frame(df, y):
    """`df` and `y` on the rows with a target, capped at SIGNAL_ROWS."""
    ok = y.notna()
    df, y = df[ok], y[ok]
    if len(df) > SIGNAL_ROWS:
        idx = np.sort(np.random.default_rng(0).choice(len(df), SIGNAL_ROWS,
                                                      replace=False))
        df, y = df.iloc[idx], y.iloc[idx]
    return df, y


def _encoded(s):
    """(values, discrete) for mutual information."""
    from logic import schema
    if schema.detect_dtype(s) == "numeric":
        v = pd.to_numeric(s, errors="coerce").astype("float64")
        v = v.where(np.isfinite(v))
        fill = v.median() if v.notna().any() else 0.0
        return v.fillna(fill).to_numpy(), False
    labels = s.astype(object).where(s.notna(), "(missing)").astype(str)
    return pd.factorize(labels)[0], True


def column_signal(values, y, task) -> float:
    """Mutual information between one column and the target: 0 when the
    column says nothing about it, higher the more it does."""
    from sklearn.feature_selection import (mutual_info_classif,
                                           mutual_info_regression)
    x, discrete = _encoded(values)
    fn = (mutual_info_classif if task == "classification"
          else mutual_info_regression)
    target = (pd.factorize(y.astype(str))[0] if task == "classification"
              else pd.to_numeric(y, errors="coerce").to_numpy())
    return float(fn(x.reshape(-1, 1), target, discrete_features=[discrete],
                    random_state=0)[0])


def signals(df, target, task, columns) -> pd.Series:
    """Each column's signal about the target, strongest first."""
    sub, y = _signal_frame(df[columns], df[target])
    out = {c: column_signal(sub[c], y, task) for c in columns}
    return pd.Series(out).sort_values(ascending=False)


def new_signal(values, df, target, task) -> float:
    """A candidate column's signal, on the same rows `signals` uses."""
    frame = pd.DataFrame({"x": values}, index=df.index)
    sub, y = _signal_frame(frame, df[target])
    return column_signal(sub["x"], y, task)


def suggest_features(df, target, task, existing_signal, *, api_key=None,
                     client=None, model=claude.MODEL, n=6) -> list:
    """Claude's ideas for new columns: [{name, code, rationale}]. Nothing is
    run here; the page runs each through `run_script`."""
    strongest = ", ".join(f"{c} ({v:.3f})"
                          for c, v in existing_signal.head(12).items())
    user = (f"Columns available:\n{column_context(df)}\n\n"
            f"Target to predict: {target} ({task}).\n"
            f"Existing columns' mutual information with the target, "
            f"strongest first: {strongest}.\n\n"
            f"Propose {n} new features that could help a model predict "
            f"{target}. Do not reuse these existing names: "
            f"{', '.join(map(str, df.columns))}.")
    try:
        got = claude.ask(_SUGGEST_SYSTEM, [{"role": "user", "content": user}],
                         _SUGGEST_SCHEMA, api_key=api_key, client=client,
                         model=model, effort="high")
    except claude.AIError as e:
        raise ScriptError(str(e)) from None
    return [{"name": s.get("name") or "New_Column",
             "code": (s.get("code") or "").strip(),
             "rationale": s.get("rationale", "")}
            for s in got.get("suggestions") or []]


def safe_name(name: str, existing) -> str:
    """A clean, non-colliding column name."""
    name = " ".join(str(name or "").split()).strip() or "New_Column"
    base, i = name, 2
    while name in existing:
        name, i = f"{base}_{i}", i + 1
    return name


def preview(values: pd.Series, df: pd.DataFrame) -> dict:
    """What the user sees before committing the column."""
    from logic import schema
    dtype = schema.detect_dtype(values)
    info = {"dtype": dtype, "missing": int(values.isna().sum()),
            "distinct": int(values.nunique(dropna=True))}
    if dtype == "numeric":
        num = pd.to_numeric(values, errors="coerce")
        if num.notna().any():
            info["min"] = float(num.min())
            info["max"] = float(num.max())
            info["mean"] = float(num.mean())
    else:
        info["top"] = values.dropna().astype(str).value_counts().head(4).to_dict()
    return info
