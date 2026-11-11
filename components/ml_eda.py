"""Machine Learning, step 2: explore the file, then prepare it.

Explore reads every row (`logic.ml_explore`); only drawings are thinned.
Prepare records steps (`logic.ml_prep`) with their numbers fixed when they
are added, so the configure page, the exported scripts and a new file being
scored all see the same preparation. The file as loaded stays untouched in
`ml_source`; `ml_df` is that file with the steps applied.

Everything heavy is computed once per version of the prepared file
(`ml_version`): Streamlit draws every tab on every run.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from logic import ml, ml_eval, ml_explore, ml_prep, ml_recipe
from components import shell
from utils import auth, guard, log, scroll
from utils.constants import ML_PALETTE

_log = log.get("ml_eda")

PREFIX = "ml_e_"
PALETTE = ML_PALETTE
MUTED = "#898781"
CODE_HEIGHT = 520
PREVIEW_ROWS = 100
DIVERGING = [[0.0, "#2a78d6"], [0.5, "#f4f3ee"], [1.0, "#e34948"]]

KINDS = {
    "drop_columns": "Drop columns",
    "drop_duplicates": "Drop duplicate rows",
    "drop_missing": "Drop rows with a gap",
    "keep_range": "Keep rows in a range",
    "keep_values": "Keep rows with chosen values",
    "cap": "Cap extreme values",
    "transform": "Transform a number",
    "bin": "Cut a number into bands",
    "as_category": "Read a number as categories",
    "as_number": "Read text as numbers",
    "fill": "Fill gaps",
    "date_parts": "Turn dates into numbers",
}


def _layout(fig, height=320, **kw):
    fig.update_layout(height=height, margin=dict(t=30, b=10, l=10, r=10),
                      **kw)
    return fig


def _chart(fig, key):
    st.plotly_chart(fig, width="stretch", key=key)


def _seed(key, default, options=None):
    ss = st.session_state
    if key not in ss or (options is not None and ss[key] not in options):
        ss[key] = default


# ── state ───────────────────────────────────────────────────────────────
def set_steps(steps):
    """The prepared file follows the steps. Anything computed from the
    previous version goes: the profile, the leakage checks, the feature
    choice's basis and any trained models, which learned from it."""
    ss = st.session_state
    ss["ml_steps"] = list(steps)
    ss["ml_df"] = ml_prep.apply_steps(ss["ml_source"], ss["ml_steps"])
    ss["ml_version"] = ss.get("ml_version", 0) + 1
    for k in ("ml_profile", "ml_leak", "ml_basis", "ml_run", "ml_e_cache"):
        ss.pop(k, None)


def _cache():
    """Per version of the prepared file."""
    ss = st.session_state
    version = ss.get("ml_version", 0)
    cache = ss.get("ml_e_cache")
    if not cache or cache.get("version") != version:
        cache = ss["ml_e_cache"] = {"version": version}
    return cache


def _cached(name, fn, *args):
    cache = _cache()
    if name not in cache:
        cache[name] = fn(*args)
    return cache[name]


def _profile(df):
    ss = st.session_state
    if "ml_profile" not in ss:
        ss["ml_profile"] = ml.profile(df)
    return ss["ml_profile"]


# ── header ──────────────────────────────────────────────────────────────
def _header(df):
    ss = st.session_state
    steps = ss.get("ml_steps") or []
    src = ss["ml_source"]
    st.markdown("## Explore & prepare")
    shape = f"{len(df):,} rows × {df.shape[1]} columns"
    if steps:
        shape += (f" after {len(steps)} preparation step"
                  f"{'' if len(steps) == 1 else 's'} (loaded: "
                  f"{len(src):,} × {src.shape[1]})")
    st.caption(f"{shape}. Every row is read; only drawings are thinned.")


# ── explore ─────────────────────────────────────────────────────────────
# Why a column is off by default, as one word for its chip.
KIND_WORD = {"id": "Identifier", "constant": "Constant", "date": "Date",
             "postal": "Postal code", "high_card": "Too many values",
             "many_levels": "Many values", "derived": "Copy",
             "duplicate": "Duplicate"}


def _overview(df, prof):
    o = _cached("overview", ml_explore.overview, df, prof)
    c = st.columns(6)
    c[0].metric("Rows", f"{o['rows']:,}", border=True)
    c[1].metric("Columns", f"{o['columns']:,}", border=True)
    c[2].metric("Numbers / categories",
                f"{o['numeric']} / {o['categorical']}", border=True)
    c[3].metric("Gaps", f"{o['gaps']:,}", border=True)
    c[4].metric("Rows with a gap", f"{o['rows_with_gap']:,}", border=True)
    c[5].metric("Duplicate rows", f"{o['duplicate_rows']:,}", border=True)
    st.caption(f"Gaps are {o['gap_share']:.1%} of cells. A gap is an empty "
               f"cell, or an infinite number, which every model reads as "
               f"empty too. In memory: {o['memory_mb']:,.1f} MB.")
    st.markdown("**First rows**")
    st.dataframe(df.head(PREVIEW_ROWS), width="stretch", height=280)
    st.caption(f"The first {min(PREVIEW_ROWS, len(df)):,} of {len(df):,} "
               f"rows; every number on this page covers all of them.")
    st.markdown("**What the configure page will do with each column**")
    shell.table(
        [("Column", "column"), ("Type", "type"),
         ("Distinct", "unique", "num"), ("Gaps", "missing", "num"),
         ("Used", "use"), ("Note", "note")],
        [{"column": r["column"], "type": r["type"],
          "unique": f"{r['unique']:,}", "missing": f"{r['missing']:,}",
          "use": shell.status("Used", "good") if r["use"] else
          shell.status(KIND_WORD.get(r["kind"], "Off"), "idle"),
          "note": r["note"]} for r in prof.to_dict("records")])


def _columns_tab(df, prof):
    table = _cached("summary", ml_explore.column_summary, df, prof)
    names = {"column": "Column", "distinct": "Distinct", "gaps": "Gaps",
             "gap_share": "Gap share", "mean": "Mean", "sd": "Std dev",
             "min": "Min", "q1": "25%", "median": "Median", "q3": "75%",
             "max": "Max", "zeros": "Zeros", "skew": "Skew",
             "top": "Commonest", "top_share": "Its share"}
    num = st.column_config.NumberColumn(format="%.4g")
    share = st.column_config.NumberColumn(format="percent")
    numbers = table[table["type"] == "numeric"]
    if len(numbers):
        st.markdown(f"**Numbers** ({len(numbers)})")
        st.dataframe(numbers[["column", "distinct", "gaps", "gap_share",
                              "mean", "sd", "min", "q1", "median", "q3",
                              "max", "zeros", "skew"]].rename(columns=names),
                     hide_index=True, width="stretch", column_config={
                         **{c: num for c in ("Mean", "Std dev", "Min", "25%",
                                             "Median", "75%", "Max",
                                             "Skew")},
                         "Gap share": share})
    cats = table[table["type"] != "numeric"]
    if len(cats):
        st.markdown(f"**Categories** ({len(cats)})")
        st.dataframe(cats[["column", "distinct", "gaps", "gap_share", "top",
                           "top_share"]].rename(columns=names),
                     hide_index=True, width="stretch", column_config={
                         "Gap share": share, "Its share": share})
    st.caption("Skew above 1 or below -1 marks a long tail: a log transform "
               "under Prepare often suits it, and the configure page's "
               "log-scale target does the same for what is predicted.")


@st.fragment
@guard.guarded("Distributions")
def _distributions(df, prof):
    cols = list(df.columns)
    if not cols:
        st.caption("The file has no columns left.")
        return
    key = f"{PREFIX}dist_col"
    numeric = ml_explore.numeric_columns(prof)
    first = (ml_explore.measured_numeric(prof) or numeric or cols)[0]
    _seed(key, first, cols)
    col = st.selectbox("Column", cols, key=key)
    if col in numeric:
        bins_key = f"{PREFIX}dist_bins"
        _seed(bins_key, 40)
        bins = st.slider("Bars", 10, 100, step=5, key=bins_key)
        lefts, rights, counts = ml_eval.histogram(
            pd.to_numeric(df[col], errors="coerce"), bins)
        if not len(counts):
            st.caption(f"{col} holds no numbers.")
            return
        fig = go.Figure(go.Bar(
            x=(lefts + rights) / 2, y=counts, width=(rights - lefts) * 0.95,
            marker_color=PALETTE[0], marker_line_width=0,
            customdata=np.column_stack([lefts, rights]),
            hovertemplate="%{customdata[0]:,.4g} to %{customdata[1]:,.4g}: "
                          "%{y:,} rows<extra></extra>"))
        _layout(fig, 300, xaxis_title=col, yaxis_title="Rows", bargap=0)
        _chart(fig, "ml_e_hist")
        five = ml_explore.five_numbers(df[col])
        box = go.Figure(go.Box(
            x=[col], q1=[five["q1"]], median=[five["median"]],
            q3=[five["q3"]], lowerfence=[five["lowerfence"]],
            upperfence=[five["upperfence"]], mean=[five["mean"]],
            marker_color=PALETTE[0], orientation="v", boxpoints=False,
            name=col))
        _layout(box, 260, showlegend=False)
        _chart(box, "ml_e_box")
        gaps = int(pd.to_numeric(df[col], errors="coerce").isna().sum())
        st.caption(f"{int(counts.sum()):,} values drawn"
                   + (f"; {gaps:,} gaps left out" if gaps else "")
                   + ". The box is the quartiles, the whiskers the furthest "
                     "values within 1.5 interquartile ranges, the dashed "
                     "line the mean: all over every value.")
    else:
        counts = ml_explore.value_counts(df[col])
        show = counts.iloc[::-1]
        fig = go.Figure(go.Bar(
            x=show["Rows"], y=show["value"], orientation="h",
            marker_color=PALETTE[0], marker_line_width=0,
            customdata=show["share"],
            hovertemplate="%{y}: %{x:,} rows (%{customdata:.1%})"
                          "<extra></extra>"))
        _layout(fig, 90 + 22 * len(show), xaxis_title="Rows", bargap=0.3,
                yaxis_type="category")
        _chart(fig, "ml_e_levels")
        st.caption(f"{int(df[col].nunique(dropna=True)):,} distinct values; "
                   f"the bars add up to all {len(df):,} rows.")


def _gaps_tab(df, prof):
    table = _cached("gaps", ml_explore.gaps_table, df, prof)
    if not len(table):
        st.caption("No gaps: every cell holds a value.")
        return
    show = table.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=show["share"], y=show["column"], orientation="h",
        marker_color=PALETTE[1], marker_line_width=0,
        customdata=show["gaps"],
        hovertemplate="%{y}: %{customdata:,} gaps (%{x:.1%})<extra></extra>"))
    _layout(fig, 90 + 22 * len(show), xaxis_title="Share of rows",
            xaxis_tickformat=".0%", bargap=0.3)
    _chart(fig, "ml_e_gaps")
    st.dataframe(table.rename(columns={"column": "Column", "gaps": "Gaps",
                                       "share": "Share"}),
                 hide_index=True, width="stretch", column_config={
                     "Share": st.column_config.NumberColumn(
                         format="percent")})
    st.caption("Missing numbers are left missing by default on the "
               "configure page: trees, forests and boosting read a gap as it "
               "is, and other models ask for a fill there, learned from the "
               "training rows only. Prepare can fill a gap with a fixed "
               "value or drop the rows instead.")


@st.fragment
@guard.guarded("Correlations")
def _correlations(df, prof):
    numeric = ml_explore.numeric_columns(prof)
    if len(numeric) < 2:
        st.caption("Correlations need at least two numeric columns.")
        return
    m_key = f"{PREFIX}corr_method"
    _seed(m_key, "spearman", ["spearman", "pearson"])
    method = st.radio("Measure", ["spearman", "pearson"], horizontal=True,
                      key=m_key, format_func={
                          "spearman": "Spearman (ranks: one extreme value "
                                      "cannot drive it)",
                          "pearson": "Pearson (straight-line)"}.get)
    c_key = f"{PREFIX}corr_cols"
    if c_key not in st.session_state:
        st.session_state[c_key] = (ml_explore.measured_numeric(prof)
                                   or numeric)[:ml_explore.MAX_CORR_COLUMNS]
    else:
        st.session_state[c_key] = [c for c in st.session_state[c_key]
                                   if c in numeric]
    cols = st.multiselect("Columns", numeric, key=c_key,
                          max_selections=ml_explore.MAX_CORR_COLUMNS)
    if len(cols) < 2:
        st.caption("Choose at least two columns.")
        return
    corr = _cached(("corr", method, tuple(cols)), ml_explore.correlations,
                   df, cols, method)
    z = corr.to_numpy()
    fig = go.Figure(go.Heatmap(
        z=z, x=cols, y=cols, zmin=-1, zmax=1, colorscale=DIVERGING,
        text=np.where(np.isnan(z), "", np.round(z, 2).astype(str))
        if len(cols) <= 15 else None,
        texttemplate="%{text}" if len(cols) <= 15 else None,
        hovertemplate="%{y} and %{x}: %{z:.3f}<extra></extra>",
        colorbar=dict(title="r")))
    _layout(fig, 140 + 26 * len(cols), yaxis_autorange="reversed")
    _chart(fig, "ml_e_corr")
    pairs = ml_explore.strongest_pairs(corr)
    st.markdown("**Pairs that move together most**")
    st.dataframe(pairs.rename(columns={"first": "Column", "second": "With",
                                       "correlation": "Correlation"}),
                 hide_index=True, width="stretch", column_config={
                     "Correlation": st.column_config.NumberColumn(
                         format="%.3f")})
    st.caption("Over the rows where both values exist. Two columns near 1 "
               "or -1 carry the same information: a model needs only one.")


@st.fragment
@guard.guarded("Target")
def _target_tab(df, prof):
    cands = _cached("targets", ml_explore.candidate_targets, prof)
    if not cands:
        st.caption("No column can serve as a target.")
        return
    t_key = f"{PREFIX}target"
    _seed(t_key, cands[0], cands)
    target = st.selectbox("The column to predict", cands, key=t_key)
    kind = ml_explore.target_kind(df, prof, target)
    if kind is None:
        st.caption(f"{target} is neither a quantity nor a handful of "
                   f"classes ({int(df[target].nunique()):,} distinct values),"
                   f" so it cannot be predicted as it is.")
        return
    positive = None
    if kind == "categorical":
        classes = sorted(ml_recipe.label_text(df[target]).dropna()
                         .unique(), key=str)
        p_key = f"{PREFIX}positive"
        _seed(p_key, ml.default_positive(classes), classes)
        positive = st.selectbox("Class of interest", classes, key=p_key)
        counts = ml_explore.value_counts(df[target])
        fig = go.Figure(go.Bar(
            x=counts["value"], y=counts["Rows"], marker_color=PALETTE[0],
            marker_line_width=0, customdata=counts["share"],
            hovertemplate="%{x}: %{y:,} rows (%{customdata:.1%})"
                          "<extra></extra>"))
        _layout(fig, 240, yaxis_title="Rows", bargap=0.35,
                xaxis_type="category")
        _chart(fig, "ml_e_tclasses")
    else:
        lefts, rights, counts = ml_eval.histogram(
            pd.to_numeric(df[target], errors="coerce"), 40)
        fig = go.Figure(go.Bar(x=(lefts + rights) / 2, y=counts,
                               width=(rights - lefts) * 0.95,
                               marker_color=PALETTE[0], marker_line_width=0,
                               hovertemplate="%{x:,.4g}: %{y:,} rows"
                                             "<extra></extra>"))
        _layout(fig, 240, xaxis_title=target, yaxis_title="Rows", bargap=0)
        _chart(fig, "ml_e_thist")
    strength = _cached(("strength", target, positive),
                       ml_explore.target_strength, df, prof, target,
                       positive)
    if not len(strength):
        st.caption("No other column can be compared with it.")
        return
    st.markdown(f"**What goes with {target}**, strongest first")
    top = strength.head(15).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top["strength"], y=top["feature"], orientation="h",
        marker_color=PALETTE[2], marker_line_width=0,
        customdata=top["measure"],
        hovertemplate="%{y}: %{x:.3f} (%{customdata})<extra></extra>"))
    _layout(fig, 90 + 24 * len(top), xaxis_title="Strength, 0 to 1",
            xaxis_range=[0, 1], bargap=0.3)
    _chart(fig, "ml_e_strength")
    st.caption("Each column alone against the target, on a 0-to-1 scale: "
               "|2·AUC − 1| for a number against two classes, η² for a "
               "number against more classes or a category against a "
               "number, Cramér's V for a category against classes, "
               "|Spearman ρ| for two numbers. A column near 1 may be the "
               "target in disguise; the configure page checks for that.")
    f_key = f"{PREFIX}tfeature"
    feats = list(strength["feature"])
    _seed(f_key, feats[0], feats)
    feature = st.selectbox("One column against the target", feats,
                           key=f_key)
    by = ml_explore.target_by_value(df, target, feature, kind, positive)
    label = (f"Share {positive}" if kind == "categorical"
             else f"Average {target}")
    fig = go.Figure(go.Bar(
        x=by["value"].astype(str), y=by["target"], marker_color=PALETTE[0],
        marker_line_width=0, customdata=by["Rows"],
        hovertemplate="%{x}: %{y:.3g} over %{customdata:,} rows"
                      "<extra></extra>"))
    _layout(fig, 300, yaxis_title=label, xaxis_title=feature, bargap=0.3,
            xaxis_type="category",
            yaxis_tickformat=".0%" if kind == "categorical" else None)
    _chart(fig, "ml_e_tby")
    st.caption("Rows with no target are left out. A number is cut into "
               "equal-count bands; a category shows its commonest values.")


@st.fragment
@guard.guarded("Outliers")
def _outliers(df, prof):
    numeric = ml_explore.measured_numeric(prof)
    if not numeric:
        st.caption("No numeric columns to measure: identifiers, postal codes "
                   "and constants have no outliers to speak of.")
        return
    table = _cached("outliers", ml_explore.outliers, df, numeric)
    st.dataframe(table.rename(columns={
        "column": "Column", "low_fence": "Low fence",
        "high_fence": "High fence", "below": "Below", "above": "Above",
        "share": "Share beyond", "beyond_3_sd": "Beyond 3 std dev"}),
        hide_index=True, width="stretch", column_config={
            "Low fence": st.column_config.NumberColumn(format="%.4g"),
            "High fence": st.column_config.NumberColumn(format="%.4g"),
            "Share beyond": st.column_config.NumberColumn(format="percent")})
    key = f"{PREFIX}out_cols"
    default = list(table["column"].head(6))
    if key not in st.session_state:
        st.session_state[key] = default
    else:
        st.session_state[key] = [c for c in st.session_state[key]
                                 if c in numeric]
    cols = st.multiselect("Boxes for", numeric, key=key, max_selections=12)
    if cols:
        # One panel per column, each on its own scale: on a shared axis a
        # column in the hundreds of thousands flattens every other box.
        fig = make_subplots(rows=1, cols=len(cols), horizontal_spacing=0.04)
        for i, c in enumerate(cols, 1):
            five = ml_explore.five_numbers(df[c])
            if not five:
                continue
            fig.add_trace(go.Box(
                x=[c], q1=[five["q1"]], median=[five["median"]],
                q3=[five["q3"]], lowerfence=[five["lowerfence"]],
                upperfence=[five["upperfence"]], mean=[five["mean"]],
                name=c, marker_color=PALETTE[(i - 1) % len(PALETTE)],
                boxpoints=False), row=1, col=i)
            fig.update_xaxes(type="category", row=1, col=i)
        _layout(fig, 320, showlegend=False)
        _chart(fig, "ml_e_boxes")
        st.caption("Each box on its own scale. The box is the quartiles, "
                   "the whiskers the furthest values within the fences, the "
                   "dashed line the mean, all over every value.")
    st.caption("Fences are 1.5 interquartile ranges beyond the quartiles. "
               "Prepare can cap values at them, or at percentiles, or keep "
               "only the rows inside a range.")


# ── prepare ─────────────────────────────────────────────────────────────
def _steps_table(source, steps):
    """Rows and columns after each step, replayed from the file as loaded."""
    rows, before = [], source
    for n, step in enumerate(steps, 1):
        after = ml_prep.apply_step(before, step, n)
        rows.append({"Step": n, "What it does": ml_prep.describe(step),
                     "Rows after": len(after),
                     "Rows dropped": len(before) - len(after),
                     "Columns after": after.shape[1]})
        before = after
    return pd.DataFrame(rows)


def _steps_list(steps):
    ss = st.session_state
    if not steps:
        st.caption("No steps yet: the configure page sees the file as "
                   "loaded.")
        return
    st.dataframe(_cached("steps", _steps_table, ss["ml_source"], steps),
                 hide_index=True, width="stretch")
    if not auth.can("edit"):
        return
    c1, c2, _ = st.columns([1, 1, 2])
    if c1.button("Undo the last step", width="stretch",
                 key="ml_e_undo"):
        set_steps(steps[:-1])
        log.event(_log, "ml_prep_step_removed", steps=len(steps) - 1)
        st.rerun()
    if c2.button("Remove every step", width="stretch", key="ml_e_clear"):
        set_steps([])
        log.event(_log, "ml_prep_step_removed", steps=0)
        st.rerun()


def _new_name(df, base):
    name, i = base, 2
    while name in df.columns:
        name, i = f"{base}_{i}", i + 1
    return name


def _builder(df, prof):
    """The widgets for one new step; returns (step, problem). Numbers are
    fixed here, from the file as it stands."""
    numeric = [c for c in ml_explore.numeric_columns(prof) if c in df.columns]
    categorical = [c for c in df.columns if c not in numeric]
    k_key = f"{PREFIX}kind"
    _seed(k_key, "drop_columns", list(KINDS))
    kind = st.selectbox("Step", list(KINDS), key=k_key, format_func=KINDS.get)

    def pick(label, options, key):
        if not options:
            return None
        _seed(f"{PREFIX}{key}", options[0], options)
        return st.selectbox(label, options, key=f"{PREFIX}{key}")

    if kind == "drop_columns":
        cols = st.multiselect("Columns to drop", list(df.columns),
                              key=f"{PREFIX}drop_cols")
        if not cols:
            return None, "Choose the columns to drop."
        if len(cols) == df.shape[1]:
            return None, "At least one column has to stay."
        return {"kind": kind, "columns": cols}, None
    if kind == "drop_duplicates":
        return {"kind": kind}, None
    if kind == "drop_missing":
        gaps = list(_cached("gaps", ml_explore.gaps_table, df, prof)["column"])
        if not gaps:
            return None, "No column has a gap."
        cols = st.multiselect("Drop the rows with a gap in", gaps,
                              key=f"{PREFIX}miss_cols")
        if not cols:
            return None, "Choose the columns."
        return {"kind": kind, "columns": cols}, None
    if kind in ("keep_range", "cap", "transform", "bin", "as_category"):
        col = pick("Column", numeric, "num_col")
        if col is None:
            return None, "The file has no numeric columns."
        v = pd.to_numeric(df[col], errors="coerce")
        v = v[np.isfinite(v)]
        if not len(v):
            return None, f"{col} holds no numbers."
        lo, hi = float(v.min()), float(v.max())
        if kind == "keep_range":
            c1, c2 = st.columns(2)
            low = c1.number_input("Lowest to keep", value=lo,
                                  key=f"{PREFIX}rng_lo_{col}", format="%g")
            high = c2.number_input("Highest to keep", value=hi,
                                   key=f"{PREFIX}rng_hi_{col}", format="%g")
            if low > high:
                return None, "The lowest value is above the highest."
            if low <= lo and high >= hi:
                return None, (f"Every row is inside that range: move a "
                              f"bound to keep fewer than all {len(df):,}.")
            return {"kind": kind, "column": col,
                    "low": None if low <= lo else float(low),
                    "high": None if high >= hi else float(high)}, None
        if kind == "cap":
            m_key = f"{PREFIX}cap_mode"
            _seed(m_key, "pct", ["pct", "iqr"])
            mode = st.radio("Cap at", ["pct", "iqr"], horizontal=True,
                            key=m_key, format_func={
                                "pct": "Percentiles",
                                "iqr": "Quartile fences"}.get)
            if mode == "pct":
                r_key = f"{PREFIX}cap_pct"
                _seed(r_key, (1.0, 99.0))
                p_lo, p_hi = st.slider("Percentiles", 0.0, 100.0, step=0.5,
                                       key=r_key)
                low = ml_prep.quantile(v, p_lo / 100) if p_lo > 0 else None
                high = ml_prep.quantile(v, p_hi / 100) if p_hi < 100 \
                    else None
            else:
                f_key = f"{PREFIX}cap_k"
                _seed(f_key, 1.5, [1.5, 3.0])
                k = st.radio("Interquartile ranges beyond the quartiles",
                             [1.5, 3.0], horizontal=True, key=f_key)
                low, high = ml_prep.iqr_fences(v, k)
                low = low if low is not None and low > lo else None
                high = high if high is not None and high < hi else None
            if low is None and high is None:
                return None, f"Nothing to cap: {col} has no values beyond."
            return {"kind": kind, "column": col, "low": low,
                    "high": high}, None
        if kind == "transform":
            h_key = f"{PREFIX}how"
            _seed(h_key, "log1p", list(ml_prep.TRANSFORMS))
            how = st.radio("Transform", list(ml_prep.TRANSFORMS),
                           horizontal=True, key=h_key,
                           format_func=ml_prep.TRANSFORMS.get)
            problem = ml_prep.domain_problem(v, how)
            if problem:
                return None, problem
            r_key = f"{PREFIX}replace"
            _seed(r_key, False)
            replace = st.checkbox(f"Replace {col} instead of adding a column",
                                  key=r_key)
            into = col if replace else _new_name(df, f"{col}_{how}")
            return {"kind": kind, "column": col, "how": how,
                    "into": into}, None
        if kind == "bin":
            b_key = f"{PREFIX}bands"
            _seed(b_key, 5)
            bands = st.slider("Bands", 2, 20, key=b_key)
            w_key = f"{PREFIX}band_how"
            _seed(w_key, "quantile", ["quantile", "width"])
            how = st.radio("Band edges", ["quantile", "width"],
                           horizontal=True, key=w_key, format_func={
                               "quantile": "Equal counts",
                               "width": "Equal widths"}.get)
            edges = ml_prep.bin_edges(v, bands, how)
            if len(edges) < 3:
                return None, f"{col} has too few distinct values to cut."
            return {"kind": kind, "column": col, "edges": edges,
                    "into": _new_name(df, f"{col}_band")}, None
        return {"kind": "as_category", "column": col}, None
    if kind == "keep_values":
        col = pick("Column", categorical, "cat_col")
        if col is None:
            return None, "The file has no categorical columns."
        counts = ml_explore.value_counts(df[col], top=200)
        values = [v for v in counts["value"] if not v.endswith(
            " other values")]
        keep = st.multiselect("Keep the rows whose value is", values,
                              key=f"{PREFIX}keep_vals_{col}")
        if not keep:
            return None, "Choose the values to keep."
        return {"kind": kind, "column": col, "values": keep}, None
    if kind == "as_number":
        col = pick("Column", categorical, "text_col")
        if col is None:
            return None, "The file has no text columns."
        return {"kind": kind, "column": col}, None
    if kind == "fill":
        gaps = list(_cached("gaps", ml_explore.gaps_table, df, prof)["column"])
        col = pick("Column with gaps", gaps, "fill_col")
        if col is None:
            return None, "No column has a gap."
        if col in numeric:
            h_key = f"{PREFIX}fill_how"
            _seed(h_key, "median", ["median", "mean", "zero", "value"])
            how = st.radio("Fill with", ["median", "mean", "zero", "value"],
                           horizontal=True, key=h_key, format_func={
                               "median": "Its median", "mean": "Its mean",
                               "zero": "Zero", "value": "A value"}.get)
            v = pd.to_numeric(df[col], errors="coerce")
            v = v[np.isfinite(v)]
            if how == "value":
                value = st.number_input("Value", value=0.0, format="%g",
                                        key=f"{PREFIX}fill_value")
            elif how == "zero":
                value = 0.0
            elif not len(v):
                return None, f"{col} holds no numbers to take a {how} of."
            else:
                value = float(v.median() if how == "median" else v.mean())
            return {"kind": kind, "column": col, "value": value}, None
        h_key = f"{PREFIX}fill_how_text"
        _seed(h_key, "mode", ["mode", "value"])
        how = st.radio("Fill with", ["mode", "value"], horizontal=True,
                       key=h_key, format_func={"mode": "Its commonest value",
                                               "value": "A value"}.get)
        if how == "mode":
            counts = df[col].dropna().astype(str).value_counts()
            if not len(counts):
                return None, f"{col} has no values."
            value = str(counts.index[0])
        else:
            value = st.text_input("Value", key=f"{PREFIX}fill_text").strip()
            if not value:
                return None, "Type the value to fill with."
        return {"kind": kind, "column": col, "value": value}, None
    if kind == "date_parts":
        col = pick("Date column", _dates_left(df, prof), "date_col")
        if col is None:
            return None, ("No date columns left. A column whose values read "
                          "as dates (2024-03-31, 31/03/2024) shows here.")
        return ml_prep.date_step(df, col)
    return None, "Choose a step."


def _dates_left(df, prof):
    """Date columns that have no date features yet."""
    return [c for c in prof.loc[prof["kind"] == "date", "column"]
            if c in df.columns and f"{c}_month" not in df.columns]


def _date_offer(df, prof, steps):
    """One click for every date column: models cannot read a date whole."""
    left = _dates_left(df, prof)
    if not left:
        return
    st.info(f"Date column{'' if len(left) == 1 else 's'} found: "
            f"{', '.join(left)}. A model cannot read a date as it is; its "
            f"year, month, weekday, days since the earliest date and the "
            f"like can predict.")
    if st.button("Turn dates into numbers", key="ml_e_dates"):
        new = []
        for c in left:
            step, problem = ml_prep.date_step(df, c)
            if step:
                new.append(step)
        set_steps(steps + new)
        log.event(_log, "ml_prep_step_added", kind="date_parts",
                  steps=len(steps) + len(new))
        st.rerun()


@st.fragment
@guard.guarded("Prepare")
def _prepare(df, prof):
    ss = st.session_state
    steps = list(ss.get("ml_steps") or [])
    st.markdown("**Steps, in order**")
    _steps_list(steps)
    if auth.denied("edit"):
        return
    _date_offer(df, prof, steps)
    st.markdown("**Add a step**")
    if "ml_run" in ss:
        st.caption("A new step clears the trained models: they learned from "
                   "the file as it is now.")
    step, problem = _builder(df, prof)
    if problem:
        st.caption(problem)
    else:
        rows, cols = ml_prep.effect(df, step)
        dropped = len(df) - rows
        line = ml_prep.describe(step) + ". "
        if ml_prep.is_row_step(step):
            line += (f"Drops {dropped:,} row{'' if dropped == 1 else 's'} "
                     f"({dropped / max(len(df), 1):.1%}); {rows:,} remain.")
        else:
            added = cols - df.shape[1]
            line += (f"Adds {added} column{'' if added == 1 else 's'}."
                     if added > 0 else f"{cols} columns after it."
                     if added < 0 else "Rows and columns unchanged.")
        st.caption(line)
        if rows == 0:
            st.caption("It would leave no rows, so it cannot be added.")
        elif st.button("Add step", type="primary", key="ml_e_add"):
            set_steps(steps + [step])
            log.event(_log, "ml_prep_step_added", kind=step["kind"],
                      steps=len(steps) + 1, rows_before=len(df),
                      rows_after=rows, columns_after=cols)
            st.rerun()
    st.caption("Fills and caps here use every row, so a test row helps "
               "choose them; the configure page's fills and caps learn from "
               "the training rows only. Rows dropped here are not scored. A "
               "file scored later gets the column steps; its rows are all "
               "kept.")
    if steps:
        code = ml_eval.prep_script(steps)
        with st.expander("Python code for this preparation"):
            st.download_button("Download the code (Python)",
                               data=code.encode("utf-8"),
                               file_name="prepare_file.py",
                               mime="text/x-python", key="ml_e_code")
            st.code(code, language="python", height=CODE_HEIGHT)


# ── page ────────────────────────────────────────────────────────────────
def render():
    ss = st.session_state
    df = ss["ml_df"]
    prof = _profile(df)
    _header(df)
    names = ["Overview", "Columns", "Distributions", "Gaps", "Correlations",
             "Target", "Outliers", "Prepare"]
    tabs = dict(zip(names, st.tabs(names)))

    def panel(name):
        # Each tab's content on one white panel.
        return st.container(border=True, key=f"card_eda_{name.lower()}")

    with tabs["Overview"], panel("Overview"):
        guard.section("Overview", _overview, df, prof)
    with tabs["Columns"], panel("Columns"):
        guard.section("Columns", _columns_tab, df, prof)
    with tabs["Distributions"], panel("Distributions"):
        _distributions(df, prof)
    with tabs["Gaps"], panel("Gaps"):
        guard.section("Gaps", _gaps_tab, df, prof)
    with tabs["Correlations"], panel("Correlations"):
        _correlations(df, prof)
    with tabs["Target"], panel("Target"):
        _target_tab(df, prof)
    with tabs["Outliers"], panel("Outliers"):
        _outliers(df, prof)
    with tabs["Prepare"], panel("Prepare"):
        _prepare(df, prof)
    _, right = st.columns([3, 1.2])
    if right.button("Continue to Feature engineering", type="primary",
                    icon=":material/arrow_forward:", icon_position="right",
                    width="stretch", key="ml_e_continue"):
        ss["page"] = "ml_features"
        scroll.request()
        st.rerun()

