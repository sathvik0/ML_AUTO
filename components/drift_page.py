"""Drift analysis: the file loaded here (as prepared) against a newer one.

The newer file gets the same preparation steps, Feature engineering's
columns included, so the two are compared like for like. Everything is
computed once per pair of files (`logic.ml_drift.analyse`) and kept until
either changes.
"""
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import snowflake_panel
from logic import ml_drift, ml_prep
from utils import guard, log
from utils.constants import ML_PALETTE
from utils.file_loader import load_file, read_path

_log = log.get("drift_page")

PREFIX = "ml_d_"
REF, CUR = ML_PALETTE[0], ML_PALETTE[1]
SEV_COLOR = {"Stable": "#B4B2A9", "Moderate": "#eda100", "Major": "#e34948"}
DIVERGING = [[0.0, "#2a78d6"], [0.5, "#f4f3ee"], [1.0, "#e34948"]]
NO_TARGET = "(none)"
RECENT_SAMPLE = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "sample_data", "insurance_churn_recent.csv")


def _layout(fig, height=340, **kw):
    fig.update_layout(height=height, margin=dict(t=30, b=10, l=10, r=10),
                      legend=dict(orientation="h", y=1.08), **kw)
    return fig


def _chart(fig, key):
    st.plotly_chart(fig, width="stretch", key=f"{PREFIX}{key}")


# ── the newer file ──────────────────────────────────────────────────────
def _source():
    """(frame, label) of the newer file, or (None, None) until there is
    one. Kept in session state, so a rerun does not read it again."""
    ss = st.session_state
    options = ["upload", "snowflake"] + (
        ["sample"] if os.path.exists(RECENT_SAMPLE) else [])
    key = f"{PREFIX}source"
    if key not in ss or ss[key] not in options:
        ss[key] = options[0]
    source = st.radio("The newer data", options, key=key, horizontal=True,
                      format_func={"upload": "Upload a file",
                                   "snowflake": "A Snowflake table",
                                   "sample": "Sample: recent customers"}.get)
    if source == "upload":
        up = st.file_uploader("A file with the same columns",
                              type=["csv", "xlsx", "xls"],
                              key=f"{PREFIX}file")
        if up is not None and ss.get(f"{PREFIX}cur_label") != up.name:
            df = load_file(up)
            if df is not None:
                ss[f"{PREFIX}cur"], ss[f"{PREFIX}cur_label"] = df, up.name
    elif source == "snowflake":
        from logic import snowflake_io
        if not snowflake_io.available():
            st.info(snowflake_io.unavailable_reason())
        else:
            loaded = guard.section("Snowflake", snowflake_panel.source_picker,
                                   "ml_d")
            if loaded is not None:
                ss[f"{PREFIX}cur"] = loaded
                ss[f"{PREFIX}cur_label"] = "the Snowflake data"
    else:
        st.caption("5,000 newer insurance customers, made to drift from the "
                   "sample: try it with the sample file loaded.")
        if ss.get(f"{PREFIX}cur_label") != "the recent-customers sample":
            ss[f"{PREFIX}cur"] = read_path(RECENT_SAMPLE)
            ss[f"{PREFIX}cur_label"] = "the recent-customers sample"
    return ss.get(f"{PREFIX}cur"), ss.get(f"{PREFIX}cur_label")


def _prepared(cur):
    """The newer file with this visit's preparation steps, or as loaded
    (with a note) when they do not fit it."""
    steps = st.session_state.get("ml_steps") or []
    if not steps:
        return cur, None
    try:
        return ml_prep.apply_steps(cur, steps), None
    except ml_prep.StepError as e:
        return cur, (f"The preparation steps could not be applied to the "
                     f"newer file ({e}); it is compared as loaded.")


def _target_choice(ref, cur):
    ss = st.session_state
    both = [c for c in ref.columns if c in cur.columns]
    task = ss.get("ml_w_task", "classification")
    guess = ss.get(f"ml_w_target_{task}")
    key = f"{PREFIX}target"
    options = [NO_TARGET] + both
    if key not in ss or ss[key] not in options:
        ss[key] = guess if guess in both else NO_TARGET
    target = st.selectbox("Target (optional): checks how each feature's "
                          "relationship with it has moved", options, key=key)
    return None if target == NO_TARGET else target


# ── sections ────────────────────────────────────────────────────────────
def _summary(r):
    cv = r["covariate"]
    c = st.columns(4)
    c[0].metric("Columns compared", f"{len(r['columns']):,}")
    c[1].metric("Major drift", int((cv["severity"] == "Major").sum()))
    c[2].metric("Moderate drift", int((cv["severity"] == "Moderate").sum()))
    clf = r["classifier"]
    c[3].metric("Drift classifier AUC",
                f"{clf['auc']:.2f}" if clf else "—")
    if clf:
        st.caption(f"Drift classifier: {ml_drift.classifier_reading(clf['auc'])}"
                   f" (0.50 means the files cannot be told apart).")
    ref_n, cur_n = r["rows"]
    notes = [f"Reference: {ref_n:,} rows; newer file: {cur_n:,} rows."]
    if r["only_ref"]:
        notes.append("Missing from the newer file: "
                     + ", ".join(r["only_ref"][:8]) + ".")
    if r["only_cur"]:
        notes.append("New in the newer file: "
                     + ", ".join(r["only_cur"][:8]) + ".")
    st.caption(" ".join(notes))


@guard.guarded("Covariate drift")
def _covariate(r):
    cv = r["covariate"]
    if cv.empty:
        st.caption("No columns to compare.")
        return
    show = cv.head(25).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=show["psi"], y=show["column"], orientation="h",
        marker_color=[SEV_COLOR[s] for s in show["severity"]],
        customdata=show["severity"],
        hovertemplate="%{y}: PSI %{x:.3f} (%{customdata})<extra></extra>"))
    for x, dash in ((ml_drift.PSI_MODERATE, "dot"), (ml_drift.PSI_MAJOR,
                                                     "dash")):
        fig.add_vline(x=x, line_dash=dash, line_color="#898781")
    _chart(_layout(fig, 90 + 22 * len(show), xaxis_title="PSI (dotted 0.10 "
                   "moderate, dashed 0.25 major)"), "psi")
    table = cv.assign(
        missing=lambda d: [f"{a:.1%} → {b:.1%}" for a, b in
                           zip(d["missing_ref"], d["missing_cur"])])
    st.dataframe(
        table[["column", "type", "psi", "severity", "test", "p_value",
               "change", "missing", "new_categories"]],
        hide_index=True, width="stretch",
        column_config={
            "psi": st.column_config.NumberColumn("PSI", format="%.3f"),
            "p_value": st.column_config.NumberColumn("p-value",
                                                     format="%.2g"),
            "change": "What moved", "missing": "Missing",
            "new_categories": "New categories"})
    st.caption("PSI measures how far the shares across bins (deciles of "
               "the reference for a number, the categories for a category) "
               "have moved. With thousands of rows, tests flag even tiny "
               "shifts as significant, so PSI is the measure to act on.")


@guard.guarded("Distributions")
def _distributions(r, ref, cur):
    cv = r["covariate"]
    if cv.empty:
        return
    cols = list(cv["column"])
    col = st.selectbox("Column (most drifted first)", cols,
                       key=f"{PREFIX}dist_col")
    if r["types"][col] == "numeric":
        a = pd.to_numeric(ref[col], errors="coerce").dropna()
        b = pd.to_numeric(cur[col], errors="coerce").dropna()
        lo, hi = np.nanpercentile(np.r_[a, b], [0.5, 99.5]) if len(a) + len(
            b) else (0, 1)
        fig = go.Figure()
        for v, name, color in ((a, "Reference", REF), (b, "Newer", CUR)):
            fig.add_trace(go.Histogram(
                x=v.clip(lo, hi), name=name, marker_color=color,
                opacity=0.6, histnorm="percent", nbinsx=40))
        fig.update_layout(barmode="overlay")
        _chart(_layout(fig, xaxis_title=col, yaxis_title="% of rows"),
               "dist_num")
    else:
        ra = ref[col].astype(object).where(ref[col].notna(), "(missing)") \
            .astype(str).value_counts(normalize=True)
        cb = cur[col].astype(object).where(cur[col].notna(), "(missing)") \
            .astype(str).value_counts(normalize=True)
        cats = list(ra.add(cb, fill_value=0).sort_values(ascending=False)
                    .index[:20])
        fig = go.Figure()
        for s, name, color in ((ra, "Reference", REF), (cb, "Newer", CUR)):
            fig.add_trace(go.Bar(x=cats, y=s.reindex(cats, fill_value=0),
                                 name=name, marker_color=color))
        fig.update_layout(barmode="group", yaxis_tickformat=".0%")
        _chart(_layout(fig, xaxis_title=col, yaxis_title="Share of rows"),
               "dist_cat")
    row = cv.set_index("column").loc[col]
    st.caption(f"PSI {row['psi']:.3f} ({row['severity']}) · {row['change']}")


@guard.guarded("Correlation drift")
def _correlation(r):
    pairs = r["pairs"]
    if pairs is None or pairs.empty:
        st.caption("Correlation drift needs at least two numeric columns.")
        return
    diff = r["corr_cur"] - r["corr_ref"]
    fig = go.Figure(go.Heatmap(
        z=diff.to_numpy(), x=list(diff.columns), y=list(diff.index),
        zmin=-1, zmax=1, colorscale=DIVERGING,
        hovertemplate="%{y} × %{x}: change %{z:+.2f}<extra></extra>"))
    _chart(_layout(fig, 140 + 22 * len(diff), title="Change in Spearman "
                   "correlation (newer − reference)"), "corr")
    st.markdown("**Pairs whose relationship moved most**")
    st.dataframe(pairs.head(15), hide_index=True, width="stretch",
                 column_config={
                     "first": "Column", "second": "With",
                     "ref_r": st.column_config.NumberColumn(
                         "Reference r", format="%.2f"),
                     "cur_r": st.column_config.NumberColumn(
                         "Newer r", format="%.2f"),
                     "change": st.column_config.NumberColumn(
                         "Change", format="%+.2f")})
    top = pairs.iloc[0]
    if abs(top["change"]) >= 0.2:
        st.caption(f"{top['first']} and {top['second']} moved from r = "
                   f"{top['ref_r']:.2f} to {top['cur_r']:.2f}: a "
                   f"relationship the model learnt no longer holds as it "
                   f"did, even where each column alone looks stable.")


@guard.guarded("Target relationships")
def _target(r):
    rel = r["target_rel"]
    if r["target"] is None:
        st.caption("Choose a target above to see how each feature's "
                   "relationship with it has moved.")
        return
    if rel.empty:
        st.caption(f"{r['target']} needs to be in both files, as a number "
                   f"or as two classes.")
        return
    show = rel.head(12)
    fig = go.Figure()
    for col, name, color in (("ref", "Reference", REF), ("cur", "Newer",
                                                         CUR)):
        fig.add_trace(go.Bar(y=show["feature"], x=show[col], name=name,
                             orientation="h", marker_color=color))
    fig.update_layout(barmode="group", yaxis_autorange="reversed")
    _chart(_layout(fig, 120 + 40 * len(show),
                   xaxis_title=f"Relationship with {r['target']}"),
           "target_rel")
    st.dataframe(rel, hide_index=True, width="stretch", column_config={
        "ref": st.column_config.NumberColumn("Reference", format="%.3f"),
        "cur": st.column_config.NumberColumn("Newer", format="%.3f"),
        "change": st.column_config.NumberColumn("Change", format="%+.3f")})
    st.caption("A feature whose relationship with the target has moved is "
               "concept drift: the model's learnt rule is out of date there, "
               "and retraining on recent data is the remedy.")


@guard.guarded("Drift classifier")
def _classifier(r):
    clf = r["classifier"]
    if not clf:
        st.caption("No columns to compare.")
        return
    st.markdown(f"**ROC AUC {clf['auc']:.3f}** — "
                f"{ml_drift.classifier_reading(clf['auc'])}.")
    imp = clf["importance"].head(15).iloc[::-1]
    fig = go.Figure(go.Bar(x=imp.to_numpy(), y=list(imp.index),
                           orientation="h", marker_color=ML_PALETTE[0]))
    _chart(_layout(fig, 90 + 22 * len(imp),
                   xaxis_title="Importance in telling the files apart"),
           "clf")
    st.caption("A random forest is trained to tell reference rows from "
               "newer rows; its out-of-fold ROC AUC measures drift across "
               "all columns at once, including combinations that no single "
               "column shows.")


def _downloads(r):
    c1, c2 = st.columns(2)
    c1.download_button("Covariate drift report (CSV)",
                       data=r["covariate"].to_csv(index=False)
                       .encode("utf-8"),
                       file_name="drift_covariate.csv", mime="text/csv",
                       key=f"{PREFIX}dl_cov", width="stretch")
    if r["pairs"] is not None and not r["pairs"].empty:
        c2.download_button("Correlation drift report (CSV)",
                           data=r["pairs"].to_csv(index=False)
                           .encode("utf-8"),
                           file_name="drift_correlation.csv",
                           mime="text/csv", key=f"{PREFIX}dl_corr",
                           width="stretch")


def render():
    ss = st.session_state
    ref = ss["ml_df"]
    st.markdown("## Drift analysis")
    st.caption(f"Compare the file loaded here ({len(ref):,} rows, as "
               f"prepared) with newer data: which columns have moved "
               f"(covariate drift) and which relationships have moved "
               f"(correlational drift). Run it before trusting a model on "
               f"new data, and regularly once it is in use.")
    cur, label = _source()
    if cur is None:
        return
    cur, note = _prepared(cur)
    if note:
        st.caption(note)
    target = _target_choice(ref, cur)
    sig = (ss.get("ml_version", 0), label, len(cur), target)
    if st.button("Analyse drift", type="primary", key=f"{PREFIX}go"):
        with st.spinner("Comparing every column"), log.timer() as t:
            ss[f"{PREFIX}result"] = {"sig": sig, "r": ml_drift.analyse(
                ref, cur, target, ss.get("ml_profile"))}
        r = ss[f"{PREFIX}result"]["r"]
        log.event(_log, "ml_drift_analysed", columns=len(r["columns"]),
                  rows_ref=len(ref), rows_cur=len(cur), ms=t.ms,
                  major=int((r["covariate"]["severity"] == "Major").sum()))
    got = ss.get(f"{PREFIX}result")
    if not got:
        return
    if got["sig"] != sig:
        st.caption("The files or the target changed since the analysis: "
                   "analyse again to update it.")
        return
    r = got["r"]
    st.divider()
    st.markdown(f"#### Against {label}")
    _summary(r)
    names = ["Covariate drift", "Distributions", "Correlation drift",
             "Target relationships", "Drift classifier"]
    tabs = dict(zip(names, st.tabs(names)))
    with tabs["Covariate drift"]:
        _covariate(r)
    with tabs["Distributions"]:
        _distributions(r, ref, cur)
    with tabs["Correlation drift"]:
        _correlation(r)
    with tabs["Target relationships"]:
        _target(r)
    with tabs["Drift classifier"]:
        _classifier(r)
    st.divider()
    _downloads(r)
