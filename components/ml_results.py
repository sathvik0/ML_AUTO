"""Machine Learning, step 3: how the models did, what drives them, and the
files to take away.

Every number on the page comes from `logic.ml_eval`, computed on every test
row (or every row, for clustering). Only drawings are thinned: a scatter
draws at most `DRAW_POINTS` points, chosen at random with a fixed seed, and
says so beside it.
"""
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from components import (export_page, ml_evaluate, ml_summary, project_panel,
                        snowflake_panel)
from logic import ml, ml_eval
from utils import guard, log
from utils.constants import ML_PALETTE
from utils.file_loader import load_file

_log = log.get("ml_results")

# Validated in utils/constants.py. The first three pass for every pair, so a
# scatter of more than three groups gives each its own marker shape too;
# three sit below 3:1 contrast on white, which is why every chart here has
# its numbers in a table as well.
PALETTE = ML_PALETTE
MUTED = "#898781"         # reference lines, folded groups
DIM = "#c3c2b7"           # the models that are not the best
SYMBOLS = ("circle", "square", "diamond", "triangle-up", "cross", "x",
           "star", "triangle-down")
SEQUENTIAL = [[0.0, "#cde2fb"], [0.5, "#5598e7"], [1.0, "#0d366b"]]
DRAW_POINTS = 2_000
# A script is several hundred lines; it scrolls inside this height rather
# than running the tab to 13,000px.
CODE_HEIGHT = 520
PAGE_SIZES = [100, 500, 1_000]
TEXT_CELLS = 10           # a confusion matrix larger than this drops its text


def _layout(fig, height=320, **kw):
    fig.update_layout(height=height, margin=dict(t=30, b=10, l=10, r=10),
                      **kw)
    return fig


def _chart(fig, key):
    st.plotly_chart(fig, width="stretch", key=key)


def _drawn(n, seed=0):
    """Positions to draw: every one up to DRAW_POINTS, a seeded random
    choice above it."""
    if n <= DRAW_POINTS:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, DRAW_POINTS,
                                                      replace=False))


def _seconds(value):
    """"about 1 second", never "1 seconds"."""
    n = max(int(round(value)), 1)
    return f"{n:,} second{'' if n == 1 else 's'}"


def _fmt(key, value):
    return ml_eval.fmt_metric(key, value)


# ── header and verdict ──────────────────────────────────────────────────
def _summary(run):
    cfg = run["config"]
    feats = len(cfg["numeric"]) + len(cfg["categorical"])
    if run.get("external"):
        what = (f"{run['n_rows']:,} rows grouped" if run["task"] ==
                "clustering" else
                f"{ml.TASK_LABELS[run['task']]} of "
                f"{ml_eval.describe_target(run)}"
                + (f" (class of interest: {run['positive']})"
                   if run.get("positive") else "")
                + f" · {run['n_test']:,} rows scored against it")
        unknown = run.get("n_unknown_class") or 0
        return (f"Uploaded model, evaluated on this file: {what}"
                + (f", {unknown:,} with a class the model does not have"
                   if unknown else "")
                + f" · {feats} input{'' if feats == 1 else 's'} · "
                  f"{run['created']}, {run['seconds']:.1f}s. "
                + ml_evaluate.frame_note(run))
    if run["task"] == "clustering":
        return (f"Clustering of {run['n_rows']:,} rows on {feats} feature"
                f"{'' if feats == 1 else 's'} · trained {run['created']} in "
                f"{run['seconds']:.1f}s")
    pos = (f" (class of interest: {run['positive']})"
           if run.get("positive") else "")
    unl = (f", {run['n_unlabelled']:,} with no target scored"
           if run["n_unlabelled"] else "")
    folds = (f" · {run['folds']}-fold cross-validation" if run["folds"]
             else "")
    return (f"{ml.TASK_LABELS[run['task']]} of "
            f"{ml_eval.describe_target(run)}{pos} · {run['n_train']:,} rows "
            f"to train, {run['n_test']:,} to test{unl} · {feats} feature"
            f"{'' if feats == 1 else 's'}{folds} · trained {run['created']} "
            f"in {run['seconds']:.1f}s")


def _verdict(run):
    v = ml_eval.verdict(run)
    if v is None:
        return
    ok, key, value, chance = v
    best = ml.result(run, run["best"])
    name = ml.metric_label(run["family"], key)
    shown = f"{chance:.2f}" if key.startswith("roc_auc") else f"{chance:g}"
    where = (f"on the {run['n_test']:,} rows with a target, where {shown} "
             f"is guessing" if run.get("external") else
             f"on the {run['n_test']:,} test rows, where {shown} is guessing")
    if ok:
        st.success(f"**{best['label']}** beats guessing: {name} "
                   f"{value:.3f} {where}.")
        if run.get("external"):
            st.caption("If the model was trained on some of these rows, "
                       "these scores overstate how it does on rows it has "
                       "not seen.")
    else:
        target = ml_eval.describe_target(run)
        st.warning(f"No model beats guessing by a meaningful margin. The "
                   f"best, {best['label']}, has {name} {value:.3f} {where}. "
                   f"On this file, the chosen features do not predict "
                   f"{target}.")


def _leaderboard(run):
    table = ml_eval.leaderboard(run).drop(columns="key")
    metric_cols = [c for c in table.columns
                   if c not in ("Model", "Status", "Settings", "Clusters")]
    config = {c: st.column_config.NumberColumn(format="%.3f")
              for c in metric_cols}
    config["Fit (s)"] = st.column_config.NumberColumn(format="%.2f")
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config=config)
    metric = run["metric"]
    ok = [r for r in run["results"] if not r.get("error")
          and r["test"].get(metric) is not None]
    if len(ok) < 2:
        return
    name = ml.metric_label(run["family"], metric)
    # Ranked, best result at the top; the chosen model in colour.
    up = ml.higher_is_better(run["family"], metric)
    ok = sorted(ok, key=lambda r: r["test"][metric], reverse=not up)
    colour = [PALETTE[0] if r["key"] == run["best"] else
              MUTED if r["key"] == ml.BASELINE else DIM for r in ok]
    fig = go.Figure(go.Bar(
        y=[r["label"] for r in ok], x=[r["test"][metric] for r in ok],
        orientation="h", marker_color=colour, marker_line_width=0,
        hovertemplate="%{y}<br>" + name + " %{x:.3f}<extra></extra>"))
    rows_ = "rows with a target" if run.get("external") else "test rows"
    _layout(fig, 70 + 34 * len(ok), xaxis_title=f"{name}, {rows_}",
            bargap=0.35)
    by = ("chosen by cross-validation, judged on the test rows"
          if run["folds"] else "chosen on the test rows")
    st.caption("The uploaded model in colour, the baseline grey, both on the "
               "rows with a target." if run.get("external") else
               f"Best model in colour, {by}; the baseline is grey.")
    _chart(fig, "ml_models")


# ── classification ──────────────────────────────────────────────────────
def _confusion_fig(table):
    z = table.to_numpy()
    totals = z.sum(axis=1, keepdims=True)
    share = np.divide(z, totals, out=np.zeros_like(z, dtype=float),
                      where=totals > 0)
    fig = go.Figure(go.Heatmap(
        z=z, x=list(table.columns), y=list(table.index),
        colorscale=SEQUENTIAL, showscale=False, customdata=share,
        hovertemplate="Actual %{y}, predicted %{x}<br>%{z:,} rows "
                      "(%{customdata:.0%} of the actual)<extra></extra>"))
    if len(table) <= TEXT_CELLS:
        # Dark text on the light cells, white on the dark ones: one text
        # colour cannot read on both ends of the ramp.
        top = z.max() or 1
        for i, actual in enumerate(table.index):
            for j, said in enumerate(table.columns):
                fig.add_annotation(
                    x=said, y=actual, showarrow=False,
                    text=f"{z[i, j]:,}<br>{share[i, j]:.0%}",
                    font=dict(color="#FFFFFF" if z[i, j] / top > 0.5
                              else "#0b0b0b"))
    fig.update_yaxes(autorange="reversed", title="Actual", type="category")
    fig.update_xaxes(title="Predicted", type="category")
    return _layout(fig, 300 + 18 * max(0, len(table) - 4))


def _diagonal(fig, x0=0, x1=1, name="Guessing"):
    fig.add_trace(go.Scatter(x=[x0, x1], y=[x0, x1], mode="lines",
                             line=dict(color=MUTED, width=1), name=name,
                             hoverinfo="skip"))


def _binary_performance(run, res):
    pos = ml_eval.rows(run, "test")
    hit = ml_eval.truth(run, pos) == run["positive"]
    score = ml_eval.positive_score(run, res, pos)
    pred = res["pred"][pos]
    c_l, c_r = st.columns(2)
    with c_l:
        st.markdown("**Confusion matrix**")
        table = ml_eval.confusion(ml_eval.truth(run, pos), pred,
                                  run["classes"])
        _chart(_confusion_fig(table), f"ml_cm_{res['key']}")
        st.caption("Rows are what happened, columns what the model said, at "
                   "a 0.50 probability; the Targeting tab moves the cut-off.")
    with c_r:
        st.markdown("**ROC curve**")
        if 0 < hit.sum() < len(hit):
            fpr, tpr, auc = ml_eval.roc(hit, score)
            x, y = ml_eval.thin(fpr, tpr)
            fig = go.Figure()
            _diagonal(fig)
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines", line=dict(color=PALETTE[0], width=2),
                name=f"{res['label']} (AUC {auc:.3f})",
                hovertemplate="False positives %{x:.0%}<br>True positives "
                              "%{y:.0%}<extra></extra>"))
            _layout(fig, xaxis_title="False positive rate",
                    yaxis_title="True positive rate",
                    legend=dict(orientation="h", y=-0.25))
            _chart(fig, f"ml_roc_{res['key']}")
        else:
            st.caption("The test rows hold one class only, so there is no "
                       "curve to draw.")
    c_l, c_r = st.columns(2)
    with c_l:
        st.markdown("**Precision and recall**")
        if hit.any():
            recall, precision, ap = ml_eval.pr(hit, score)
            x, y = ml_eval.thin(recall, precision)
            fig = go.Figure()
            rate = hit.mean()
            fig.add_trace(go.Scatter(x=[0, 1], y=[rate, rate], mode="lines",
                                     line=dict(color=MUTED, width=1),
                                     name=f"Guessing ({rate:.0%})",
                                     hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines", line=dict(color=PALETTE[0], width=2),
                name=f"{res['label']} (PR AUC {ap:.3f})",
                hovertemplate="Recall %{x:.0%}<br>Precision %{y:.0%}"
                              "<extra></extra>"))
            _layout(fig, xaxis_title="Recall: share of positives found",
                    yaxis_title="Precision: share of flags right",
                    legend=dict(orientation="h", y=-0.25))
            _chart(fig, f"ml_pr_{res['key']}")
    with c_r:
        st.markdown("**Calibration**")
        cal = ml_eval.calibration(hit, score)
        fig = go.Figure()
        _diagonal(fig, name="Perfectly calibrated")
        fig.add_trace(go.Scatter(
            x=cal["predicted"], y=cal["observed"], mode="lines+markers",
            line=dict(color=PALETTE[0], width=2), marker=dict(size=8),
            name=res["label"], customdata=cal["records"],
            hovertemplate="Predicted %{x:.0%}<br>Observed %{y:.0%}<br>"
                          "%{customdata:,} rows<extra></extra>"))
        _layout(fig, xaxis_title="Predicted probability (bin average)",
                yaxis_title="Observed share",
                legend=dict(orientation="h", y=-0.25))
        _chart(fig, f"ml_cal_{res['key']}")
        st.caption("Ten equal bins of predicted probability. On the line, a "
                   "70% prediction comes true 70% of the time.")
    st.markdown("**Probabilities by actual class**")
    edges = np.linspace(0, 1, 21)
    left, right, on = ml_eval.histogram(score[hit], bins=edges)
    _, _, off = ml_eval.histogram(score[~hit], bins=edges)
    fig = go.Figure()
    # Side by side, not overlaid: two translucent colours over each other
    # read as a third.
    for counts, name, colour in ((on, f"Actual {run['positive']}",
                                  PALETTE[0]),
                                 (off, "Actual other", PALETTE[1])):
        if len(counts):
            fig.add_trace(go.Bar(x=(left + right) / 2, y=counts, name=name,
                                 marker_color=colour, marker_line_width=0,
                                 hovertemplate="%{x:.2f}: %{y:,} rows"
                                               "<extra></extra>"))
    _layout(fig, 260, barmode="group", bargap=0.25, bargroupgap=0.05,
            xaxis_title="Predicted probability", xaxis_range=[0, 1],
            yaxis_title="Test rows", legend=dict(orientation="h", y=-0.3))
    _chart(fig, f"ml_hist_{res['key']}")
    with st.expander("Tables"):
        st.dataframe(table, width="stretch")
        st.dataframe(cal.rename(columns={"predicted": "Predicted",
                                         "observed": "Observed",
                                         "records": "Rows"}),
                     width="stretch", hide_index=True)


@st.fragment
@guard.guarded("Targeting")
def _targeting(run, res, df):
    pos = ml_eval.rows(run, "test")
    hit = ml_eval.truth(run, pos) == run["positive"]
    score = ml_eval.positive_score(run, res, pos)
    st.markdown("**Gains: how many positives the top of the list holds**")
    g = ml_eval.gains(hit, score)
    c_l, c_r = st.columns(2)
    with c_l:
        fig = go.Figure()
        _diagonal(fig, name="Targeting at random")
        fig.add_trace(go.Scatter(
            x=[0] + list(g["targeted"]), y=[0] + list(g["captured"]),
            mode="lines+markers", line=dict(color=PALETTE[0], width=2),
            marker=dict(size=8), name=res["label"],
            hovertemplate="Top %{x:.0%} of rows<br>%{y:.0%} of positives"
                          "<extra></extra>"))
        _layout(fig, xaxis_title="Share of rows targeted, best scores first",
                yaxis_title="Share of positives reached",
                xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                legend=dict(orientation="h", y=-0.25))
        _chart(fig, f"ml_gain_{res['key']}")
    with c_r:
        fig = go.Figure(go.Bar(
            x=g["group"], y=g["lift"], marker_color=PALETTE[0],
            marker_line_width=0,
            hovertemplate="Group %{x}<br>lift %{y:.2f}<extra></extra>"))
        fig.add_hline(y=1, line_color=MUTED, line_width=1)
        _layout(fig, xaxis_title="Tenth of rows, 1 = highest scores",
                yaxis_title="Lift over the average rate", bargap=0.35)
        fig.update_xaxes(dtick=1)
        _chart(fig, f"ml_lift_{res['key']}")
    top = g.iloc[0]
    st.caption(f"Targeting the top tenth of test rows by score reaches "
               f"{top['captured']:.0%} of the positives, {top['lift']:.1f} "
               f"times the rate of targeting at random.")
    shares = ("rate", "captured", "targeted")
    table = g.assign(**{c: g[c] * 100 for c in shares}).rename(columns={
        "group": "Tenth", "records": "Rows", "positives": "Positives",
        "rate": "Rate", "captured": "Positives reached (cumulative)",
        "targeted": "Rows targeted (cumulative)", "lift": "Lift"})
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        c: st.column_config.NumberColumn(format="%.1f%%") for c in
        ("Rate", "Positives reached (cumulative)",
         "Rows targeted (cumulative)")} | {
        "Lift": st.column_config.NumberColumn(format="%.2f")})
    st.markdown("**Cut-off: who the model flags**")
    t = st.slider("Flag rows with a probability of at least", 0.01, 0.99,
                  0.50, 0.01, key=f"ml_r_threshold_{res['key']}")
    stats = ml_eval.at_threshold(hit, score, t)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", "n/a" if stats["precision"] is None
              else f"{stats['precision']:.0%}")
    m2.metric("Recall", "n/a" if stats["recall"] is None
              else f"{stats['recall']:.0%}")
    m3.metric("F1", "n/a" if stats["f1"] is None else f"{stats['f1']:.3f}")
    m4.metric("Test rows flagged", f"{stats['flagged']:,}")
    every = ml_eval.positive_score(run, res, slice(None)) >= t
    st.caption(f"Precision: the share of flagged test rows in class "
               f"{run['positive']}. Recall: the share of class "
               f"{run['positive']} test rows that are flagged. Across the "
               f"whole file this cut-off flags {int(every.sum()):,} of "
               f"{len(every):,} rows ({every.mean():.1%}).")
    st.dataframe(pd.DataFrame(
        {"Flagged": [stats["tp"], stats["fp"]],
         "Not flagged": [stats["fn"], stats["tn"]]},
        index=[f"Actual {run['positive']}", "Actual other"]),
        width="content")


def _multiclass_performance(run, res):
    pos = ml_eval.rows(run, "test")
    y = ml_eval.truth(run, pos)
    pred = res["pred"][pos]
    st.markdown("**Confusion matrix**")
    table = ml_eval.confusion(y, pred, run["classes"])
    _chart(_confusion_fig(table), f"ml_cm_{res['key']}")
    st.caption("Rows are what happened, columns the class the model gave the "
               "highest probability.")
    st.markdown("**By class**")
    st.dataframe(ml_eval.per_class(y, pred, run["classes"]),
                 hide_index=True, width="stretch", column_config={
                     c: st.column_config.NumberColumn(format="%.3f")
                     for c in ("Precision", "Recall", "F1")})
    if len(run["classes"]) <= len(PALETTE):
        st.markdown("**ROC curve of each class against the rest**")
        fig = go.Figure()
        _diagonal(fig)
        for j, c in enumerate(run["classes"]):
            hit = y == c
            if not 0 < hit.sum() < len(hit):
                continue
            fpr, tpr, auc = ml_eval.roc(hit, res["proba"][pos, j])
            x, yy = ml_eval.thin(fpr, tpr)
            fig.add_trace(go.Scatter(x=x, y=yy, mode="lines",
                                     line=dict(color=PALETTE[j], width=2),
                                     name=f"{c} (AUC {auc:.3f})"))
        _layout(fig, 360, xaxis_title="False positive rate",
                yaxis_title="True positive rate")
        _chart(fig, f"ml_roc_{res['key']}")
    else:
        st.caption(f"{len(run['classes'])} classes are too many to tell "
                   f"apart by colour; the table above has every one.")


# ── regression ──────────────────────────────────────────────────────────
def _regression_performance(run, res):
    pos = ml_eval.rows(run, "test")
    actual = ml_eval.truth(run, pos).astype("float64")
    pred = res["pred"][pos]
    resid = actual - pred
    drawn = _drawn(len(pos), run["base"]["seed"])
    c_l, c_r = st.columns(2)
    with c_l:
        st.markdown("**Predicted against actual**")
        lo = float(np.nanmin([actual.min(), pred.min()]))
        hi = float(np.nanmax([actual.max(), pred.max()]))
        fig = go.Figure()
        _diagonal(fig, lo, hi, name="Perfect prediction")
        fig.add_trace(go.Scatter(
            x=actual[drawn], y=pred[drawn], mode="markers", name="Test row",
            marker=dict(color=PALETTE[0], size=5, opacity=0.5),
            hovertemplate="Actual %{x:,.4g}<br>Predicted %{y:,.4g}"
                          "<extra></extra>"))
        _layout(fig, xaxis_title=f"Actual {run['target']}",
                yaxis_title=f"Predicted {run['target']}",
                legend=dict(orientation="h", y=-0.25))
        _chart(fig, f"ml_pva_{res['key']}")
    with c_r:
        st.markdown("**Errors against the prediction**")
        fig = go.Figure(go.Scatter(
            x=pred[drawn], y=resid[drawn], mode="markers",
            marker=dict(color=PALETTE[0], size=5, opacity=0.5),
            hovertemplate="Predicted %{x:,.4g}<br>Actual minus predicted "
                          "%{y:,.4g}<extra></extra>"))
        fig.add_hline(y=0, line_color=MUTED, line_width=1)
        _layout(fig, xaxis_title=f"Predicted {run['target']}",
                yaxis_title="Actual minus predicted")
        _chart(fig, f"ml_res_{res['key']}")
    if len(pos) > DRAW_POINTS:
        st.caption(f"Drawn: {DRAW_POINTS:,} of the {len(pos):,} test rows, "
                   f"chosen at random. Every number on this page covers all "
                   f"of them.")
    c_l, c_r = st.columns(2)
    with c_l:
        st.markdown("**Spread of the errors**")
        left, right, counts = ml_eval.histogram(resid)
        fig = go.Figure(go.Bar(
            x=(left + right) / 2, y=counts, width=(right - left) * 0.9,
            marker_color=PALETTE[0], marker_line_width=0,
            hovertemplate="%{x:,.4g}: %{y:,} rows<extra></extra>"))
        _layout(fig, 280, xaxis_title="Actual minus predicted",
                yaxis_title="Test rows")
        _chart(fig, f"ml_rhist_{res['key']}")
    with c_r:
        st.markdown("**Average by tenth of the prediction**")
        dec = ml_eval.regression_deciles(actual, pred)
        fig = go.Figure()
        for col, name, colour in (("predicted", "Predicted", PALETTE[0]),
                                  ("actual", "Actual", PALETTE[1])):
            fig.add_trace(go.Bar(x=dec["group"], y=dec[col], name=name,
                                 marker_color=colour, marker_line_width=0))
        _layout(fig, 280, barmode="group", bargap=0.3,
                xaxis_title="Tenth of rows, 1 = highest predictions",
                yaxis_title=f"Average {run['target']}",
                legend=dict(orientation="h", y=-0.3))
        fig.update_xaxes(dtick=1)
        _chart(fig, f"ml_rdec_{res['key']}")
    with st.expander("Table"):
        st.dataframe(dec.rename(columns={"group": "Tenth", "records": "Rows",
                                         "predicted": "Average predicted",
                                         "actual": "Average actual"}),
                     hide_index=True, width="stretch")
    st.caption("ML_Gap in the predictions file is predicted minus actual: "
               "above zero, a row scores less than rows like it do. "
               "Rows marked out of fold or test are scored by a model that "
               "never saw them.")


# ── what drives the model ───────────────────────────────────────────────
@st.fragment
@guard.guarded("What drives the model")
def _drivers(run, res, df):
    cache = run.setdefault("importance", {})
    imp = cache.get(res["key"])
    name = ml.metric_label(run["family"], run["metric"])
    if imp is None:
        repeats = ml_eval.importance_repeats(run)
        cost = ml_eval.importance_seconds(run, res)
        st.caption(f"Shuffles each feature in turn on the {run['n_test']:,} "
                   f"test rows, {repeats} times, and measures how much "
                   f"{name} drops: about {_seconds(cost)} for this "
                   f"model.")
        if st.button("Measure feature importance",
                     key=f"ml_r_imp_{res['key']}"):
            with st.spinner("Measuring feature importance"):
                cache[res["key"]] = ml_eval.permutation_importance(run, res,
                                                                   df)
            imp = cache[res["key"]]
    if imp is not None:
        st.markdown(f"**Feature importance: the drop in {name} when a "
                    f"feature is shuffled**")
        show = imp.iloc[::-1]
        fig = go.Figure(go.Bar(
            x=show["importance"], y=show["feature"], orientation="h",
            marker_color=PALETTE[0], marker_line_width=0,
            error_x=dict(type="data", array=show["spread"], color=MUTED,
                         thickness=1),
            hovertemplate="%{y}<br>drop %{x:.4f}<extra></extra>"))
        _layout(fig, 120 + 22 * len(show), bargap=0.35,
                xaxis_title=f"Drop in {name}")
        _chart(fig, f"ml_imp_{res['key']}")
        st.caption("Measured on the test rows, in the metric that picked the "
                   "model. Near zero, the model does not lean on a feature; a "
                   "column it leans on heavily deserves a leakage check. "
                   "Associations, not causes.")
    if run["task"] == "classification" and len(run["classes"]) != 2:
        return
    feats = run["base"]["numeric"] + run["base"]["categorical"]
    if not feats:
        return
    st.markdown("**Effect of one feature**")
    key = f"ml_r_pd_{res['key']}"
    if key not in st.session_state or st.session_state[key] not in feats:
        st.session_state[key] = (imp["feature"].iloc[0] if imp is not None
                                 else feats[0])
    feature = st.selectbox("Feature", feats, key=key)
    pd_cache = run.setdefault("pd", {})
    pkey = (res["key"], feature)
    if pkey not in pd_cache:
        with st.spinner("Computing the effect"):
            pd_cache[pkey] = ml_eval.partial_dependence(run, res, df, feature)
    curve = pd_cache[pkey]
    what = (f"P({run['positive']})" if run["task"] == "classification"
            else f"Predicted {run['target']}")
    if feature in run["base"]["categorical"]:
        fig = go.Figure(go.Bar(x=curve["value"].astype(str),
                               y=curve["prediction"],
                               marker_color=PALETTE[0], marker_line_width=0))
        fig.update_xaxes(type="category")
    else:
        fig = go.Figure(go.Scatter(x=curve["value"], y=curve["prediction"],
                                   mode="lines+markers",
                                   line=dict(color=PALETTE[0], width=2),
                                   marker=dict(size=8)))
    _layout(fig, 280, xaxis_title=feature, yaxis_title=f"Average {what}",
            bargap=0.35)
    _chart(fig, f"ml_pd_{res['key']}")
    st.caption(f"The average {what} over every test row with {feature} set "
               f"to each value in turn and everything else left as it is.")


# ── clustering ──────────────────────────────────────────────────────────
def _cluster_colours(names):
    """Fixed by cluster name, so a cluster keeps its colour in every chart;
    past the palette, clusters share the muted grey and keep their shape."""
    out = {}
    for name in names:
        if name == "Noise":
            out[name] = MUTED
            continue
        i = int(str(name)[1:]) - 1
        out[name] = PALETTE[i] if i < len(PALETTE) else MUTED
    return out


def _clusters(run, res, df):
    df = ml_eval.analysis_frame(run, df)
    labels = pd.Series(res["labels"])
    counts = labels.value_counts()
    order = sorted(counts.index, key=ml_eval._cluster_key)
    colours = _cluster_colours(order)
    c_l, c_r = st.columns([1, 2])
    with c_l:
        st.markdown("**Rows per cluster**")
        fig = go.Figure(go.Bar(
            x=order, y=[counts[c] for c in order],
            marker_color=[colours[c] for c in order], marker_line_width=0,
            customdata=[counts[c] / len(labels) for c in order],
            hovertemplate="%{x}<br>%{y:,} rows (%{customdata:.1%})"
                          "<extra></extra>"))
        _layout(fig, 340, bargap=0.35, yaxis_title="Rows")
        fig.update_xaxes(type="category")
        _chart(fig, f"ml_csize_{res['key']}")
    with c_r:
        st.markdown("**The rows on two axes**")
        coords = run.get("coords")
        if coords is None:
            st.caption("One feature: nothing to project.")
        else:
            drawn = _drawn(len(labels), run["base"]["seed"])
            fig = go.Figure()
            shapes = len(order) > 3
            for i, c in enumerate(order):
                idx = drawn[labels.to_numpy()[drawn] == c]
                fig.add_trace(go.Scattergl(
                    x=coords[idx, 0], y=coords[idx, 1], mode="markers",
                    name=c, marker=dict(
                        color=colours[c], size=6, opacity=0.6,
                        symbol=SYMBOLS[i % len(SYMBOLS)] if shapes
                        else "circle"),
                    hovertemplate=f"{c}<extra></extra>"))
            _layout(fig, 340, xaxis_title="First principal component",
                    yaxis_title="Second principal component")
            _chart(fig, f"ml_cmap_{res['key']}")
            note = ("The two directions along which the prepared features "
                    "vary most.")
            if len(labels) > DRAW_POINTS:
                note += (f" Drawn: {DRAW_POINTS:,} of {len(labels):,} rows, "
                         f"chosen at random; the counts cover every one.")
            st.caption(note)
    m = res["test"]
    parts = [f"{ml.metric_label('clustering', k)} {_fmt(k, m.get(k))}"
             for k in ("silhouette", "calinski_harabasz", "davies_bouldin")]
    note = ("" if m.get("silhouette") is not None else
            f" Silhouette compares every pair of rows, so it is computed up "
            f"to {ml.SILHOUETTE_MAX_ROWS:,} rows and not estimated from a "
            f"subset above that.")
    st.caption(" · ".join(parts) + ". Higher silhouette and "
               "Calinski-Harabasz, and lower Davies-Bouldin, mean tighter, "
               "better separated clusters." + note)


def _profiles(run, res, df):
    df = ml_eval.analysis_frame(run, df)
    numeric, categorical = run["base"]["numeric"], run["base"]["categorical"]
    if numeric:
        st.markdown("**What sets each cluster apart**")
        heat = ml_eval.cluster_heat(df, res["labels"], numeric)
        span = float(np.nanmax(np.abs(heat.to_numpy()))) if heat.size else 1
        span = span if np.isfinite(span) and span > 0 else 1
        fig = go.Figure(go.Heatmap(
            z=heat.to_numpy(), x=list(heat.columns), y=list(heat.index),
            colorscale="RdBu_r", zmid=0, zmin=-span, zmax=span,
            colorbar=dict(title="SD"),
            hovertemplate="%{y}, %{x}<br>%{z:+.2f} SD from the average"
                          "<extra></extra>"))
        fig.update_yaxes(autorange="reversed", type="category")
        _layout(fig, 140 + 30 * len(heat))
        _chart(fig, f"ml_heat_{res['key']}")
        st.caption("Each cluster's average, in standard deviations from the "
                   "whole file's average: red above, blue below, white "
                   "typical.")
    st.markdown("**Profiles**")
    prof = ml_eval.cluster_profiles(df, res["labels"], numeric, categorical)
    prof["Share"] = prof["Share"] * 100
    cfg = {"Share": st.column_config.NumberColumn(format="%.1f%%")}
    cfg.update({c: st.column_config.NumberColumn(format="%.4g")
                for c in numeric})
    st.dataframe(prof, hide_index=True, width="stretch", column_config=cfg)
    st.caption("Averages over every row in the cluster; for a category, its "
               "most common value and that value's share.")
    sep = ml_eval.cluster_separation(df, res["labels"], numeric, categorical)
    if len(sep):
        st.markdown("**How strongly each feature separates the clusters**")
        show = sep.iloc[::-1]
        fig = go.Figure(go.Bar(x=show["strength"], y=show["feature"],
                               orientation="h", marker_color=PALETTE[0],
                               marker_line_width=0,
                               hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
        _layout(fig, 120 + 22 * len(show), bargap=0.35,
                xaxis_title="0 = no difference, 1 = fully separated",
                xaxis_range=[0, 1])
        _chart(fig, f"ml_sep_{res['key']}")
        st.caption("For a number, the share of its variation that lies "
                   "between cluster averages; for a category, Cramér's V.")


def _k_scan(run):
    scan = run.get("scan")
    if not scan:
        st.caption("The comparison of cluster counts was not run. Turn it on "
                   "under Number of clusters and train again.")
        return
    table = scan["table"]
    st.markdown(f"**Suggested: {scan['suggested']} clusters**, by "
                f"{scan['why']}.")
    panels = [("inertia", "Within-cluster spread (lower is tighter)"),
              ("calinski_harabasz", "Calinski-Harabasz (higher is better)"),
              ("davies_bouldin", "Davies-Bouldin (lower is better)")]
    if table["silhouette"].notna().any():
        panels.insert(1, ("silhouette", "Silhouette (higher is better)"))
    fig = make_subplots(rows=1, cols=len(panels),
                        subplot_titles=[t for _, t in panels],
                        horizontal_spacing=0.06)
    for i, (col, _) in enumerate(panels, 1):
        fig.add_trace(go.Scatter(
            x=table["k"], y=table[col], mode="lines+markers",
            line=dict(color=PALETTE[0], width=2), marker=dict(size=8),
            showlegend=False,
            hovertemplate="k = %{x}<br>%{y:,.4g}<extra></extra>"),
            row=1, col=i)
        fig.add_vline(x=scan["suggested"], line_color=MUTED, line_width=1,
                      row=1, col=i)
        fig.update_xaxes(dtick=1, title_text="Clusters", row=1, col=i)
    fig.update_annotations(font_size=12)
    _layout(fig, 300)
    _chart(fig, "ml_kscan")
    st.dataframe(table.rename(columns={"k": "Clusters",
                                       "inertia": "Within-cluster spread",
                                       "silhouette": "Silhouette",
                                       "calinski_harabasz": "Calinski-Harabasz",
                                       "davies_bouldin": "Davies-Bouldin"}),
                 hide_index=True, width="stretch")


# ── predictions ─────────────────────────────────────────────────────────
def _id_columns(df):
    prof = st.session_state.get("ml_profile")
    if prof is None:
        return []
    return [c for c in prof.loc[prof["kind"] == "id", "column"]
            if c in df.columns]


@st.fragment
@guard.guarded("Predictions")
def _predictions(run, res, df):
    cols = ml_eval.prediction_columns(run, res, df.columns)
    cols.index = df.index
    keep = _id_columns(df)
    if run["task"] != "clustering" and run["target"] not in keep:
        keep.append(run["target"])
    table = pd.concat([df[keep], cols], axis=1)
    score_col = next((c for c in cols.columns if c.startswith(
        ("ML_Probability", "ML_Confidence"))), None)
    if run["task"] == "regression":
        score_col = cols.columns[0]
    options = ["File order"] + ([f"Highest {score_col} first"]
                                if score_col else [])
    c1, c2, c3 = st.columns([2, 1, 1])
    order = c1.selectbox("Order", options, key=f"ml_r_order_{res['key']}")
    size = c2.selectbox("Rows per page", PAGE_SIZES,
                        key=f"ml_r_size_{res['key']}")
    pages = export_page.page_count(len(table), size)
    page_key = f"ml_r_page_{res['key']}"
    if st.session_state.get(page_key, 1) > pages:
        st.session_state[page_key] = pages   # a larger page size: fewer pages
    page = c3.number_input(f"Page (of {pages:,})", min_value=1,
                           max_value=pages, step=1, key=page_key)
    if order != "File order":
        table = table.iloc[ml_eval.ranked(table[score_col].to_numpy(
            dtype="float64"))]
    st.dataframe(export_page.page_slice(table, int(page), size),
                 width="stretch", hide_index=True)
    st.caption(f"{len(table):,} rows: every row in the file. The "
               f"predictions file below holds them all, with every original "
               f"column.")


# ── SHAP ────────────────────────────────────────────────────────────────
def _shap_rows(run, res, cls):
    """Test rows ranked by the score SHAP explains, highest first."""
    test = ml_eval.rows(run, "test")
    if run["task"] == "regression":
        score = res["pred"][test]
    else:
        score = res["proba"][test, run["classes"].index(cls)]
    return test[ml_eval.ranked(score)]


def _shap_summary_charts(exp, res, df):
    summary = ml_eval.shap_summary(exp).head(15)
    st.markdown(f"**Average effect of each feature**, in {exp['units']}")
    show = summary.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=show["mean_abs"], y=show["feature"], orientation="h",
        marker_color=PALETTE[0], marker_line_width=0,
        hovertemplate="%{y}<br>mean |SHAP| %{x:.4f}<extra></extra>"))
    _layout(fig, 120 + 22 * len(show), bargap=0.35,
            xaxis_title=f"Mean absolute SHAP value ({exp['units']})")
    _chart(fig, f"ml_shapbar_{res['key']}")
    top = list(summary["feature"].head(12))
    vals = pd.DataFrame(exp["values"], columns=exp["features"])
    drawn = _drawn(len(vals))
    feats = df.iloc[exp["rows"][drawn]]
    xs, ys, colour, cat_x, cat_y = [], [], [], [], []
    jitter = np.random.default_rng(1).uniform(-0.3, 0.3, len(drawn))
    for i, f in enumerate(top[::-1]):
        v = vals[f].to_numpy()[drawn]
        if f in res["recipe"]["numeric"]:
            raw = pd.to_numeric(feats[f], errors="coerce").to_numpy()
            xs.extend(v)
            ys.extend(i + jitter)
            colour.extend(pd.Series(raw).rank(pct=True).to_numpy())
        else:
            cat_x.extend(v)
            cat_y.extend(i + jitter)
    fig = go.Figure()
    if xs:
        fig.add_trace(go.Scattergl(
            x=xs, y=ys, mode="markers", name="Number",
            marker=dict(size=4, opacity=0.6, color=colour,
                        colorscale="RdBu_r", cmin=0, cmax=1,
                        colorbar=dict(title="Feature value",
                                      tickvals=[0, 1],
                                      ticktext=["Low", "High"])),
            hovertemplate="SHAP %{x:.4f}<extra></extra>"))
    if cat_x:
        fig.add_trace(go.Scattergl(
            x=cat_x, y=cat_y, mode="markers", name="Category",
            marker=dict(size=4, opacity=0.6, color=MUTED),
            hovertemplate="SHAP %{x:.4f}<extra></extra>"))
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    fig.update_yaxes(tickvals=list(range(len(top))), ticktext=top[::-1])
    _layout(fig, 120 + 26 * len(top), showlegend=False,
            xaxis_title=f"SHAP value ({exp['units']})")
    _chart(fig, f"ml_shapdots_{res['key']}")
    note = ("Each dot is one test row: right of the line pushes the "
            "prediction up, left pushes it down. Numbers are coloured from "
            "low (blue) to high (red); categories are grey.")
    if len(vals) > DRAW_POINTS:
        note += (f" Drawn: {DRAW_POINTS:,} of {len(vals):,} rows, chosen at "
                 f"random; the averages above cover every one.")
    st.caption(note)


def _waterfall(run, res, exp, df):
    row = int(exp["rows"][0])
    steps = ml_eval.shap_waterfall(exp, 0)
    base = float(exp["base"][0])
    total = base + float(exp["values"][0].sum())
    labels = ["Average row"] + list(steps["feature"]) + ["This row"]
    # In running order — the base, each step, the total — and drawn top
    # down; a waterfall adds up in the order its data comes.
    fig = go.Figure(go.Waterfall(
        orientation="h", y=labels,
        x=[base] + list(steps["value"]) + [total],
        measure=["absolute"] + ["relative"] * len(steps) + ["total"],
        increasing=dict(marker_color=PALETTE[7]),
        decreasing=dict(marker_color=PALETTE[6]),
        totals=dict(marker_color=MUTED),
        hovertemplate="%{y}: %{x:+.4f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", type="category")
    _layout(fig, 140 + 26 * len(labels),
            xaxis_title=f"{exp['units']}")
    _chart(fig, f"ml_shapwf_{res['key']}")
    said = f"{total:,.4g} {exp['units']}"
    if exp["units"] == "log-odds" and len(run["classes"]) == 2:
        said += f", a probability of {1 / (1 + np.exp(-total)):.2f}"
    ids = _id_columns(df)
    who = (", ".join(f"{c} {df[c].iloc[row]}" for c in ids[:2])
           if ids else f"row {row + 1:,} of the file")
    st.caption(f"{who}: the model's output is {said}. Red bars push it up, "
               f"blue bars down, from the average row's {base:,.4g}.")


@st.fragment
@guard.guarded("SHAP")
def _shap(run, res, df):
    why = ml_eval.shap_problem(run, res)
    if why:
        st.caption(why)
        return
    cls = None
    if run["task"] == "classification":
        if len(run["classes"]) > 2:
            key = f"ml_r_shap_cls_{res['key']}"
            if key not in st.session_state or \
                    st.session_state[key] not in run["classes"]:
                st.session_state[key] = run["classes"][0]
            cls = st.selectbox("Explain the score for class", run["classes"],
                               key=key)
        else:
            cls = run["positive"]
    units = ml_eval.shap_units(run, res)
    cache = run.setdefault("shap", {})
    exp = cache.get((res["key"], cls if len(run["classes"]) > 2 else None))
    st.markdown("**SHAP: how much each feature moves each prediction**")
    if exp is None:
        costs = run.setdefault("shap_cost", {})
        if res["key"] not in costs:
            costs[res["key"]] = ml_eval.shap_seconds(run, res, df)
        st.caption(f"Exact SHAP values, in {units}, for every one of the "
                   f"{run['n_test']:,} test rows: about "
                   f"{_seconds(costs[res['key']])} for this model.")
        if st.button("Explain every test row",
                     key=f"ml_r_shap_go_{res['key']}"):
            bar = st.progress(0.0, text="SHAP")
            try:
                with log.timer() as t:
                    exp = ml_eval.explain(
                        run, res, df, ml_eval.rows(run, "test"), cls,
                        progress=lambda f, text: bar.progress(f, text=text))
            except ValueError:
                log.event(_log, "ml_shap_failed", level=logging.WARNING,
                          exc_info=True, model=res["key"])
                st.warning("The SHAP values did not add up to the model's "
                           "own output, so they are not shown.")
                return
            log.event(_log, "ml_shap", model=res["key"], task=run["task"],
                      rows=len(exp["rows"]), ms=t.ms)
            cache[(res["key"], cls if len(run["classes"]) > 2 else None)] = \
                exp
            bar.empty()
    if exp is not None:
        _shap_summary_charts(exp, res, df)
    st.markdown("**One row**")
    ranked = _shap_rows(run, res, cls or (run.get("positive")))
    choice = st.radio("Which row", ["Highest score in the test rows",
                                    "Lowest score in the test rows",
                                    "A row of the file"],
                      horizontal=True, key=f"ml_r_shap_who_{res['key']}")
    if choice.startswith("Highest"):
        row = int(ranked[0])
    elif choice.startswith("Lowest"):
        row = int(ranked[-1])
    else:
        row = int(st.number_input("Row number, as in the file", min_value=1,
                                  max_value=run["n_rows"], step=1,
                                  key=f"ml_r_shap_row_{res['key']}")) - 1
    one = ml_eval.explain(run, res, df, [row], cls)
    _waterfall(run, res, one, df)


# ── the Python code ─────────────────────────────────────────────────────
def _code(run, res):
    if run.get("external"):
        code = ml_eval.scoring_script(run, res)
        st.caption("Standalone Python that scores a file with the uploaded "
                   "model: it builds the model's inputs from the file's "
                   "columns the way this page did, and writes every row's "
                   "prediction. Run it with `python score_file.py "
                   "your_model_file your_file.csv`.")

        @log.logged_download(_log, "ml_scoring_script", task=run["task"])
        def _scoring():
            return code.encode("utf-8")

        st.download_button("Scoring code (Python)", data=_scoring,
                           file_name="score_file.py", mime="text/x-python",
                           key=f"ml_code_score_{res['key']}")
        st.code(code, language="python", height=CODE_HEIGHT)
        return
    one = ml_eval.script(run, res)
    every = ml_eval.run_script(run)
    st.caption("Standalone Python: it splits the rows the way the app does, "
               "trains with the same settings, prints each model's test "
               "result, which matches the leaderboard, and writes every "
               "row's prediction. Run it with "
               "`python train_model.py your_file.csv`.")
    fields = {"task": run["task"], "model": res["key"]}

    @log.logged_download(_log, "ml_script", **fields)
    def _one():
        return one.encode("utf-8")

    @log.logged_download(_log, "ml_run_script", **fields)
    def _every():
        return every.encode("utf-8")

    c1, c2 = st.columns(2)
    c1.download_button(f"{res['label']} (Python)", data=_one,
                       file_name="train_model.py", mime="text/x-python",
                       width="stretch", key=f"ml_code_one_{res['key']}")
    c2.download_button("Every model of the run (Python)", data=_every,
                       file_name="train_all_models.py",
                       mime="text/x-python", width="stretch",
                       key=f"ml_code_all_{res['key']}")
    with st.expander(f"{res['label']}: train_model.py", expanded=True):
        st.code(one, language="python", height=CODE_HEIGHT)
    with st.expander("Every model of the run: train_all_models.py"):
        st.code(every, language="python", height=CODE_HEIGHT)


# ── downloads ───────────────────────────────────────────────────────────
def _downloads(run, res, df):
    st.markdown("#### Downloads")
    key, task = res["key"], run["task"]
    importance = (run.get("importance") or {}).get(key)
    fields = {"task": task, "model": key}

    @log.logged_download(_log, "ml_predictions", rows=len(df), **fields)
    def _predictions_csv():
        return ml_eval.with_predictions(run, res, df).to_csv(index=False) \
            .encode("utf-8")

    @log.logged_download(_log, "ml_leaderboard", **fields)
    def _leaderboard_csv():
        return ml_eval.leaderboard(run).drop(columns="key").to_csv(
            index=False).encode("utf-8")

    @log.logged_download(_log, "ml_model_card", **fields)
    def _card():
        return ml_eval.model_card(run, res).encode("utf-8")

    @log.logged_download(_log, "ml_script", **fields)
    def _script():
        return ml_eval.script(run, res).encode("utf-8")

    @log.logged_download(_log, "ml_model", **fields)
    def _model():
        return ml_eval.model_bytes(run, res)

    @log.logged_download(_log, "ml_run_script", **fields)
    def _run_script():
        return ml_eval.run_script(run).encode("utf-8")

    @log.logged_download(_log, "ml_pptx", **fields)
    def _pptx():
        return ml_summary.build_pptx(run, res, df)

    @log.logged_download(_log, "ml_bundle", **fields)
    def _bundle():
        return ml_eval.zip_bundle(run, res, df, importance)

    if run.get("external"):
        @log.logged_download(_log, "ml_scoring_script", **fields)
        def _scoring():
            return ml_eval.scoring_script(run, res).encode("utf-8")

        c1, c2, c3 = st.columns(3)
        c1.download_button("Predictions (CSV)", data=_predictions_csv,
                           file_name="ml_predictions.csv", mime="text/csv",
                           width="stretch", key=f"ml_dl_pred_{key}")
        c2.download_button("Leaderboard (CSV)", data=_leaderboard_csv,
                           file_name="ml_leaderboard.csv", mime="text/csv",
                           width="stretch", key=f"ml_dl_lb_{key}")
        c3.download_button("Summary (PowerPoint)", data=_pptx,
                           file_name="ml_summary.pptx",
                           mime="application/vnd.openxmlformats-"
                                "officedocument.presentationml.presentation",
                           width="stretch", key=f"ml_dl_pptx_{key}")
        c1.download_button("Model card (Markdown)", data=_card,
                           file_name="model_card.md", mime="text/markdown",
                           width="stretch", key=f"ml_dl_card_{key}")
        c2.download_button("Scoring code (Python)", data=_scoring,
                           file_name="score_file.py", mime="text/x-python",
                           width="stretch", key=f"ml_dl_score_{key}")
        c3.download_button("Everything (zip)", data=_bundle,
                           file_name="ml_results.zip", mime="application/zip",
                           width="stretch", key=f"ml_dl_zip_{key}")
        st.caption(f"For {res['label']}. The predictions file is the whole "
                   f"dataset with the model's columns added; the scoring "
                   f"code scores another file with the uploaded model.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("Predictions (CSV)", data=_predictions_csv,
                       file_name="ml_predictions.csv", mime="text/csv",
                       width="stretch", key=f"ml_dl_pred_{key}")
    c2.download_button("Leaderboard (CSV)", data=_leaderboard_csv,
                       file_name="ml_leaderboard.csv", mime="text/csv",
                       width="stretch", key=f"ml_dl_lb_{key}")
    c3.download_button("Summary (PowerPoint)", data=_pptx,
                       file_name="ml_summary.pptx",
                       mime="application/vnd.openxmlformats-officedocument."
                            "presentationml.presentation",
                       width="stretch", key=f"ml_dl_pptx_{key}")
    c4.download_button("Model card (Markdown)", data=_card,
                       file_name="model_card.md", mime="text/markdown",
                       width="stretch", key=f"ml_dl_card_{key}")
    c1.download_button("This model's code (Python)", data=_script,
                       file_name="train_model.py", mime="text/x-python",
                       width="stretch", key=f"ml_dl_script_{key}")
    c2.download_button("Every model's code (Python)", data=_run_script,
                       file_name="train_all_models.py",
                       mime="text/x-python", width="stretch",
                       key=f"ml_dl_runscript_{key}")
    c3.download_button("Model file (joblib)", data=_model,
                       file_name="model.joblib",
                       mime="application/octet-stream", width="stretch",
                       key=f"ml_dl_model_{key}")
    c4.download_button("Everything (zip)", data=_bundle,
                       file_name="ml_results.zip", mime="application/zip",
                       width="stretch", key=f"ml_dl_zip_{key}")
    st.caption(f"For {res['label']}. The predictions file is the whole "
               f"dataset with the model's columns added; the PowerPoint "
               f"summarises the run for the people it goes to; each script "
               f"retrains on a file with the same columns and reproduces "
               f"the test results; the model file needs scikit-learn "
               f"{run['sklearn']}"
               + (f" and {res['label']}" if key in ("xgboost", "lightgbm")
                  else "") + " to load.")
    _to_snowflake(lambda: ml_eval.with_predictions(run, res, df),
                  f"ml_r_sf_{key}")


def _to_snowflake(frame, key):
    """Write-back behind a toggle: the frame is only built when asked for."""
    if not st.toggle("Write to Snowflake", key=key):
        return
    from logic import snowflake_io
    if not snowflake_io.available():
        st.info(snowflake_io.unavailable_reason())
        return
    guard.section("Snowflake", snowflake_panel.writer, frame(), key)


@guard.guarded("Score a new file")
def _score_new(run, res):
    if not ml_eval.can_score(run, res):
        st.caption(f"{res['label']} cannot place new rows into its clusters; "
                   f"K-means and Gaussian mixture can.")
        return
    up = st.file_uploader("A file with the same feature columns",
                          type=["csv", "xlsx", "xls"],
                          key=f"ml_r_score_{res['key']}")
    if up is None:
        return
    new = load_file(up)
    if new is None:
        return
    with log.timer() as t:
        scored, notes = ml_eval.score_file(run, res, new)
    if scored is None:
        for n in notes:
            st.warning(n)
        return
    log.event(_log, "ml_scored_file", rows=len(new), task=run["task"],
              model=res["key"], notes=len(notes), ms=t.ms)
    st.success(f"Scored {len(scored):,} rows with {res['label']}.")
    for n in notes:
        st.caption(n)
    st.dataframe(scored.head(20), width="stretch", hide_index=True)

    @log.logged_download(_log, "ml_scored", rows=len(scored),
                         task=run["task"], model=res["key"])
    def _scored_csv():
        return scored.to_csv(index=False).encode("utf-8")

    st.download_button("Download the scored file (CSV)", data=_scored_csv,
                       file_name="ml_scored.csv", mime="text/csv",
                       key=f"ml_dl_scored_{res['key']}")
    _to_snowflake(lambda: scored, f"ml_r_sf_scored_{res['key']}")


# ── the page ────────────────────────────────────────────────────────────
def _header(run):
    st.markdown("## Model results")
    st.caption(_summary(run))


def render():
    ss = st.session_state
    run = ss["ml_run"]
    # An uploaded model keeps the file it was evaluated on: with its own
    # preparation replayed, that is not the file prepared on this visit.
    df = run["frame"] if run.get("frame") is not None else ss["ml_df"]
    models = ml.usable(run)
    best = ml.result(run, run["best"]) if run.get("best") else None
    _header(run)
    if best is None:
        st.warning("No model could be trained. The leaderboard below gives "
                   "each one's reference; changing the settings and training "
                   "again may help.")
        _leaderboard(run)
        return
    _verdict(run)
    st.markdown("#### Leaderboard")
    _leaderboard(run)
    st.divider()
    keys = [r["key"] for r in models]
    pick_key = "ml_r_model"
    if pick_key not in ss or ss[pick_key] not in keys:
        ss[pick_key] = run["best"]
    pick = st.selectbox("Model to inspect", keys, key=pick_key,
                        format_func=lambda k: ml.result(run, k)["label"]
                        + ("  (best)" if k == run["best"] else ""))
    res = ml.result(run, pick)
    task = run["task"]
    if task == "clustering":
        tabs = st.tabs(["Clusters", "Profiles", "Number of clusters",
                        "Predictions", "Python code"])
        with tabs[0]:
            _clusters(run, res, df)
        with tabs[1]:
            _profiles(run, res, df)
        with tabs[2]:
            _k_scan(run)
        with tabs[3]:
            _predictions(run, res, df)
        with tabs[4]:
            _code(run, res)
    else:
        binary = task == "classification" and len(run["classes"]) == 2
        names = ["Performance"] + (["Targeting"] if binary else []) + [
            "What drives it", "SHAP", "Predictions", "Python code"] + (
            ["Tuning"] if res.get("tuned") else [])
        tabs = dict(zip(names, st.tabs(names)))
        with tabs["Performance"]:
            if task == "regression":
                _regression_performance(run, res)
            elif binary:
                _binary_performance(run, res)
            else:
                _multiclass_performance(run, res)
        if binary:
            with tabs["Targeting"]:
                _targeting(run, res, df)
        with tabs["What drives it"]:
            _drivers(run, res, df)
        with tabs["SHAP"]:
            _shap(run, res, df)
        with tabs["Predictions"]:
            _predictions(run, res, df)
        with tabs["Python code"]:
            _code(run, res)
        if res.get("tuned"):
            with tabs["Tuning"]:
                st.caption(f"{len(res['search'])} settings compared by "
                           f"cross-validated "
                           f"{ml.metric_label(run['family'], run['metric'])}"
                           f", best first. The best was then trained on "
                           f"every training row.")
                st.dataframe(res["search"].rename(columns={
                    "cv_mean": "CV mean", "cv_std": "CV spread"}),
                    hide_index=True, width="stretch")
    st.divider()
    _downloads(run, res, df)
    project_panel.register(run, res)
    with st.expander("Score a new file with this model"):
        _score_new(run, res)
