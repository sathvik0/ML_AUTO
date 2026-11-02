"""A PowerPoint summary of a Machine Learning run, for the people the
results go to.

Built from the slide helpers in `summary_export`: one matplotlib
chart per slide, drawn headless, and a chart that fails is logged and its
slide falls back to the numbers as a table — a deck with a table where a chart
should be beats no deck. Every number comes from `logic.ml_eval`, the same
functions the results page draws from.
"""
import io
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402

from components import summary_export as se  # noqa: E402
from logic import ml, ml_eval  # noqa: E402
from utils.constants import ML_PALETTE  # noqa: E402

_FIG = (9, 5)
_MUTED, _DIM = "#898781", "#c3c2b7"
_DRAW_POINTS = 2_000


# ── charts ──────────────────────────────────────────────────────────────
def _models_chart(run):
    metric = run["metric"]
    ok = [r for r in run["results"] if not r.get("error")
          and r["test"].get(metric) is not None]
    if len(ok) < 2:
        return None
    up = ml.higher_is_better(run["family"], metric)
    ok = sorted(ok, key=lambda r: r["test"][metric], reverse=up)[::-1]
    fig, ax = plt.subplots(figsize=_FIG)
    colours = [ML_PALETTE[0] if r["key"] == run["best"] else
               _MUTED if r["key"] == ml.BASELINE else _DIM for r in ok]
    ax.barh([r["label"] for r in ok], [r["test"][metric] for r in ok],
            color=colours)
    for i, r in enumerate(ok):
        ax.text(r["test"][metric], i, f" {r['test'][metric]:.3f}",
                va="center", fontsize=9)
    ax.set_xlabel(f"{ml.metric_label(run['family'], metric)}, test rows")
    return se._fig_png(fig)


def _confusion_chart(table):
    z = table.to_numpy()
    fig, ax = plt.subplots(figsize=_FIG)
    ax.imshow(z, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    top = z.max() or 1
    if len(table) <= 10:
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                ax.text(j, i, f"{z[i, j]:,}", ha="center", va="center",
                        fontsize=9,
                        color="white" if z[i, j] / top > 0.5 else "#0b0b0b")
    return se._fig_png(fig)


def _roc_chart(hit, score, label):
    fpr, tpr, auc = ml_eval.roc(hit, score)
    fig, ax = plt.subplots(figsize=_FIG)
    ax.plot([0, 1], [0, 1], color=_MUTED, lw=1, label="Guessing")
    ax.plot(fpr, tpr, color=ML_PALETTE[0], lw=2,
            label=f"{label} (AUC {auc:.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=9, loc="lower right")
    return se._fig_png(fig)


def _gains_chart(g, label):
    fig, ax = plt.subplots(figsize=_FIG)
    ax.plot([0, 1], [0, 1], color=_MUTED, lw=1, label="Targeting at random")
    ax.plot([0] + list(g["targeted"]), [0] + list(g["captured"]),
            color=ML_PALETTE[0], lw=2, marker="o", label=label)
    ax.set_xlabel("Share of rows targeted, best scores first")
    ax.set_ylabel("Share of positives reached")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(fontsize=9, loc="lower right")
    return se._fig_png(fig)


def _pva_chart(actual, pred, target, seed):
    pos = (np.arange(len(actual)) if len(actual) <= _DRAW_POINTS else
           np.sort(np.random.default_rng(seed).choice(
               len(actual), _DRAW_POINTS, replace=False)))
    lo = float(min(actual.min(), pred.min()))
    hi = float(max(actual.max(), pred.max()))
    fig, ax = plt.subplots(figsize=_FIG)
    ax.plot([lo, hi], [lo, hi], color=_MUTED, lw=1)
    ax.scatter(actual[pos], pred[pos], s=8, alpha=0.5, color=ML_PALETTE[0])
    ax.set_xlabel(f"Actual {target}")
    ax.set_ylabel(f"Predicted {target}")
    return se._fig_png(fig)


def _deciles_chart(dec, target):
    x = np.arange(len(dec))
    fig, ax = plt.subplots(figsize=_FIG)
    ax.bar(x - 0.2, dec["predicted"], 0.4, color=ML_PALETTE[0],
           label="Predicted")
    ax.bar(x + 0.2, dec["actual"], 0.4, color=ML_PALETTE[1], label="Actual")
    ax.set_xticks(x)
    ax.set_xticklabels(dec["group"])
    ax.set_xlabel("Tenth of rows, 1 = highest predictions")
    ax.set_ylabel(f"Average {target}")
    ax.legend(fontsize=9)
    return se._fig_png(fig)


def _bars_chart(labels, values, xlabel, colours=None):
    fig, ax = plt.subplots(figsize=_FIG)
    ax.barh(list(labels)[::-1], list(values)[::-1],
            color=(colours[::-1] if colours else ML_PALETTE[0]))
    ax.set_xlabel(xlabel)
    return se._fig_png(fig)


def _cluster_sizes_chart(labels):
    counts = pd.Series(labels).value_counts()
    order = sorted(counts.index, key=ml_eval._cluster_key)
    colours = [(_MUTED if c == "Noise" else
                ML_PALETTE[int(str(c)[1:]) - 1]
                if int(str(c)[1:]) <= len(ML_PALETTE) else _MUTED)
               for c in order]
    fig, ax = plt.subplots(figsize=_FIG)
    ax.bar(order, [counts[c] for c in order], color=colours)
    for i, c in enumerate(order):
        ax.text(i, counts[c], f"{counts[c]:,}", ha="center", va="bottom",
                fontsize=9)
    ax.set_ylabel("Rows")
    return se._fig_png(fig)


def _heat_chart(heat):
    span = float(np.nanmax(np.abs(heat.to_numpy()))) if heat.size else 1.0
    span = span if np.isfinite(span) and span > 0 else 1.0
    fig, ax = plt.subplots(figsize=(9, 1.2 + 0.5 * len(heat)))
    im = ax.imshow(heat.to_numpy(), cmap="RdBu_r", vmin=-span, vmax=span,
                   aspect="auto")
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index)
    fig.colorbar(im, ax=ax, fraction=0.046, label="SD from the average")
    return se._fig_png(fig)


def _scan_chart(scan):
    table = scan["table"]
    cols = [("inertia", "Within-cluster spread")]
    if table["silhouette"].notna().any():
        cols.append(("silhouette", "Silhouette"))
    cols.append(("calinski_harabasz", "Calinski-Harabasz"))
    fig, axes = plt.subplots(1, len(cols), figsize=(11, 4))
    for ax, (col, title) in zip(np.atleast_1d(axes), cols):
        ax.plot(table["k"], table[col], color=ML_PALETTE[0], marker="o")
        ax.axvline(scan["suggested"], color=_MUTED, lw=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Clusters")
    return se._fig_png(fig)


# ── slides ──────────────────────────────────────────────────────────────
def _chart_or_table(prs, title, png, table, note=None):
    """A chart slide, or the same numbers as a table when the chart could
    not be drawn."""
    s = se._slide(prs)
    se._title(s, title)
    if png is not None:
        s.shapes.add_picture(png, Inches(2.0), Inches(1.4),
                             height=Inches(5.3))
    elif table is not None and len(table):
        se._table(s, table.head(14), 0.6, 1.5, 12.1, 4.5, size=11)
    if note:
        tb = s.shapes.add_textbox(Inches(0.6), Inches(6.8), Inches(12.1),
                                  Inches(0.5))
        p = tb.text_frame.paragraphs[0]
        p.text = note
        p.font.size = Pt(11)
        p.font.color.rgb = se._GREY
    return s


def _fmt_table(df):
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:,.3f}")
        else:
            out[c] = out[c].astype(str)
    return out


def _verdict_lines(run):
    v = ml_eval.verdict(run)
    best = ml.result(run, run["best"]) if run.get("best") else None
    lines = []
    if v is not None and best is not None:
        ok, key, value, chance = v
        name = ml.metric_label(run["family"], key)
        shown = f"{chance:.2f}" if key.startswith("roc_auc") else f"{chance:g}"
        rows_ = ("on the rows with a target" if run.get("external")
                 else "on the test rows")
        lines.append(f"{best['label']} {'beats' if ok else 'does not beat'} "
                     f"guessing: {name} {value:.3f} {rows_}, where "
                     f"{shown} is guessing.")
    elif best is not None:
        lines.append(f"Best model: {best['label']}, by "
                     f"{ml.metric_label(run['family'], run['metric'])}.")
    if run["task"] == "clustering":
        lines.append(f"{run['n_rows']:,} rows grouped on "
                     f"{len(run['base']['numeric']) + len(run['base']['categorical'])}"
                     f" features.")
    elif run.get("external"):
        lines.append(f"An uploaded model, not trained here, scored on the "
                     f"{run['n_test']:,} rows with a target; "
                     f"{run['n_rows'] - run['n_test']:,} more predicted "
                     f"without one.")
        lines.append("If it was trained on some of these rows, its scores "
                     "here are higher than it would earn on new ones.")
    else:
        lines.append(f"{run['n_labelled']:,} rows with a target: "
                     f"{run['n_train']:,} to train, {run['n_test']:,} to "
                     f"test; {run['n_unlabelled']:,} without one, scored by "
                     f"the model.")
        lines.append("Cross-validation: " + (f"{run['folds']} folds, which "
                                             f"chose the best model"
                                             if run["folds"] else "none"))
        base = next((r for r in run["results"]
                     if r["key"] == ml.BASELINE and not r.get("error")), None)
        if base is not None:
            m = run["metric"]
            lines.append(f"A baseline that ignores every feature scores "
                         f"{ml_eval.fmt_metric(m, base['test'].get(m))} on "
                         f"{ml.metric_label(run['family'], m)}.")
    return lines


def build_pptx(run, res, df) -> bytes:
    """The deck for one run, its selected model in the detail slides."""
    prs = Presentation()
    prs.slide_width, prs.slide_height = se._W, se._H
    task = run["task"]

    s = se._slide(prs)
    se._title(s, "Machine Learning summary")
    sub = s.shapes.add_textbox(Inches(0.5), Inches(1.1), Inches(12),
                               Inches(0.4))
    sp = sub.text_frame.paragraphs[0]
    sp.text = (f"{ml.TASK_LABELS[task]} of {ml_eval.describe_target(run)} · "
               f"{run['n_rows']:,} rows · generated {date.today():%d %b %Y}")
    sp.font.size = Pt(14)
    sp.font.color.rgb = se._GREY
    se._bullets(s, _verdict_lines(run), 0.6, 1.8, 12.1, 5, size=16)

    board = ml_eval.leaderboard(run).drop(columns="key")
    keep = ["Model", "Status"] + [c for c in board.columns
                                  if c not in ("Model", "Status", "Settings",
                                               "Fit (s)")][:5]
    s = se._slide(prs)
    se._title(s, "Leaderboard")
    se._table(s, _fmt_table(board[keep]), 0.5, 1.5, 12.3,
              0.4 * (len(board) + 1), size=11)
    _chart_or_table(prs, "Models compared", se._safe(_models_chart, run),
                    None)

    label = res["label"]
    if task == "classification":
        pos = ml_eval.rows(run, "test")
        y = ml_eval.truth(run, pos)
        table = ml_eval.confusion(y, res["pred"][pos], run["classes"])
        _chart_or_table(prs, f"{label}: what it gets right",
                        se._safe(_confusion_chart, table),
                        table.reset_index().rename(columns={
                            "index": "Actual"}),
                        "Rows are what happened, columns what the model "
                        "said.")
        if len(run["classes"]) == 2:
            hit = y == run["positive"]
            score = ml_eval.positive_score(run, res, pos)
            if 0 < hit.sum() < len(hit):
                _chart_or_table(prs, f"{label}: ROC curve",
                                se._safe(_roc_chart, hit, score, label), None)
            g = ml_eval.gains(hit, score)
            top = g.iloc[0]
            _chart_or_table(
                prs, "Who to call first", se._safe(_gains_chart, g, label),
                _fmt_table(g),
                f"Targeting the top tenth by score reaches "
                f"{top['captured']:.0%} of the positives, {top['lift']:.1f} "
                f"times the rate of targeting at random.")
        else:
            s = se._slide(prs)
            se._title(s, f"{label}: by class")
            se._table(s, _fmt_table(ml_eval.per_class(
                y, res["pred"][pos], run["classes"])), 0.5, 1.5, 12.3,
                0.4 * (len(run["classes"]) + 1), size=11)
    elif task == "regression":
        pos = ml_eval.rows(run, "test")
        actual = ml_eval.truth(run, pos).astype("float64")
        pred = res["pred"][pos]
        _chart_or_table(prs, f"{label}: predicted against actual",
                        se._safe(_pva_chart, actual, pred, run["target"],
                                 run["base"]["seed"]), None,
                        f"Test rows; at most {_DRAW_POINTS:,} drawn, chosen "
                        f"at random. The metrics cover every one.")
        dec = ml_eval.regression_deciles(actual, pred)
        _chart_or_table(prs, "Average by tenth of the prediction",
                        se._safe(_deciles_chart, dec, run["target"]),
                        _fmt_table(dec))
    else:
        numeric = run["base"]["numeric"]
        categorical = run["base"]["categorical"]
        df = ml_eval.analysis_frame(run, df)
        _chart_or_table(prs, f"{label}: rows per cluster",
                        se._safe(_cluster_sizes_chart, res["labels"]), None)
        if numeric:
            heat = ml_eval.cluster_heat(df, res["labels"], numeric)
            _chart_or_table(prs, "What sets each cluster apart",
                            se._safe(_heat_chart, heat),
                            _fmt_table(heat.reset_index().rename(
                                columns={"index": "Cluster"})),
                            "Each cluster's average in standard deviations "
                            "from the whole file's: red above, blue below.")
        prof = ml_eval.cluster_profiles(df, res["labels"], numeric,
                                        categorical)
        s = se._slide(prs)
        se._title(s, "Cluster profiles")
        se._table(s, _fmt_table(prof.iloc[:, :8]), 0.5, 1.5, 12.3,
                  0.4 * (len(prof) + 1), size=10)
        if run.get("scan"):
            _chart_or_table(prs, f"Number of clusters: {run['scan']['suggested']}"
                                 f" suggested", se._safe(_scan_chart,
                                                         run["scan"]),
                            _fmt_table(run["scan"]["table"]),
                            f"Suggested by {run['scan']['why']}.")

    imp = (run.get("importance") or {}).get(res["key"])
    if imp is not None:
        name = ml.metric_label(run["family"], run["metric"])
        top = imp.head(12)
        _chart_or_table(prs, "What drives the model",
                        se._safe(_bars_chart, top["feature"],
                                 top["importance"], f"Drop in {name} when "
                                 f"the feature is shuffled"),
                        _fmt_table(top),
                        "Measured on the test rows. Associations, not "
                        "causes.")
    exp = next((v for (k, _), v in (run.get("shap") or {}).items()
                if k == res["key"]), None)
    if exp is not None:
        summary = ml_eval.shap_summary(exp).head(12)
        _chart_or_table(prs, "SHAP: average effect of each feature",
                        se._safe(_bars_chart, summary["feature"],
                                 summary["mean_abs"],
                                 f"Mean absolute SHAP value, in "
                                 f"{exp['units']}"),
                        _fmt_table(summary),
                        f"Over the {len(exp['rows']):,} test rows.")

    s = se._slide(prs)
    se._title(s, f"{label}: how it was built")
    card = ml_eval.model_card(run, res).splitlines()
    keep = [ln[2:] for ln in card if ln.startswith("- ")][:14]
    se._bullets(s, keep, 0.6, 1.5, 12.1, 5.5, size=13)

    s = se._slide(prs)
    se._title(s, "Using the results")
    se._bullets(s, [
        "The predictions file holds every row with the model's columns; "
        "rows marked out of fold or test were scored by a model that never "
        "saw them.",
        ("The scoring script scores another file with the uploaded model, "
         "building its inputs the same way." if run.get("external") else
         "The training script retrains the model on a file with the same "
         "columns and reproduces the test result."),
        "A new file is scored on the results page; values the model never "
        "saw are counted per column.",
        "Associations, not causes: a feature that predicts the outcome does "
        "not make it happen. Retrain when the population changes."],
        0.6, 1.5, 12.1, 5, size=15)

    out = io.BytesIO()
    prs.save(out)
    return out.getvalue()
