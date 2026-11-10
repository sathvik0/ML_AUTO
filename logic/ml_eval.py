"""What the model results page shows and hands out, computed from a run.

Pure: every function takes a training run from `logic.ml` (and the file it was
trained on) and returns numbers, frames, text or bytes. The page only draws
them, which is what lets the smoke and stress suites recompute every number a
second way.

Curves and tables are computed on every test row. Only a *drawing* is ever
thinned (`thin`), and the caption beside it says so.
"""
import ast
import inspect
import io
import json
import pprint
import zipfile

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.inspection import partial_dependence as _sk_partial_dependence
from sklearn.inspection import permutation_importance as _sk_permutation
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

from logic import ml, ml_prep, ml_recipe, ml_upload

GAIN_GROUPS = 10
MAX_CURVE_POINTS = 400
IMPORTANCE_REPEATS = 5
IMPORTANCE_REPEATS_LARGE = 3     # above IMPORTANCE_LARGE test rows
IMPORTANCE_LARGE = 20_000
PD_GRID = 20


# ── which rows are which ────────────────────────────────────────────────
def rows(run, which):
    """Row positions of the test rows, the rows with a target, or all."""
    split = run.get("split")
    if which == "test":
        return np.flatnonzero(split == "test")
    if which == "labelled":
        return np.flatnonzero(split != "no target")
    return np.arange(run["n_rows"])


def truth(run, positions):
    """The target as trained, at these row positions."""
    return run["y"].to_numpy()[positions]


def positive_score(run, res, positions):
    """P(positive class) at these rows, for a two-class run."""
    j = run["classes"].index(run["positive"])
    return res["proba"][positions, j]


# ── curves ──────────────────────────────────────────────────────────────
def thin(x, y, max_points=MAX_CURVE_POINTS):
    """At most `max_points` of a curve for drawing, first and last kept."""
    x, y = np.asarray(x), np.asarray(y)
    if len(x) <= max_points:
        return x, y
    keep = np.unique(np.linspace(0, len(x) - 1, max_points).round()
                     .astype(int))
    return x[keep], y[keep]


def roc(hit, score):
    """(false positive rate, true positive rate, AUC) over every row."""
    fpr, tpr, _ = roc_curve(hit, score)
    return fpr, tpr, float(roc_auc_score(hit, score))


def pr(hit, score):
    """(recall, precision, average precision) over every row."""
    precision, recall, _ = precision_recall_curve(hit, score)
    return recall, precision, float(average_precision_score(hit, score))


def ranked(score):
    """Row order by descending score; ties keep file order, so the groups
    below are the same on every run."""
    return np.argsort(-np.asarray(score, dtype="float64"), kind="mergesort")


def gains(hit, score, groups=GAIN_GROUPS) -> pd.DataFrame:
    """Rows ranked by score and cut into equal groups, best scores first.

    `captured` is the share of all positives found by targeting down to the
    end of that group; `lift` is how many times the group's positive rate is
    the overall rate. The question a targeting team asks of a model: call the
    top 10% and how many positives do you reach?
    """
    hit = np.asarray(hit, dtype=bool)
    n = len(hit)
    order = ranked(score)
    group = np.minimum(np.arange(n) * groups // max(n, 1), groups - 1) + 1
    frame = pd.DataFrame({"group": group, "hit": hit[order]})
    out = frame.groupby("group")["hit"].agg(records="size", positives="sum")
    out = out.reindex(range(1, groups + 1), fill_value=0)
    total = max(int(hit.sum()), 1)
    base = hit.mean() if n else 0
    out["rate"] = out["positives"] / out["records"].replace(0, np.nan)
    out["captured"] = out["positives"].cumsum() / total
    out["targeted"] = out["records"].cumsum() / max(n, 1)
    out["lift"] = out["rate"] / base if base else np.nan
    return out.reset_index()


def calibration(hit, score, bins=10) -> pd.DataFrame:
    """Predicted probability against what happened, in equal-count bins."""
    hit = np.asarray(hit, dtype=float)
    score = np.asarray(score, dtype="float64")
    order = ranked(-score)                     # ascending score
    b = np.minimum(np.arange(len(score)) * bins // max(len(score), 1),
                   bins - 1)
    frame = pd.DataFrame({"bin": b, "predicted": score[order],
                          "observed": hit[order]})
    return (frame.groupby("bin").agg(predicted=("predicted", "mean"),
                                     observed=("observed", "mean"),
                                     records=("observed", "size"))
            .reset_index(drop=True))


THRESHOLD_GOALS = {"f1": "Best F1 (precision and recall together)",
                   "youden": "Best balance of hits and false alarms "
                             "(Youden's J)",
                   "cost": "Lowest cost of mistakes"}


def best_threshold(hit, score, goal="f1", miss_cost=1.0, alarm_cost=1.0):
    """The cut-off (0.01 to 0.99, in hundredths, as the slider moves) that
    does best on `goal` for these rows: F1, Youden's J (recall minus the
    false-alarm rate), or the lowest total cost of missed positives and
    false alarms."""
    hit = np.asarray(hit, dtype=bool)
    score = np.asarray(score, dtype="float64")
    p = int(hit.sum())
    n = len(hit) - p
    cuts = np.round(np.arange(0.01, 1.0, 0.01), 2)
    # Rows flagged at each cut-off, counted at once from the sorted scores.
    order = np.sort(score[hit]), np.sort(score[~hit])
    tp = p - np.searchsorted(order[0], cuts, side="left")
    fp = n - np.searchsorted(order[1], cuts, side="left")
    fn = p - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        if goal == "f1":
            value = np.where(tp > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        elif goal == "youden":
            value = (tp / max(p, 1)) - (fp / max(n, 1))
        elif goal == "cost":
            value = -(fn * float(miss_cost) + fp * float(alarm_cost))
        else:
            raise ValueError(f"Unknown goal: {goal}")
    return float(cuts[int(np.argmax(value))])


def at_threshold(hit, score, threshold) -> dict:
    """The confusion matrix and its rates when flagging score >= threshold."""
    hit = np.asarray(hit, dtype=bool)
    flag = np.asarray(score, dtype="float64") >= threshold
    tp = int((flag & hit).sum())
    fp = int((flag & ~hit).sum())
    fn = int((~flag & hit).sum())
    tn = int((~flag & ~hit).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    n = len(hit)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision,
            "recall": recall, "f1": f1,
            "accuracy": (tp + tn) / n if n else None,
            "flagged": int(flag.sum()), "share": flag.mean() if n else 0.0}


def confusion(y_true, y_pred, classes) -> pd.DataFrame:
    """Counts: rows are what happened, columns what the model said."""
    table = pd.crosstab(pd.Categorical(y_true, categories=classes),
                        pd.Categorical(y_pred, categories=classes),
                        dropna=False)
    table.index = [str(c) for c in classes]
    table.columns = [str(c) for c in classes]
    return table


def per_class(y_true, y_pred, classes) -> pd.DataFrame:
    """Precision, recall and F1 of each class, with how many rows hold it."""
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    rows_ = []
    for c in classes:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        said = int((y_pred == c).sum())
        held = int((y_true == c).sum())
        precision = tp / said if said else None
        recall = tp / held if held else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall else 0.0)
        rows_.append({"Class": str(c), "Rows": held, "Predicted": said,
                      "Precision": precision, "Recall": recall, "F1": f1})
    return pd.DataFrame(rows_)


def regression_deciles(y_true, y_pred, groups=GAIN_GROUPS) -> pd.DataFrame:
    """Rows ranked by prediction, best first: does a higher prediction mean
    a higher actual value, group by group?"""
    y_true = np.asarray(y_true, dtype="float64")
    order = ranked(y_pred)
    n = len(order)
    group = np.minimum(np.arange(n) * groups // max(n, 1), groups - 1) + 1
    frame = pd.DataFrame({"group": group,
                          "predicted": np.asarray(y_pred)[order],
                          "actual": y_true[order]})
    return (frame.groupby("group").agg(records=("actual", "size"),
                                       predicted=("predicted", "mean"),
                                       actual=("actual", "mean"))
            .reset_index())


def histogram(values, bins=40):
    """(left edges, right edges, counts), binned here so the browser is sent
    bars rather than every value."""
    v = np.asarray(values, dtype="float64")
    v = v[np.isfinite(v)]
    if not len(v):
        return np.array([]), np.array([]), np.array([], dtype=int)
    counts, edges = np.histogram(v, bins=bins)
    return edges[:-1], edges[1:], counts


# ── what drives the model ───────────────────────────────────────────────
def _features(run):
    base = run["base"]
    return base["numeric"], base["categorical"]


def features_frame(run, df):
    """These rows' features as the model reads them: the app's preparation
    for a model trained here, and for an uploaded one each file column under
    the model's own name for it."""
    base = run["base"]
    if run.get("external"):
        return ml_upload.model_frame(df, base["columns"], base["kinds"])
    return ml_recipe.prepare_features(df, base["numeric"], base["categorical"])


def analysis_frame(run, df):
    """The frame the cluster views read features from: the file for a model
    trained here, the model's own inputs for an uploaded one."""
    return features_frame(run, df) if run.get("external") else df


def source_columns(run):
    """The file columns the model reads."""
    base = run["base"]
    if run.get("external"):
        return list(dict.fromkeys(base["columns"].values()))
    return list(base["numeric"]) + list(base["categorical"])


def test_frame(run, df):
    """The test rows' features, as the model reads them."""
    pos = rows(run, "test")
    return features_frame(run, df.iloc[pos]), truth(run, pos)


def importance_repeats(run) -> int:
    return (IMPORTANCE_REPEATS if run["n_test"] <= IMPORTANCE_LARGE
            else IMPORTANCE_REPEATS_LARGE)


def importance_seconds(run, res) -> float:
    """What `permutation_importance` should take: one prediction pass per
    shuffle, plus the unshuffled one. `predict_s` timed two passes (labels
    and probabilities), the scorer makes one. Measured at 300,000 rows: 15.6s
    estimated, 17.1s taken."""
    base = run["base"]
    shuffles = (len(base["numeric"]) + len(base["categorical"])) \
        * importance_repeats(run)
    return res["predict_s"] / 2 * (shuffles + 1)


def permutation_importance(run, res, df) -> pd.DataFrame:
    """How much the test result drops when one column's values are shuffled.

    Measured on every test row, in the selection metric, on the user's own
    columns rather than on encoded ones — a category's columns are shuffled
    together because the whole original column is.
    """
    X, y = test_frame(run, df)
    task, family, metric = run["task"], run["family"], run["metric"]
    sign = 1 if ml.higher_is_better(family, metric) else -1

    def scorer(est, X_, y_):
        value = ml.one_metric(task, metric, X_, y_, est, run["classes"],
                              run["positive"])
        return sign * (value if value is not None else 0.0)

    out = _sk_permutation(res["pipeline"], X, y, scoring=scorer,
                          n_repeats=importance_repeats(run),
                          random_state=run["base"]["seed"])
    return (pd.DataFrame({"feature": list(X.columns),
                          "importance": out.importances_mean,
                          "spread": out.importances_std})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))


def cluster_separation(df, labels, numeric, categorical) -> pd.DataFrame:
    """0 to 1: how strongly each column differs between the clusters.

    Numbers: the share of the column's variation that lies between cluster
    averages. Categories: Cramér's V of the column against the clusters.
    """
    labels = pd.Series(labels, index=df.index)
    rows_ = []
    for c in numeric:
        v = pd.to_numeric(df[c], errors="coerce").astype("float64")
        v = v.where(np.isfinite(v))
        ok = v.notna()
        total = float(((v[ok] - v[ok].mean()) ** 2).sum())
        within = float(((v[ok] - v[ok].groupby(labels[ok]).transform("mean"))
                        ** 2).sum())
        rows_.append({"feature": c, "kind": "number",
                      "strength": 1 - within / total if total > 0 else 0.0})
    for c in categorical:
        table = pd.crosstab(ml_recipe.label_text(df[c]).fillna("(blank)"),
                            labels).to_numpy(dtype="float64")
        n = table.sum()
        r, k = table.shape
        if n == 0 or min(r, k) < 2:
            rows_.append({"feature": c, "kind": "category", "strength": 0.0})
            continue
        expected = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) \
            / n
        chi2 = float(((table - expected) ** 2 / np.where(expected > 0,
                                                         expected, 1)).sum())
        rows_.append({"feature": c, "kind": "category",
                      "strength": float(np.sqrt(chi2 / (n * (min(r, k) - 1))))})
    out = pd.DataFrame(rows_, columns=["feature", "kind", "strength"])
    return out.sort_values("strength", ascending=False).reset_index(drop=True)


def cluster_profiles(df, labels, numeric, categorical) -> pd.DataFrame:
    """One row per cluster: its size, the average of every numeric feature
    and the most common value of every categorical one, over all its rows."""
    labels = pd.Series(labels, index=df.index, name="Cluster")
    order = sorted(labels.unique(), key=_cluster_key)
    out = pd.DataFrame({"Rows": labels.value_counts()}).reindex(order)
    out["Share"] = out["Rows"] / len(labels)
    for c in numeric:
        v = pd.to_numeric(df[c], errors="coerce").astype("float64")
        out[c] = v.where(np.isfinite(v)).groupby(labels).mean().reindex(order)
    for c in categorical:
        text = ml_recipe.label_text(df[c]).fillna("(blank)")
        tab = pd.crosstab(labels, text)
        top = tab.idxmax(axis=1)
        share = tab.max(axis=1) / tab.sum(axis=1)
        out[c] = [f"{top[g]} ({share[g]:.0%})" for g in order]
    out.index.name = "Cluster"
    return out.reset_index()


def _cluster_key(name):
    return (1, 0) if name == "Noise" else (0, int(str(name)[1:]))


def cluster_heat(df, labels, numeric) -> pd.DataFrame:
    """Each cluster's average of each numeric feature, in standard
    deviations from the whole file's average: what sets a cluster apart."""
    labels = pd.Series(labels, index=df.index)
    order = sorted(labels.unique(), key=_cluster_key)
    cols = {}
    for c in numeric:
        v = pd.to_numeric(df[c], errors="coerce").astype("float64")
        v = v.where(np.isfinite(v))
        sd = v.std()
        means = v.groupby(labels).mean().reindex(order)
        cols[c] = (means - v.mean()) / sd if sd and sd > 0 else means * 0
    return pd.DataFrame(cols, index=order)


def partial_dependence(run, res, df, feature) -> pd.DataFrame:
    """The average prediction as one feature moves and the rest stay as they
    are, over every test row. Two-class runs give P(positive)."""
    X, _ = test_frame(run, df)
    _, categorical = _features(run)
    is_cat = feature in categorical
    # A category's grid is its values, given explicitly: scikit-learn sorts
    # the column to find them, and text beside a missing value cannot sort.
    custom = ({feature: np.array(sorted(X[feature].dropna().unique(),
                                        key=str), dtype=object)}
              if is_cat else None)
    out = _sk_partial_dependence(
        res["pipeline"], X, [feature],
        categorical_features=[feature] if is_cat else None,
        custom_values=custom, grid_resolution=PD_GRID, kind="average")
    grid = out["grid_values"][0]
    avg = np.asarray(out["average"])[0]
    if run["task"] == "classification":
        # scikit-learn reports the second class of a two-class model
        seen = ml_recipe.model_classes(res["pipeline"], run["classes"])
        if run["positive"] == seen[0]:
            avg = 1 - avg
    return pd.DataFrame({"value": grid, "prediction": avg})


# ── SHAP ────────────────────────────────────────────────────────────────
# Exact explainers only: TreeExplainer for the tree models and boosters, and
# LinearExplainer for the linear ones. A model-agnostic explainer would need a
# background sample and thousands of model calls per row; for those models
# permutation importance and the effect of one feature say what drives them.
SHAP_TREES = ("tree", "random_forest", "extra_trees", "hist_gb", "xgboost",
              "lightgbm")
SHAP_LINEAR = ("logistic", "linear", "ridge", "lasso", "elastic_net")
SHAP_CHUNK = 2_000


SHAP_TOLERANCE = 1e-3


CALIBRATION = {"sigmoid": "Platt scaling", "isotonic": "isotonic"}


def shap_problem(run, res):
    """Why SHAP cannot explain this model here, or None when it can."""
    missing = ml.package_problem("shap")
    if missing:
        return missing
    if run["task"] == "clustering":
        return "Clustering has no prediction to explain."
    if run.get("external"):
        return ("SHAP is computed for the models trained here, where every "
                "step between the file and the model is known; for an "
                "uploaded model, permutation importance shows what drives "
                "it.")
    if (res["recipe"].get("calibration") or "none") != "none":
        return ("Calibration averages three copies of the model behind a "
                "curve that SHAP's exact explainers cannot read; "
                "permutation importance shows what drives it.")
    if res["key"] not in SHAP_TREES + SHAP_LINEAR:
        return (f"SHAP is computed exactly for trees, forests, boosting and "
                f"linear models; for {res['label']}, permutation importance "
                f"shows what drives it.")
    if ml_recipe.native_categories(res["recipe"]) and \
            res["recipe"]["categorical"]:
        # Measured: values off by up to 5.9 log-odds with category splits,
        # exact to 1e-6 on numbers alone.
        return ("SHAP's tree explainer misreads gradient boosting's category "
                "splits: in testing its values did not add up to the model's "
                "output. XGBoost and LightGBM encode categories as columns "
                "and are explained exactly, as is gradient boosting on "
                "numbers alone.")
    return None


def feature_map(pipe, recipe) -> list:
    """The user's column behind each column the model reads, in order.

    Counted from the fitted transformers rather than parsed from encoded
    names: "Specialty_Onc" cannot say whether the column was Specialty or
    Specialty_Onc. A one-hot input yields one column per category it kept,
    with its rare categories, when grouped, folded into one.
    """
    out = []
    for name, trans, cols in pipe.named_steps["prep"].transformers_:
        cols = list(cols)
        if name == "remainder" or (isinstance(trans, str) and
                                   trans == "drop") or not cols:
            continue
        if name == "cat":
            enc = trans.named_steps["encode"]
            if isinstance(enc, OneHotEncoder):
                try:
                    rare = enc.infrequent_categories_
                except AttributeError:
                    rare = [None] * len(cols)
                for col, cats, few in zip(cols, enc.categories_, rare):
                    width = len(cats) - (len(few) - 1 if few is not None
                                         else 0)
                    out += [col] * width
            elif (isinstance(enc, TargetEncoder)
                  and enc.target_type_ == "multiclass"):
                for col in cols:             # one column per class
                    out += [col] * len(enc.classes_)
            else:
                out += cols                  # one column each
        elif name == "num":
            out += cols
            imp = (trans.named_steps.get("impute")
                   if isinstance(trans, Pipeline) else None)
            if imp is not None and getattr(imp, "indicator_", None) \
                    is not None:
                out += [cols[i] for i in imp.indicator_.features_]
        elif name == "gaps":
            out += [cols[i] for i in trans.features_]
    return out


def _explained_estimator(pipe):
    """The fitted model SHAP reads; a log-scale regressor is explained on the
    scale it was fitted on."""
    est = pipe[-1]
    if isinstance(est, TransformedTargetRegressor):
        return est.regressor_
    return est


def shap_units(run, res) -> str:
    """What the values add up to, so a chart can say so."""
    if run["task"] == "regression":
        return (f"log(1 + {run['target']})"
                if res["recipe"]["prep"].get("log_target") else run["target"])
    if res["key"] in ("tree", "random_forest", "extra_trees"):
        return "probability"
    return "log-odds"


def _raw(run, res, z, cls):
    """What a row's SHAP values add up to, read from the model itself: the
    class probability for forests and trees, the log-odds for boosting and
    logistic regression, the prediction (on its fitted scale) for a
    regressor."""
    est = _explained_estimator(res["pipeline"])
    if run["task"] == "regression":
        return np.asarray(est.predict(z), dtype="float64")
    seen = ml_recipe.model_classes(res["pipeline"], run["classes"])
    if shap_units(run, res) == "probability":
        return est.predict_proba(z)[:, seen.index(cls)]
    if res["key"] == "xgboost":
        margin = est.predict(z, output_margin=True)
    elif res["key"] == "lightgbm":
        margin = est.predict(z, raw_score=True)
    else:
        margin = est.decision_function(z)
    margin = np.asarray(margin, dtype="float64")
    if margin.ndim == 2:
        return margin[:, seen.index(cls)]
    return margin if cls == seen[-1] else -margin


def raw_output(run, res, X, cls=None):
    """`_raw` for prepared rows; the default class is the class of
    interest."""
    seen = (ml_recipe.model_classes(res["pipeline"], run["classes"])
            if run["task"] == "classification" else [])
    cls = cls or run.get("positive") or (seen[0] if seen else None)
    return _raw(run, res, res["pipeline"][:-1].transform(X), cls)


def explain(run, res, df, positions, cls=None, progress=None) -> dict:
    """Exact SHAP values for the rows at `positions`, one column per user
    feature: an encoded category's columns are summed, which SHAP's additivity
    allows. Every row asked for is explained; the chunks only pace the work.

    {"features", "values" (rows x features), "base" (per row), "rows",
    "units", "class"}; each row's values plus its base add up to
    `raw_output` for that row.
    """
    import shap

    numeric, categorical = _features(run)
    pipe = res["pipeline"]
    front, est = pipe[:-1], _explained_estimator(pipe)
    names = feature_map(pipe, res["recipe"])
    order = list(dict.fromkeys(names))
    onto = np.zeros((len(names), len(order)))
    onto[np.arange(len(names)), [order.index(n) for n in names]] = 1.0
    tree = res["key"] in SHAP_TREES
    if tree:
        explainer = shap.TreeExplainer(est)
    else:
        # Against the average training row: one reference row, computed over
        # every training row, rather than a sampled background.
        train = np.flatnonzero(np.char.startswith(
            np.asarray(run["split"], dtype=str), "train"))
        Xtr = ml_recipe.prepare_features(df.iloc[train], numeric, categorical)
        mean = np.asarray(front.transform(Xtr), dtype="float64").mean(
            axis=0, keepdims=True)
        explainer = shap.LinearExplainer(est, shap.maskers.Independent(mean))
    seen = (ml_recipe.model_classes(pipe, run["classes"])
            if run["task"] == "classification" else [])
    cls = cls or run.get("positive") or (seen[0] if seen else None)
    positions = np.asarray(positions)
    chunks = np.array_split(positions, max(1, -(-len(positions)
                                                 // SHAP_CHUNK)))
    values, base = [], []
    for i, chunk in enumerate(chunks, 1):
        X = ml_recipe.prepare_features(df.iloc[chunk], numeric, categorical)
        z = np.asarray(front.transform(X), dtype="float64")
        out = explainer(z, check_additivity=False) if tree else explainer(z)
        v = np.asarray(out.values, dtype="float64")
        b = np.asarray(out.base_values, dtype="float64")
        if v.ndim == 3:                       # one slice per class
            j = seen.index(cls)
            v = v[:, :, j]
            b = b[:, j] if b.ndim == 2 else b
        elif seen and cls == seen[0] and len(seen) == 2:
            v, b = -v, -b                     # the log-odds of the other class
        # One base for every row (a linear model's) or one per row; never
        # float() of a one-row array, which NumPy 2.4 refuses.
        b = b.reshape(-1)
        b = np.full(len(chunk), b[0]) if b.size == 1 else \
            np.broadcast_to(b, (len(chunk),)).copy()
        # Shown only if it adds up: every row's values plus its base must be
        # the model's own output, or something misread the model.
        raw = _raw(run, res, z, cls)
        gap = np.abs(v.sum(axis=1) + b - raw)
        if (gap > SHAP_TOLERANCE * np.maximum(1.0, np.abs(raw))).any():
            raise ValueError("SHAP values do not add up to the model's "
                             "output")
        values.append((v @ onto).astype("float32"))
        base.append(b)
        if progress:
            progress(i / len(chunks),
                     f"SHAP: {min(i * SHAP_CHUNK, len(positions)):,} of "
                     f"{len(positions):,} rows")
    return {"features": order, "values": np.vstack(values),
            "base": np.concatenate(base), "rows": positions,
            "units": shap_units(run, res), "class": cls}


def shap_seconds(run, res, df) -> float:
    """Roughly how long `explain` takes over every test row.

    A one-row call is the fixed cost (building the explainer, and for a
    linear model the average training row); 200 rows less that give the cost
    per row. Timing 50 rows and scaling up counted the fixed cost 1,200 times
    over at 60,000 test rows: 198 seconds predicted for a logistic
    regression.
    """
    import time
    test = rows(run, "test")
    if not len(test):
        return 0.0
    t = time.perf_counter()
    explain(run, res, df, test[:1])
    fixed = time.perf_counter() - t
    some = test[:200]
    t = time.perf_counter()
    explain(run, res, df, some)
    per_row = max(0.0, time.perf_counter() - t - fixed) / len(some)
    return fixed + per_row * len(test)


def shap_summary(exp) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, largest first."""
    # In float64: the values are stored as float32, and a float32 mean over
    # tens of thousands of rows drifts in the sixth digit.
    mean_abs = np.abs(exp["values"].astype("float64")).mean(axis=0)
    return (pd.DataFrame({"feature": exp["features"], "mean_abs": mean_abs})
            .sort_values("mean_abs", ascending=False)
            .reset_index(drop=True))


def shap_waterfall(exp, i, top=10) -> pd.DataFrame:
    """One row's contributions, largest first, the rest folded together."""
    row = pd.Series(exp["values"][i], index=exp["features"], dtype="float64")
    order = row.abs().sort_values(ascending=False).index
    shown = row[order[:top]]
    rest = row[order[top:]]
    out = pd.DataFrame({"feature": shown.index, "value": shown.to_numpy()})
    if len(rest):
        out.loc[len(out)] = [f"{len(rest)} other features", float(rest.sum())]
    return out


# ── leaderboard ─────────────────────────────────────────────────────────
def leaderboard(run) -> pd.DataFrame:
    """One row per model: the test result of every metric, the
    cross-validated selection metric, and timings. Raw numbers; the page
    formats them."""
    family, metric = run["family"], run["metric"]
    rows_ = []
    for r in run["results"]:
        row = {"Model": r["label"], "key": r["key"]}
        if r.get("error"):
            row["Status"] = f"Could not be trained. Reference {r['error']}."
            rows_.append(row)
            continue
        row["Status"] = ("Best" if r["key"] == run["best"] else
                         "Reference" if r["key"] == ml.BASELINE else "")
        for k, label, _ in ml.METRICS[family]:
            row[label] = r["test"].get(k)
        if r.get("cv"):
            vals = [v for v in r["cv"][metric] if v is not None]
            name = ml.metric_label(family, metric)
            row[f"CV {name}"] = float(np.mean(vals)) if vals else None
            row[f"CV {name} spread"] = float(np.std(vals)) if vals else None
        if family == "clustering":
            row["Clusters"] = r.get("n_clusters")
            if "inertia" in r["test"]:
                row["Inertia"] = r["test"]["inertia"]
            if "bic" in r["test"]:
                row["BIC"] = r["test"]["bic"]
        if not run.get("external"):     # an uploaded model was fitted elsewhere
            row["Fit (s)"] = r.get("fit_s")
        row["Settings"] = settings_text(run["task"], r)
        rows_.append(row)
    return pd.DataFrame(rows_)


def settings_text(task, res) -> str:
    if res.get("external") and res["key"] != ml.BASELINE:
        return "As trained before upload"
    if res["key"] == ml.BASELINE:
        return ("Always the most common class" if task == "classification"
                else "Always the average")
    if res["key"] == ml.STACK:
        text = "; ".join(
            f"{ml.model_label(task, m['model'])} ("
            + settings_text(task, {"key": m["model"], "params": m["params"]})
            + ")" for m in res["params"]["members"]) +             "; combined by a " + ("logistic regression" if task ==
                                  "classification" else "ridge regression")
    else:
        specs = {p["name"]: p
                 for p in ml.MODELS[task][res["key"]]["params"]}
        parts = [f"{specs[k]['label']} {ml.show_value(specs[k], v)}"
                 for k, v in res["params"].items() if k in specs]
        text = ", ".join(parts) or "Defaults"
    how = (res.get("recipe") or {}).get("calibration") or "none"
    if how != "none":
        text += f"; probabilities calibrated ({CALIBRATION[how]})"
    return text


# The verdict's yardstick: a metric whose chance value does not depend on the
# baseline. Macro F1 against an always-the-commonest-class baseline called a
# chance-level model a win (the baseline scores near zero on every class but
# one), so the verdict reads AUC or R² whatever metric picked the model.
CHANCE = {"binary": ("roc_auc", 0.5, 0.55), "multiclass": ("roc_auc_ovr", 0.5,
                                                           0.55),
          "regression": ("r2", 0.0, 0.05)}


def verdict(run):
    """(beats guessing, metric key, value, chance value) for the best model
    on the test rows, or None when there is nothing to judge."""
    if run["family"] not in CHANCE or not run.get("best"):
        return None
    key, chance, needed = CHANCE[run["family"]]
    value = ml.result(run, run["best"])["test"].get(key)
    if value is None:
        return None
    return value >= needed, key, value, chance


# ── outputs ─────────────────────────────────────────────────────────────
def _free(name, taken):
    """`name`, or `name_2`, `name_3`, when the file already has it."""
    if name not in taken:
        return name
    i = 2
    while f"{name}_{i}" in taken:
        i += 1
    return f"{name}_{i}"


def prediction_columns(run, res, taken=(), detail=True) -> pd.DataFrame:
    """The model's output for every row, as new columns.

    `detail` adds the split and, for several classes, each class's
    probability; without it, the prediction and its score only.
    """
    taken = set(taken)
    cols = {}

    def add(name, values):
        key = _free(name, taken | set(cols))
        cols[key] = values

    if run["task"] == "clustering":
        add("ML_Cluster", res["labels"])
        return pd.DataFrame(cols)
    add("ML_Prediction", predictions(run, res))
    if run["task"] == "classification":
        if len(run["classes"]) == 2:
            add("ML_Probability", positive_score(run, res, slice(None)))
        else:
            add("ML_Confidence", res["proba"].max(axis=1))
            if detail:
                for j, c in enumerate(run["classes"]):
                    add(f"ML_Probability_{c}", res["proba"][:, j])
    else:
        actual = run["y"].to_numpy(dtype="float64")
        # Predicted minus actual: above zero, the row scores less than
        # rows like it do. Honest only where the prediction is out of
        # sample, which the split column records.
        add("ML_Gap", res["pred"] - actual)
    if detail:
        split = run["split"]
        if run.get("external"):
            split = np.where(split == "test", "scored against the target",
                             split)
        add("ML_Split", split)
    return pd.DataFrame(cols)


def predictions(run, res):
    """The predicted labels for every row, at the saved cut-off if any."""
    if run["task"] != "classification":
        return res["pred"]
    return ml.with_cutoff(res["pred"], res.get("proba"), run["classes"],
                          run.get("positive"), res.get("threshold"))


def with_predictions(run, res, df, detail=True) -> pd.DataFrame:
    """The file as it was, plus the model's columns."""
    extra = prediction_columns(run, res, df.columns, detail=detail)
    extra.index = df.index
    return pd.concat([df, extra], axis=1)


def metric_lines(run, res) -> list:
    family = run["family"]
    out = []
    for k, label, _ in ml.METRICS[family]:
        v = res["test"].get(k)
        if v is not None:
            out.append(f"| {label} | {fmt_metric(k, v)} |")
    return out


def fmt_metric(key, value) -> str:
    if value is None:
        return "not computed"
    if key in ("rmse", "mae", "medae"):
        return f"{value:,.4g}"
    if key == "mape":
        return f"{value:.1f}%"
    if key in ("calinski_harabasz", "inertia", "bic"):
        return f"{value:,.1f}"
    return f"{value:.3f}"


def describe_target(run) -> str:
    if run["task"] == "clustering":
        return "no target (clustering)"
    if run.get("rule"):
        return f"yes/no: {run['rule']['describe']}"
    return run["target"]


def _external_card(run, res) -> str:
    """What was uploaded, what it read, and how it did on this file."""
    up = run["upload"]
    base = run["base"]
    task = run["task"]
    kind = {"app": "a model file from this app",
            "estimator": "a scikit-learn model or pipeline",
            "xgboost": "an XGBoost model file",
            "lightgbm": "a LightGBM model file"}.get(up["kind"], up["kind"])
    lines = [f"# Model card: {res['label']}", "",
             f"Uploaded and evaluated {run['created']} on the file loaded "
             f"here; it was not trained here.", "", "## The model", "",
             f"- {up['label']}: {kind}"]
    if up.get("trained"):
        lines.append(f"- Trained {up['trained']}"
                     + (f" with scikit-learn {up['sklearn']}"
                        if up.get("sklearn") else ""))
    lines += ["", "## Task", "", f"- {ml.TASK_LABELS[task]}"]
    if task != "clustering":
        lines.append(f"- Target: {describe_target(run)}")
        if run.get("positive"):
            lines.append(f"- Class of interest: {run['positive']}")
        lines.append(f"- {run['n_test']:,} rows scored against the target; "
                     f"{run['n_rows'] - run['n_test']:,} predicted without "
                     f"one")
        if run.get("n_unknown_class"):
            lines.append(f"- {run['n_unknown_class']:,} rows hold a class "
                         f"the model does not have, and are left out of "
                         f"the scores")
    lines += ["", "## Inputs", ""]
    for name, col in base["columns"].items():
        lines.append(f"- {name}" + (f" from {col}" if col != name else "")
                     + f" ({base['kinds'][name]})")
    steps = run.get("prep_steps") or []
    if steps:
        lines += ["", "## Steps applied to the file first", ""] + [
            f"- {n}. {ml_prep.describe(step)}"
            for n, step in enumerate(steps, 1) if not ml_prep.is_row_step(
                step)]
    lines += ["", "## Results" if task == "clustering"
              else "## Results on the rows with a target", "",
              "| Metric | Value |", "| --- | --- |"] + metric_lines(run, res)
    base_res = next((r for r in run["results"]
                     if r["key"] == ml.BASELINE and not r.get("error")), None)
    if base_res is not None:
        m = run["metric"]
        lines += ["", f"The baseline, which ignores every feature, scores "
                      f"{fmt_metric(m, base_res['test'].get(m))} on "
                      f"{ml.metric_label(run['family'], m)}."]
    lines += ["", "## Read with care", "",
              "- If the model was trained on some of these rows, its scores "
              "here overstate how it does on rows it has not seen.",
              "- Associations, not causes: a column that predicts the target "
              "does not make it happen."]
    return "\n".join(lines) + "\n"


def model_card(run, res) -> str:
    """A Markdown record of what was trained, on what, and how it did."""
    if run.get("external"):
        return _external_card(run, res)
    base = run["base"]
    task = run["task"]
    lines = [f"# Model card: {res['label']}", "",
             f"Trained {run['created']} with scikit-learn {run['sklearn']}.",
             "", "## Task", "",
             f"- {ml.TASK_LABELS[task]}",
             f"- Target: {describe_target(run)}"]
    if run.get("positive"):
        lines.append(f"- Class of interest: {run['positive']}")
        if res.get("threshold") is not None:
            lines.append(f"- Flagged when its probability is at least "
                         f"{res['threshold']:.2f} (chosen cut-off)")
    if task == "classification":
        counts = pd.Series(truth(run, rows(run, "labelled"))).value_counts()
        lines.append("- Classes: " + ", ".join(
            f"{k} ({v:,})" for k, v in counts.items()))
    lines += ["", "## Data", "",
              f"- {run['n_rows']:,} rows in the file"]
    if task != "clustering":
        lines += [f"- {run['n_labelled']:,} with a target: "
                  f"{run['n_train']:,} to train, {run['n_test']:,} to test",
                  f"- {run['n_unlabelled']:,} without a target, scored by "
                  f"the model"]
    lines += [f"- Numeric features ({len(base['numeric'])}): "
              + (", ".join(base["numeric"]) or "none"),
              f"- Categorical features ({len(base['categorical'])}): "
              + (", ".join(base["categorical"]) or "none")]
    steps = run.get("prep_steps") or []
    if steps:
        lines += ["", "## Steps applied to the file first", ""] + [
            f"- {n}. {ml_prep.describe(step)}"
            for n, step in enumerate(steps, 1)]
    lines += ["", "## Preparation", ""]
    prep = base["prep"]
    filled = {"none": "left missing", "median": "median", "mean": "mean",
              "zero": "zero", "most_frequent": "most frequent"}
    lines += [f"- Missing numbers: "
              f"{filled.get(prep.get('num_missing', 'none'), 'left missing')}"
              + (", with a flag column per gap"
                 if prep.get("missing_flags") else ""),
              "- Missing categories: "
              + ("most frequent" if prep.get("cat_missing") == "most_frequent"
                 else "their own category"),
              f"- Scaling: {prep.get('scale')}",
              "- Extreme numbers capped: " + {
                  "pct": "at the 1st and 99th percentiles of the training "
                         "rows",
                  "iqr": "at the training rows' quartile fences (1.5 × IQR)"
              }.get(prep.get("cap", "none"), "no"),
              f"- Rare categories grouped below: "
              f"{100 * float(prep.get('rare_share') or 0):g}% of rows"]
    if task == "regression":
        lines.append(f"- Target on a log scale: "
                     f"{'yes' if prep.get('log_target') else 'no'}")
    if task == "clustering" and prep.get("pca"):
        lines.append(f"- Principal components: {prep['pca']}")
    if task != "clustering":
        val = base["validation"]
        lines += ["", "## Validation", "",
                  f"- Test share: {float(val['test_size']):.0%}"
                  + (f", whole {val['group']} groups held out"
                     if val.get("group") else
                     ", stratified by class" if val.get("stratify")
                     and task == "classification" else ""),
                  "- Cross-validation: "
                  + (f"{run['folds']} folds" if run["folds"] else "none"),
                  f"- Seed: {base['seed']}",
                  f"- Best model chosen by: "
                  f"{ml.metric_label(run['family'], run['metric'])}"
                  + (" (cross-validated)" if run["folds"] else
                     " (test rows)")]
    lines += ["", "## Settings", "", f"- {settings_text(task, res)}"]
    if res.get("tuned"):
        how = (" (Bayesian search, Optuna)"
               if "stopped_early" in res["search"] else "")
        lines.append(f"- Tuned: {len(res['search'])} settings compared by "
                     f"cross-validation{how}")
    if base.get("balance"):
        lines.append("- Classes balanced by weight")
    resampled = {"smote": "oversampled with SMOTE",
                 "under": "undersampled"}.get(base.get("resample"))
    if resampled:
        lines.append(f"- Training rows {resampled} at each fit (the test "
                     f"rows never were)")
    lines += ["", "## Results on the test rows" if task != "clustering"
              else "## Cluster quality", "", "| Metric | Value |",
              "| --- | --- |"] + metric_lines(run, res)
    base_res = next((r for r in run["results"]
                     if r["key"] == ml.BASELINE and not r.get("error")), None)
    if base_res is not None and res["key"] != ml.BASELINE:
        m = run["metric"]
        lines += ["", f"The baseline, which ignores every feature, scores "
                      f"{fmt_metric(m, base_res['test'].get(m))} on "
                      f"{ml.metric_label(run['family'], m)}."]
    lines += ["", "## Using the model file", "",
              "```python",
              "import joblib, pandas as pd",
              "bundle = joblib.load('model.joblib')",
              "df = pd.read_csv('new_file.csv')",
              "# the training script defines prepare_features; it turns the",
              "# columns into the types the model was trained on",
              "X = prepare_features(df, bundle['recipe']['numeric'],",
              "                     bundle['recipe']['categorical'])",
              "bundle['pipeline'].predict(X)",
              "```", "",
              "Load a model file only from a source you trust: loading one "
              "can run code.", "", "## Limits", "",
              "- Learned from this file. A population that changes (a new "
              "launch, a new territory map) needs retraining.",
              "- Associations, not causes: a column that predicts the target "
              "does not make it happen."]
    return "\n".join(lines) + "\n"


def _prep_source() -> str:
    """ml_prep.py as the scripts carry it: no docstring, no import of
    label_text (the recipe above defines it), and none of the app's
    step-building helpers."""
    src = inspect.getsource(ml_prep)
    src = src.split("# ── building steps", 1)[0].rstrip() + "\n"
    tree = ast.parse(src)
    first = tree.body[0]
    lines = src.splitlines()
    if isinstance(first, ast.Expr) and isinstance(getattr(first, "value",
                                                          None), ast.Constant):
        lines = lines[first.end_lineno:]
    keep = [ln for ln in lines if ln.strip() not in (
        "from logic.ml_recipe import label_text", "import numpy as np",
        "import pandas as pd")]
    return "\n".join(keep).strip("\n") + "\n"


def _comment(text) -> str:
    """Text for one comment line. A column name may hold a newline, which
    would end the comment and leave the rest of the name to run as code."""
    return " ".join("".join(ch if ch.isprintable() else " "
                            for ch in str(text)).split())


def _steps_literal(steps) -> str:
    """The steps as a Python literal, each with its description above it."""
    if not steps:
        return "PREP_STEPS = []\n"
    out = ["PREP_STEPS = ["]
    for n, step in enumerate(steps, 1):
        out.append(f"    # {n}. {_comment(ml_prep.describe(step))}")
        body = pprint.pformat(step, width=72, sort_dicts=False)
        out += ["    " + ln for ln in body.splitlines()]
        out[-1] += ","
    out.append("]")
    return "\n".join(out) + "\n"


def prep_script(steps) -> str:
    """The Explore & prepare page's steps as a standalone script."""
    label = inspect.getsource(ml_recipe.label_text)
    return ('"""Prepare a file the way the Machine Learning page did.\n\n'
            "    python THIS_FILE.py your_file.csv\n\n"
            "Applies the steps below in order, their numbers fixed when "
            "they were added,\nand writes your_file_prepared.csv. Needs "
            'numpy and pandas.\n"""\n'
            "import sys\n\nimport numpy as np\nimport pandas as pd\n\n\n"
            + label + "\n\n" + _prep_source() + "\n\n"
            + _steps_literal(steps)
            + "\n\nif __name__ == \"__main__\":\n"
              "    if len(sys.argv) < 2:\n"
              "        sys.exit(\"Usage: python %s your_file.csv\" % "
              "sys.argv[0])\n"
              "    path = sys.argv[1]\n"
              "    df = (pd.read_excel(path) if path.lower().endswith("
              "(\".xlsx\", \".xls\"))\n"
              "          else pd.read_csv(path))\n"
              "    out = apply_steps(df, PREP_STEPS)\n"
              "    target = path.rsplit(\".\", 1)[0] + \"_prepared.csv\"\n"
              "    out.to_csv(target, index=False)\n"
              "    print(\"Wrote %s: %d rows, %d columns\" % (target, len(out),"
              " out.shape[1]))\n")


def _recipe_source() -> str:
    """ml_recipe.py without its module docstring, which speaks of the app."""
    src = inspect.getsource(ml_recipe)
    tree = ast.parse(src)
    first = tree.body[0]
    if isinstance(first, ast.Expr) and isinstance(getattr(first, "value",
                                                          None), ast.Constant):
        src = "\n".join(src.splitlines()[first.end_lineno:]).lstrip("\n")
    return src


_SCRIPT_MAIN = """

# Scikit-learn's scorer for each metric the app can choose a model by, used
# when SEARCHES asks for tuning. Mean absolute % error here counts rows with
# a zero target, which the app leaves out.
SCORING = {"roc_auc": "roc_auc", "roc_auc_ovr": "roc_auc_ovr",
           "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy",
           "f1_macro": "f1_macro", "log_loss": "neg_log_loss",
           "r2": "r2", "rmse": "neg_root_mean_squared_error",
           "mae": "neg_mean_absolute_error",
           "mape": "neg_mean_absolute_percentage_error"}
CLASSES = []


def load(path):
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def cluster_names(codes):
    values, counts = np.unique(codes, return_counts=True)
    order = [v for v, _ in sorted(zip(values, counts), key=lambda t: -t[1])
             if v != -1]
    names = {int(v): "C%d" % (i + 1) for i, v in enumerate(order)}
    names[-1] = "Noise"
    return names


def scorer(metric, codes):
    if metric not in ("f1", "average_precision"):
        return SCORING[metric]
    pos = CLASSES.index(POSITIVE) if codes else POSITIVE
    if metric == "f1":
        return make_scorer(f1_score, pos_label=pos)
    return make_scorer(average_precision_score, pos_label=pos,
                       response_method="predict_proba")


def fit(recipe, X, y, groups):
    pipe = build_pipeline(with_caps(recipe, X))
    search = SEARCHES.get(recipe["model"])
    if not search:
        return pipe.fit(*resample(recipe, X, y))
    log_scale = recipe["task"] == "regression" and recipe["prep"].get(
        "log_target")
    prefix = "model__regressor__" if log_scale else "model__"
    grid = {prefix + k: v for k, v in search["space"].items()}
    cv = folds(recipe, groups is not None)
    score = scorer(METRIC, recipe["task"] == "classification"
                   and recipe["model"] in INTEGER_LABELS)
    if search["mode"] == "grid":
        found = GridSearchCV(pipe, grid, cv=cv, scoring=score)
    else:
        # "bayes" (Optuna, in the app) samples at random here, so this
        # script needs no extra package.
        found = RandomizedSearchCV(pipe, grid, n_iter=search["n_iter"],
                                   cv=cv, scoring=score,
                                   random_state=recipe["seed"])
    found.fit(X, y, groups=groups)
    print("  %s: best settings %s" % (recipe["model"], found.best_params_))
    if (recipe.get("resample") or "none") != "none":
        # Settings chosen on the rows as they are (the search's folds
        # cannot resample), then the final model fitted on rebalanced rows.
        return found.best_estimator_.fit(*resample(recipe, X, y))
    return found.best_estimator_


def report(recipe, pipe, X, y):
    if recipe["task"] == "classification":
        pred = predicted_labels(pipe, X, CLASSES)
        scores = {"accuracy": accuracy_score(y, pred)}
        if len(CLASSES) == 2:
            hit = y == POSITIVE
            if 0 < hit.sum() < len(hit):
                pos = model_classes(pipe, CLASSES).index(POSITIVE)
                scores["roc_auc"] = roc_auc_score(
                    hit, pipe.predict_proba(X)[:, pos])
    else:
        err = pipe.predict(X) - y
        total = ((y - y.mean()) ** 2).sum()
        scores = {"r2": 1 - (err ** 2).sum() / total,
                  "rmse": np.sqrt((err ** 2).mean())}
    print("%s: %s" % (recipe["model"], ", ".join(
        "%s %.4f" % (k, v) for k, v in scores.items())))


def main(path):
    global CLASSES
    df = apply_steps(load(path), PREP_STEPS)
    base = RECIPES[0]
    X = prepare_features(df, base["numeric"], base["categorical"])
    out = df.copy()
    if base["task"] == "clustering":
        for recipe in RECIPES:
            codes = np.asarray(build_pipeline(with_caps(recipe, X))
                               .fit_predict(X))
            names = cluster_names(codes)
            print("%s: %d clusters" % (recipe["model"], len(names) - 1))
            if recipe["model"] == BEST:
                out["ML_Cluster"] = pd.Series(codes).map(names).values
    else:
        y_all = target_values(df, base)
        have = y_all.notna().to_numpy()
        y = y_all.to_numpy()[have]
        if base["task"] == "regression":
            y = y.astype("float64")
        else:
            CLASSES = sorted(pd.unique(y).tolist(), key=str)
        group = base["validation"].get("group")
        groups = None
        if group:
            groups = label_text(df[group]).to_numpy()[have]
            groups = np.where(pd.isna(groups), "(blank)", groups).astype(str)
        train, test = split(y, groups, base)
        Xl = X.iloc[np.flatnonzero(have)]
        best = None
        for recipe in RECIPES:
            pipe = fit(recipe, Xl.iloc[train],
                       fit_labels(recipe, y[train], CLASSES),
                       None if groups is None else groups[train])
            report(recipe, pipe, Xl.iloc[test], y[test])
            if recipe["model"] == BEST:
                best = pipe
        if APP_TEST_RESULTS:
            print("The app reported:", APP_TEST_RESULTS)
        if best is not None:
            # Every row from the model trained above. The app gave training
            # rows their out-of-fold prediction instead.
            if base["task"] == "classification":
                out["ML_Prediction"] = predicted_labels(best, X, CLASSES)
                if len(CLASSES) == 2:
                    pos = model_classes(best, CLASSES).index(POSITIVE)
                    out["ML_Probability"] = best.predict_proba(X)[:, pos]
            else:
                out["ML_Prediction"] = best.predict(X)
    target = path.rsplit(".", 1)[0] + "_predictions.csv"
    out.to_csv(target, index=False)
    print("Wrote", target)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python %s your_file.csv" % sys.argv[0])
    main(sys.argv[1])
"""


def _script(recipes, *, title, what, best, positive, metric, app=None,
            searches=None, sklearn_version=None, steps=None):
    """The one template every exported script comes from.

    The recipes are literals built with `repr` (through pprint), so a column
    named O'Brien or one holding a newline stays one string. `ml_recipe.py` itself is the body, so
    the script builds every model the way the app does.
    """
    uses = sorted({r["model"] for r in recipes
                   if r["model"] in ("xgboost", "lightgbm")})
    needs = "numpy, pandas and scikit-learn" + (
        "".join(f", {'XGBoost' if u == 'xgboost' else 'LightGBM'}"
                for u in uses))
    version = (f" The app used scikit-learn {sklearn_version}."
               if sklearn_version else "")
    prepared = (f"It first applies the {len(steps)} preparation step"
                f"{'' if len(steps) == 1 else 's'} from Explore & prepare "
                f"(PREP_STEPS), then\nsplits" if steps else "It splits")
    head = (f'"""{title}\n\n{what}\n\n'
            "    python THIS_FILE.py your_file.csv\n\n"
            f"{prepared} the rows the way the app does (same seed), trains, "
            "prints each\nmodel's test result, and writes every row's "
            "prediction to\nyour_file_predictions.csv. Needs "
            f"{needs}.{version}\n"
            '"""\n'
            "import sys\n\n"
            "from sklearn.metrics import (accuracy_score, "
            "average_precision_score, f1_score,\n"
            "                             make_scorer, roc_auc_score)\n"
            "from sklearn.model_selection import GridSearchCV, "
            "RandomizedSearchCV\n\n")
    tail = ("\n\nRECIPES = " + pprint.pformat(list(recipes), width=79,
                                               sort_dicts=False)
            + "\n\n# The model whose predictions are written out."
            + "\nBEST = " + repr(best)
            + "\n\nPOSITIVE = " + repr(positive)
            + "\n\nMETRIC = " + repr(metric)
            + "\n\nSEARCHES = " + pprint.pformat(searches or {}, width=79,
                                                  sort_dicts=False)
            + "\n\nAPP_TEST_RESULTS = " + pprint.pformat(app or {}, width=79,
                                                          sort_dicts=False)
            + "\n")
    return (head + _recipe_source() + "\n\n" + _prep_source() + tail
            + "\n" + _steps_literal(steps) + _SCRIPT_MAIN)


_SCORING_MAIN = """

def load_model(path):
    if FORMAT == "xgboost":
        import xgboost
        model = (xgboost.XGBClassifier() if TASK == "classification"
                 else xgboost.XGBRegressor())
        model.load_model(path)
        return model
    if FORMAT == "lightgbm":
        import lightgbm
        return lightgbm.Booster(model_file=path)
    import joblib
    obj = joblib.load(path)
    return obj["pipeline"] if isinstance(obj, dict) else obj


def inputs(df):
    data = {}
    for name, column in MAPPING.items():
        if KINDS[name] == "numeric":
            v = pd.to_numeric(df[column], errors="coerce").astype("float64")
            data[name] = v.where(np.isfinite(v))
        else:
            data[name] = label_text(df[column])
    X = pd.DataFrame(data, index=df.index, columns=list(MAPPING))
    return X.to_numpy() if BY_POSITION or FORMAT == "lightgbm" else X


def as_labels(values, model):
    values = np.asarray(values)
    if CODES:
        return np.asarray(CLASSES, dtype=object)[values.astype("int64")]
    return label_text(pd.Series(values)).to_numpy(dtype=object)


def main(model_path, path):
    df = (pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls"))
          else pd.read_csv(path))
    df = apply_steps(df, PREP_STEPS, rows=False)
    X = inputs(df)
    model = load_model(model_path)
    out = df.copy()
    if TASK == "clustering":
        codes = pd.Series(np.asarray(model.predict(X)), index=df.index)
        out["ML_Cluster"] = codes.map(CLUSTER_NAMES).fillna("Noise")
        proba = None
    elif FORMAT == "lightgbm":
        raw = np.asarray(model.predict(X))
        if TASK == "classification":
            proba = np.column_stack([1 - raw, raw]) if raw.ndim == 1 else raw
            out["ML_Prediction"] = as_labels(proba.argmax(axis=1), model)
        else:
            out["ML_Prediction"] = raw
            proba = None
    else:
        out["ML_Prediction"] = (as_labels(model.predict(X), model)
                                if TASK == "classification"
                                else model.predict(X))
        proba = (model.predict_proba(X) if TASK == "classification"
                 and len(CLASSES) == 2 else None)
    if proba is not None and POSITIVE is not None:
        seen = (list(range(len(CLASSES))) if FORMAT == "lightgbm"
                else list(np.asarray(model.classes_)))
        names = ([CLASSES[int(i)] for i in seen] if CODES else
                 [str(c) for c in label_text(pd.Series(seen))])
        out["ML_Probability"] = proba[:, names.index(POSITIVE)]
    target = path.rsplit(".", 1)[0] + "_scored.csv"
    out.to_csv(target, index=False)
    print("Wrote", target)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("Usage: python %s your_model_file your_file.csv"
                 % sys.argv[0])
    main(sys.argv[1], sys.argv[2])
"""


def scoring_script(run, res) -> str:
    """Scores a file with the uploaded model, outside the app: the same
    inputs, built the same way from the same columns."""
    up = run["upload"]
    base = run["base"]
    fmt = {"xgboost": "xgboost", "lightgbm": "lightgbm"}.get(up["kind"],
                                                           "joblib")
    codes = bool(ml_recipe.answers_in_codes(res["pipeline"])) \
        if run["task"] == "classification" else False
    label = inspect.getsource(ml_recipe.label_text)
    steps = [s_ for s_ in (run.get("prep_steps") or [])
             if not ml_prep.is_row_step(s_)]
    head = ('"""Score a file with the model evaluated on the Machine Learning '
            'page.\n\n'
            "    python THIS_FILE.py your_model_file your_file.csv\n\n"
            "Builds the inputs the model reads from the file's columns "
            "(MAPPING), predicts\nevery row, and writes "
            "your_file_scored.csv. Load a model file only from a\nsource "
            'you trust: loading a joblib or pickle file can run code.\n"""\n'
            "import sys\n\nimport numpy as np\nimport pandas as pd\n\n\n")
    consts = ("\n\nFORMAT = " + repr(fmt)
              + "\nTASK = " + repr(run["task"])
              + "\nCODES = " + repr(codes)
              + "\nCLASSES = " + repr(list(run.get("classes") or []))
              + "\nPOSITIVE = " + repr(run.get("positive"))
              + "\n# The name each cluster number is shown under."
              + "\nCLUSTER_NAMES = " + repr(
                  {int(k): str(v) for k, v in (res.get("names") or {})
                   .items()} if run["task"] == "clustering" else {})
              + "\nBY_POSITION = " + repr(bool(base.get("by_position")))
              + "\n# The model's input, and the file column it is read from."
              + "\nMAPPING = " + pprint.pformat(dict(base["columns"]),
                                                width=79, sort_dicts=False)
              + "\nKINDS = " + pprint.pformat(dict(base["kinds"]), width=79,
                                              sort_dicts=False) + "\n")
    return (head + label + "\n\n" + _prep_source() + consts + "\n"
            + _steps_literal(steps) + _SCORING_MAIN)


def _app_results(results):
    return {r["key"]: {k: round(v, 6) for k, v in r["test"].items()
                       if k in ("accuracy", "roc_auc", "r2", "rmse")
                       and isinstance(v, float)}
            for r in results}


def script(run, res) -> str:
    """Retrains one model: the one on screen. An uploaded model was not
    trained here, so its script scores a file with it instead."""
    if run.get("external"):
        return scoring_script(run, res)
    return _script(
        [res["recipe"]],
        title=f"Retrain {res['label']} "
              f"({ml.TASK_LABELS[run['task']].lower()}).",
        what=f"Exported from Machine Learning, {run['created']}, with the "
             f"settings the app\ntrained. Run it on a file with the same "
             f"columns:",
        best=res["key"], positive=run.get("positive"), metric=run["metric"],
        app=_app_results([res]), sklearn_version=run["sklearn"],
        steps=run.get("prep_steps"))


def run_script(run) -> str:
    """Retrains every model of the run, the baseline first, and reproduces
    each one's test result."""
    if run.get("external"):
        return scoring_script(run, ml.result(run, run["best"]))
    done = [r for r in run["results"] if not r.get("error")]
    return _script(
        [r["recipe"] for r in done],
        title=f"Retrain every model of a {run['task']} run.",
        what=f"Exported from Machine Learning, {run['created']}: "
             f"{len(done)} models with the\nsettings the app trained. Run "
             f"it on a file with the same columns:",
        best=run.get("best"), positive=run.get("positive"),
        metric=run["metric"], app=_app_results(done),
        sklearn_version=run["sklearn"], steps=run.get("prep_steps"))


def config_script(df, config, steps=None) -> str:
    """The configure page's choices as a script, before anything trains: the
    models with their settings as entered, and any tuning as a scikit-learn
    search over the values chosen."""
    rule = (ml.resolve_rule(df, config["target"], config["target_rule"])
            if config.get("target_rule") else None)
    base = ml.base_recipe(config, rule)
    recipes = []
    if config["task"] != "clustering":
        recipes.append({**base, "model": ml.BASELINE, "params": {},
                        "prep": {**base["prep"], "log_target": False}})
    recipes += [{**base, "model": m["key"], "params": dict(m["params"])}
                for m in config["models"]]
    searches = {m["key"]: {"mode": m["search"]["mode"],
                           "space": m["search"]["space"],
                           "n_iter": m["search"].get("n_iter", 10)}
                for m in config["models"] if m.get("search")}
    labels = [r for r in recipes if r["model"] != ml.BASELINE]
    family = ml.metric_family(config["task"], None if config["task"] !=
                              "classification" else 2)
    metric = config.get("metric") or (ml.SELECTABLE.get(family) or [None])[0]
    positive = (ml_recipe.YES if rule else config.get("positive"))
    return _script(
        recipes, title=f"Train the {config['task']} models as configured.",
        what="Written by the Machine Learning configure page from the "
             "choices on screen,\nbefore training. Run it on a file with the "
             "same columns:",
        best=labels[0]["model"] if labels else None, positive=positive,
        metric=metric, searches=searches, steps=steps)


def model_bytes(run, res) -> bytes:
    bundle = {"pipeline": res["pipeline"], "recipe": res["recipe"],
              "classes": run["classes"], "positive": run["positive"],
              "cluster_names": res.get("names"), "sklearn": run["sklearn"],
              "created": run["created"],
              "prep_steps": list(run.get("prep_steps") or []),
              # The cut-off chosen on the results page, if any: every place
              # that scores with this file flags at it.
              "threshold": res.get("threshold")}
    buf = io.BytesIO()
    joblib.dump(bundle, buf)
    return buf.getvalue()


def zip_bundle(run, res, df, importance=None) -> bytes:
    """Everything one model produced, in one download."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("predictions.csv",
                   with_predictions(run, res, df).to_csv(index=False))
        z.writestr("leaderboard.csv",
                   leaderboard(run).drop(columns="key").to_csv(index=False))
        z.writestr("model_card.md", model_card(run, res))
        if run.get("external"):
            z.writestr("score_file.py", scoring_script(run, res))
        else:
            z.writestr("train_model.py", script(run, res))
            z.writestr("train_all_models.py", run_script(run))
            z.writestr("model.joblib", model_bytes(run, res))
        if importance is not None:
            z.writestr("feature_importance.csv",
                       importance.to_csv(index=False))
        if run["task"] == "clustering":
            numeric, categorical = _features(run)
            z.writestr("cluster_profiles.csv", cluster_profiles(
                analysis_frame(run, df), res["labels"], numeric,
                categorical).to_csv(index=False))
        if res.get("search") is not None:
            z.writestr("tuning.csv", res["search"].to_csv(index=False))
        z.writestr("settings.json", json.dumps(
            {"recipe": res["recipe"], "positive": run.get("positive"),
             "threshold": res.get("threshold")},
            default=str, indent=2))
    return buf.getvalue()


def can_score(run, res) -> bool:
    return hasattr(res["pipeline"], "predict")


def score_file(run, res, df_new):
    """(scored frame, notes) for a new file with the model's columns, or
    (None, [reasons]) when it cannot be scored.

    A category the model never saw is scored as unknown — counted per
    column, so a new specialty name is visible rather than silent.
    """
    steps = run.get("prep_steps") or []
    if steps:
        # The column steps only: every row in a file being scored gets a
        # prediction, so the steps that drop rows are left out.
        try:
            df_new = ml_prep.apply_steps(df_new, steps, rows=False)
        except ml_prep.StepError as e:
            return None, [f"The preparation could not be applied to this "
                          f"file. {e}"]
    numeric, categorical = _features(run)
    missing = [c for c in source_columns(run) if c not in df_new.columns]
    if missing:
        return None, [f"Not in this file: {', '.join(missing)}."]
    if not can_score(run, res):
        return None, [f"{res['label']} cannot place new rows into its "
                      f"clusters; K-means and Gaussian mixture can."]
    X = features_frame(run, df_new)
    notes = []
    if steps:
        n_rows = sum(ml_prep.is_row_step(step) for step in steps)
        n_cols = len(steps) - n_rows

        def counted(n, what):
            return (f"{n} {what} step{'' if n == 1 else 's'} "
                    f"{'was' if n == 1 else 'were'}")

        note = (f"From Explore & prepare, {counted(n_cols, 'column')} "
                f"applied" if n_cols else "")
        if n_rows:
            note += ("; " if note else "From Explore & prepare, ") + (
                f"{counted(n_rows, 'row')} not, so every row in the file "
                f"is scored")
        notes.append(note + ".")
    seen = _training_categories(res)
    for c in categorical:
        known = seen.get(c)
        if known is None:
            continue
        values = X[c].dropna()
        unseen = values[~values.isin(known)]
        if len(unseen):
            examples = ", ".join(map(str, pd.unique(unseen)[:3]))
            notes.append(f"{c}: {len(unseen):,} row"
                         f"{'' if len(unseen) == 1 else 's'} with a value the "
                         f"model never saw ({examples}), scored as unknown.")
    gaps = {c: int(n) for c, n in X[numeric].isna().sum().items() if n}
    filled = res["recipe"]["prep"].get("num_missing", "none") != "none"
    if gaps and not filled and not ml.reads_gaps(res) \
            and not run.get("external"):
        # Trained with no fill on a file without gaps: it has nothing to put
        # in one, and the model would raise on the first.
        listed = ", ".join(f"{c} ({n:,})" for c, n in list(gaps.items())[:3])
        if len(gaps) > 3:
            listed += f" and {len(gaps) - 3} more"
        return None, [f"{res['label']} was trained with no fill for missing "
                      f"numbers and cannot read a gap; this file has gaps: "
                      f"{listed}. Train it again with a fill under Missing "
                      f"numbers, or score with a model that reads gaps."]
    for c, n in gaps.items():
        notes.append(f"{c}: {n:,} missing, " + (
            "passed to the model as gaps." if run.get("external") else
            "filled by the missing-value rule." if filled else
            "left missing, as in training: the model reads gaps."))
    pipe = res["pipeline"]
    out = df_new.copy()
    if run["task"] == "clustering":
        codes = np.asarray(pipe.predict(X))
        names = res["names"]
        out[_free("ML_Cluster", set(out.columns))] = pd.Series(
            codes, index=out.index).map(names).to_numpy(dtype=object)
        return out, notes
    pred, proba = ml.predict(run["task"], pipe, X, run["classes"])
    pred = ml.with_cutoff(pred, proba, run["classes"], run.get("positive"),
                          res.get("threshold"))
    out[_free("ML_Prediction", set(out.columns))] = pred
    if proba is not None:
        if len(run["classes"]) == 2:
            j = run["classes"].index(run["positive"])
            out[_free("ML_Probability", set(out.columns))] = proba[:, j]
        else:
            out[_free("ML_Confidence", set(out.columns))] = proba.max(axis=1)
    return out, notes


def _training_categories(res) -> dict:
    """{column: categories seen in training}, from the fitted encoder."""
    try:
        cat = res["pipeline"].named_steps["prep"].named_transformers_["cat"]
        enc = cat.named_steps["encode"]
    except (KeyError, AttributeError):
        return {}
    # In the recipe's order: the encoder sees an array when an imputer runs
    # first, so it has no column names of its own.
    return {c: {str(v) for v in cats if not pd.isna(v)}
            for c, cats in zip(res["recipe"]["categorical"], enc.categories_)}
