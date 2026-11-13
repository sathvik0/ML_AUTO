"""Machine learning on a tabular file: classify, predict a number, or group.

The configure page decides nothing about the data itself. This module profiles
the columns, proposes a target and the features, flags a column that would
leak the answer, checks a configuration, and trains. Models are built from
recipes by `logic.ml_recipe` — the exported script embeds that file, so
nothing here builds an estimator any other way.

**Every row is used.** The project's rule is never to sample records, so an
algorithm whose cost grows with the square of the rows is not offered past a
measured size (`max_rows`), rather than handed a subset. Measured on the
sample's 62 encoded columns: a support vector machine takes 30s per fit at
10,000 rows; agglomerative clustering adds 760 MB at 10,000 rows; DBSCAN is
left out altogether, because its memory follows the radius, not the rows (2.75
GB at 10,000 rows on a radius of 10), and nothing checkable up front bounds it.

**Every score is honest.** Labelled rows are split into train and test. With
cross-validation on, each training row's prediction comes from the fold that
did not see it, so the predictions file holds an out-of-sample score for every
row with a target, and the model is judged on the test rows it never saw. A
baseline that predicts without looking at any feature is always trained
alongside, because a model is only useful if it beats guessing.
"""
import importlib
import logging
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, calinski_harabasz_score,
                             davies_bouldin_score, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score,
                             silhouette_score)
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.pipeline import Pipeline

from logic import ml_prep, ml_recipe, schema
from utils import log

_log = log.get("ml")

TASKS = ("classification", "regression", "clustering")
TASK_LABELS = {"classification": "Classification", "regression": "Regression",
               "clustering": "Clustering"}

MAX_CLASSES = 30          # a classification target with more is refused
MAX_LEVELS = 50           # categorical features above this are off by default
MAX_ENCODE_LEVELS = 200   # and above this cannot be encoded at all
MIN_LABELLED = 20         # rows with a target needed to train and test
MAX_FITS = 500            # a tuning plan above this is refused
SILHOUETTE_MAX_ROWS = 20_000   # pairwise distances: n² work
LEAKAGE_ERROR_SHARE = 0.10     # a column alone leaves under 10% of the errors
LEAKAGE_R2 = 0.98
POSTAL_HINTS = ("zip", "postal", "postcode", "fips")
OUTCOME_HINTS = ("default", "fraud", "respon", "adopt", "convert", "churn", "outcome",
                 "target", "label", "flag")
VOLUME_HINTS = ("revenue", "sales", "premium", "spend", "value", "amount",
                "price", "income", "volume", "units")
GEO_HINTS = ("latitude", "longitude", "lat", "lon", "lng")

BASELINE = "baseline"
# Models that read a missing number as it is. With "None" under Missing
# numbers, any other model is refused when a feature has a gap — it would
# fail on the first one.
GAP_MODELS = ("baseline", "tree", "random_forest", "extra_trees", "hist_gb",
              "xgboost", "lightgbm")


# ── the catalogue ───────────────────────────────────────────────────────
def _p(name, label, values, default, kind="ordered", show=None):
    """One hyperparameter: its choices, its default, how it is drawn.

    `ordered` choices are drawn as a slider, `choice` as a dropdown; `show`
    names a value where its plain form would not read well (None is
    "Unlimited", 1.0 features per split is "All").
    """
    return {"name": name, "label": label, "values": list(values),
            "default": default, "kind": kind, "show": show or {}}


_UNLIMITED = {None: "Unlimited"}
_DEPTH = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, None]
_LEAF = [1, 2, 5, 10, 20, 50, 100]
_TREES = [25, 50, 100, 200, 300, 500]
_FEATURES = ["sqrt", "log2", 0.33, 0.5, 1.0]
_FEATURES_SHOW = {"sqrt": "Square root", "log2": "Log2", 0.33: "A third",
                  0.5: "Half", 1.0: "All"}
_ALPHA = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
_K = list(range(2, 16))


def _forest(max_features):
    return [_p("n_estimators", "Trees", _TREES, 100),
            # Bounded by default: an unlimited forest on 240,000 rows grows
            # trees of ~100k nodes each, over a gigabyte for 100 of them.
            _p("max_depth", "Maximum depth", _DEPTH, 12, show=_UNLIMITED),
            _p("min_samples_leaf", "Minimum rows per leaf", _LEAF, 5),
            _p("max_features", "Features tried per split", _FEATURES,
               max_features, kind="choice", show=_FEATURES_SHOW)]


_TREE = [_p("max_depth", "Maximum depth", _DEPTH, 6, show=_UNLIMITED),
         _p("min_samples_leaf", "Minimum rows per leaf", _LEAF, 10)]
_BOOST = [_p("learning_rate", "Learning rate",
             [0.01, 0.03, 0.05, 0.1, 0.2, 0.3], 0.1),
          _p("max_iter", "Boosting rounds", [50, 100, 200, 300, 500, 1000],
             200),
          _p("max_leaf_nodes", "Leaves per tree", [7, 15, 31, 63, 127], 31),
          _p("min_samples_leaf", "Minimum rows per leaf",
             [5, 10, 20, 50, 100], 20),
          _p("l2_regularization", "L2 penalty", [0.0, 0.01, 0.1, 1.0, 10.0],
             0.0)]
_KNN = [_p("n_neighbors", "Neighbours", [3, 5, 10, 15, 25, 50, 100], 15),
        _p("weights", "Weighting", ["uniform", "distance"], "uniform",
           kind="choice",
           show={"uniform": "Equal", "distance": "Closer counts more"})]
_SVM = [_p("C", "Penalty (C)", [0.01, 0.1, 1.0, 10.0, 100.0], 1.0),
        _p("kernel", "Kernel", ["rbf", "linear", "poly"], "rbf",
           kind="choice", show={"rbf": "Radial (RBF)", "linear": "Linear",
                                "poly": "Polynomial"}),
        _p("gamma", "Kernel width", ["scale", "auto"], "scale", kind="choice",
           show={"scale": "Scaled to the data", "auto": "1 / features"})]
# Rows and features per tree default to all of them: lowering either is a
# regulariser to reach for, not a default that leaves rows out of a tree.
_XGB = [_p("n_estimators", "Trees", [100, 200, 300, 500, 1000], 300),
        _p("learning_rate", "Learning rate",
           [0.01, 0.03, 0.05, 0.1, 0.2, 0.3], 0.1),
        _p("max_depth", "Maximum depth", [2, 3, 4, 6, 8, 10], 6),
        _p("min_child_weight", "Minimum weight per leaf", [1, 3, 5, 10], 1),
        _p("subsample", "Share of rows per tree", [0.5, 0.7, 0.8, 0.9, 1.0],
           1.0),
        _p("colsample_bytree", "Share of features per tree",
           [0.5, 0.7, 0.8, 1.0], 1.0),
        _p("reg_lambda", "L2 penalty", [0.0, 0.1, 1.0, 10.0], 1.0)]
_LGBM = [_p("n_estimators", "Trees", [100, 200, 300, 500, 1000], 300),
         _p("learning_rate", "Learning rate",
            [0.01, 0.03, 0.05, 0.1, 0.2, 0.3], 0.1),
         _p("num_leaves", "Leaves per tree", [7, 15, 31, 63, 127], 31),
         _p("min_child_samples", "Minimum rows per leaf",
            [5, 10, 20, 50, 100], 20),
         _p("subsample", "Share of rows per tree",
            [0.5, 0.7, 0.8, 0.9, 1.0], 1.0),
         _p("colsample_bytree", "Share of features per tree",
            [0.5, 0.7, 0.8, 1.0], 1.0),
         _p("reg_lambda", "L2 penalty", [0.0, 0.1, 1.0, 10.0], 0.0)]
_XGB_ABOUT = ("Gradient-boosted trees from the XGBoost library: "
              "regularised, reads gaps directly, fast on large files.")
_LGBM_ABOUT = ("Gradient-boosted trees grown leaf by leaf: very fast on "
               "large files, reads gaps directly.")
_MLP = [_p("hidden_layer_sizes", "Hidden layers",
           [(32,), (64,), (128,), (64, 32), (128, 64), (256, 128, 64)],
           (64, 32), kind="choice",
           show={(32,): "32", (64,): "64", (128,): "128", (64, 32): "64, 32",
                 (128, 64): "128, 64", (256, 128, 64): "256, 128, 64"}),
        _p("alpha", "L2 penalty", [0.00001, 0.0001, 0.001, 0.01, 0.1],
           0.0001),
        _p("learning_rate_init", "Learning rate", [0.0001, 0.001, 0.01],
           0.001),
        _p("max_iter", "Training passes", [100, 200, 500], 200)]

MODELS = {
    "classification": {
        "logistic": {
            "label": "Logistic regression", "default": True,
            "about": "A weighted sum of the features turned into a "
                     "probability. Fast, and its weights are easy to explain.",
            "params": [_p("C", "Penalty strength (C, lower is simpler)",
                          [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], 1.0)]},
        "tree": {
            "label": "Decision tree",
            "about": "A short list of yes/no questions. Easy to read; one "
                     "tree on its own overfits easily.", "params": _TREE},
        "random_forest": {
            "label": "Random forest", "default": True,
            "about": "Many trees on random slices of the data, averaged. A "
                     "strong, low-maintenance default.",
            "params": _forest("sqrt")},
        "extra_trees": {
            "label": "Extra trees",
            "about": "A random forest with random split points: faster to "
                     "train and often as accurate.",
            "params": _forest("sqrt")},
        "hist_gb": {
            "label": "Gradient boosting", "default": True,
            "about": "Trees built one after another, each correcting the "
                     "last. Usually the most accurate, and fast on large "
                     "files.", "params": _BOOST},
        "knn": {
            "label": "K-nearest neighbours", "max_rows": 100_000,
            "about": "Predicts from the most similar rows. Every prediction "
                     "compares against every training row.",
            "params": _KNN},
        "svm": {
            "label": "Support vector machine", "max_rows": 10_000,
            "about": "Finds the widest margin between the classes. Accurate "
                     "on small files; training time grows with the square of "
                     "the rows.", "params": _SVM},
        "naive_bayes": {
            "label": "Naive Bayes",
            "about": "Treats each feature as independent evidence. Very fast; "
                     "a useful sanity check.",
            "params": [_p("var_smoothing", "Smoothing",
                          [1e-11, 1e-9, 1e-7, 1e-5], 1e-9)]},
        "mlp": {
            "label": "Neural network",
            "about": "Layers of weighted sums with a bend between them. "
                     "Flexible; needs more data than the others.",
            "params": _MLP},
        "xgboost": {"label": "XGBoost", "package": "xgboost",
                    "about": _XGB_ABOUT, "params": _XGB},
        "lightgbm": {"label": "LightGBM", "package": "lightgbm",
                     "about": _LGBM_ABOUT, "params": _LGBM},
    },
    "regression": {
        "linear": {
            "label": "Linear regression",
            "about": "A straight-line fit of the target on the features. The "
                     "reference point for every other model.", "params": []},
        "ridge": {
            "label": "Ridge regression", "default": True,
            "about": "Linear regression with a penalty on large weights; "
                     "steadier when features move together.",
            "params": [_p("alpha", "Penalty strength", _ALPHA, 1.0)]},
        "lasso": {
            "label": "Lasso regression",
            "about": "Linear regression that can set a weight to exactly "
                     "zero, dropping a feature it does not need.",
            "params": [_p("alpha", "Penalty strength", _ALPHA, 1.0)]},
        "elastic_net": {
            "label": "Elastic net",
            "about": "A blend of the Ridge and Lasso penalties.",
            "params": [_p("alpha", "Penalty strength", _ALPHA, 1.0),
                       _p("l1_ratio", "Lasso share of the penalty",
                          [0.1, 0.25, 0.5, 0.75, 0.9], 0.5)]},
        "tree": {
            "label": "Decision tree",
            "about": "A short list of yes/no questions. Easy to read; one "
                     "tree on its own overfits easily.", "params": _TREE},
        "random_forest": {
            "label": "Random forest", "default": True,
            "about": "Many trees on random slices of the data, averaged. A "
                     "strong, low-maintenance default.",
            # A third of the features per split, not scikit-learn's all of
            # them: 154s against 49s per fit at 240,000 rows, on one core.
            "params": _forest(0.33)},
        "extra_trees": {
            "label": "Extra trees",
            "about": "A random forest with random split points: faster to "
                     "train and often as accurate.", "params": _forest(0.33)},
        "hist_gb": {
            "label": "Gradient boosting", "default": True,
            "about": "Trees built one after another, each correcting the "
                     "last. Usually the most accurate, and fast on large "
                     "files.", "params": _BOOST},
        "knn": {
            "label": "K-nearest neighbours", "max_rows": 100_000,
            "about": "Predicts from the most similar rows. Every prediction "
                     "compares against every training row.", "params": _KNN},
        "svm": {
            "label": "Support vector machine", "max_rows": 10_000,
            "about": "Fits within a margin of error and ignores what falls "
                     "inside it. Training time grows with the square of the "
                     "rows.", "params": [p for p in _SVM]},
        "mlp": {
            "label": "Neural network",
            "about": "Layers of weighted sums with a bend between them. "
                     "Flexible; needs more data than the others.",
            "params": _MLP},
        "xgboost": {"label": "XGBoost", "package": "xgboost",
                    "about": _XGB_ABOUT, "params": _XGB},
        "lightgbm": {"label": "LightGBM", "package": "lightgbm",
                     "about": _LGBM_ABOUT, "params": _LGBM},
    },
    "clustering": {
        "kmeans": {
            "label": "K-means", "default": True,
            "about": "Splits the rows into k groups, each row to its nearest "
                     "group centre.",
            "params": [_p("n_clusters", "Clusters", _K, 4),
                       _p("n_init", "Restarts (best one kept)",
                          [1, 3, 5, 10], 3)]},
        "gmm": {
            "label": "Gaussian mixture",
            "about": "Like k-means with soft, oval groups: each row gets a "
                     "probability of belonging to each.",
            "params": [_p("n_components", "Clusters", _K, 4),
                       _p("covariance_type", "Group shape",
                          ["diag", "full", "tied", "spherical"], "diag",
                          kind="choice",
                          show={"diag": "Axis-aligned ovals",
                                "full": "Any oval (slow on many columns)",
                                "tied": "One shared oval",
                                "spherical": "Circles"})]},
        "agglomerative": {
            "label": "Agglomerative", "max_rows": 5_000,
            "about": "Merges the closest rows, then the closest groups, until "
                     "k remain. Memory grows with the square of the rows.",
            "params": [_p("n_clusters", "Clusters", _K, 4),
                       _p("linkage", "Merge rule",
                          ["ward", "complete", "average", "single"], "ward",
                          kind="choice",
                          show={"ward": "Least added variance (Ward)",
                                "complete": "Farthest members",
                                "average": "Average distance",
                                "single": "Nearest members"})]},
    },
}

BASELINE_LABEL = "Baseline (no model)"


STACK = ml_recipe.STACK
STACK_TOP = (2, 3, 4, 5)        # how many of the best models a stack holds
# XGBoost learns class codes, not labels; inside a stack it would see labels.
STACK_LEAVES_OUT = ("xgboost",)


def reads_gaps(res):
    """Whether a trained model reads a missing number as it is."""
    if res["key"] == STACK:
        return all(m["model"] in GAP_MODELS
                   for m in res["params"]["members"])
    return res["key"] in GAP_MODELS


def model_label(task, key):
    if key == BASELINE:
        return BASELINE_LABEL
    if key == STACK:
        return "Stack of the top models"
    return MODELS[task][key]["label"]


def default_params(task, key):
    return {p["name"]: p["default"] for p in MODELS[task][key]["params"]}


_PACKAGE_PROBLEMS = {}


def package_problem(name):
    """None when an optional package imports here, otherwise why not.

    Imported once per process. A booster can be installed and still fail:
    LightGBM links the system's OpenMP library, which a slim image lacks
    (OSError), and XGBoost raises its own error, a ValueError, when its
    native library will not load (libomp on a Mac without Homebrew's).
    """
    if name not in _PACKAGE_PROBLEMS:
        label = {"xgboost": "XGBoost", "lightgbm": "LightGBM",
                 "shap": "SHAP"}.get(name, name)
        try:
            importlib.import_module(name)
            _PACKAGE_PROBLEMS[name] = None
        except ImportError:
            _PACKAGE_PROBLEMS[name] = (f"{label} is not installed here; "
                                       f"pip install -r requirements.txt "
                                       f"adds it.")
        except OSError:
            _PACKAGE_PROBLEMS[name] = (f"{label} is installed but cannot load "
                                       f"its OpenMP library: apt-get install "
                                       f"libgomp1 on Debian or Ubuntu, brew "
                                       f"install libomp on a Mac.")
        except Exception:
            _PACKAGE_PROBLEMS[name] = (
                f"{label} is installed but could not be loaded here."
                + (" On a Mac it also needs brew install libomp."
                   if name == "xgboost" else ""))
    return _PACKAGE_PROBLEMS[name]


def unavailable(task, n_rows) -> dict:
    """{model: reason} for the models this file is too large for, or whose
    optional package is not here."""
    out = {}
    for key, m in MODELS[task].items():
        missing = m.get("package") and package_problem(m["package"])
        if missing:
            out[key] = missing
            continue
        limit = m.get("max_rows")
        if limit and n_rows > limit:
            out[key] = (f"{m['label']} is offered up to {limit:,} rows; this "
                        f"file has {n_rows:,}. Every row is used, so it is "
                        f"not run on a subset.")
    return out


def show_value(spec, value) -> str:
    if value in spec["show"]:
        return spec["show"][value]
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


# ── metrics ─────────────────────────────────────────────────────────────
# (key, label, higher is better)
METRICS = {
    "binary": [("roc_auc", "ROC AUC", True),
               ("average_precision", "PR AUC", True),
               ("f1", "F1", True),
               ("balanced_accuracy", "Balanced accuracy", True),
               ("accuracy", "Accuracy", True),
               ("precision", "Precision", True),
               ("recall", "Recall", True),
               ("log_loss", "Log loss", False)],
    "multiclass": [("f1_macro", "F1 (macro)", True),
                   ("balanced_accuracy", "Balanced accuracy", True),
                   ("accuracy", "Accuracy", True),
                   ("roc_auc_ovr", "ROC AUC (one vs rest)", True),
                   ("precision_macro", "Precision (macro)", True),
                   ("recall_macro", "Recall (macro)", True),
                   ("log_loss", "Log loss", False)],
    "regression": [("r2", "R²", True), ("rmse", "RMSE", False),
                   ("mae", "Mean absolute error", False),
                   ("medae", "Median absolute error", False),
                   ("mape", "Mean absolute % error", False),
                   ("wape", "Weighted absolute % error (WAPE)", False)],
    "clustering": [("silhouette", "Silhouette", True),
                   ("calinski_harabasz", "Calinski-Harabasz", True),
                   ("davies_bouldin", "Davies-Bouldin", False)],
}
# The metrics a person picks from to choose the best model; the rest are
# reported alongside.
SELECTABLE = {"binary": ("roc_auc", "average_precision", "f1",
                         "balanced_accuracy", "accuracy", "log_loss"),
              "multiclass": ("f1_macro", "balanced_accuracy", "accuracy",
                             "roc_auc_ovr", "log_loss"),
              "regression": ("r2", "rmse", "mae", "mape", "wape")}


def metric_family(task, n_classes=None):
    if task == "classification":
        return "binary" if n_classes == 2 else "multiclass"
    return task


def metric_label(family, key):
    return next(lbl for k, lbl, _ in METRICS[family] if k == key)


def higher_is_better(family, key):
    return next(up for k, _, up in METRICS[family] if k == key)


def classification_metrics(y_true, y_pred, proba, classes, positive=None):
    """Every classification number the page reports, from one set of
    predictions. A number that cannot be computed (an AUC with one class
    present) is None, never a stand-in value."""
    classes = list(classes)
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    # The class codes 0 to k - 1, not the labels. scikit-learn sorts its
    # labels on every call (sorting 300,000 strings was 5.8 of 7.7 seconds),
    # and it pairs probability columns with labels in sorted order, which
    # for text is not the model's: "10" sorts before "2". The codes are the
    # column order of `proba` by construction.
    index = {c: i for i, c in enumerate(classes)}
    t = pd.Series(y_true).map(index)
    p = pd.Series(y_pred).map(index)
    if t.isna().any() or p.isna().any():
        t, p, labels, pos_label = y_true, y_pred, classes, positive
    else:
        t, p = t.to_numpy(dtype="int64"), p.to_numpy(dtype="int64")
        labels, pos_label = list(range(len(classes))), index.get(positive)
    out = {"accuracy": float(accuracy_score(t, p)),
           "balanced_accuracy": float(balanced_accuracy_score(t, p))}
    present = set(y_true)
    if len(classes) == 2:
        pos = classes.index(positive)
        truth = y_true == positive
        kw = dict(pos_label=pos_label, zero_division=0)
        out["precision"] = float(precision_score(t, p, **kw))
        out["recall"] = float(recall_score(t, p, **kw))
        out["f1"] = float(f1_score(t, p, **kw))
        both = 0 < truth.sum() < len(truth)
        out["roc_auc"] = (float(roc_auc_score(truth, proba[:, pos]))
                          if both else None)
        out["average_precision"] = (
            float(average_precision_score(truth, proba[:, pos]))
            if truth.any() else None)
    else:
        kw = dict(average="macro", zero_division=0, labels=labels)
        out["precision_macro"] = float(precision_score(t, p, **kw))
        out["recall_macro"] = float(recall_score(t, p, **kw))
        out["f1_macro"] = float(f1_score(t, p, **kw))
        out["roc_auc_ovr"] = (
            float(roc_auc_score(t, proba, multi_class="ovr", labels=labels))
            if present == set(classes) else None)
    out["log_loss"] = float(log_loss(t, proba, labels=labels))
    return out


# Metrics read from probabilities alone: scoring one of them needs no labels.
PROBABILITY_METRICS = ("roc_auc", "average_precision", "log_loss",
                       "roc_auc_ovr")


def one_metric(task, key, X, y_true, pipe, classes, positive):
    """One metric of a fitted model on (X, y): a single prediction pass.

    Permutation importance scores the model once per feature and repeat;
    computing every metric and both prediction passes each time made it three
    times slower than it needs to be at 300,000 rows.
    """
    if task == "regression":
        return regression_metrics(y_true, pipe.predict(X))[key]
    y_true = np.asarray(y_true, dtype=object)
    if key in PROBABILITY_METRICS:
        proba = _proba(pipe, X, classes)
        if key == "log_loss":
            return float(log_loss(y_true, proba, labels=list(classes)))
        if key == "roc_auc_ovr":
            return (float(roc_auc_score(y_true, proba, multi_class="ovr",
                                        labels=list(classes)))
                    if set(y_true) == set(classes) else None)
        truth = y_true == positive
        score = proba[:, list(classes).index(positive)]
        if key == "roc_auc":
            return (float(roc_auc_score(truth, score))
                    if 0 < truth.sum() < len(truth) else None)
        return float(average_precision_score(truth, score)) \
            if truth.any() else None
    pred = ml_recipe.predicted_labels(pipe, X, classes)
    if key == "accuracy":
        return float(np.mean(pred == y_true))
    if key == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, pred))
    one = dict(pos_label=positive, zero_division=0)
    macro = dict(average="macro", zero_division=0, labels=list(classes))
    fn, kw = {"f1": (f1_score, one), "precision": (precision_score, one),
              "recall": (recall_score, one), "f1_macro": (f1_score, macro),
              "precision_macro": (precision_score, macro),
              "recall_macro": (recall_score, macro)}[key]
    return float(fn(y_true, pred, **kw))


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    err = y_pred - y_true
    total = float(((y_true - y_true.mean()) ** 2).sum())
    nonzero = y_true != 0
    return {
        "r2": (1 - float((err ** 2).sum()) / total) if total > 0 else None,
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(np.abs(err).mean()),
        "medae": float(np.median(np.abs(err))),
        # Over the rows with a non-zero target: a percentage of zero is not
        # a number, and a stand-in epsilon turns one such row into billions.
        "mape": (float((np.abs(err[nonzero]) / np.abs(y_true[nonzero])).mean()
                       * 100) if nonzero.any() else None),
        # All the errors over all the actuals: days with zero sales do not
        # blow it up the way they do a mean of percentages.
        "wape": (float(np.abs(err).sum() / np.abs(y_true).sum() * 100)
                 if np.abs(y_true).sum() > 0 else None),
    }


def clustering_metrics(z, codes):
    """Cluster quality on the matrix the model clustered.

    Silhouette compares every pair of rows, so above `SILHOUETTE_MAX_ROWS`
    it is not computed rather than estimated from a subset.
    """
    k = len(np.unique(codes))
    out = {"silhouette": None, "calinski_harabasz": None,
           "davies_bouldin": None}
    if k < 2 or k >= len(codes):
        return out
    out["calinski_harabasz"] = float(calinski_harabasz_score(z, codes))
    out["davies_bouldin"] = float(davies_bouldin_score(z, codes))
    if len(codes) <= SILHOUETTE_MAX_ROWS:
        out["silhouette"] = float(silhouette_score(z, codes))
    return out


# ── profiling ───────────────────────────────────────────────────────────
def _numeric_like(series):
    return schema.detect_dtype(series) == "numeric"


def _hint(name, hints):
    low = str(name).lower().replace(" ", "_")
    return any(h in low for h in hints)


def _codes(series):
    return pd.factorize(series, use_na_sentinel=False)[0]


def _same_information(df, columns):
    """{later column: earlier column} for categorical pairs that are one
    relabelling of each other (Region_ID and Region_Name): one-hot
    encoding both doubles the columns and adds nothing."""
    codes = {c: _codes(df[c]) for c in columns}
    levels = {c: int(codes[c].max()) + 1 if len(codes[c]) else 0
              for c in columns}
    out = {}
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            if b in out or levels[a] != levels[b] or levels[a] < 2:
                continue
            pair = codes[a].astype("int64") * levels[b] + codes[b]
            if len(np.unique(pair)) == levels[a]:
                out[b] = a
    return out


def profile(df) -> pd.DataFrame:
    """One row per column: its type, size, gaps, and whether it is used.

    `use` is the default for the feature list; `note` says why a column is
    off by default, in words the configure page shows as they are.
    """
    n = len(df)
    rows = []
    for col in df.columns:
        s = df[col]
        dtype = "numeric" if _numeric_like(s) else "categorical"
        uniq = int(s.nunique(dropna=True))
        miss = int(s.isna().sum())
        role = schema._guess_role(str(col), dtype)
        note, use, kind = "", True, "feature"
        base = schema.base_column(str(col))
        if base != str(col) and base in df.columns:
            note, use, kind = (f"A rescaled copy of {base}: {base} "
                               f"itself is enough.", False, "derived")
        elif uniq <= 1:
            note, use, kind = "Constant: carries no information.", False, \
                "constant"
        elif ml_prep.looks_like_date(s):
            # Read whole, a date is a category that never repeats in a new
            # file; its parts (month, weekday, days since) are what predict.
            done = f"{col}_month" in df.columns
            note, use, kind = ((f"A date: its features ({col}_month, "
                                f"{col}_days_since, …) are used instead.")
                               if done else
                               ("A date: Explore & prepare → Prepare → Turn "
                                "dates into numbers makes features from it."),
                               False, "date")
        elif role == "id" or (uniq == n and n > MIN_LABELLED and (
                dtype == "categorical"
                or pd.api.types.is_integer_dtype(s))):
            note, use, kind = ("Identifier: kept in the downloads, not used "
                               "to learn.", False, "id")
        elif _hint(col, POSTAL_HINTS):
            note, use, kind = ("Postal code: a label, not a quantity. The "
                               "territory or latitude and longitude carry the "
                               "geography.", False, "postal")
        elif dtype == "categorical" and uniq > MAX_ENCODE_LEVELS:
            note, use, kind = (f"{uniq:,} categories: too many to encode.",
                               False, "high_card")
        elif dtype == "categorical" and uniq > MAX_LEVELS:
            note, use, kind = (f"{uniq:,} categories: off by default, each "
                               f"becomes its own column when encoded.",
                               False, "many_levels")
        if miss and use:
            share = 100 * miss / n
            note = (f"{share:.1f}% missing, filled by the missing-value rule."
                    if share >= 0.1 else
                    f"{miss:,} missing, filled by the missing-value rule.")
        rows.append({"column": str(col), "type": dtype, "unique": uniq,
                     "missing": miss, "use": use, "kind": kind, "note": note})
    out = pd.DataFrame(rows)
    cats = [r["column"] for r in rows
            if r["type"] == "categorical" and r["use"]]
    for later, earlier in _same_information(df, cats).items():
        i = out.index[out["column"] == later][0]
        out.loc[i, ["use", "kind", "note"]] = [
            False, "duplicate",
            f"Same information as {earlier}: one of the two is enough."]
    return out


def target_candidates(prof, task) -> list:
    """Columns that can be this task's target, best guess first."""
    ok = prof[~prof["kind"].isin(["id", "constant", "postal"])]
    if task == "regression":
        rows = ok[(ok["type"] == "numeric") & (ok["unique"] > 2)]
        cols = list(rows["column"])
        # Money before counts: revenue or premium is the usual thing to
        # predict, whatever order the file lists its columns in.
        keyed = sorted((c for c in cols if _hint(c, VOLUME_HINTS)),
                       key=lambda c: min(i for i, h in enumerate(VOLUME_HINTS)
                                         if h in str(c).lower()
                                         .replace(" ", "_")))
        rest = [c for c in cols if c not in keyed and not _hint(c, GEO_HINTS)]
        geo = [c for c in cols if c not in keyed and c not in rest]
        return keyed + rest + geo
    if task != "classification":
        return []
    labelled = ok[(ok["unique"] >= 2) & (ok["unique"] <= MAX_CLASSES)]
    numeric_wide = ok[(ok["type"] == "numeric") & (ok["unique"] > MAX_CLASSES)]
    binary = list(labelled[labelled["unique"] == 2]["column"])
    named = [c for c in binary if _hint(c, OUTCOME_HINTS)]
    multi = list(labelled[labelled["unique"] > 2]["column"])
    first = named + [c for c in binary if c not in named]
    return first + multi + list(numeric_wide["column"])


def needs_rule(prof, target) -> bool:
    """A numeric target with too many values to be classes: it is turned
    into yes/no at a cut-off before it can be classified."""
    row = prof[prof["column"] == target].iloc[0]
    return row["type"] == "numeric" and row["unique"] > MAX_CLASSES


def resolve_rule(df, target, rule) -> dict:
    """{"at_least": cut-off, "describe": words} for a yes/no target cut from
    a numeric column — by the top share or at a value."""
    v = pd.to_numeric(df[target], errors="coerce").astype("float64")
    v = v[np.isfinite(v)]
    if rule["mode"] == "top_share":
        share = float(rule["value"])
        cut = float(v.quantile(1 - share / 100)) if len(v) else 0.0
        hit = float((v >= cut).mean() * 100) if len(v) else 0.0
        describe = (f"{target} of at least {cut:,.4g}: the top {hit:.1f}% "
                    f"(asked for {share:g}%; ties at the cut-off are kept "
                    f"together)" if abs(hit - share) >= 0.05 else
                    f"{target} of at least {cut:,.4g}: the top {share:g}%")
    else:
        cut = float(rule["value"])
        hit = float((v >= cut).mean() * 100) if len(v) else 0.0
        describe = f"{target} of at least {cut:,.4g}: {hit:.1f}% of rows"
    return {"at_least": cut, "describe": describe}


def default_positive(classes) -> str:
    """The class of interest: an obvious yes when there is one, otherwise
    the rarer class — usually the one worth finding."""
    for yes in ("1", "Yes", "yes", "Y", "y", "True", "true"):
        if yes in classes:
            return yes
    return classes[-1]


def leakage(df, target, task, features, rule=None) -> dict:
    """{feature: note} for columns that predict the target nearly on their
    own.

    A column recorded after the outcome, or computed from it, makes a model
    look far better than it will be on a new file. The test is deliberately
    simple: how much of the target one column explains by itself, with a
    numeric column cut into up to 100 equal-count bins so a deterministic
    threshold still shows (one mixed bin in a hundred).
    """
    recipe = {"task": task, "target": target, "target_rule": rule}
    y = ml_recipe.target_values(df, recipe)
    keep = y.notna()
    y = y[keep]
    n = len(y)
    if n < MIN_LABELLED:
        return {}
    bins = max(2, min(100, n // 20))
    out = {}
    for col in features:
        s = df.loc[keep, col]
        if s.notna().sum() < MIN_LABELLED:
            continue
        if _numeric_like(s) and s.nunique() > bins:
            num = pd.to_numeric(s, errors="coerce")
            key = pd.qcut(num.rank(method="first"), bins, labels=False)
        else:
            key = pd.Series(_codes(s), index=s.index)
        key = key.fillna(-1)
        if task == "classification":
            base = float(y.value_counts(normalize=True).iloc[0])
            purity = float(pd.crosstab(key, y).max(axis=1).sum() / n)
            if base < 1 and (1 - purity) <= LEAKAGE_ERROR_SHARE * (1 - base):
                out[col] = (f"Predicts the target almost on its own: alone it "
                            f"removes {100 * (purity - base) / (1 - base):.0f}"
                            f"% of the errors of guessing. A column recorded "
                            f"after the outcome, or derived from it, makes a "
                            f"model look better than it will be.")
        else:
            yv = y.astype("float64")
            total = float(((yv - yv.mean()) ** 2).sum())
            if total <= 0:
                continue
            within = float(((yv - yv.groupby(key).transform("mean")) ** 2)
                           .sum())
            explained = 1 - within / total
            if explained >= LEAKAGE_R2:
                out[col] = (f"Explains {100 * explained:.1f}% of the target's "
                            f"variation on its own. A column recorded after "
                            f"the outcome, or derived from it, makes a model "
                            f"look better than it will be.")
    return out


# ── configuration ───────────────────────────────────────────────────────
def search_combinations(model) -> int:
    search = model.get("search")
    if not search:
        return 1
    grid = 1
    for values in search["space"].values():
        grid *= max(1, len(values))
    if search["mode"] in ("random", "bayes"):
        return min(int(search.get("n_iter", 10)), grid)
    return grid


def plan(config) -> dict:
    """How many fits a configuration costs, before any is run."""
    folds = int(config["validation"]["folds"]) \
        if config["task"] != "clustering" else 0
    fits, combos = 0, {}
    for m in config["models"]:
        c = search_combinations(m)
        combos[m["key"]] = c
        fits += c * max(folds, 0) + 1
    if config["task"] != "clustering":
        fits += max(folds, 0) + 1          # the baseline
    return {"fits": fits, "combinations": combos, "folds": folds}


# Seconds per fit per 100,000 training rows, measured 2026-10-30 on four cores
# with the sample's 17 features (62 columns once encoded). An estimate to warn
# before a long run, not a promise: a two-core server takes about twice as
# long for the forests, and more features cost more.
_RATE = {"classification": {"baseline": 0.05, "logistic": 1.0, "tree": 1.0,
                            "random_forest": 2.6, "extra_trees": 2.2,
                            "hist_gb": 1.1, "naive_bayes": 0.5, "mlp": 8.6,
                            "xgboost": 2.2, "lightgbm": 1.6},
         "regression": {"baseline": 0.05, "linear": 0.3, "ridge": 0.3,
                        "lasso": 0.6, "elastic_net": 0.6, "tree": 1.0,
                        "random_forest": 5.3, "extra_trees": 3.5,
                        "hist_gb": 0.8, "mlp": 8.6, "xgboost": 1.75,
                        "lightgbm": 1.2},
         "clustering": {"kmeans": 0.6, "gmm": 0.9}}
# Models whose cost grows with the square of the rows: (rows, seconds).
_QUADRATIC = {"knn": (100_000, 10.0), "svm": (10_000, 30.0),
              "agglomerative": (5_000, 1.4)}
_SAMPLE_FEATURES = 17


def estimate_seconds(config, n_rows) -> float:
    """Roughly how long a configuration takes to train on `n_rows`."""
    task = config["task"]
    supervised = task != "clustering"
    share = 1 - float(config["validation"]["test_size"]) if supervised else 1
    n_train = n_rows * share
    folds = int(config["validation"]["folds"]) if supervised else 0
    width = max(1.0, (len(config["numeric"]) + len(config["categorical"]))
                / _SAMPLE_FEATURES)
    models = list(config["models"]) + (
        [{"key": BASELINE}] if supervised else [])
    total = 0.0
    for m in models:
        fits = search_combinations(m) * folds + 1
        if m["key"] in _QUADRATIC:
            rows, secs = _QUADRATIC[m["key"]]
            per = secs * (n_train / rows) ** 2
        else:
            per = _RATE[task].get(m["key"], 1.0) * n_train / 100_000
        total += fits * per * width
    return total


def base_recipe(config, rule=None):
    """The recipe every model in a run shares: everything but the model."""
    return {"task": config["task"], "target": config.get("target"),
            "target_rule": ({"at_least": rule["at_least"]} if rule else None),
            "numeric": list(config["numeric"]),
            "categorical": list(config["categorical"]),
            "prep": dict(config["prep"]),
            "validation": dict(config.get("validation") or {}),
            "seed": int(config.get("seed", 42)),
            "balance": bool(config.get("balance")),
            "resample": (config.get("resample") or "none")
            if config["task"] == "classification" else "none",
            "calibration": (config.get("calibration") or "none")
            if config["task"] == "classification" else "none",
            "model": None, "params": {}}


IMBALANCED = 0.20       # the smallest class below this share of rows


def smallest_share(y):
    """The smallest class's share of the rows with a target, or None."""
    counts = pd.Series(y).dropna().astype(str).value_counts()
    return float(counts.min() / counts.sum()) if len(counts) > 1 else None


def _classes(y):
    return sorted(pd.unique(y[pd.notna(y)]).tolist(), key=str)


def _gap_error(df, config):
    """With no fill for missing numbers, a model that cannot read a gap would
    fail on the first one; say which, and what to do, before anything runs."""
    if config["prep"].get("num_missing", "none") != "none" \
            or not config["numeric"]:
        return None
    task = config["task"]
    blocked = [m["key"] for m in config["models"]
               if m["key"] not in GAP_MODELS]
    if not blocked:
        return None
    gaps = ml_recipe.prepare_features(df, config["numeric"], []).isna().sum()
    gaps = gaps[gaps > 0].sort_values(ascending=False)
    if not len(gaps):
        return None
    listed = ", ".join(f"{c} ({n:,})" for c, n in gaps.head(3).items())
    if len(gaps) > 3:
        listed += f" and {len(gaps) - 3} more"
    names = [model_label(task, k) for k in blocked]
    who = (names[0] if len(names) == 1 else
           ", ".join(names[:-1]) + " and " + names[-1])
    readers = [MODELS[task][k]["label"] for k in GAP_MODELS
               if k in MODELS.get(task, {})
               and k not in unavailable(task, len(df))]
    advice = "Choose a fill under Missing numbers"
    if readers:
        advice += (", or use a model that reads gaps: "
                   + ", ".join(readers[:-1]) + " and " + readers[-1]
                   if len(readers) > 1 else
                   f", or use a model that reads gaps: {readers[0]}")
    return (f"{who} cannot read a missing number, and these features have "
            f"gaps (missing or infinite values): {listed}. {advice}.")


def validate(df, config) -> list:
    """Readable reasons a configuration cannot be trained, or []."""
    errors = []
    task = config["task"]
    features = list(config["numeric"]) + list(config["categorical"])
    n = len(df)
    if not features:
        errors.append("Choose at least one feature.")
    missing = [c for c in features if c not in df.columns]
    if missing:
        errors.append(f"Not in this file: {', '.join(missing)}.")
        return errors
    if not config["models"]:
        errors.append("Choose at least one model.")
    for key, why in unavailable(task, n).items():
        if any(m["key"] == key for m in config["models"]):
            errors.append(why)
    for c in config["categorical"]:
        levels = df[c].nunique(dropna=True)
        if levels > MAX_ENCODE_LEVELS:
            errors.append(f"{c} has {levels:,} categories; up to "
                          f"{MAX_ENCODE_LEVELS} can be encoded. Leave it out, "
                          f"or treat it as a number if it is one.")
    for c in features:
        if df[c].notna().sum() == 0:
            errors.append(f"{c} has no values.")
    gap_error = _gap_error(df, config)
    if gap_error:
        errors.append(gap_error)
    if task == "clustering":
        if n < 3:
            errors.append(f"Clustering needs at least 3 rows; this file has "
                          f"{n}.")
        for m in config["models"]:
            p = m["params"]
            k = int(p.get("n_clusters", p.get("n_components", 2)))
            if k >= n:
                errors.append(f"{model_label(task, m['key'])}: {k} clusters "
                              f"need more than {k} rows; this file has {n}.")
        pca = int(config["prep"].get("pca") or 0)
        if pca and pca > len(features):
            errors.append(f"Principal components ({pca}) cannot exceed the "
                          f"number of features ({len(features)}).")
        elif pca and pca >= n:
            errors.append(f"Principal components ({pca}) need more than "
                          f"{pca} rows; this file has {n}.")
        return errors

    target = config.get("target")
    if not target or target not in df.columns:
        errors.append("Choose a target.")
        return errors
    if target in features:
        errors.append(f"{target} is the target, so it cannot be a feature "
                      f"too.")
    rule = config.get("target_rule")
    recipe = base_recipe(config, resolve_rule(df, target, rule)
                         if rule else None)
    y = ml_recipe.target_values(df, recipe)
    labelled = int(y.notna().sum())
    if labelled < MIN_LABELLED:
        errors.append(f"Training needs at least {MIN_LABELLED} rows with a "
                      f"value for {target}; this file has {labelled:,}.")
        return errors
    val = config["validation"]
    k = int(val["folds"])
    tuned = [m for m in config["models"] if m.get("search")]
    if tuned and k < 2:
        errors.append("Tuning compares settings by cross-validation: choose "
                      "3 or more folds, or use the settings as entered.")
    if task == "classification":
        counts = y.value_counts()
        if len(counts) < 2:
            errors.append(f"{target} has one class among the rows with a "
                          f"value; there is nothing to tell apart.")
        elif len(counts) > MAX_CLASSES:
            errors.append(f"{target} has {len(counts):,} classes; up to "
                          f"{MAX_CLASSES} can be predicted.")
        else:
            need = max(k, 2) + 1
            small = counts[counts < need]
            for cls, cnt in small.items():
                errors.append(f"Class {cls} of {target} has {cnt:,} row"
                              f"{'' if cnt == 1 else 's'}; at least {need} "
                              f"are needed to train and test"
                              + (f" with {k} folds." if k >= 2 else "."))
    else:
        if y.dropna().nunique() < 2:
            errors.append(f"{target} has one value among the rows with a "
                          f"value; there is nothing to predict.")
        if config["prep"].get("log_target") and (y.dropna() < 0).any():
            errors.append(f"{target} has negative values, so it cannot be "
                          f"modelled on a log scale.")
    share = float(val["test_size"])
    group = val.get("group")
    time = val.get("time")
    if time:
        errors += _time_problems(df, recipe, y, time, share, k, group)
    elif group:
        g = ml_recipe.label_text(df[group])[y.notna()].fillna("(blank)")
        n_groups = int(g.nunique())
        # GroupShuffleSplit sends the test share of the groups, rounded up,
        # to the test side; the folds then need one group each.
        left = n_groups - int(np.ceil(share * n_groups))
        if left < max(k, 1):
            errors.append(f"Holding out whole {group} groups leaves {left} "
                          f"of its {n_groups} groups for training, and "
                          f"training needs at least {max(k, 1)}. Choose "
                          f"fewer folds, a smaller test share or a column "
                          f"with more groups.")
    elif task == "classification" and val.get("stratify"):
        n_classes = int(y.dropna().nunique())
        n_test = int(np.ceil(share * labelled))
        if 1 < n_classes <= MAX_CLASSES and (
                n_test < n_classes or labelled - n_test < n_classes):
            errors.append(f"A {share:.0%} test share leaves {n_test:,} test "
                          f"rows for {n_classes} classes; each class needs "
                          f"a row on both sides. Raise the test share or turn "
                          f"stratification off.")
    for m in config["models"]:
        if m["key"] == "knn":
            per_fold = labelled * (1 - float(val["test_size"])) * (
                (k - 1) / k if k >= 2 else 1)
            if m["params"].get("n_neighbors", 1) > per_fold:
                errors.append(f"K-nearest neighbours asks for "
                              f"{m['params']['n_neighbors']} neighbours; each "
                              f"fit has about {int(per_fold):,} rows.")
    fits = plan(config)["fits"]
    if fits > MAX_FITS:
        errors.append(f"This plan trains {fits:,} models; the limit is "
                      f"{MAX_FITS}. Try fewer values, fewer models or random "
                      f"search.")
    return errors


def _time_problems(df, recipe, y, time, share, k, group):
    """What stops a time split on `time`, in words."""
    if time not in df.columns:
        return [f"{time} is not in this file."]
    if group:
        return ["Hold out either by time or by group, not both: with a "
                "time split the test rows are simply the latest ones."]
    col = df[time]
    if not _numeric_like(col):
        recipe = {**recipe, "validation": {
            **recipe["validation"], "time_format": ml_prep.date_format(col)}}
    times = ml_recipe.time_values(df, recipe)[y.notna().to_numpy()]
    gaps = int(np.isnan(times).sum())
    if gaps:
        return [f"{gaps:,} rows with a target have no {time} (or one that "
                f"does not read as a date or number). Drop them first "
                f"(Explore & prepare → Prepare → Drop rows with a gap), so "
                f"no row lands on the wrong side of the split."]
    distinct = len(np.unique(times))
    if distinct < max(k, 1) + 2:
        return [f"{time} has {distinct} distinct value"
                f"{'' if distinct == 1 else 's'}: a time split needs at "
                f"least {max(k, 1) + 2} (one for the test, and one more "
                f"than the folds for training)."]
    train, test = ml_recipe.split(y.dropna().to_numpy(), None, recipe,
                                  times)
    if len(train) == 0:
        return [f"Every row shares the latest {time}, so nothing is left "
                f"to train on. Choose a smaller test share."]
    return []


# ── training ────────────────────────────────────────────────────────────
class _Ticker:
    """Progress as (done, total, message); the page draws it."""

    def __init__(self, total, fn):
        self.done, self.total, self.fn = 0, max(1, total), fn

    def step(self, text, n=1):
        self.done += n
        if self.fn:
            self.fn(min(self.done, self.total) / self.total, text)

    def say(self, text):
        if self.fn:
            self.fn(min(self.done, self.total) / self.total, text)


def _proba(pipe, X, classes):
    """Class probabilities in `classes` order; a class a fold never saw gets
    a column of zeros rather than shifting the others.

    Each row is rescaled to sum to exactly one: XGBoost and LightGBM answer
    in float32, a few parts in ten million off, which scikit-learn's log
    loss warns about on every call.
    """
    raw = pipe.predict_proba(X)
    seen = ml_recipe.model_classes(pipe, classes)
    out = np.zeros((len(X), len(classes)))
    for j, c in enumerate(classes):
        if c in seen:
            out[:, j] = raw[:, seen.index(c)]
    total = out.sum(axis=1, keepdims=True)
    return out / np.where(total > 0, total, 1.0)


def flag(proba, classes, positive, threshold):
    """Two-class labels at a chosen cut-off: the class of interest where its
    probability reaches `threshold`, the other class elsewhere."""
    j = classes.index(positive)
    other = classes[1 - j]
    return np.where(proba[:, j] >= threshold, positive, other).astype(object)


def with_cutoff(pred, proba, classes, positive, threshold):
    """`pred`, or the labels at the model's saved cut-off when it has one
    (two classes only; without one, the likelier class wins)."""
    if threshold is None or proba is None or len(classes) != 2 \
            or positive not in classes:
        return pred
    return flag(proba, list(classes), positive, float(threshold))


def score_predictions(task, y_true, pred, proba, classes, positive):
    if task == "classification":
        return classification_metrics(y_true, pred, proba, classes, positive)
    return regression_metrics(y_true, pred)


def predict(task, pipe, X, classes):
    if task == "classification":
        return (ml_recipe.predicted_labels(pipe, X, classes),
                _proba(pipe, X, classes))
    return np.asarray(pipe.predict(X), dtype="float64"), None


def _cross_validate(recipe, X, y, splits, classes, positive, ticker, what,
                    on_fold=None):
    """Out-of-fold predictions and per-fold metrics, on fixed folds shared by
    every model and setting in the run — so their scores compare like for
    like. `on_fold(i, scores so far)` may stop it early by raising."""
    task = recipe["task"]
    oof = (np.empty(len(y), dtype=object) if task == "classification"
           else np.full(len(y), np.nan))
    oof_p = np.zeros((len(y), len(classes))) if classes else None
    per_fold = []
    for i, (tr, va) in enumerate(splits, 1):
        pipe = ml_recipe.build_pipeline(ml_recipe.with_caps(recipe,
                                                            X.iloc[tr]))
        Xf, yf = ml_recipe.resample(recipe, X.iloc[tr], y[tr])
        pipe.fit(Xf, ml_recipe.fit_labels(recipe, yf, classes))
        pred, proba = predict(task, pipe, X.iloc[va], classes)
        oof[va] = pred
        if proba is not None:
            oof_p[va] = proba
        per_fold.append(score_predictions(task, y[va], pred, proba, classes, positive))
        ticker.step(f"{what}: fold {i} of {len(splits)}")
        if on_fold is not None:
            on_fold(i, per_fold)
    scores = {k: [f[k] for f in per_fold] for k in per_fold[0]}
    return {"scores": scores, "oof": oof, "oof_proba": oof_p}


def _mean(values):
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def _better(family, metric, a, b) -> bool:
    """Whether score a beats score b (None loses to anything)."""
    if a is None:
        return False
    if b is None:
        return True
    return a > b if higher_is_better(family, metric) else a < b


def _fit_supervised(m, base, data, splits, family, metric, ticker):
    task = base["task"]
    key = m["key"]
    label = m.get("label") or model_label(task, key)
    params = dict(m.get("params") or {})
    recipe = {**base, "model": key, "params": params}
    if key == BASELINE:
        # Guessing the plain mean, so the baseline's R² reads as zero: on a
        # log scale it would guess a geometric mean instead.
        recipe["prep"] = {**base["prep"], "log_target": False}
    classes, positive = data["classes"], data["positive"]
    Xtr, ytr = data["Xtr"], data["ytr"]
    search_rows, cv = None, None
    t0 = time.perf_counter()
    search = m.get("search")
    mode = (search or {}).get("mode")
    if mode == "bayes" and package_problem("optuna"):
        mode = "random"                # same number of settings, at random
    if search and splits and mode == "bayes":
        best, search_rows = _bayes_search(m, recipe, params, data, splits,
                                          family, metric, ticker, label)
        _, recipe, cv = best
    elif search and splits:
        space = {k: list(v) for k, v in search["space"].items() if v}
        cands = (list(ParameterGrid(space)) if search["mode"] == "grid" else
                 list(ParameterSampler(space, n_iter=search_combinations(m),
                                       random_state=base["seed"])))
        best = None
        search_rows = []
        for i, cand in enumerate(cands, 1):
            r = {**recipe, "params": {**params, **cand}}
            res = _cross_validate(r, Xtr, ytr, splits, classes, positive,
                                  ticker, f"{label}: setting {i} of "
                                          f"{len(cands)}")
            score = _mean(res["scores"][metric])
            search_rows.append({**cand, "cv_mean": score,
                                "cv_std": float(np.std(
                                    [v for v in res["scores"][metric]
                                     if v is not None] or [0]))})
            if best is None or _better(family, metric, score, best[0]):
                best = (score, r, res)
        _, recipe, cv = best
    elif splits:
        cv = _cross_validate(recipe, Xtr, ytr, splits, classes, positive,
                             ticker, label)
    recipe = ml_recipe.with_caps(recipe, Xtr)
    pipe = ml_recipe.build_pipeline(recipe)
    t1 = time.perf_counter()
    Xf, yf = ml_recipe.resample(recipe, Xtr, ytr)
    pipe.fit(Xf, ml_recipe.fit_labels(recipe, yf, classes))
    fit_s = time.perf_counter() - t1
    ticker.step(f"{label}: final fit")
    t2 = time.perf_counter()
    pred_te, proba_te = predict(task, pipe, data["Xte"], classes)
    predict_s = time.perf_counter() - t2
    test = score_predictions(task, data["yte"], pred_te, proba_te, classes, positive)
    # Every row gets a prediction: test rows from the model that never saw
    # them, training rows from their fold's model when there are folds, rows
    # with no target from the final model.
    n = data["n"]
    pred_all = (np.empty(n, dtype=object) if task == "classification"
                else np.full(n, np.nan))
    proba_all = np.zeros((n, len(classes))) if classes else None
    tr_rows, te_rows = data["train_rows"], data["test_rows"]
    if cv is not None:
        pred_all[tr_rows] = cv["oof"]
        if proba_all is not None:
            proba_all[tr_rows] = cv["oof_proba"]
        gone = ~data.get("covered", np.ones(len(tr_rows), dtype=bool))
        if gone.any():
            p, pp = predict(task, pipe, Xtr.iloc[np.flatnonzero(gone)],
                            classes)
            pred_all[tr_rows[gone]] = p
            if proba_all is not None:
                proba_all[tr_rows[gone]] = pp
    else:
        p, pp = predict(task, pipe, Xtr, classes)
        pred_all[tr_rows] = p
        if proba_all is not None:
            proba_all[tr_rows] = pp
    pred_all[te_rows] = pred_te
    if proba_all is not None:
        proba_all[te_rows] = proba_te
    if len(data["unl_rows"]):
        p, pp = predict(task, pipe, data["Xunl"], classes)
        pred_all[data["unl_rows"]] = p
        if proba_all is not None:
            proba_all[data["unl_rows"]] = pp
    table = None
    if search_rows:
        table = pd.DataFrame(search_rows)
        table = table.sort_values("cv_mean",
                                  ascending=not higher_is_better(family,
                                                                 metric),
                                  na_position="last").reset_index(drop=True)
    return {"key": key, "label": label, "recipe": recipe,
            "params": recipe["params"], "tuned": bool(search_rows),
            "search": table, "cv": cv["scores"] if cv else None,
            "test": test, "pipeline": pipe, "pred": pred_all,
            "proba": proba_all, "fit_s": fit_s, "predict_s": predict_s,
            "total_s": time.perf_counter() - t0, "error": None}


def _bayes_search(m, recipe, params, data, splits, family, metric, ticker,
                  label):
    """Settings chosen by Optuna's TPE sampler, which learns from the scores
    of the settings already tried where to look next. Every setting is
    scored by the same cross-validation as grid and random search; one that
    trails the median of earlier settings after a fold is stopped early,
    and the search ends at `n_iter` settings or `seconds`, whichever comes
    first. Seeded, though a time limit makes the count depend on speed.

    Returns ((score, recipe, cv) of the best, the rows of the search table).
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    search = m["search"]
    space = {k: list(v) for k, v in search["space"].items() if v}
    total = search_combinations(m)
    up = higher_is_better(family, metric)
    study = optuna.create_study(
        direction="maximize" if up else "minimize",
        sampler=optuna.samplers.TPESampler(seed=recipe.get("seed", 0)),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                           n_warmup_steps=1))
    tried, rows, best = {}, [], []
    classes, positive = data["classes"], data["positive"]

    def objective(trial):
        cand = {k: trial.suggest_categorical(k, v) for k, v in space.items()}
        key = repr(sorted(cand.items(), key=lambda kv: kv[0]))
        if key in tried:                   # the sampler came back to one
            return tried[key]
        r = {**recipe, "params": {**params, **cand}}

        def on_fold(i, per_fold):
            so_far = _mean([f[metric] for f in per_fold])
            if so_far is None:
                return
            trial.report(so_far, i)
            if trial.should_prune():
                rows.append({**cand, "cv_mean": None, "cv_std": None,
                             "stopped_early": True})
                tried[key] = so_far
                raise optuna.TrialPruned()

        res = _cross_validate(r, data["Xtr"], data["ytr"], splits, classes,
                              positive, ticker,
                              f"{label}: setting {trial.number + 1} of "
                              f"{total}", on_fold)
        values = [v for v in res["scores"][metric] if v is not None]
        score = _mean(values)
        rows.append({**cand, "cv_mean": score,
                     "cv_std": float(np.std(values or [0])),
                     "stopped_early": False})
        if score is not None and (not best or _better(family, metric, score,
                                                      best[0][0])):
            best[:] = [(score, r, res)]
        tried[key] = float("nan") if score is None else score
        return tried[key]

    seconds = float(search.get("seconds") or 0) or None
    study.optimize(objective, n_trials=total, timeout=seconds)
    if not best:
        raise ValueError("No setting could be scored.")
    return best[0], rows


def _failed(key, task, ref):
    return {"key": key, "label": model_label(task, key), "error": ref}


def train(df, config, progress=None, source=None, steps=None) -> dict:
    """Train every model in `config` on `df`; returns the run.

    One model failing does not end the run: it is reported with a reference
    and logged, and the others train as normal. `progress(fraction, text)`
    is called as fits finish.

    With `source` (the file as loaded) and `steps` (the preparation that
    turned it into `df`), the steps' learnt values (fills, caps, bands) are
    worked out again from the training rows once the split is known, and
    the run keeps those steps (`prep_steps`) and that file (`frame`).
    """
    t0 = time.perf_counter()
    task = config["task"]
    ticker = _Ticker(plan(config)["fits"], progress)
    if task == "clustering":
        run = _train_clustering(df, config, ticker)
    else:
        run = _train_supervised(df, config, ticker, source, steps)
    run["seconds"] = time.perf_counter() - t0
    run["sklearn"] = sklearn.__version__
    run["created"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    ok = [r for r in run["results"] if not r.get("error")]
    log.event(_log, "ml_trained", task=task,
              models=[r["key"] for r in run["results"]],
              failed=len(run["results"]) - len(ok), rows=len(df),
              features=len(config["numeric"]) + len(config["categorical"]),
              folds=int((config.get("validation") or {}).get("folds") or 0),
              tuned=sum(1 for m in config["models"] if m.get("search")),
              ms=round(run["seconds"] * 1000))
    return run


def _refit_frame(df, config, source, steps, train_rows):
    """(steps, file) with the steps' learnt values from the training rows,
    or (None, None) when no step learns or the rows would not line up."""
    if source is None or not any(s.get("learn") for s in steps or []):
        return None, None
    features = set(config["numeric"]) | set(config["categorical"])
    learnt = ml_prep.refit(source, steps, df.index[train_rows], features)
    again = ml_prep.apply_steps(source, learnt)
    if not again.index.equals(df.index):
        return None, None
    return learnt, again


def _train_supervised(df, config, ticker, source=None, steps=None):
    task = config["task"]
    target = config["target"]
    rule = (resolve_rule(df, target, config["target_rule"])
            if config.get("target_rule") else None)
    base = base_recipe(config, rule)
    X = ml_recipe.prepare_features(df, base["numeric"], base["categorical"])
    y_all = ml_recipe.target_values(df, base)
    labelled = y_all.notna().to_numpy()
    lab_rows = np.flatnonzero(labelled)
    unl_rows = np.flatnonzero(~labelled)
    y = y_all.to_numpy()[labelled]
    if task == "regression":
        y = y.astype("float64")
    group = base["validation"].get("group")
    groups = (ml_recipe.label_text(df[group]).to_numpy()[labelled]
              if group else None)
    if groups is not None:
        groups = np.where(pd.isna(groups), "(blank)", groups).astype(str)
    times = None
    if base["validation"].get("time"):
        col = base["validation"]["time"]
        if "time_format" not in base["validation"]:
            # Fixed now, so the script and a retrain read the dates alike.
            base["validation"]["time_format"] = ml_prep.date_format(df[col]) \
                if not _numeric_like(df[col]) else None
        times = ml_recipe.time_values(df, base)[labelled]
    tr, te = ml_recipe.split(y, groups, base, times)
    learnt, again = _refit_frame(df, config, source, steps, lab_rows[tr])
    if again is not None:
        # The features as the training rows alone would prepare them; the
        # target, the split and the rows are unchanged.
        df = again
        X = ml_recipe.prepare_features(df, base["numeric"],
                                       base["categorical"])
    classes = [str(c) for c in _classes(y)] if task == "classification" \
        else []
    positive = None
    if len(classes) == 2:
        positive = ml_recipe.YES if rule else config.get("positive")
        if positive not in classes:
            positive = default_positive(classes)
    family = metric_family(task, len(classes))
    metric = config.get("metric")
    if metric not in {k for k, _, _ in METRICS[family]}:
        metric = SELECTABLE[family][0]
    Xl = X.iloc[lab_rows]
    data = {"n": len(df), "classes": classes, "positive": positive,
            "Xtr": Xl.iloc[tr], "ytr": y[tr], "Xte": Xl.iloc[te],
            "yte": y[te], "Xunl": X.iloc[unl_rows],
            "train_rows": lab_rows[tr], "test_rows": lab_rows[te],
            "unl_rows": unl_rows}
    k = int(base["validation"]["folds"])
    splits = []
    if k >= 2:
        splitter = ml_recipe.folds(base, groups is not None,
                                   None if times is None else times[tr])
        splits = (splitter if isinstance(splitter, list) else
                  list(splitter.split(data["Xtr"], data["ytr"],
                                      groups[tr] if groups is not None
                                      else None)))
    # Walk-forward folds never validate the earliest rows: those keep an
    # in-sample prediction and say so.
    covered = np.zeros(len(tr), dtype=bool)
    for _, va in splits:
        covered[va] = True
    data["covered"] = covered
    results = []
    for m in [{"key": BASELINE, "params": {}}] + list(config["models"]):
        try:
            results.append(_fit_supervised(m, base, data, splits, family,
                                           metric, ticker))
        except Exception:
            ref = log.incident_id()
            log.event(_log, "ml_model_failed", level=logging.ERROR,
                      exc_info=True, model=m["key"], task=task, ref=ref)
            results.append(_failed(m["key"], task, ref))
    stack = _stack_member_list(results, config, family, metric, bool(splits))
    if stack:
        try:
            results.append(_fit_supervised(stack, base, data, splits, family,
                                           metric, ticker))
        except Exception:
            ref = log.incident_id()
            log.event(_log, "ml_model_failed", level=logging.ERROR,
                      exc_info=True, model=STACK, task=task, ref=ref)
            results.append(_failed(STACK, task, ref))
    split = np.full(len(df), "no target", dtype=object)
    split[data["train_rows"]] = np.where(covered, "train (out of fold)",
                                         "train (in sample)")
    split[data["test_rows"]] = "test"
    time_split = None
    if times is not None:
        col = base["validation"]["time"]
        lab = df.iloc[lab_rows]
        last_train = lab.iloc[tr][col]
        first_test = lab.iloc[te][col]
        order_tr, order_te = times[tr], times[te]
        time_split = {
            "column": col,
            "train_from": str(last_train.iloc[int(np.argmin(order_tr))]),
            "train_to": str(last_train.iloc[int(np.argmax(order_tr))]),
            "test_from": str(first_test.iloc[int(np.argmin(order_te))]),
            "test_to": str(first_test.iloc[int(np.argmax(order_te))]),
            "unvalidated": int((~covered).sum()) if splits else 0}
    return {"task": task, "config": config, "base": base, "rule": rule,
            "target": target, "classes": classes, "positive": positive,
            "family": family, "metric": metric, "y": y_all, "split": split,
            "n_rows": len(df), "n_labelled": int(labelled.sum()),
            "n_train": len(tr), "n_test": len(te),
            "n_unlabelled": len(unl_rows), "folds": len(splits),
            "time_split": time_split,
            **({"prep_steps": learnt, "frame": df,
                "prep_learnt": True} if learnt is not None else {}),
            "results": results,
            "best": _best_supervised(results, family, metric, bool(splits))}


def _stack_member_list(results, config, family, metric, by_cv):
    """The stack to fit, as a model entry, or None: the best `top` models by
    cross-validated score, each with the settings it won with. Needs folds,
    so the choice never looks at the test rows."""
    top = int(config.get("stack") or 0)
    if top < 2 or not by_cv:
        return None
    ranked = [r for r in results
              if not r.get("error") and r["key"] != BASELINE
              and r["key"] not in STACK_LEAVES_OUT and r.get("cv")
              and _mean(r["cv"][metric]) is not None]
    ranked.sort(key=lambda r: _mean(r["cv"][metric]),
                reverse=higher_is_better(family, metric))
    chosen = ranked[:top]
    if len(chosen) < 2:
        return None
    return {"key": STACK, "search": None,
            "label": "Stack: " + " + ".join(r["label"] for r in chosen),
            "params": {"members": [{"model": r["key"],
                                    "params": dict(r["params"])}
                                   for r in chosen]}}


def _best_supervised(results, family, metric, by_cv):
    """Chosen by cross-validation when there are folds — the test rows are
    then the honest check of that choice, not what made it."""
    best, best_score = None, None
    for r in results:
        if r.get("error") or r["key"] == BASELINE:
            continue
        score = _mean(r["cv"][metric]) if by_cv and r["cv"] else \
            r["test"].get(metric)
        if best is None or _better(family, metric, score, best_score):
            best, best_score = r["key"], score
    return best


def _cluster_names(codes):
    """C1 is the largest cluster; noise, where an algorithm leaves it, is
    named as such."""
    values, counts = np.unique(codes, return_counts=True)
    order = [v for v, _ in sorted(zip(values, counts), key=lambda t: -t[1])
             if v != -1]
    names = {int(v): f"C{i + 1}" for i, v in enumerate(order)}
    names[-1] = "Noise"
    return names


def _elbow(ks, inertia):
    """The k after which adding clusters buys little: the point farthest
    from the line joining the first and last points of the curve."""
    if len(ks) < 3:
        return ks[0]
    x = (np.asarray(ks) - ks[0]) / (ks[-1] - ks[0])
    y = np.asarray(inertia, dtype="float64")
    span = y[0] - y[-1]
    if span <= 0:
        return ks[0]
    y = (y - y[-1]) / span
    return ks[int(np.argmax(np.abs(1 - x - y)))]


def _train_clustering(df, config, ticker):
    base = base_recipe(config)
    X = ml_recipe.prepare_features(df, base["numeric"], base["categorical"])
    # One preprocessing pass for every model in the run: clustering has no
    # held-out rows, so all of them share the same fitted transform.
    first = ml_recipe.with_caps(
        {**base, "model": config["models"][0]["key"],
         "params": dict(config["models"][0]["params"])}, X)
    front = ml_recipe.build_pipeline(first)[:-1]
    ticker.say("Preparing the features")
    z = front.fit_transform(X)
    results = []
    for m in config["models"]:
        key = m["key"]
        label = model_label("clustering", key)
        # The shared preparation's cap levels go with every model's recipe.
        recipe = {**first, "model": key, "params": dict(m["params"])}
        try:
            est = ml_recipe.build_estimator(recipe)
            t1 = time.perf_counter()
            codes = np.asarray(est.fit_predict(z))
            fit_s = time.perf_counter() - t1
            names = _cluster_names(codes)
            labels = pd.Series(codes).map(names).to_numpy(dtype=object)
            metrics = clustering_metrics(z, codes)
            extra = {}
            if key == "kmeans":
                extra["inertia"] = float(est.inertia_)
            if key == "gmm":
                extra["bic"] = float(est.bic(z))
            pipe = Pipeline(list(front.steps) + [("model", est)])
            results.append({"key": key, "label": label, "recipe": recipe,
                            "params": recipe["params"], "pipeline": pipe,
                            "labels": labels, "names": names,
                            "n_clusters": len(names) - 1,
                            "test": {**metrics, **extra}, "fit_s": fit_s,
                            "error": None})
        except Exception:
            ref = log.incident_id()
            log.event(_log, "ml_model_failed", level=logging.ERROR,
                      exc_info=True, model=key, task="clustering", ref=ref)
            results.append(_failed(key, "clustering", ref))
        ticker.step(f"{label}: fitted")
    scan = None
    if config.get("k_scan"):
        scan = _k_scan(z, base, int(config["k_scan"]["max_k"]), ticker)
    coords = None
    if z.shape[1] >= 2:
        coords = PCA(n_components=2, random_state=base["seed"]) \
            .fit_transform(z).astype("float32")
    elif z.shape[1] == 1:
        coords = np.column_stack([z[:, 0], np.zeros(len(z))]) \
            .astype("float32")
    ok = [r for r in results if not r.get("error")]
    by = "silhouette" if ok and all(r["test"]["silhouette"] is not None
                                    for r in ok) else "calinski_harabasz"
    best = None
    for r in ok:
        if best is None or _better("clustering", by, r["test"][by],
                                   best["test"][by]):
            best = r
    return {"task": "clustering", "config": config, "base": base,
            "rule": None, "target": None, "classes": [], "positive": None,
            "family": "clustering", "metric": by, "n_rows": len(df),
            "n_labelled": 0, "n_train": len(df), "n_test": 0,
            "n_unlabelled": 0, "folds": 0, "results": results,
            "best": best["key"] if best else None, "scan": scan,
            "coords": coords, "z_columns": int(z.shape[1])}


def _k_scan(z, base, max_k, ticker):
    """K-means at every k from 2 to `max_k`, on the matrix the models saw.
    Suggests k by the silhouette where it is computed, otherwise by the
    elbow of the within-cluster spread."""
    from sklearn.cluster import KMeans
    rows = []
    for k in range(2, max(2, min(max_k, len(z) - 1)) + 1):
        est = KMeans(n_clusters=k, n_init=3, random_state=base["seed"])
        codes = est.fit_predict(z)
        rows.append({"k": k, "inertia": float(est.inertia_),
                     **clustering_metrics(z, codes)})
        ticker.say(f"Comparing cluster counts: k = {k}")
    table = pd.DataFrame(rows)
    if table.empty:
        return None
    if table["silhouette"].notna().all():
        pick = int(table.loc[table["silhouette"].idxmax(), "k"])
        why = "the highest silhouette"
    else:
        pick = _elbow(list(table["k"]), list(table["inertia"]))
        why = "the elbow of the within-cluster spread"
    return {"table": table, "suggested": pick, "why": why}


def result(run, key):
    return next(r for r in run["results"] if r["key"] == key)


def usable(run) -> list:
    """Models that trained, best first, the baseline last."""
    ok = [r for r in run["results"] if not r.get("error")]
    order = sorted(ok, key=lambda r: (r["key"] == BASELINE,
                                      r["key"] != run["best"]))
    return order
