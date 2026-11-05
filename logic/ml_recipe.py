"""How a model is built from its recipe: the one place that does it.

A recipe is a plain dict — task, model, hyperparameters, the feature columns,
the preprocessing choices, the seed and the split — and `build_pipeline` turns
it into a scikit-learn Pipeline. `logic.ml` trains with it, and the training
script the results page exports carries **this file's source** verbatim, so
the app and the script build the same model by construction rather than by
two copies kept in step.

That is why this module imports nothing but numpy, pandas and scikit-learn,
and why the pipelines hold only scikit-learn objects: the script must run
anywhere those three are installed, and a downloaded model must load there
too. A transformer defined in this app would make every saved model need it.
XGBoost and LightGBM are imported only when a recipe names them, so a script
for any other model never needs either.
"""
import importlib

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import (ElasticNet, Lasso, LinearRegression,
                                  LogisticRegression, Ridge)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold,
                                     StratifiedGroupKFold, StratifiedKFold,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import (KNeighborsClassifier, KNeighborsRegressor,
                               NearestNeighbors)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler,
                                   OneHotEncoder, TargetEncoder,
                                   OrdinalEncoder, PowerTransformer,
                                   RobustScaler, StandardScaler)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# (task, model) -> (estimator class, fixed arguments). The fixed arguments are
# the ones no setting on the configure page changes.
ESTIMATORS = {
    ("classification", "baseline"): (DummyClassifier, {"strategy": "prior"}),
    ("classification", "logistic"): (LogisticRegression, {"max_iter": 2000}),
    ("classification", "tree"): (DecisionTreeClassifier, {}),
    # Forests use every core, as gradient boosting already does: at 240,000
    # rows one forest fit took 49s on one core and 12.6s on four.
    ("classification", "random_forest"): (RandomForestClassifier,
                                          {"n_jobs": -1}),
    ("classification", "extra_trees"): (ExtraTreesClassifier, {"n_jobs": -1}),
    ("classification", "hist_gb"): (HistGradientBoostingClassifier, {}),
    ("classification", "knn"): (KNeighborsClassifier, {"n_jobs": -1}),
    ("classification", "svm"): (SVC, {"probability": True}),
    ("classification", "naive_bayes"): (GaussianNB, {}),
    ("classification", "mlp"): (MLPClassifier, {"early_stopping": True}),
    ("regression", "baseline"): (DummyRegressor, {"strategy": "mean"}),
    ("regression", "linear"): (LinearRegression, {}),
    ("regression", "ridge"): (Ridge, {}),
    ("regression", "lasso"): (Lasso, {"max_iter": 5000}),
    ("regression", "elastic_net"): (ElasticNet, {"max_iter": 5000}),
    ("regression", "tree"): (DecisionTreeRegressor, {}),
    ("regression", "random_forest"): (RandomForestRegressor, {"n_jobs": -1}),
    ("regression", "extra_trees"): (ExtraTreesRegressor, {"n_jobs": -1}),
    ("regression", "hist_gb"): (HistGradientBoostingRegressor, {}),
    ("regression", "knn"): (KNeighborsRegressor, {"n_jobs": -1}),
    ("regression", "svm"): (SVR, {}),
    ("regression", "mlp"): (MLPRegressor, {"early_stopping": True}),
    # Optional packages, named rather than imported: see _resolve.
    ("classification", "xgboost"): ("xgboost:XGBClassifier",
                                    {"tree_method": "hist", "verbosity": 0}),
    ("regression", "xgboost"): ("xgboost:XGBRegressor",
                                {"tree_method": "hist", "verbosity": 0}),
    # subsample_freq 1: without it LightGBM ignores a subsample below 1.
    ("classification", "lightgbm"): ("lightgbm:LGBMClassifier",
                                     {"verbose": -1, "subsample_freq": 1}),
    ("regression", "lightgbm"): ("lightgbm:LGBMRegressor",
                                 {"verbose": -1, "subsample_freq": 1}),
    ("clustering", "kmeans"): (KMeans, {}),
    ("clustering", "gmm"): (GaussianMixture, {}),
    ("clustering", "agglomerative"): (AgglomerativeClustering, {}),
    ("clustering", "dbscan"): (DBSCAN, {}),
}

# Gradient boosting reads categories natively; one-hot columns would only make
# it slower. Every other model gets one column per category.
NATIVE_CATEGORIES = ("hist_gb",)
# XGBoost's classifier is fitted on class codes 0 to k - 1: it refuses text
# labels ("Invalid classes inferred from unique values of y").
INTEGER_LABELS = ("xgboost",)

YES, NO = "Yes", "No"


def _resolve(cls):
    """An estimator class, importing an optional package's only when named."""
    if isinstance(cls, str):
        module, name = cls.split(":")
        return getattr(importlib.import_module(module), name)
    return cls


def fit_labels(recipe, y, classes):
    """The labels a model is fitted on: class codes for XGBoost, the classes
    themselves for every other model."""
    if recipe["task"] == "classification" and \
            recipe["model"] in INTEGER_LABELS:
        return pd.Series(y).map({c: i for i, c in enumerate(classes)}) \
            .to_numpy(dtype="int64")
    return y


def answers_in_codes(model):
    """Whether a classifier answers in class codes 0 to k - 1, standing for
    `classes` in order, rather than in the labels themselves: XGBoost always
    does (it refuses text labels), and so does a model that says so. A model
    trained elsewhere on the labels 1, 2 and 3 answers in those labels."""
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    return bool(getattr(est, "answers_in_codes", False)) or \
        type(est).__module__.split(".")[0] == "xgboost"


def model_classes(pipe, classes):
    """The classes a fitted model predicts, as labels, in its own order."""
    seen = np.asarray(pipe.classes_)
    if answers_in_codes(pipe):
        return [classes[int(i)] for i in seen]
    return [str(c) for c in label_text(pd.Series(seen))]


def predicted_labels(pipe, X, classes):
    """A classifier's predictions as labels, whichever way it was fitted."""
    pred = np.asarray(pipe.predict(X))
    if answers_in_codes(pipe):
        return np.asarray(classes, dtype=object)[pred.astype("int64")]
    return label_text(pd.Series(pred)).to_numpy(dtype=object)


def label_text(series):
    """Category labels as text, a missing value kept missing.

    A flag read as 1 in one file and as 1.0 in the next (one blank turns an
    integer column float) has to be the same category, or every row in the
    second file holds a value the model never saw — so whole numbers print
    without a decimal.
    """
    out = pd.Series(np.nan, index=series.index, dtype=object)
    keep = series.notna()
    if not keep.any():
        return out
    values = series[keep]
    if pd.api.types.is_bool_dtype(values):
        text = values.astype(str)
    elif pd.api.types.is_numeric_dtype(values):
        f = values.astype("float64")
        whole = (np.isfinite(f) & (f % 1 == 0) & (np.abs(f) < 2 ** 53)).all()
        text = f.astype("int64").astype(str) if whole else values.astype(str)
    else:
        text = values.astype(str)
    out[keep] = text.to_numpy()
    return out


def prepare_features(df, numeric, categorical):
    """The feature columns as a model reads them.

    Numbers are coerced (text that does not parse becomes missing) and an
    infinity becomes missing too, which the imputer can fill; categories
    become text. Run on the training file and on every file scored later, so
    the two always agree.
    """
    cols = {}
    for c in categorical:
        cols[c] = label_text(df[c])
    for c in numeric:
        v = pd.to_numeric(df[c], errors="coerce").astype("float64")
        cols[c] = v.where(np.isfinite(v))
    return pd.DataFrame(cols, index=df.index,
                        columns=list(categorical) + list(numeric))


def target_values(df, recipe):
    """The target as the model learns it; missing where a row has none."""
    target = recipe["target"]
    if recipe["task"] == "regression":
        v = pd.to_numeric(df[target], errors="coerce").astype("float64")
        return v.where(np.isfinite(v))
    rule = recipe.get("target_rule")
    if rule:
        # A numeric column split into yes/no at a value fixed when the model
        # was trained, so a later file is cut at the same place.
        v = pd.to_numeric(df[target], errors="coerce").astype("float64")
        v = v.where(np.isfinite(v))
        out = pd.Series(np.nan, index=df.index, dtype=object)
        ok = v.notna()
        out[ok] = np.where(v[ok] >= rule["at_least"], YES, NO)
        return out
    return label_text(df[target])


def encoding(recipe):
    """How categories reach the model: "onehot", "ordinal" or "target".
    Target encoding learns from the target, so clustering, which has none,
    falls back to one-hot."""
    how = recipe["prep"].get("cat_encoding", "onehot")
    if how == "target" and recipe["task"] == "clustering":
        return "onehot"
    return how


def native_categories(recipe):
    """Gradient boosting reads categories itself unless they arrive target
    encoded, as numbers."""
    return recipe["model"] in NATIVE_CATEGORIES and encoding(recipe) != "target"


def with_caps(recipe, X):
    """The recipe with its cap levels learnt from `X`, the rows about to be
    fitted: each fold's training rows in cross-validation, the training rows
    for the final model. So no test or validation row helps set a cap.
    "pct" caps at the 1st and 99th percentiles, "iqr" at 1.5 interquartile
    ranges beyond the quartiles; a gap stays a gap."""
    how = recipe["prep"].get("cap", "none")
    if how == "none" or not recipe["numeric"]:
        return recipe
    low, high = [], []
    for c in recipe["numeric"]:
        v = pd.to_numeric(X[c], errors="coerce").to_numpy(dtype="float64")
        v = v[np.isfinite(v)]
        if not len(v):
            lo, hi = -np.inf, np.inf
        elif how == "pct":
            lo, hi = np.quantile(v, [0.01, 0.99])
        else:
            q1, q3 = np.quantile(v, [0.25, 0.75])
            lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        low.append(float(lo))
        high.append(float(hi))
    return {**recipe, "caps": {"low": low, "high": high}}


SMOTE_NEIGHBOURS = 5


def resample(recipe, X, y):
    """The rows a model is fitted on, rebalanced: "smote" adds synthetic
    rows to every class smaller than the largest, "under" keeps a random
    sample of every class the size of the smallest. Only ever given the rows
    being fitted (a fold's training rows, or the training rows), so no row
    being scored is copied or dropped. Seeded, so a run repeats exactly.

    SMOTE here is SMOTE-NC: a new row lies between a real row and one of
    its nearest neighbours of the same class on every number, and takes the
    commonest value among those neighbours for every category. Distance
    reads the numbers scaled to unit spread and counts a differing category
    as the median spread, so mixed columns weigh fairly.
    """
    how = recipe.get("resample") or "none"
    if how == "none" or recipe["task"] != "classification":
        return X, y
    y = np.asarray(y, dtype=object)
    rng = np.random.default_rng(recipe.get("seed", 0))
    labels, counts = np.unique(y.astype(str), return_counts=True)
    if len(labels) < 2:
        return X, y
    if how == "under":
        keep = np.concatenate([
            rng.choice(np.flatnonzero(y.astype(str) == c), counts.min(),
                       replace=False) for c in labels])
        keep.sort()
        return X.iloc[keep], y[keep]
    numeric = [c for c in recipe["numeric"] if c in X.columns]
    categorical = [c for c in recipe["categorical"] if c in X.columns]
    num = X[numeric].to_numpy(dtype="float64")
    centre = np.nanmedian(num, axis=0) if numeric else np.zeros(0)
    spread = np.nanstd(num, axis=0) if numeric else np.zeros(0)
    spread = np.where(np.isfinite(spread) & (spread > 0), spread, 1.0)
    filled = np.where(np.isnan(num), centre, num) / spread
    # Scaled numbers have a spread of one, so a differing category (two
    # indicator columns apart) adds one squared unit, as SMOTE-NC asks.
    cats = X[categorical].astype(object).where(X[categorical].notna(),
                                               "(missing)")
    dummies = (pd.get_dummies(cats).to_numpy(dtype="float64") / np.sqrt(2)
               if categorical else np.zeros((len(X), 0)))
    space = np.hstack([filled, dummies])
    new_X, new_y = [], []
    for c, n in zip(labels, counts):
        need = counts.max() - n
        rows = np.flatnonzero(y.astype(str) == c)
        if need == 0 or n < 2:
            continue
        k = min(SMOTE_NEIGHBOURS, n - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(space[rows])
        near = nn.kneighbors(space[rows], return_distance=False)[:, 1:]
        base = rng.integers(0, n, need)
        pick = near[base, rng.integers(0, k, need)]
        gap = rng.random((need, 1))
        a, b = num[rows[base]], num[rows[pick]]
        values = np.where(np.isnan(a) | np.isnan(b), a, a + gap * (b - a))
        part = pd.DataFrame(values, columns=numeric)
        for col in categorical:
            codes, uniques = pd.factorize(X[col].to_numpy(dtype=object)[rows],
                                          use_na_sentinel=False)
            votes = codes[near[base]]                # need × k
            # The commonest code in each row; a tie goes to the nearest.
            agree = (votes[:, :, None] == votes[:, None, :]).sum(axis=2)
            won = votes[np.arange(need), agree.argmax(axis=1)]
            part[col] = np.asarray(uniques, dtype=object)[won]
        new_X.append(part[list(X.columns)])
        new_y.append(np.full(need, y[rows[0]], dtype=object))
    if not new_X:
        return X, y
    out = pd.concat([X.reset_index(drop=True)]
                    + [p.astype(X.dtypes.to_dict()) for p in new_X],
                    ignore_index=True)
    return out, np.concatenate([y] + new_y)


def build_preprocessor(recipe):
    prep = recipe["prep"]
    numeric, categorical = list(recipe["numeric"]), list(recipe["categorical"])
    rare = float(prep.get("rare_share") or 0) or None
    native = native_categories(recipe)
    how = encoding(recipe)
    parts = []
    # Categories first: gradient boosting is told which output columns are
    # categorical by position, and these come out in front.
    if categorical:
        steps = []
        if prep.get("cat_missing") == "most_frequent":
            steps.append(("impute", SimpleImputer(strategy="most_frequent",
                                                  keep_empty_features=True)))
        if native:
            steps.append(("encode", OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=np.nan,
                encoded_missing_value=np.nan, min_frequency=rare)))
        elif how == "target":
            # Each category becomes the target's mean over the training rows
            # (one column per class for several classes), cross-fitted so a
            # row never sees its own target.
            steps.append(("encode", TargetEncoder(
                random_state=recipe.get("seed", 0))))
        elif how == "ordinal":
            # Codes in sorted order; unseen and missing values share -1, so
            # a model that cannot read gaps still can read these.
            steps.append(("encode", OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1,
                encoded_missing_value=-1, min_frequency=rare)))
        else:
            steps.append(("encode", OneHotEncoder(
                handle_unknown="infrequent_if_exist" if rare else "ignore",
                min_frequency=rare, sparse_output=False)))
        parts.append(("cat", Pipeline(steps), categorical))
    if numeric:
        how = prep.get("num_missing", "none")
        flags = bool(prep.get("missing_flags"))
        steps = []
        caps = recipe.get("caps")
        if caps:
            # numpy's own clip with fixed levels: a saved model needs
            # nothing but scikit-learn and numpy to load.
            steps.append(("cap", FunctionTransformer(
                np.clip, feature_names_out="one-to-one",
                kw_args={"a_min": np.asarray(caps["low"]),
                         "a_max": np.asarray(caps["high"])})))
        if how != "none":
            steps.append(("impute", SimpleImputer(
                strategy="constant" if how == "zero" else how,
                fill_value=0 if how == "zero" else None,
                add_indicator=flags, keep_empty_features=True)))
        # Every scaler here skips a gap when it fits and keeps it when it
        # transforms, so with no fill the gaps reach the model as gaps.
        scaler = {"standard": StandardScaler(), "robust": RobustScaler(),
                  "minmax": MinMaxScaler(),
                  "power": PowerTransformer(method="yeo-johnson"),
                  "none": None}[prep.get("scale", "standard")]
        if scaler is not None:
            steps.append(("scale", scaler))
        parts.append(("num", Pipeline(steps) if steps else "passthrough",
                       numeric))
        if how == "none" and flags:
            # No imputer to add the flags, so they come from their own step.
            parts.append(("gaps", MissingIndicator(error_on_new=False),
                          numeric))
    return ColumnTransformer(parts, sparse_threshold=0.0)


def build_estimator(recipe):
    task, model = recipe["task"], recipe["model"]
    cls, fixed = ESTIMATORS[(task, model)]
    cls = _resolve(cls)
    args = dict(fixed)
    args.update(recipe.get("params") or {})
    accepted = cls().get_params()
    if "random_state" in accepted:
        args["random_state"] = recipe.get("seed", 0)
    if recipe.get("balance") and "class_weight" in accepted:
        args["class_weight"] = "balanced"
    if model in NATIVE_CATEGORIES:
        n_cat = len(recipe["categorical"]) if native_categories(recipe) else 0
        args["categorical_features"] = list(range(n_cat)) if n_cat else None
    return cls(**args)


def build_pipeline(recipe):
    """preprocessing -> (principal components) -> model, unfitted."""
    steps = [("prep", build_preprocessor(recipe))]
    prep = recipe["prep"]
    if recipe["task"] == "clustering" and prep.get("pca"):
        steps.append(("pca", PCA(n_components=int(prep["pca"]),
                                 random_state=recipe.get("seed", 0))))
    model = build_estimator(recipe)
    if recipe["task"] == "regression" and prep.get("log_target"):
        # Fitted on log(1 + y) and turned back on the way out, so every
        # number reported is on the target's own scale.
        model = TransformedTargetRegressor(regressor=model, func=np.log1p,
                                           inverse_func=np.expm1,
                                           check_inverse=False)
    steps.append(("model", model))
    return Pipeline(steps)


def split(y, groups, recipe):
    """Train and test positions among the rows that have a target.

    Seeded, so the same file and recipe always split the same way. Whole
    groups go to one side when a group column is set, and the classes keep
    their shares on both sides when the split is stratified.
    """
    val = recipe["validation"]
    seed, share = recipe.get("seed", 0), float(val["test_size"])
    positions = np.arange(len(y))
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=share, random_state=seed)
        train, test = next(gss.split(positions, y, groups))
        return np.sort(train), np.sort(test)
    strat = (y if recipe["task"] == "classification" and val.get("stratify")
             else None)
    train, test = train_test_split(positions, test_size=share,
                                   random_state=seed, stratify=strat)
    return np.sort(train), np.sort(test)


def folds(recipe, grouped):
    """The cross-validation splitter for the training rows."""
    val = recipe["validation"]
    k, seed = int(val["folds"]), recipe.get("seed", 0)
    classify = recipe["task"] == "classification" and val.get("stratify")
    if grouped:
        return (StratifiedGroupKFold(k, shuffle=True, random_state=seed)
                if classify else GroupKFold(k))
    return (StratifiedKFold(k, shuffle=True, random_state=seed)
            if classify else KFold(k, shuffle=True, random_state=seed))
