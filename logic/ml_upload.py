"""A model trained elsewhere, or downloaded from this app, evaluated on the
file loaded here.

**Loading a pickle runs code.** A joblib or pickle file can name any Python
function and have it called while it loads, so `joblib.load` on an uploaded
file would hand the server to whoever wrote the file. Here such a file is
read by an unpickler that resolves only an allow-list of model components:
scikit-learn, XGBoost and LightGBM estimator classes, the exact internal
types their files are made of (collected by loading this app's own model
files and common outside pipelines), and numpy's array machinery. Anything
else stops the load before it is called, and the page names what was asked
for. numpy's ufunc loader is never resolved as is: it imports any module and
returns any attribute, so a substitute hands back numpy ufuncs only.

What remains is the libraries' own code reading their own state — a tree's
arrays, a booster's binary — which an allow-list cannot vet. XGBoost's JSON
and LightGBM's text model formats run no Python at all and are the safest
way in. A deployment can turn upload off: ML_MODEL_UPLOAD=off.

Nothing here raises to the page: `read` returns (model, None) or
(None, sentence).
"""
import importlib
import io
import json
import re
import time

import numpy as np
import pandas as pd

from logic import ml, ml_prep, ml_recipe
from utils import log

_log = log.get("ml_upload")

KEY = "uploaded"
EXTENSIONS = (".joblib", ".pkl", ".pickle", ".json", ".ubj", ".txt")
MAX_BYTES = 500 * 1024 * 1024


class Refused(Exception):
    """The file asked for something that is not a model component."""


# ── what a model file may name, and what it may call ────────────────────
# Collected by loading every model this app trains and common outside
# pipelines (calibrated, stacked, voting, grid-searched, feature-selected,
# log-target, nearest-neighbour, XGBoost and LightGBM) and recording each
# name they asked for. The smoke suite reloads that corpus through this
# list and fires attack files at it.
_SAFE_BUILTINS = ("dict", "list", "tuple", "set", "frozenset", "slice",
                  "complex", "object", "str", "int", "float", "bool",
                  "range")
_NUMPY = {("numpy", "ndarray"), ("numpy", "dtype"),
          ("numpy.core.multiarray", "_reconstruct"),
          ("numpy._core.multiarray", "_reconstruct"),
          ("numpy.core.multiarray", "scalar"),
          ("numpy._core.multiarray", "scalar"),
          ("numpy.core.numeric", "_frombuffer"),
          ("numpy._core.numeric", "_frombuffer"),
          ("numpy.ma", "MaskedArray"), ("numpy.ma.core", "MaskedArray"),
          ("numpy.ma.core", "_mareconstruct"),
          ("numpy.random._mt19937", "MT19937"),
          ("numpy.random._pcg64", "PCG64"),
          ("numpy.random._philox", "Philox"), ("numpy.random._sfc64", "SFC64"),
          ("numpy.random.mtrand", "RandomState"),
          ("numpy.random._generator", "Generator"),
          ("numpy.random._pickle", "__bit_generator_ctor"),
          ("numpy.random._pickle", "__generator_ctor"),
          ("numpy.random._pickle", "__randomstate_ctor"),
          ("numpy.random.bit_generator", "SeedSequence"),
          ("numpy.random.bit_generator", "__pyx_unpickle_SeedSequence"),
          ("joblib.numpy_pickle", "NumpyArrayWrapper")}
# Inside the estimator packages, beyond the estimator classes themselves:
# modules whose classes are model parts (a tree's arrays, a loss, a link, a
# distance, a splitter), and the functions an estimator may hold as a
# setting or call to rebuild a part.
_PART_MODULES = ("sklearn._loss.", "sklearn.tree._tree",
                 "sklearn.ensemble._hist_gradient_boosting.",
                 "sklearn.neighbors._kd_tree", "sklearn.neighbors._ball_tree",
                 "sklearn.metrics._dist_metrics",
                 "sklearn.neural_network._stochastic_optimizers",
                 "sklearn.model_selection._split", "sklearn.calibration",
                 "sklearn.metrics._scorer", "sklearn.utils._bunch")
_FUNCTION_MODULES = ("sklearn.metrics._classification",
                     "sklearn.metrics._regression", "sklearn.metrics._ranking",
                     "sklearn.feature_selection._univariate_selection",
                     "sklearn.feature_selection._mutual_info")
_FUNCTIONS = {("sklearn.utils.validation", "check_array")}
# Read from a model's own state, never called: their constructors accept a
# file path, so a file that called one could make the server read a file.
_STATE_ONLY = {("xgboost.core", "Booster"), ("lightgbm.basic", "Booster")}


def _safe_ufunc(module, name):
    """numpy's old ufunc loader, minus the arbitrary import."""
    obj = getattr(np, str(name), None)
    if not isinstance(obj, np.ufunc):
        raise Refused(f"numpy.{name}")
    return obj


def _safe_bytearray(*args):
    if args and isinstance(args[0], int) and args[0] > MAX_BYTES:
        raise Refused("a buffer larger than the file")
    return bytearray(*args)


def _safe_reconstructor(cls, base, state):
    """copyreg's rebuild of an old-protocol object, for plain bases only: it
    calls `base.__init__` with what the file holds."""
    import copyreg
    if base not in (object, dict, list, tuple, set, frozenset, int, float,
                    str, bytes):
        raise Refused(f"rebuilding on {getattr(base, '__name__', base)}")
    return copyreg._reconstructor(cls, base, state)


def _safe_partial(func, *args, **kwargs):
    import functools
    if not _callable(func):
        raise Refused(f"a deferred call of {_name(func)}")
    return functools.partial(func, *args, **kwargs)


def _safe_defaultdict(factory=None, *args):
    import collections
    if factory is not None and not _callable(factory):
        raise Refused(f"a default made by {_name(factory)}")
    return collections.defaultdict(factory, *args)


_SUBSTITUTES = {("numpy.core", "_ufunc_reconstruct"): _safe_ufunc,
                ("numpy._core", "_ufunc_reconstruct"): _safe_ufunc,
                ("builtins", "bytearray"): _safe_bytearray,
                ("copyreg", "_reconstructor"): _safe_reconstructor,
                ("functools", "partial"): _safe_partial,
                ("collections", "defaultdict"): _safe_defaultdict}


def _name(obj):
    return f"{getattr(obj, '__module__', '?')}." \
           f"{getattr(obj, '__qualname__', type(obj).__name__)}"


def _resolve(module, name):
    """The object a model file names, if it is on the list. Nothing outside
    the estimator packages and numpy is imported to find out."""
    module, name = str(module), str(name)
    key = (module, name)
    if key in _SUBSTITUTES:
        return _SUBSTITUTES[key]
    if module == "builtins" and name in _SAFE_BUILTINS:
        return getattr(__import__("builtins"), name)
    if module == "collections" and name == "OrderedDict":
        import collections
        return collections.OrderedDict
    if key in _NUMPY:
        return getattr(importlib.import_module(module), name)
    if module == "numpy" and name.isidentifier():
        obj = getattr(np, name, None)
        if isinstance(obj, np.ufunc) or (isinstance(obj, type) and issubclass(
                obj, (np.number, np.bool_))):
            return obj
    if module == "numpy.dtypes" and name.isidentifier():
        obj = getattr(importlib.import_module("numpy.dtypes"), name, None)
        if isinstance(obj, type) and issubclass(obj, np.dtype):
            return obj
    if not module.startswith(("sklearn.", "xgboost.", "lightgbm.")) or \
            not name.isidentifier():
        raise Refused(f"{module}.{name}")
    try:
        obj = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError):
        raise Refused(f"{module}.{name}") from None
    from sklearn.base import BaseEstimator
    if isinstance(obj, type) and (issubclass(obj, BaseEstimator)
                                  or key in _STATE_ONLY
                                  or module.startswith(_PART_MODULES)):
        return obj
    if callable(obj) and not isinstance(obj, type) and (
            key in _FUNCTIONS or module in _FUNCTION_MODULES
            or (module.startswith(_PART_MODULES)
                and (name == "newObj" or name.startswith("__pyx_unpickle_")))):
        return obj
    raise Refused(f"{module}.{name}")


def _callable(obj):
    """Whether a file may call `obj` with arguments it chose (the REDUCE,
    INST and OBJ opcodes, and a partial or a default factory). Anything that
    resolved may be built empty and given its state; calling is narrower."""
    if isinstance(obj, type):
        if (getattr(obj, "__module__", ""), obj.__name__) in _STATE_ONLY:
            return False
        return True              # resolved classes only reach here
    return callable(obj)


def _guarded_load(data):
    """joblib.load with every name resolved through `_resolve` and every call
    checked by `_callable`. joblib 1.6 reads an object array's nested pickle
    through the same unpickler's find_class; earlier versions did not, so
    they are refused rather than trusted."""
    import functools
    import pickle

    import joblib
    from joblib import numpy_pickle as npk
    major, minor = (int(x) for x in joblib.__version__.split(".")[:2])
    if (major, minor) < (1, 6):
        raise Refused("joblib before 1.6 (pip install -U joblib)")

    class Guarded(npk.NumpyUnpickler):
        dispatch = npk.NumpyUnpickler.dispatch.copy()

        def find_class(self, module, name):
            return _resolve(module, name)

        def load_reduce(self):
            args = self.stack.pop()
            func = self.stack[-1]
            if not _callable(func):
                raise Refused(f"a call of {_name(func)}")
            self.stack[-1] = func(*args)

        def _instantiate(self, klass, args):
            if not _callable(klass):
                raise Refused(f"a call of {_name(klass)}")
            return super()._instantiate(klass, args)

        def load_build(self):
            state, inst = self.stack[-1], self.stack[-2]
            if isinstance(inst, functools.partial):
                func = state[0] if isinstance(state, tuple) and state else None
                if not _callable(func):
                    raise Refused(f"a deferred call of {_name(func)}")
            npk.NumpyUnpickler.load_build(self)

        def persistent_load(self, pid):
            raise Refused("a reference outside the file")

        dispatch[pickle.REDUCE[0]] = load_reduce
        dispatch[pickle.BUILD[0]] = load_build

    with npk._validate_fileobject_and_memmap(io.BytesIO(data), "", None) as (
            fobj, _):
        if isinstance(fobj, str):
            raise Refused("a joblib format from before version 0.10")
        return Guarded("", fobj, ensure_native_byte_order=True).load()


# ── reading a file ──────────────────────────────────────────────────────
def _final(model):
    """The estimator at the end of a pipeline, or the model itself."""
    steps = getattr(model, "steps", None)
    return steps[-1][1] if steps else model


def _task_of(model):
    from sklearn.base import is_classifier, is_regressor
    if is_classifier(model):
        return "classification"
    if is_regressor(model):
        return "regression"
    # scikit-learn 1.6 moved the estimator type into tags, and 1.9 no
    # longer sets `_estimator_type` on a clusterer
    est = _final(model)
    try:
        from sklearn.utils import get_tags
        kind = get_tags(est).estimator_type
    except Exception:
        kind = getattr(est, "_estimator_type", None)
    # a Gaussian mixture is tagged a density estimator, and it assigns each
    # row a component: clustering, as on the configure page
    if kind in ("clusterer", "density_estimator") and hasattr(est,
                                                               "predict"):
        return "clustering"
    if hasattr(model, "predict_proba"):
        return "classification"
    return None


def _names(model):
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        return None
    return [str(n) for n in names]


def _n_features(model):
    n = getattr(model, "n_features_in_", None)
    return int(n) if n is not None else None


def _describe_estimator(model, kind):
    est = _final(model)
    label = type(est).__name__
    if getattr(model, "steps", None):
        label = f"Pipeline ending in {label}"
    return {"kind": kind, "model": model, "task": _task_of(model),
            "label": label, "features": _names(model),
            "n_features": _n_features(model),
            "classes": (list(ml_recipe.label_text(pd.Series(
                np.asarray(getattr(model, "classes_")))))
                if hasattr(model, "classes_") else None),
            "codes": ml_recipe.answers_in_codes(model)}


def _from_bundle(bundle):
    recipe = bundle["recipe"]
    task = recipe["task"]
    model = bundle["pipeline"]
    return {"kind": "app", "model": model, "task": task,
            "label": ml.model_label(task, recipe["model"]),
            "features": list(recipe["categorical"]) + list(recipe["numeric"]),
            "n_features": len(recipe["categorical"]) + len(recipe["numeric"]),
            "classes": list(bundle.get("classes") or []) or None,
            "codes": ml_recipe.answers_in_codes(model), "recipe": recipe,
            "positive": bundle.get("positive"),
            "cluster_names": bundle.get("cluster_names"),
            "prep_steps": list(bundle.get("prep_steps") or []),
            "threshold": bundle.get("threshold"),
            "trained": bundle.get("created"),
            "sklearn": bundle.get("sklearn")}


def _xgboost(data):
    import xgboost
    booster = xgboost.Booster()
    booster.load_model(bytearray(data))
    config = json.loads(booster.save_config())
    objective = config["learner"]["objective"]["name"]
    if objective.startswith(("binary:", "multi:")):
        model, task = xgboost.XGBClassifier(), "classification"
    elif objective.startswith(("reg:", "count:", "survival:")):
        model, task = xgboost.XGBRegressor(), "regression"
    else:
        raise Refused(f"an XGBoost {objective} model (ranking is not "
                      f"evaluated here)")
    model.load_model(bytearray(data))
    names = booster.feature_names
    return {"kind": "xgboost", "model": model, "task": task,
            "label": f"XGBoost ({objective})",
            "features": list(names) if names else None,
            "n_features": int(booster.num_features()),
            "classes": None, "codes": task == "classification"}


class LightGBMText:
    """A LightGBM booster read from its text format, with the methods the
    results page calls. Classification answers in class codes, like XGBoost:
    the target's values in sorted order."""
    answers_in_codes = True

    def __init__(self, booster, task, n_classes):
        self.booster = booster
        self.task = task
        self.n_classes = n_classes
        self._estimator_type = ("classifier" if task == "classification"
                                else "regressor")
        if task == "classification":
            self.classes_ = np.arange(n_classes)
        self.n_features_in_ = booster.num_feature()

    def __sklearn_tags__(self):
        from sklearn.utils import Tags, ClassifierTags, RegressorTags, \
            TargetTags
        tags = Tags(estimator_type=self._estimator_type,
                    target_tags=TargetTags(required=True))
        if self.task == "classification":
            tags.classifier_tags = ClassifierTags()
        else:
            tags.regressor_tags = RegressorTags()
        return tags

    def fit(self, X, y=None):
        raise TypeError("A model read from a file is not refitted here.")

    def _raw(self, X):
        return np.asarray(self.booster.predict(np.asarray(X, dtype="float64")))

    def predict_proba(self, X):
        p = self._raw(X)
        if p.ndim == 1:
            return np.column_stack([1 - p, p])
        return p

    def predict(self, X):
        if self.task == "classification":
            return self.predict_proba(X).argmax(axis=1)
        return self._raw(X)


def _lightgbm(data):
    import lightgbm
    text = data.decode("utf-8")
    booster = lightgbm.Booster(model_str=text)
    found = re.search(r"^objective=(\S+)", text, re.M)
    objective = found.group(1) if found else "regression"
    classes = re.search(r"^num_class=(\d+)", text, re.M)
    n_classes = int(classes.group(1)) if classes else 1
    if objective in ("binary", "cross_entropy", "xentropy"):
        task, n_classes = "classification", 2
    elif objective.startswith(("multiclass", "softmax", "multiclassova")):
        task = "classification"
    elif objective.startswith(("lambdarank", "rank_xendcg")):
        raise Refused(f"a LightGBM {objective} model (ranking is not "
                      f"evaluated here)")
    else:
        task = "regression"
    return {"kind": "lightgbm", "model": LightGBMText(booster, task,
                                                     n_classes),
            "task": task, "label": f"LightGBM ({objective})",
            "features": list(booster.feature_name()),
            "n_features": int(booster.num_feature()), "classes": None,
            "codes": task == "classification"}


def read(data, filename):
    """(description, None) or (None, why): never raises.

    The description: kind ("app", "estimator", "xgboost", "lightgbm"), the
    fitted model, its task, a label, the feature names it was fitted with
    (or None, when it was fitted by position), how many it reads, its
    classes, and for this app's own files the recipe and preparation.
    """
    t0 = time.perf_counter()
    name = str(filename or "").lower()
    if len(data) > MAX_BYTES:
        return None, (f"The file is {len(data) / 1e6:,.0f} MB; files up to "
                      f"{MAX_BYTES / 1e6:,.0f} MB are read.")
    try:
        if name.endswith((".json", ".ubj")):
            if ml.package_problem("xgboost"):
                return None, ml.package_problem("xgboost")
            info = _xgboost(data)
        elif name.endswith(".txt") or data[:5] == b"tree\n":
            if ml.package_problem("lightgbm"):
                return None, ml.package_problem("lightgbm")
            info = _lightgbm(data)
        else:
            obj = _guarded_load(data)
            if isinstance(obj, dict) and {"pipeline", "recipe"} <= set(obj):
                info = _from_bundle(obj)
            elif hasattr(obj, "predict"):
                info = _describe_estimator(obj, "estimator")
            else:
                return None, (f"The file holds a {type(obj).__name__}, not "
                              f"a model with a predict method.")
    except Refused as e:
        log.event(_log, "ml_model_refused", error_kind="not_allowed")
        return None, (f"Not loaded: the file asks for {e}, which is not "
                      f"a part of a scikit-learn, XGBoost or LightGBM model, "
                      f"so the load stopped there and nothing in the file "
                      f"was run. A model that needs a function or class of "
                      f"its own cannot be read here: rebuild it from "
                      f"standard steps, or save a booster in its own format "
                      f"(.json, .ubj or .txt).")
    except Exception as e:
        log.event(_log, "ml_model_refused", error_kind=type(e).__name__)
        return None, (f"The file could not be read as a model "
                      f"({type(e).__name__}). Supported: a joblib or pickle "
                      f"file of a scikit-learn, XGBoost or LightGBM model "
                      f"(this app's model file is one), an XGBoost .json or "
                      f".ubj model, or a LightGBM .txt model.")
    if not hasattr(info["model"], "predict"):
        return None, (f"{info['label']} cannot place rows it has not seen: "
                      f"it has no predict step. Agglomerative clustering is "
                      f"like that; K-means and Gaussian mixture models can "
                      f"be evaluated.")
    if info["task"] is None:
        info["task"] = "regression"
        info["task_guessed"] = True
    log.event(_log, "ml_model_uploaded", kind=info["kind"], task=info["task"],
              features=info["n_features"],
              ms=round((time.perf_counter() - t0) * 1000))
    return info, None


# ── which file column feeds which model input ───────────────────────────
def _norm(text):
    return re.sub(r"[^0-9a-z]", "", str(text).casefold())


def auto_map(names, columns):
    """{model input: file column or None}: the same name, then the same
    letters and digits regardless of case and punctuation."""
    columns = [str(c) for c in columns]
    exact = set(columns)
    folded = {}
    for c in columns:
        folded.setdefault(_norm(c), c)
    return {n: (n if n in exact else folded.get(_norm(n))) for n in names}


def feature_kinds(info, columns, prof):
    """"numeric" or "categorical" for each model input: the recipe's for this
    app's files, numbers for the boosters' own formats, and the file
    column's type otherwise."""
    if info["kind"] == "app":
        cats = set(info["recipe"]["categorical"])
        return {n: "categorical" if n in cats else "numeric" for n in columns}
    if info["kind"] in ("xgboost", "lightgbm"):
        return {n: "numeric" for n in columns}
    types = prof.set_index("column")["type"]
    return {n: types.get(c, "categorical") for n, c in columns.items()}


def as_array(X):
    """A frame's values, for a model fitted without column names."""
    return X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)


def positional(model, X):
    """The model behind a step that hands it a plain array: fitted on one
    without names, it would warn at every call given a frame."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer
    step = FunctionTransformer(as_array).fit(X.head(2))
    return Pipeline([("by_position", step), ("model", model)])


def model_frame(df, columns, kinds):
    """The inputs as the model reads them: each file column under the
    model's name for it, numbers coerced (an infinity is a gap), categories
    as text."""
    data = {}
    for name, col in columns.items():
        s = df[col]
        if kinds[name] == "numeric":
            v = pd.to_numeric(s, errors="coerce").astype("float64")
            data[name] = v.where(np.isfinite(v))
        else:
            data[name] = ml_recipe.label_text(s)
    return pd.DataFrame(data, index=df.index, columns=list(columns))


# ── the evaluation, as a run the results page can show ──────────────────
def file_for(info, source, prepared):
    """(frame, None) or (None, why): the file an uploaded model is evaluated
    on. This app's model file carries the preparation it was trained after,
    and its column steps are replayed on the file as loaded — never on top
    of the steps made on this visit, which would apply a step such as
    "replace Revenue with its log" twice. Its row steps are not replayed: every
    row gets a prediction. Any other model reads the file as prepared here.
    """
    steps = info.get("prep_steps") or []
    if not steps:
        return prepared, None
    try:
        return ml_prep.apply_steps(source, steps, rows=False), None
    except ml_prep.StepError as e:
        return None, f"The model's preparation does not fit this file. {e}"


def evaluate(df, info, columns, task, target=None, positive=None, prof=None,
             by_position=False):
    """Every row predicted by the uploaded model; the rows with a target
    scored against it, beside a baseline that ignores every feature. `df`
    is the file from `file_for`; the run keeps it as its `frame`, which the
    results page reads. Returns (run, None) or (None, sentence)."""
    from sklearn.dummy import DummyClassifier, DummyRegressor
    t0 = time.perf_counter()
    steps = info.get("prep_steps") or []
    kinds = feature_kinds(info, columns, prof)
    model = info["model"]
    recipe = info.get("recipe") or {}
    rule = recipe.get("target_rule") if info["kind"] == "app" else None
    numeric = [n for n in columns if kinds[n] == "numeric"]
    categorical = [n for n in columns if kinds[n] == "categorical"]
    base = {"task": task, "model": KEY, "target": target,
            "target_rule": rule, "numeric": numeric,
            "categorical": categorical, "prep": dict(recipe.get("prep") or {}),
            "params": {}, "seed": 0, "balance": False,
            "validation": {"test_size": 0.0, "folds": 0, "stratify": False,
                           "group": None},
            "columns": dict(columns), "kinds": kinds,
            "by_position": bool(by_position)}
    X = model_frame(df, columns, kinds)
    if by_position:
        model = positional(model, X)
    n = len(df)
    if task == "clustering":
        return _evaluate_clusters(df, info, model, X, base, t0, steps)
    if target is None:
        return None, "Choose the column with the actual values."
    if rule:
        y = ml_recipe.target_values(df, {**recipe, "target": target})
    elif task == "regression":
        y = pd.to_numeric(df[target], errors="coerce").astype("float64")
        y = y.where(np.isfinite(y))
    else:
        y = ml_recipe.label_text(df[target])
    classes = []
    try:
        if task == "classification":
            classes, why = _classes(info, model, y)
            if why:
                return None, why
        pred, proba = ml.predict(task, model, X, classes)
    except Exception as e:
        return None, (f"The model could not read these columns: "
                      f"{type(e).__name__}: {str(e)[:300]}")
    predict_s = time.perf_counter() - t0
    has_target = y.notna().to_numpy()
    unknown = (has_target & ~y.isin(classes).to_numpy()
               if task == "classification" else np.zeros(n, dtype=bool))
    labelled = has_target & ~unknown
    lab = np.flatnonzero(labelled)
    split = np.where(labelled, "test", "no target").astype(object)
    split[unknown] = "class unknown to the model"
    if not len(lab):
        if task == "classification" and has_target.any():
            held = sorted(pd.unique(y.dropna()).tolist(), key=str)
            return None, (f"No row of {target} holds one of the model's "
                          f"classes: the model knows {_listing(classes)}, "
                          f"and {target} holds {_listing(held)}.")
        return None, f"No row has a value in {target} to score against."
    family = ml.metric_family(task, len(classes))
    if task == "classification" and len(classes) == 2:
        positive = positive if positive in classes else (
            info.get("positive") if info.get("positive") in classes
            else ml.default_positive(classes))
    else:
        positive = None
    y_lab = y.to_numpy()[lab]
    dummy = (DummyClassifier(strategy="prior") if task == "classification"
             else DummyRegressor(strategy="mean"))
    dummy.fit(np.zeros((len(lab), 1)), y_lab)
    b_pred, b_proba = ml.predict(task, dummy, np.zeros((n, 1)), classes)
    results = []
    for key, label, pipe, p, pp in (
            (ml.BASELINE, "Baseline (ignores every feature)", dummy, b_pred,
             b_proba),
            (KEY, f"{info['label']} (uploaded)", model, pred, proba)):
        test = ml.score_predictions(task, y_lab, p[lab],
                                    None if pp is None else pp[lab],
                                    classes, positive)
        results.append({"key": key, "label": label, "recipe": base,
                        "params": {}, "tuned": False, "search": None,
                        "cv": None, "test": test, "pipeline": pipe,
                        "pred": p, "proba": pp, "fit_s": 0.0,
                        "predict_s": predict_s if key == KEY else 0.0,
                        "total_s": predict_s, "error": None,
                        "external": True})
    metric = ml.SELECTABLE[family][0]
    seconds = time.perf_counter() - t0
    run = {"task": task, "external": True,
           "upload": {"kind": info["kind"], "label": info["label"],
                      "trained": info.get("trained"),
                      "sklearn": info.get("sklearn"),
                      "by_position": bool(by_position),
                      "mapped": {n: c for n, c in columns.items() if n != c}},
           "config": {"numeric": numeric, "categorical": categorical,
                      "models": [], "task": task, "target": target},
           "base": base, "rule": rule, "target": target, "classes": classes,
           "positive": positive, "family": family, "metric": metric,
           "y": y,
           "split": split, "n_rows": n, "n_labelled": int(labelled.sum()),
           "n_train": 0, "n_test": int(labelled.sum()),
           "n_unlabelled": int(n - labelled.sum()),
           "n_unknown_class": int(unknown.sum()), "folds": 0,
           "results": results, "best": KEY, "seconds": seconds,
           "sklearn": _sklearn_version(),
           "created": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
           "prep_steps": steps, "importance": {}, "frame": df}
    log.event(_log, "ml_model_evaluated", kind=info["kind"], task=task,
              rows=n, labelled=int(labelled.sum()),
              ms=round(seconds * 1000))
    return run, None


def predict_only(df, info, columns, task, prof=None, by_position=False):
    """Every row of `df` predicted by the uploaded model, with no target
    needed: (the file with the model's columns added, None) or (None,
    sentence). Classes come from the model itself, or from this app's file
    for a model that answers in codes."""
    from logic.ml_eval import _free
    if prof is None and info["kind"] not in ("app", "xgboost", "lightgbm"):
        prof = ml.profile(df)
    kinds = feature_kinds(info, columns, prof)
    model = info["model"]
    missing = [c for c in columns.values() if c not in df.columns]
    if missing:
        return None, f"Not in this file: {_listing(missing)}."
    X = model_frame(df, columns, kinds)
    if by_position:
        model = positional(model, X)
    out = df.copy()
    taken = set(out.columns)
    t0 = time.perf_counter()
    try:
        if task == "clustering":
            name = _free("ML_Cluster", taken)
            out[name] = np.asarray(model.predict(X))
        elif task == "classification":
            if info["kind"] == "app" and info.get("classes"):
                classes = list(info["classes"])
            else:
                classes = [str(c) for c in ml_recipe.label_text(
                    pd.Series(np.asarray(getattr(model, "classes_", []))))]
            pred, proba = ml.predict(task, model, X, classes)
            pred = ml.with_cutoff(pred, proba, classes, info.get("positive"),
                                  info.get("threshold"))
            out[_free("ML_Prediction", taken)] = pred
            if proba is not None:
                for j, c in enumerate(classes):
                    out[_free(f"ML_Probability_{c}", set(out.columns))] = \
                        proba[:, j]
        else:
            pred, _ = ml.predict(task, model, X, [])
            out[_free("ML_Prediction", taken)] = pred
    except Exception as e:
        return None, (f"The model could not read these columns: "
                      f"{type(e).__name__}: {str(e)[:300]}")
    log.event(_log, "ml_model_predicted", kind=info["kind"], task=task,
              rows=len(df), ms=round((time.perf_counter() - t0) * 1000))
    return out, None


def _listing(values, most=6):
    values = [str(v) for v in values]
    shown = ", ".join(values[:most])
    return shown + (f" and {len(values) - most} more" if len(values) > most
                    else "")


def _sklearn_version():
    import sklearn
    return sklearn.__version__


def _classes(info, model, y):
    """(classes in the run's order, None) or ([], why). A model that answers
    in codes is matched to the target's values in sorted order — the
    convention XGBoost's own label encoding follows — or to this app's
    recorded classes for its own files."""
    known = sorted(y.dropna().unique().tolist(), key=str)
    if not info.get("codes"):
        return ml_recipe.model_classes(model, known), None
    k = len(np.asarray(getattr(model, "classes_", [])))
    if info["kind"] == "app" and info.get("classes"):
        classes = list(info["classes"])
    elif len(known) == k:
        classes = known
    elif all(str(v).isdigit() and int(v) < k for v in known):
        classes = [str(i) for i in range(k)]
    else:
        return [], (f"The model answers in {k} class codes and the target "
                    f"holds {len(known)} values, so the codes cannot be "
                    f"matched to them.")
    if len(classes) != k:
        return [], (f"The model answers in {k} class codes and its file "
                    f"names {len(classes)} classes.")
    return classes, None


def _evaluate_clusters(df, info, model, X, base, t0, steps):
    """Clusters for every row, measured in the space the model groups in:
    after its own preparation when it is a pipeline, the inputs otherwise."""
    from sklearn.decomposition import PCA
    try:
        codes = np.asarray(model.predict(X))
        steps_ = getattr(model, "steps", None)
        z = (model[:-1].transform(X) if steps_ and len(steps_) > 1
             else np.asarray(X, dtype="float64"))
        z = np.asarray(z, dtype="float64")
    except Exception as e:
        return None, (f"The model could not read these columns: "
                      f"{type(e).__name__}: {str(e)[:300]}")
    names = info.get("cluster_names") or ml._cluster_names(codes)
    labels = pd.Series(codes).map(names).fillna("Noise").to_numpy(
        dtype=object)
    metrics = (ml.clustering_metrics(z, codes) if np.isfinite(z).all()
               and len(set(codes)) > 1 else {})
    coords = None
    if np.isfinite(z).all() and z.shape[1] >= 2:
        coords = PCA(n_components=2, random_state=0).fit_transform(z) \
            .astype("float32")
    seconds = time.perf_counter() - t0
    res = {"key": KEY, "label": f"{info['label']} (uploaded)",
           "recipe": base, "params": {}, "tuned": False, "search": None,
           "cv": None, "test": metrics, "pipeline": model, "labels": labels,
           "names": names, "n_clusters": len(set(labels) - {"Noise"}),
           "fit_s": 0.0, "predict_s": seconds, "total_s": seconds,
           "error": None, "external": True}
    run = {"task": "clustering", "external": True,
           "upload": {"kind": info["kind"], "label": info["label"],
                      "trained": info.get("trained"),
                      "sklearn": info.get("sklearn"),
                      "by_position": bool(base["by_position"]),
                      "mapped": {n: c for n, c in base["columns"].items()
                                 if n != c}},
           "config": {"numeric": base["numeric"],
                      "categorical": base["categorical"], "models": [],
                      "task": "clustering", "target": None},
           "base": base, "rule": None, "target": None, "classes": [],
           "positive": None, "family": "clustering",
           "metric": ("silhouette" if metrics.get("silhouette") is not None
                      else "calinski_harabasz"),
           "n_rows": len(df), "n_labelled": 0, "n_train": 0, "n_test": 0,
           "n_unlabelled": 0, "folds": 0, "results": [res], "best": KEY,
           "scan": None, "coords": coords, "z_columns": int(z.shape[1]),
           "seconds": seconds, "sklearn": _sklearn_version(),
           "created": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
           "prep_steps": steps, "importance": {}, "frame": df}
    log.event(_log, "ml_model_evaluated", kind=info["kind"],
              task="clustering", rows=len(df), labelled=0,
              ms=round(seconds * 1000))
    return run, None
