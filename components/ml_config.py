"""Machine Learning, step 2: what to learn, from which columns, and how.

The page asks for the task, the target, the features, how to prepare them, how
to validate and which models with which settings, then trains. What it
proposes comes from `logic.ml` — the column profile, the target candidates,
the leakage check — so the page only draws the choices and keeps them.

**Choices survive a trip to the results page.** Streamlit drops a widget's
state on any run that does not draw it, so every `ml_w_` key is copied into
`ml_saved` at the end of a run here and put back before the widgets are drawn
— the problem `dash_show_nt_pref` solves for one toggle. Widgets are seeded
through session state only, never with a default argument as well: setting
both makes Streamlit print a warning on the page.
"""
import html
import os

import pandas as pd
import streamlit as st

from logic import asof, ml, ml_eval, ml_recipe
from utils import auth, guard, log, scroll

_log = log.get("ml_config")

PREFIX = "ml_w_"
SAVED = "ml_saved"
# Permutation importance is measured for the best model as training ends when
# it is expected to take no longer than this; otherwise the results page
# offers it as a button with its cost stated.
AUTO_IMPORTANCE_S = 20
LARGE_FILE = 100_000      # above this, cross-validation defaults to 3 folds
# A script is several hundred lines: shown whole, the open expander ran to
# 13,684px. It scrolls inside this instead.
CODE_HEIGHT = 520

TASK_HELP = {
    "classification": "Predict a category for each row: will the customer churn, a "
                      "segment, a risk band. Every row gets a probability, and the "
                      "results rank who to call first.",
    "regression": "Predict a number for each row: premium, revenue, spend. "
                  "The gap between a row's value and what similar rows show "
                  "is a measure of headroom.",
    "clustering": "Group rows that look alike, with no target: segments to "
                  "profile, and to target differently.",
}
# None first and by default: nothing is filled in unless asked for.
NUM_MISSING = {"none": "None (left missing)", "median": "Median",
               "mean": "Mean", "zero": "Zero",
               "most_frequent": "Most frequent"}
CAT_MISSING = {"category": "None (missing is its own category)",
               "most_frequent": "Most frequent"}
SCALING = {"standard": "Standard (z-score)",
           "robust": "Robust (median and IQR)",
           "minmax": "Min-max (0 to 1)", "power": "Power (Yeo-Johnson)",
           "none": "None"}
ENCODING = {"onehot": "One-hot (a column per category)",
            "ordinal": "Ordinal (one code per category)",
            "target": "Target encoding (the target's mean per category)"}
ENCODING_HELP = {
    "onehot": "Safe for every model; wide when a column has many categories "
              "(group the rare ones to keep it in check).",
    "ordinal": "One column per feature, in sorted order. Trees and boosting "
               "handle it well; for linear models, neighbours and neural "
               "networks it invents an order the categories do not have.",
    "target": "One column per feature (one per class for several classes), "
              "learnt from the training rows only and cross-fitted so a row "
              "never sees its own target. Suits many-category columns.",
}
CAPS = {"none": "Off",
        "pct": "At the 1st and 99th percentiles",
        "iqr": "At the quartile fences (1.5 × IQR)"}
SPLITS = {"random": "Random", "time": "By time: test on the latest rows"}
RESAMPLE = {"none": "Off",
            "smote": "Oversample the smaller classes (SMOTE)",
            "under": "Undersample the larger classes"}
RARE = {0.0: "Off", 0.005: "Below 0.5%", 0.01: "Below 1%", 0.02: "Below 2%",
        0.05: "Below 5%"}
PCA = {0: "Off", 2: "2", 3: "3", 5: "5", 10: "10"}
FOLDS = {0: "None", 3: "3 folds", 5: "5 folds", 10: "10 folds"}
TUNING = {"fixed": "As set below",
          "grid": "Grid search: every combination",
          "random": "Random search: a sample of combinations",
          "bayes": "Bayesian search: learns where to look next"}
NO_GROUP = "(none)"


def upload_allowed():
    """A deployment can turn model upload off: ML_MODEL_UPLOAD=off."""
    return os.environ.get("ML_MODEL_UPLOAD", "on").strip().lower() not in (
        "off", "0", "false", "no")


# ── keeping the choices ─────────────────────────────────────────────────
def _restore():
    ss = st.session_state
    for k, v in (ss.get(SAVED) or {}).items():
        if k not in ss:
            ss[k] = v


def _save():
    ss = st.session_state
    ss[SAVED] = {k: ss[k] for k in list(ss.keys())
                 if isinstance(k, str) and k.startswith(PREFIX)}


def _seed(key, default, options=None):
    """Give a widget its default unless it already holds a valid value; a
    persisted value missing from this run's options raises in Streamlit."""
    ss = st.session_state
    if key not in ss or (options is not None and ss[key] not in options):
        ss[key] = default


def _seed_many(key, default, options):
    ss = st.session_state
    if key not in ss:
        ss[key] = list(default)
    else:
        ss[key] = [v for v in ss[key] if v in options]


def leave():
    """Back to the start page, with every piece of Machine Learning state
    dropped: the dataset, the choices and any trained models."""
    ss = st.session_state
    for k in [k for k in ss.keys() if isinstance(k, str)
              and k.startswith("ml_")]:
        del ss[k]
    ss["page"] = "start"
    scroll.request()
    st.rerun()


# ── profiling, cached for the dataset ───────────────────────────────────
def _profile(df):
    ss = st.session_state
    if "ml_profile" not in ss:
        ss["ml_profile"] = ml.profile(df)
    return ss["ml_profile"]


def _leakage(df, task, target, rule, pool):
    """Computed once per target and rule: it reads every candidate column."""
    cache = st.session_state.setdefault("ml_leak", {})
    resolved = ml.resolve_rule(df, target, rule) if rule else None
    key = (task, target, None if not resolved else resolved["at_least"])
    if key not in cache:
        cache[key] = ml.leakage(df, target, task, pool,
                                {"at_least": resolved["at_least"]}
                                if resolved else None)
    return cache[key]


# ── sections ────────────────────────────────────────────────────────────
def _header(df):
    ss = st.session_state
    st.markdown("## Configure & train")
    steps = ss.get("ml_steps") or []
    st.caption(f"Step 4 of 5. {len(df):,} rows × {df.shape[1]} columns"
               + (f", after {len(steps)} preparation step"
                  f"{'' if len(steps) == 1 else 's'}" if steps else "")
               + ". Choose what to learn, the columns to learn from and "
                 "the models — or evaluate a model of your own instead. "
                 "Every row is used.")


def _task():
    key = f"{PREFIX}task"
    _seed(key, "classification", ml.TASKS)
    task = st.radio("What to learn", ml.TASKS, horizontal=True, key=key,
                    format_func=lambda t: ml.TASK_LABELS[t])
    st.caption(TASK_HELP[task])
    return task


def _classes(df, target, rule):
    if rule:
        return [ml_recipe.NO, ml_recipe.YES]
    values = ml_recipe.label_text(df[target]).dropna().unique().tolist()
    return sorted(map(str, values))


def _target(df, prof, task):
    """(target, rule, positive, classes), or None when nothing can be one."""
    cands = ml.target_candidates(prof, task)
    if not cands:
        what = ("a category with 2 to {0} values, or a number"
                .format(ml.MAX_CLASSES) if task == "classification"
                else "a number with more than two values")
        st.warning(f"No column in this file can be the target: {task} needs "
                   f"{what} that is not an identifier.")
        return None
    st.markdown("#### Target")
    key = f"{PREFIX}target_{task}"
    _seed(key, cands[0], cands)
    target = st.selectbox("The column to predict", cands, key=key)
    rule = None
    if task == "classification" and ml.needs_rule(prof, target):
        st.caption(f"{target} is a number with many values, so the model "
                   f"predicts yes or no: whether a row's {target} reaches "
                   f"a cut-off.")
        c_mode, c_value = st.columns(2)
        mode_key = f"{PREFIX}rule_mode"
        _seed(mode_key, "top_share", ["top_share", "at_least"])
        mode = c_mode.radio(
            "Yes when", ["top_share", "at_least"], key=mode_key,
            horizontal=True,
            format_func={"top_share": "In the top share",
                         "at_least": "At or above a value"}.get)
        if mode == "top_share":
            share_key = f"{PREFIX}rule_share"
            _seed(share_key, 20)
            value = c_value.slider("Top share of rows (%)", 1, 50,
                                   key=share_key)
        else:
            v = pd.to_numeric(df[target], errors="coerce")
            value_key = f"{PREFIX}rule_value_{target}"
            _seed(value_key, float(v.median()) if v.notna().any() else 0.0)
            value = c_value.number_input(f"{target} at least",
                                         key=value_key)
        rule = {"mode": mode, "value": value}
        st.caption(f"Yes: {ml.resolve_rule(df, target, rule)['describe']}.")
    classes = (_classes(df, target, rule) if task == "classification"
               else [])
    positive = None
    if len(classes) == 2:
        if rule:
            positive = ml_recipe.YES
        else:
            pos_key = f"{PREFIX}positive_{task}_{target}"
            _seed(pos_key, ml.default_positive(classes), classes)
            positive = st.selectbox("Class of interest", classes,
                                    key=pos_key)
            st.caption("Probabilities, gains and lift are for this class.")
    _target_summary(df, task, target, rule, positive)
    return target, rule, positive, classes


def _target_summary(df, task, target, rule, positive):
    recipe = {"task": task, "target": target,
              "target_rule": ({"at_least": ml.resolve_rule(
                  df, target, rule)["at_least"]} if rule else None)}
    y = ml_recipe.target_values(df, recipe)
    missing = int(y.isna().sum())
    if task == "classification":
        counts = y.value_counts()
        if 1 < len(counts) <= ml.MAX_CLASSES:
            table = pd.DataFrame({"Class": counts.index.astype(str),
                                  "Rows": counts.to_numpy(),
                                  "Share": counts.to_numpy() / counts.sum()})
            st.dataframe(table, hide_index=True, width="content",
                         column_config={"Share": st.column_config
                                        .NumberColumn(format="percent")})
            share = (counts.get(positive, 0) / counts.sum() if positive
                     else counts.min() / counts.sum())
            if share < 0.10:
                who = (f"The class of interest is {share:.1%} of rows"
                       if positive else
                       f"The smallest class is {share:.1%} of rows")
                st.caption(f"{who}: accuracy alone flatters a model on a "
                           f"file like this. ROC AUC and PR AUC are better "
                           f"guides, and balancing the classes can help.")
    else:
        v = y.dropna()
        if len(v):
            st.caption(f"{target}: mean {v.mean():,.4g} · median "
                       f"{v.median():,.4g} · min {v.min():,.4g} · max "
                       f"{v.max():,.4g}.")
            if v.min() >= 0 and len(v) > 2 and v.skew() > 2:
                st.caption(f"{target} is strongly skewed: a few rows hold "
                           f"much of it. A log scale (under Preparation) may "
                           f"suit it.")
    if missing:
        st.caption(f"{missing:,} row{'' if missing == 1 else 's'} ha"
                   f"{'s' if missing == 1 else 've'} no value for {target}: "
                   f"left out of training and testing, and scored by the "
                   f"model.")


def _past_columns():
    """Columns made by a "lags" step: past values, which predict the
    target strongly by design and never contain the row's own value."""
    out = set()
    for s in st.session_state.get("ml_steps") or []:
        if s["kind"] == "lags":
            out |= {f"{s['column']}_lag_{k}" for k in s.get("lags") or []}
            out |= {f"{s['column']}_mean_{w}"
                    for w in s.get("windows") or []}
    return out


def _default_features(prof, task, target, leaks):
    ok = prof[prof["use"] & (prof["column"] != target)]
    past = _past_columns()
    cols = [c for c in ok["column"] if c not in leaks or c in past]
    if task == "clustering":
        # Segments by behaviour: geography and categories would otherwise
        # decide the clusters (a region code separated the sample's best).
        num = [c for c in cols
               if ok.set_index("column").loc[c, "type"] == "numeric"
               and not ml._hint(c, ml.GEO_HINTS)]
        return num or cols
    return cols


def _features(df, prof, task, target, rule):
    ss = st.session_state
    st.markdown("#### Features")
    pool = [c for c in prof["column"] if c != target]
    leaks = (_leakage(df, task, target, rule, pool)
             if task != "clustering" else {})
    defaults = _default_features(prof, task, target, leaks)
    basis = (task, target, rule is not None)
    key = f"{PREFIX}features"
    if ss.get("ml_basis") != basis:
        # A new task or target re-picks the features: what leaks, and what
        # the target itself is, have changed with it.
        ss[key] = list(defaults)
        ss["ml_basis"] = basis
    _seed_many(key, defaults, pool)
    features = st.multiselect("The columns to learn from", pool, key=key,
                              placeholder="Choose columns")
    types = prof.set_index("column")["type"]
    numeric_pool = [c for c in features if types[c] == "numeric"]
    cat_key = f"{PREFIX}as_category"
    _seed_many(cat_key, [], numeric_pool)
    as_cat = st.multiselect(
        "Numbers to read as categories", numeric_pool, key=cat_key,
        placeholder="None") if numeric_pool else []
    if numeric_pool:
        st.caption("For codes stored as numbers, where a higher value is not "
                   "more of anything.")
    shown = [c for c in features if c in leaks]
    if shown:
        st.warning("Possible leakage: " + " ".join(
            f"**{c}**: {leaks[c]}" for c in shown))
    left_out = [c for c in leaks if c not in features]
    if left_out:
        st.caption("Left out as possible leakage: " + ", ".join(left_out)
                   + ". Each carries the target, on its own or through what "
                     "it was built from; Column details gives each reason. "
                     "Add one back above to use it anyway.")
    with st.expander("Column details"):
        table = prof.assign(used=prof["column"].isin(features))
        # The leakage reason in place of the profile's note: it is the one
        # that decided the default.
        table["note"] = [leaks.get(c, n) for c, n in zip(table["column"],
                                                         table["note"])]
        table = table[["column", "type", "unique", "missing", "used",
                       "note"]].rename(columns={
                           "column": "Column", "type": "Type",
                           "unique": "Distinct", "missing": "Missing",
                           "used": "Used", "note": "Note"})
        st.dataframe(table, hide_index=True, width="stretch")
    numeric = [c for c in features if types[c] == "numeric"
               and c not in as_cat]
    categorical = [c for c in features if c not in numeric]
    return numeric, categorical


def _choice(label, options, key, default, column=None):
    _seed(key, default, list(options))
    where = column or st
    return where.selectbox(label, list(options), key=key,
                           format_func=options.get)


def _prep(task, df, target):
    st.markdown("#### Preparation")
    c1, c2, c3, c4 = st.columns(4)
    # A file with numeric gaps starts on a median fill: left missing, the
    # default model set (with a linear model) would be refused outright.
    gaps = bool(df.select_dtypes("number").isna().to_numpy().any())
    prep = {
        "num_missing": _choice("Missing numbers", NUM_MISSING,
                               f"{PREFIX}num_missing",
                               "median" if gaps else "none", c1),
        "cat_missing": _choice("Missing categories", CAT_MISSING,
                               f"{PREFIX}cat_missing", "category", c2),
        "scale": _choice("Scaling", SCALING, f"{PREFIX}scale", "standard",
                         c3),
        "rare_share": _choice("Group rare categories", RARE,
                              f"{PREFIX}rare_share", 0.0, c4),
    }
    enc_options = {k: v for k, v in ENCODING.items()
                   if task != "clustering" or k != "target"}
    e1, e2 = st.columns([1, 3])
    prep["cat_encoding"] = _choice("Categorical encoding", enc_options,
                                   f"{PREFIX}cat_encoding", "onehot", e1)
    e2.caption(ENCODING_HELP[prep["cat_encoding"]]
               + (" Gradient boosting reads categories natively unless "
                  "they are target encoded."
                  if prep["cat_encoding"] != "target" else ""))
    k1, k2 = st.columns([1, 3])
    prep["cap"] = _choice("Cap extreme numbers", CAPS, f"{PREFIX}cap", "none",
                          k1)
    k2.caption("Levels learnt from the training rows of each fit, never the "
               "test rows, and fixed in the saved model. Helps linear "
               "models, neighbours and neural networks; trees and boosting "
               "are unaffected by it.")
    flags_key = f"{PREFIX}missing_flags"
    _seed(flags_key, False)
    prep["missing_flags"] = st.checkbox(
        "Add a flag column for each numeric gap", key=flags_key)
    st.caption("With no fill, a gap reaches the model as a gap: trees, "
               "forests, gradient boosting, XGBoost and LightGBM read it "
               "directly, and any other model asks for a fill when a feature "
               "has one. Scaling matters to linear models, neighbours, "
               "support vector machines, neural networks and clustering; "
               "trees are unaffected by it.")
    prep["log_target"] = False
    prep["pca"] = 0
    if task == "regression":
        log_key = f"{PREFIX}log_target"
        _seed(log_key, False)
        prep["log_target"] = st.checkbox("Model the target on a log scale",
                                         key=log_key)
        st.caption("Suits a skewed, non-negative target such as revenue. "
                   "Predictions and every reported number stay on the "
                   "target's own scale.")
    if task == "clustering":
        prep["pca"] = _choice("Reduce to principal components first", PCA,
                              f"{PREFIX}pca", 0)
    return prep


def _smallest_share(df, task, target, rule):
    """The smallest class's share of the rows, for a classification target."""
    if task != "classification":
        return None
    recipe = {"task": task, "target": target,
              "target_rule": ({"at_least": ml.resolve_rule(
                  df, target, rule)["at_least"]} if rule else None)}
    return ml.smallest_share(ml_recipe.target_values(df, recipe))


def _validation(df, prof, task, target, rule, classes):
    if task == "clustering":
        st.markdown("#### Number of clusters")
        scan_key = f"{PREFIX}k_scan"
        _seed(scan_key, True)
        scan = st.checkbox("Also compare K-means at 2 up to a maximum number "
                           "of clusters", key=scan_key)
        k_scan = None
        if scan:
            max_key = f"{PREFIX}k_max"
            _seed(max_key, 8)
            k_scan = {"max_k": st.slider("Largest number of clusters to "
                                         "compare", 3, 12, key=max_key)}
            st.caption("The comparison suggests a number of clusters; set it "
                       "on the model below and train again to use it.")
        return {"test_size": 0.0, "folds": 0, "stratify": False,
                "group": None}, None, False, k_scan
    st.markdown("#### Validation")
    # Dates first: a time split is what they are for. Numbers that count
    # time (a month index, a year) work too.
    dates = [c for c in prof.loc[prof["kind"] == "date", "column"]
             if c != target]
    numbers = [c for c, t in zip(prof["column"], prof["type"])
               if t == "numeric" and c != target and c not in dates]
    s_key = f"{PREFIX}split"
    _seed(s_key, "random", list(SPLITS))
    how = st.radio("Split", list(SPLITS), key=s_key, horizontal=True,
                   format_func=SPLITS.get)
    time = None
    if how == "time":
        t_key = f"{PREFIX}time_col"
        _seed(t_key, (dates + numbers + [None])[0], dates + numbers)
        time = st.selectbox("Time column", dates + numbers, key=t_key,
                            help="The date (or number) each row's outcome "
                                 "belongs to: a renewal date, a month.")
        st.caption("The latest rows are the test rows, and every training "
                   "row comes before them. Cross-validation walks forward: "
                   "each fold trains on the rows before it and validates "
                   "on the next block, so the earliest block is only "
                   "trained on. Use it whenever new data will come from "
                   "later dates than the training data.")
    c1, c2, c3 = st.columns(3)
    share_key = f"{PREFIX}test_share"
    _seed(share_key, 20)
    test_share = c1.slider("Test share (%)", 10, 40, step=5, key=share_key)
    folds = _choice("Cross-validation on the training rows", FOLDS,
                    f"{PREFIX}folds", 3 if len(df) > LARGE_FILE else 5, c2)
    groups = [c for c, t, u in zip(prof["column"], prof["type"],
                                   prof["unique"])
              if t == "categorical" and 2 <= u <= ml.MAX_LEVELS
              and c != target]
    group_key = f"{PREFIX}group"
    _seed(group_key, NO_GROUP, [NO_GROUP] + groups)
    group = c3.selectbox("Hold out whole groups", [NO_GROUP] + groups,
                         key=group_key, disabled=time is not None,
                         help="Not with a time split: the test rows are "
                              "the latest ones." if time else None)
    if time:
        group = NO_GROUP
    if group != NO_GROUP:
        st.caption(f"Every row of a {group} lands on the same side, so the "
                   f"test shows how the model does on {group} values it has "
                   f"never seen.")
    family = ml.metric_family(task, len(classes) if classes else None)
    share = _smallest_share(df, task, target, rule)
    rare = share is not None and share < ml.IMBALANCED
    # A rare class starts on PR AUC (two classes) and on class weights:
    # accuracy and ROC AUC both look good on a model that ignores it.
    suffix = "_rare" if rare else ""
    m_key = f"{PREFIX}metric_{family}{suffix}"
    options = list(ml.SELECTABLE[family])
    _seed(m_key, "average_precision" if rare and family == "binary"
          else options[0], options)
    c4, c5, c6 = st.columns(3)
    metric = c4.selectbox("Choose the best model by", options, key=m_key,
                          format_func=lambda k: ml.metric_label(family, k))
    seed_key = f"{PREFIX}seed"
    _seed(seed_key, 42)
    seed = c5.number_input("Random seed", min_value=0, max_value=1_000_000,
                           step=1, key=seed_key)
    stratify, balance, resample = False, False, "none"
    if task == "classification":
        strat_key = f"{PREFIX}stratify"
        _seed(strat_key, True)
        stratify = c6.checkbox("Same class shares in training and test",
                               key=strat_key,
                               disabled=group != NO_GROUP or time is not None)
        if rare:
            st.info(f"The smallest class is {share:.1%} of the rows (below "
                    f"{ml.IMBALANCED:.0%}), so this starts on "
                    + ("PR AUC to choose the best model, " if family ==
                       "binary" else "")
                    + "weights that balance the classes, and offers "
                      "resampling. The cut-off on the results page matters "
                      "most on a file like this.")
        bal_key = f"{PREFIX}balance{suffix}"
        _seed(bal_key, rare)
        balance = st.checkbox("Balance the classes by weight", key=bal_key)
        if balance:
            st.caption("Gives the rarer class more weight in training. "
                       "Logistic regression, trees, forests, gradient "
                       "boosting, LightGBM and support vector machines "
                       "support it; the others, XGBoost among them, train as "
                       "usual.")
        r1, r2 = st.columns([1.3, 3])
        resample = _choice("Resample the training rows", RESAMPLE,
                           f"{PREFIX}resample", "none", r1)
        r2.caption("Only rows being fitted are resampled (each fold's "
                   "training rows, then the training rows), never the rows "
                   "a score is measured on. SMOTE adds rows between "
                   "neighbours of the same class; undersampling drops rows "
                   "of the larger classes at random."
                   + (" With weights on too, the smaller class counts "
                      "twice over." if balance and resample != "none"
                      else ""))
    return ({"test_size": test_share / 100, "folds": int(folds),
             "stratify": bool(stratify) and group == NO_GROUP and not time,
             "group": None if group == NO_GROUP else group,
             "time": time},
            {"seed": int(seed), "metric": metric, "resample": resample},
            balance, None)


def _grid_default(spec):
    """The default and its neighbours: a small first grid to widen."""
    values = spec["values"]
    i = values.index(spec["default"])
    return values[max(0, i - 1):i + 2]


# "No limit" is a value a setting can take (a tree's maximum depth), but None
# in a widget's session state reads as "nothing chosen", so it travels
# through the widgets as this word and is turned back on the way out.
_NONE = "unlimited"


def _to_widget(v):
    return _NONE if v is None else v


def _from_widget(v):
    return None if v == _NONE else v


def _param_widget(task, key, spec, tune):
    name = spec["name"]
    options = [_to_widget(v) for v in spec["values"]]

    def show(v):
        return ml.show_value(spec, _from_widget(v))

    if tune:
        k = f"{PREFIX}t_{task}_{key}_{name}"
        _seed_many(k, [_to_widget(v) for v in _grid_default(spec)], options)
        picked = st.multiselect(f"{spec['label']}: values to try", options,
                                key=k, format_func=show)
        return [_from_widget(v) for v in picked]
    k = f"{PREFIX}p_{task}_{key}_{name}"
    _seed(k, _to_widget(spec["default"]), options)
    if spec["kind"] == "choice":
        value = st.selectbox(spec["label"], options, key=k, format_func=show)
    else:
        value = st.select_slider(spec["label"], options, key=k,
                                 format_func=show)
    return _from_widget(value)


def _models(task, n_rows):
    st.markdown("#### Models")
    blocked = ml.unavailable(task, n_rows)
    avail = {k: m for k, m in ml.MODELS[task].items() if k not in blocked}
    key = f"{PREFIX}models_{task}"
    _seed_many(key, [k for k, m in avail.items() if m.get("default")],
               list(avail))
    chosen = st.multiselect("Models to train", list(avail), key=key,
                            format_func=lambda k: avail[k]["label"],
                            placeholder="Choose models")
    if task != "clustering":
        st.caption("A baseline that ignores every feature is trained "
                   "alongside: a model is only worth using if it beats it.")
    if blocked:
        st.caption(" ".join(blocked.values()))
    mode = "fixed"
    n_iter, minutes = 10, 0
    if task != "clustering":
        tuning = {k: v for k, v in TUNING.items()
                  if k != "bayes" or not ml.package_problem("optuna")}
        mode_key = f"{PREFIX}tune_mode"
        _seed(mode_key, "fixed", list(tuning))
        mode = st.radio("Settings", list(tuning), key=mode_key,
                        horizontal=True, format_func=tuning.get)
        if mode in ("random", "bayes"):
            c1, c2 = st.columns(2)
            it_key = f"{PREFIX}n_iter"
            _seed(it_key, 10 if mode == "random" else 30)
            n_iter = c1.number_input("Combinations to try", min_value=2,
                                     max_value=100, step=1, key=it_key)
        if mode == "bayes":
            min_key = f"{PREFIX}tune_minutes"
            _seed(min_key, 0)
            minutes = c2.number_input("Time limit per model (minutes, 0 for "
                                      "none)", min_value=0, max_value=240,
                                      step=5, key=min_key)
            st.caption("Optuna's TPE sampler picks each next combination "
                       "from the scores of those already tried, and stops "
                       "one early when it trails the others after a fold. "
                       "It finds good settings in fewer tries than random "
                       "search on wide grids.")
        if mode != "fixed":
            st.caption("Each combination is scored by cross-validation on "
                       "the training rows; the best is then trained on all of "
                       "them and judged on the test rows.")
    out = []
    for k in chosen:
        m = ml.MODELS[task][k]
        params, space = ml.default_params(task, k), {}
        with st.expander(f"{m['label']} settings"):
            st.caption(m["about"])
            if not m["params"]:
                st.caption("No settings to choose.")
            for spec in m["params"]:
                value = _param_widget(task, k, spec, mode != "fixed")
                if mode == "fixed":
                    params[spec["name"]] = value
                else:
                    space[spec["name"]] = list(value) or [spec["default"]]
        search = ({"mode": mode, "space": space, "n_iter": int(n_iter),
                   "seconds": int(minutes) * 60}
                  if mode != "fixed" and space else None)
        out.append({"key": k, "params": params, "search": search})
    return out


def _duration(seconds) -> str:
    if seconds < 90:
        return f"{int(round(seconds, -1)) or 10} seconds"
    minutes = max(int(round(seconds / 60)), 1)
    return f"{minutes} minute{'' if minutes == 1 else 's'}"


CALIBRATE = {"none": "Off", "sigmoid": "Platt scaling (small files)",
             "isotonic": "Isotonic (large files)"}
STACK = {0: "Off", **{n: f"Stack the best {n}" for n in ml.STACK_TOP}}


def _combine(task, folds):
    """Calibration (classification) and a stack of the best models."""
    st.markdown("#### Combine and calibrate")
    c1, c2 = st.columns(2)
    stack = _choice("Stack the best models", STACK, f"{PREFIX}stack", 0, c1)
    if stack and not folds:
        c1.caption("Needs cross-validation: the best models are picked by "
                   "their cross-validated score, never the test rows.")
    elif stack:
        c1.caption("Fits the best models by cross-validated score together "
                   "under a simple model that weighs them; it joins the "
                   "leaderboard and wins only if it scores better. Takes "
                   "about four times as long as its members. XGBoost is "
                   "left out of stacks.")
    calibration = "none"
    if task == "classification":
        calibration = _choice("Calibrate probabilities", CALIBRATE,
                              f"{PREFIX}calibration", "none", c2)
        if calibration != "none":
            c2.caption("Makes a probability of 0.3 mean three in ten: worth "
                       "it when probabilities are read as risks or summed "
                       "into forecasts. Rankings, and so AUC, barely move. "
                       "After resampling it calibrates to the resampled "
                       "shares.")
    return {"stack": int(stack), "calibration": calibration}


TUNED = {"grid": "grid search", "random": "random search",
         "bayes": "Bayesian search"}


def _summary_rows(config, n_rows):
    """(label, value) lines for what is about to be trained."""
    task, val = config["task"], config["validation"]
    rows = [("Task", ml.TASK_LABELS[task])]
    if config.get("target"):
        rows.append(("Target", config["target"] + (
            f" ({config['positive']})" if config.get("positive") else "")))
    rows.append(("Features", f"{len(config['numeric'])} numeric, "
                             f"{len(config['categorical'])} categorical"))
    if task != "clustering":
        rows.append(("Validation", f"{val['test_size']:.0%} "
                     + ("latest as test" if val.get("time") else "test")
                     + (f", {val['folds']} "
                        + ("walk-forward folds" if val.get("time")
                           else "folds") if val["folds"] else ", no folds")))
        if val.get("time"):
            rows.append(("Time column", val["time"]))
        family = ml.metric_family(task, None)
        if config.get("metric"):
            for fam in ml.SELECTABLE:
                if config["metric"] in ml.SELECTABLE[fam]:
                    family = fam
            rows.append(("Best model by",
                         ml.metric_label(family, config["metric"])))
    extras = []
    if config.get("balance"):
        extras.append("class weights")
    if (config.get("resample") or "none") != "none":
        extras.append(RESAMPLE[config["resample"]].split(" (")[0].lower())
    if (config["prep"].get("cap") or "none") != "none":
        extras.append("caps")
    if config.get("stack"):
        extras.append(f"stack of {config['stack']}")
    if (config.get("calibration") or "none") != "none":
        extras.append("calibration")
    if extras:
        rows.append(("Also", ", ".join(extras)))
    return rows


def _summary_card(df, config, errors):
    """The right-hand panel: the plan, anything in the way, and Train.
    Returns (train in the background, train here)."""
    st.markdown("#### Training plan")
    lines = "".join(
        f"<div><span>{html.escape(k)}</span><b>{html.escape(str(v))}</b>"
        f"</div>" for k, v in _summary_rows(config, len(df)))
    st.markdown(f"<div class='plan-rows'>{lines}</div>",
                unsafe_allow_html=True)
    names = "".join(
        f"<li>{html.escape(ml.model_label(config['task'], m['key']))}"
        + (f" <span>{TUNED[m['search']['mode']]}</span>"
           if m.get("search") else "") + "</li>"
        for m in config["models"])
    st.markdown(f"<div class='plan-models'><span>Models "
                f"({len(config['models'])})</span><ul>{names}</ul></div>"
                if names else "<div class='plan-models'><span>No models "
                "chosen</span></div>", unsafe_allow_html=True)
    for e in errors:
        st.warning(e)
    if not errors:
        _plan_line(config, len(df))
        # Known when the prediction is made? Cautions only: the person who
        # knows the data decides.
        for c in asof.cautions(df, config,
                               st.session_state.get("ml_steps")):
            st.warning(c, icon=":material/schedule:")
    from components import jobs_panel
    blocked = bool(errors) or not auth.can("train")
    background = False
    if jobs_panel.available():
        background = st.button(
            "Train in the background", type="primary", width="stretch",
            disabled=blocked, key="ml_train_bg",
            help="Runs on a worker: keep working, leave the page, or close "
                 "the tab; the run is saved to the project.")
        go = st.button("Train here and wait", width="stretch",
                       disabled=blocked, key="ml_train",
                       help=auth.why_not("train") or "Fine for quick runs.")
    else:
        go = st.button("Train models", type="primary", width="stretch",
                       disabled=blocked, help=auth.why_not("train"),
                       key="ml_train")
    return background, go


def _plan_line(config, n_rows):
    p = ml.plan(config)
    n = len(config["models"])
    if config["task"] == "clustering":
        line = (f"Plan: {n} model{'' if n == 1 else 's'} on all "
                f"{n_rows:,} rows")
        if config.get("k_scan"):
            line += (f", and K-means at 2 to {config['k_scan']['max_k']} "
                     f"clusters")
    else:
        n_train = int(n_rows * (1 - config["validation"]["test_size"]))
        folds = config["validation"]["folds"]
        line = (f"Plan: {n} model{'' if n == 1 else 's'} and the baseline"
                + (f", {folds}-fold cross-validation" if folds else "")
                + f": {p['fits']:,} fits on about {n_train:,} training rows")
    est = ml.estimate_seconds(config, n_rows)
    if est >= 20:
        line += (f". Estimated time: about {_duration(est)}, measured on a "
                 f"four-core machine; a smaller server takes longer")
    st.caption(line + ".")


def _code(df, config):
    """The choices on screen as a standalone script, before training."""
    code = ml_eval.config_script(df, config,
                                 st.session_state.get("ml_steps"))

    @log.logged_download(_log, "ml_config_script", task=config["task"],
                         models=len(config["models"]))
    def _download():
        return code.encode("utf-8")

    with st.expander("Python code for these choices"):
        st.caption("Trains the models as set above, with any tuning as a "
                   "scikit-learn search, on a file with the same columns. "
                   "After training, the results page has the code with the "
                   "settings the run chose.")
        st.download_button("Download the code (Python)", data=_download,
                           file_name="train_configured.py",
                           mime="text/x-python", key="ml_cfg_code")
        st.code(code, language="python", height=CODE_HEIGHT)


def _train(df, config):
    ss = st.session_state
    bar = st.progress(0.0, text="Training models")

    def progress(fraction, text):
        bar.progress(min(max(float(fraction), 0.0), 1.0), text=text)

    steps = list(ss.get("ml_steps") or [])
    run = ml.train(df, config, progress, source=ss.get("ml_source"),
                   steps=steps)
    # The scripts, the model file and a file scored later replay these:
    # with learnt values from the training rows when training set them.
    run.setdefault("prep_steps", steps)
    run["importance"] = {}
    if run.get("frame") is not None:
        df = run["frame"]
    if run["task"] != "clustering" and run["best"]:
        res = ml.result(run, run["best"])
        if ml_eval.importance_seconds(run, res) <= AUTO_IMPORTANCE_S:
            bar.progress(1.0, text="Measuring what drives the best model")
            run["importance"][run["best"]] = \
                ml_eval.permutation_importance(run, res, df)
    ss["ml_run"] = run
    ss["page"] = "ml_results"
    scroll.request()
    st.rerun()


def render():
    ss = st.session_state
    df = ss["ml_df"]
    _restore()
    prof = _profile(df)
    _header(df)
    if ss.get("ml_job"):
        from components import jobs_panel
        jobs_panel.watch()
    from components import assistant_panel   # imports this module
    guard.section("Assistant", assistant_panel.render, df, prof)
    # The choices on the left, one panel each; what they add up to, and the
    # Train button, stay in view on the right.
    main, side = st.columns([2.5, 1], gap="medium")

    def panel(name):
        return st.container(border=True, key=f"card_cfg_{name}")

    target = rule = positive = None
    classes = []
    with main:
        with panel("target"):
            task = _task()
            picked = (_target(df, prof, task) if task != "clustering"
                      else ())
        if picked is None:
            _save()
            return
        if picked:
            target, rule, positive, classes = picked
        with panel("features"):
            numeric, categorical = _features(df, prof, task, target, rule)
        with panel("prep"):
            prep = _prep(task, df, target)
        with panel("validation"):
            validation, extra, balance, k_scan = _validation(
                df, prof, task, target, rule, classes)
        with panel("models"):
            models = _models(task, len(df))
        combine = {}
        if task != "clustering":
            with panel("combine"):
                combine = _combine(task, validation.get("folds"))
    config = {"task": task, "target": target, "positive": positive,
              "target_rule": rule, "numeric": numeric,
              "categorical": categorical, "prep": prep,
              "validation": validation,
              "seed": (extra or {}).get("seed", 42),
              "metric": (extra or {}).get("metric"), "balance": balance,
              "resample": (extra or {}).get("resample", "none"),
              "models": models, "k_scan": k_scan, **combine,
              # The page as it stands, so a run can be opened here again
              # with the same choices ("Clone settings" on the Runs page).
              "widgets": {k: v for k, v in ss.items()
                          if isinstance(k, str) and k.startswith(PREFIX)}}
    errors = ml.validate(df, config)
    with side, st.container(border=True, key="card_plan"):
        background, go = _summary_card(df, config, errors)
    if not errors:
        with main:
            _code(df, config)
    from components import jobs_panel
    _save()
    if background:
        guard.action("Queueing", "The training job could not be queued.",
                     jobs_panel.submit, config)
    if go:
        guard.action("Training", "The models could not be trained.", _train,
                     df, config)
