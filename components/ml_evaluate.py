"""Use own model: evaluate a model you already have on the file loaded here
and see it on the results page, or have it predict every row of this file,
a new file or a Snowflake table.

The file is read by `logic.ml_upload`, which never runs what a model file
asks it to unless it is on the allow-list; XGBoost's .json/.ubj and
LightGBM's .txt formats run no code at all. Each of the model's inputs is
matched to a file column (by name where the model kept its column names, by
position otherwise), the target is chosen, and the model predicts every row.
"""
import hashlib

import pandas as pd
import streamlit as st

from components import ml_config, snowflake_panel
from logic import ml, ml_prep, ml_recipe, ml_upload
from utils import guard, log, scroll
from utils.file_loader import load_file

_log = log.get("ml_evaluate")

PREFIX = "ml_u_"
CHOOSE = "(choose a column)"
NO_TARGET = "(none)"


def _seed(key, default, options=None):
    ss = st.session_state
    if key not in ss or (options is not None and ss[key] not in options):
        ss[key] = default


def _header(df):
    st.markdown("## Use own model")
    st.caption(f"Instead of training: {len(df):,} rows × {df.shape[1]} "
               f"columns. Evaluate a model you already have against the "
               f"actual values here, beside a baseline that ignores every "
               f"feature — or have it predict every row of this file, a new "
               f"file or a Snowflake table.")


def _read(upload):
    """Read once per file: a rerun hands the same upload back."""
    ss = st.session_state
    data = upload.getvalue()
    digest = hashlib.md5(data).hexdigest()
    if ss.get("ml_u_digest") != digest:
        info, why = ml_upload.read(data, upload.name)
        ss["ml_u_digest"] = digest
        ss["ml_u_info"] = info
        ss["ml_u_error"] = why
        for k in [k for k in ss if isinstance(k, str)
                  and k.startswith(PREFIX + "map")]:
            del ss[k]
    return ss.get("ml_u_info"), ss.get("ml_u_error")


def _frame(info, df):
    """The file the model is evaluated on (`ml_upload.file_for`), built once
    per model file: replaying its steps on every rerun would cost seconds at
    300,000 rows."""
    ss = st.session_state
    if not info.get("prep_steps"):
        return df, None
    if ss.get("ml_u_frame_for") != ss.get("ml_u_digest"):
        ss["ml_u_frame"] = ml_upload.file_for(info, ss["ml_source"], df)
        ss["ml_u_frame_for"] = ss.get("ml_u_digest")
    return ss["ml_u_frame"]


def _profile(df):
    """The profile of the file being evaluated: the configure page's own
    when that is the file, otherwise one of its own."""
    ss = st.session_state
    if df is ss["ml_df"]:
        if ss.get("ml_profile") is None:
            ss["ml_profile"] = ml.profile(df)
        return ss["ml_profile"]
    if ss.get("ml_u_profile_for") != ss.get("ml_u_digest"):
        ss["ml_u_profile"] = ml.profile(df)
        ss["ml_u_profile_for"] = ss.get("ml_u_digest")
    return ss["ml_u_profile"]


def _found(info):
    kind = {"app": "This app's model file",
            "estimator": "A scikit-learn model",
            "xgboost": "An XGBoost model file",
            "lightgbm": "A LightGBM model file"}[info["kind"]]
    lines = [f"**{info['label']}** · {kind} · "
             f"{ml.TASK_LABELS[info['task']].lower()}"]
    n = info["n_features"]
    if info["features"]:
        lines.append(f"Reads {n} column{'' if n == 1 else 's'} by name.")
    elif n:
        lines.append(f"Reads {n} column{'' if n == 1 else 's'} by position: "
                     f"it was fitted without column names.")
    if info.get("classes"):
        shown = ", ".join(map(str, info["classes"][:8]))
        lines.append(f"Classes: {shown}"
                     + (f" and {len(info['classes']) - 8} more"
                        if len(info["classes"]) > 8 else "") + ".")
    if info["kind"] == "app":
        recipe = info["recipe"]
        if recipe.get("target"):
            lines.append(f"Trained {info.get('trained') or ''} to predict "
                         f"{recipe['target']}.".replace("  ", " "))
        if info.get("prep_steps"):
            k = len(info["prep_steps"])
            lines.append(
                f"It carries the {k} preparation step"
                f"{'' if k == 1 else 's'} it was trained after: the column "
                f"steps are replayed on the file as loaded, and every row is "
                f"kept"
                + ("; the steps made on this visit are set aside"
                   if st.session_state.get("ml_steps") else "") + ".")
    if info.get("task_guessed"):
        lines.append("Whether it classifies or predicts a number could not "
                     "be read from it: choose below.")
    st.markdown("  \n".join(lines))


def _mapping(df, info):
    """{model input: file column}, every input matched, or None."""
    columns = [str(c) for c in df.columns]
    if info["features"]:
        names = list(info["features"])
        auto = ml_upload.auto_map(names, columns)
        missing = [n for n, c in auto.items() if c is None]
        if not missing:
            st.caption("Every input matched a column of the same name.")
            return auto, False
        st.markdown(f"**{len(missing)} of the model's inputs have no column "
                    f"of the same name**: choose the column each is read "
                    f"from.")
        out = dict(auto)
        for name in missing:
            key = f"{PREFIX}map_{hashlib.md5(name.encode()).hexdigest()[:10]}"
            _seed(key, CHOOSE, [CHOOSE] + columns)
            out[name] = st.selectbox(name, [CHOOSE] + columns, key=key)
        if any(v == CHOOSE for v in out.values()):
            return None, False
        return out, False
    n = info["n_features"] or 0
    if not n:
        st.warning("The model does not say how many columns it reads.")
        return None, True
    key = f"{PREFIX}map_position"
    if key not in st.session_state:
        st.session_state[key] = []
    picked = st.multiselect(f"The {n} columns it reads, in the order it was "
                            f"trained on", columns, key=key, max_selections=n)
    if len(picked) < n:
        st.caption(f"{len(picked)} of {n} chosen.")
        return None, True
    return {f"input {i + 1}": c for i, c in enumerate(picked)}, True


def _target(df, info, task):
    columns = [str(c) for c in df.columns]
    key = f"{PREFIX}target"
    default = NO_TARGET
    recipe = info.get("recipe") or {}
    if recipe.get("target") in columns:
        default = recipe["target"]
    else:
        cands = ml.target_candidates(_profile(df), task)
        default = cands[0] if cands else NO_TARGET
    _seed(key, default, [NO_TARGET] + columns)
    target = st.selectbox("The column with the actual values", [NO_TARGET]
                          + columns, key=key)
    if recipe.get("target_rule") and target == recipe.get("target"):
        rule = recipe["target_rule"]
        st.caption(f"Read as the model was trained: {rule.get('describe')}.")
    return None if target == NO_TARGET else target


@guard.guarded("Evaluate a model")
def render():
    ss = st.session_state
    df = ss["ml_df"]
    _header(df)
    if not ml_config.upload_allowed():
        st.info("Model upload is turned off on this deployment "
                "(ML_MODEL_UPLOAD=off).")
        return
    st.caption("Accepted: this app's model file (.joblib), a scikit-learn, "
               "XGBoost or LightGBM model saved with joblib or pickle, an "
               "XGBoost .json or .ubj model, or a LightGBM .txt model. A "
               "joblib or pickle file can run code as it loads, so it is "
               "read through an allow-list: only model parts are built, and "
               "the load stops at anything else before it runs. The .json, "
               ".ubj and .txt formats run nothing.")
    upload = st.file_uploader("Model file", type=[e.lstrip(".") for e in
                                                  ml_upload.EXTENSIONS],
                              key=f"{PREFIX}file")
    if upload is None:
        return
    info, why = _read(upload)
    if why:
        st.warning(why)
        return
    _found(info)
    df, why = _frame(info, df)
    if why:
        st.warning(why)
        return
    task = info["task"]
    if info.get("task_guessed"):
        t_key = f"{PREFIX}task"
        _seed(t_key, "regression", ["classification", "regression"])
        task = st.radio("It predicts", ["classification", "regression"],
                        horizontal=True, key=t_key,
                        format_func=ml.TASK_LABELS.get)
    st.markdown("#### Its inputs")
    mapping, position = _mapping(df, info)
    m_key = f"{PREFIX}mode"
    _seed(m_key, "evaluate", ["evaluate", "predict"])
    mode = st.radio("What to do", ["evaluate", "predict"], key=m_key,
                    horizontal=True, format_func={
                        "evaluate": "Evaluate against actual values",
                        "predict": "Predict (no actual values needed)"}.get)
    if mode == "predict":
        if mapping is None:
            st.caption("Match every input to a column first.")
            return
        _predict(df, info, mapping, task, position)
        return
    target = positive = None
    if task != "clustering":
        st.markdown("#### What it predicts")
        target = _target(df, info, task)
        if target is None:
            st.caption("Choose the column holding the actual values: the "
                       "model is scored against it.")
        elif task == "classification":
            known = sorted(ml_recipe.label_text(df[target]).dropna()
                           .unique().tolist(), key=str)
            if len(known) == 2:
                p_key = f"{PREFIX}positive"
                default = (info.get("positive")
                           if info.get("positive") in known
                           else ml.default_positive(known))
                _seed(p_key, default, known)
                positive = st.selectbox("Class of interest", known,
                                        key=p_key)
    ready = mapping is not None and (task == "clustering" or target)
    go = st.button("Evaluate on this file", type="primary",
                   disabled=not ready, key="ml_u_go")
    if go:
        with st.spinner("Predicting every row"):
            run, why = ml_upload.evaluate(df, info, mapping, task, target,
                                          positive, _profile(df), position)
        if why:
            st.warning(why)
            return
        ss["ml_run"] = run
        ss["page"] = "ml_results"
        scroll.request()
        st.rerun()


def _prediction_source(df, info):
    """(frame, label) to predict: this file, an upload or a Snowflake
    table. A new file gets this app's model file's own preparation."""
    ss = st.session_state
    s_key = f"{PREFIX}source"
    _seed(s_key, "this", ["this", "upload", "snowflake"])
    source = st.radio("Rows to predict", ["this", "upload", "snowflake"],
                      key=s_key, horizontal=True, format_func={
                          "this": "This file", "upload": "Upload a file",
                          "snowflake": "A Snowflake table"}.get)
    if source == "this":
        return df, "this file"
    if source == "upload":
        up = st.file_uploader("A file with the model's input columns",
                              type=["csv", "xlsx", "xls"],
                              key=f"{PREFIX}pred_file")
        new = load_file(up) if up is not None else None
        label = up.name if up is not None else None
    else:
        from logic import snowflake_io
        if not snowflake_io.available():
            st.info(snowflake_io.unavailable_reason())
            return None, None
        loaded = guard.section("Snowflake", snowflake_panel.source_picker,
                               "ml_u_pred")
        if loaded is not None:
            ss[f"{PREFIX}sf_frame"] = loaded
        new, label = ss.get(f"{PREFIX}sf_frame"), "the Snowflake data"
    if new is None:
        return None, None
    steps = info.get("prep_steps") or []
    if steps:
        try:
            new = ml_prep.apply_steps(new, steps, rows=False)
        except ml_prep.StepError as e:
            st.warning(f"The model's preparation does not fit this file. {e}")
            return None, None
    return new, label


@guard.guarded("Predict")
def _predict(df, info, mapping, task, position):
    ss = st.session_state
    st.markdown("#### Predict")
    data, label = _prediction_source(df, info)
    if data is None:
        return
    if st.button(f"Predict {len(data):,} rows", type="primary",
                 key=f"{PREFIX}predict"):
        with st.spinner("Predicting every row"):
            out, why = ml_upload.predict_only(
                data, info, mapping, task,
                _profile(df) if info["kind"] not in ("app", "xgboost",
                                                     "lightgbm") else None,
                position)
        if why:
            st.warning(why)
            return
        ss[f"{PREFIX}predicted"] = {"frame": out, "label": label}
    got = ss.get(f"{PREFIX}predicted")
    if not got:
        return
    out = got["frame"]
    col = next(c for c in reversed(out.columns)
               if str(c).startswith(("ML_Prediction", "ML_Cluster")))
    st.success(f"Predicted {len(out):,} rows of {got['label']}.")
    if task == "regression":
        v = pd.to_numeric(out[col], errors="coerce")
        st.caption(f"{col}: mean {v.mean():,.4g} · median {v.median():,.4g}"
                   f" · min {v.min():,.4g} · max {v.max():,.4g}.")
    else:
        counts = out[col].astype(str).value_counts()
        st.caption(f"{col}: " + " · ".join(
            f"{k} {n:,} ({n / len(out):.1%})" for k, n in counts.head(8)
            .items()))
    new_cols = [c for c in out.columns if str(c).startswith("ML_")]
    st.dataframe(out[new_cols + [c for c in out.columns
                                 if c not in new_cols]].head(20),
                 hide_index=True, width="stretch")

    @log.logged_download(_log, "ml_predictions_own_model", rows=len(out),
                         task=task)
    def _csv():
        return out.to_csv(index=False).encode("utf-8")

    st.download_button("Download the predictions (CSV)", data=_csv,
                       file_name="predictions.csv", mime="text/csv",
                       key=f"{PREFIX}pred_dl")
    if st.toggle("Write to Snowflake", key=f"{PREFIX}pred_sf"):
        from logic import snowflake_io
        if not snowflake_io.available():
            st.info(snowflake_io.unavailable_reason())
        else:
            guard.section("Snowflake", snowflake_panel.writer, out,
                          "ml_u_pred_w")


def frame_note(run):
    """One line for the results page: where the inputs came from."""
    up = run.get("upload") or {}
    mapped = up.get("mapped") or {}
    if up.get("by_position"):
        return "Inputs read by position."
    if mapped:
        pairs = ", ".join(f"{n} from {c}" for n, c in list(mapped.items())[:4])
        more = len(mapped) - 4
        return f"Inputs matched by hand: {pairs}" + (
            f" and {more} more." if more > 0 else ".")
    return ""

