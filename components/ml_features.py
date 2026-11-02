"""Machine Learning, step 3: feature engineering with the assistant.

Describe a column, Claude writes the script, you edit it, preview it against
the real data, and add it. Columns stack up as pending until you confirm —
nothing touches the dataset until then, so a bad idea costs nothing. On
confirm each column becomes a `script` step (`logic.ml_prep`), so the
configure page, the exported scripts and a file scored later all build it
the same way.

Layout: prompt and script pane on top, the data table beneath so you can see
what you are building against.
"""
import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_ace import st_ace

from components import input_keys, ml_eda, voice_input
from logic import derive, ml, ml_prep, schema
from utils import guard, log, scroll, secrets
from utils.constants import ML_PALETTE
from utils.frames import centre_numbers

_log = log.get("ml_features")

PREFIX = "ml_f"       # every key starts ml_, so a new file clears them
TABLE_ROWS = 200      # rows rendered in the preview grid (st.dataframe is the
                      # most expensive widget on the page at 300k rows)


def _ss():
    return st.session_state


def _k(name):
    return f"{PREFIX}_{name}"


def _pending():
    return _ss().setdefault(_k("pending"), {})


def _working_df():
    """The prepared file plus whatever columns are staged but not yet
    confirmed."""
    df = _ss()["ml_df"]
    pend = _pending()
    if not pend:
        return df
    out = df.copy(deep=False)
    for name, item in pend.items():
        out[name] = item["values"]
    return out


def _generate(prompt, key):
    df = _working_df()
    names = ", ".join(map(str, df.columns))
    try:
        with log.timer() as t:
            got = derive.generate_script(
                prompt, df, api_key=key,
                existing_note=f"Do not reuse these existing names: {names}.\n")
    except Exception as e:
        log.event(_log, "column_script_failed", level=logging.WARNING,
                  exc_info=True, ms=t.ms)
        _ss()[_k("error")] = derive.friendly_error(e)
        return
    # The timing and that it worked — never the prompt or the script.
    log.event(_log, "column_script_generated", ms=t.ms)
    _ss()[_k("error")] = None
    _ss()[_k("script")] = got["code"]
    _ss()[_k("name")] = derive.safe_name(got["name"], set(df.columns))
    _ss()[_k("note")] = got.get("explanation", "")
    _ss()[_k("nonce")] = _ss().get(_k("nonce"), 0) + 1   # re-seed editor


def _stage(code, name):
    """Run the script and hold the column as pending."""
    df = _working_df()
    try:
        values = derive.run_script(code, df)
    except derive.ScriptError as e:
        _ss()[_k("error")] = str(e)
        return
    name = derive.safe_name(name, set(df.columns))
    _pending()[name] = {"values": values, "code": code}
    for k in ("script", "name", "note", "error"):
        _ss().pop(_k(k), None)
    _ss()[_k("nonce")] = _ss().get(_k("nonce"), 0) + 1


def _confirm():
    """Every pending column becomes a script step on the prepared file."""
    pend = _pending()
    steps = list(_ss().get("ml_steps") or [])
    steps += [{"kind": "script", "into": name, "code": item["code"]}
              for name, item in pend.items()]
    ml_eda.set_steps(steps)
    log.event(_log, "ml_columns_confirmed", columns=len(pend),
              steps=len(steps))
    _ss()[_k("pending")] = {}


def _created():
    """Columns already added, each removable."""
    ss = _ss()
    steps = list(ss.get("ml_steps") or [])
    made = [(i, s) for i, s in enumerate(steps) if s["kind"] == "script"]
    if not made:
        return
    st.markdown(f"**Added to the dataset ({len(made)})**")
    for i, s in made:
        r1, r2 = st.columns([0.9, 0.1])
        r1.code(f"{s['into']}  =  {s['code'].splitlines()[-1].strip()}",
                language="python")
        if r2.button("✕", key=_k(f"remove_{i}"), help="Remove from the "
                     "dataset"):
            rest = steps[:i] + steps[i + 1:]
            try:     # checked first: set_steps keeps the steps it is given
                ml_prep.apply_steps(ss["ml_source"], rest)
            except ml_prep.StepError:
                st.warning(f"{s['into']} is used by a later step; remove "
                           f"that step first.")
                return
            ml_eda.set_steps(rest)
            log.event(_log, "ml_column_removed", steps=len(rest))
            st.rerun()


# ── AI feature ideas ────────────────────────────────────────────────────
def _task_for(df, target):
    """Classification for a category or a number with few values,
    regression otherwise — as the configure page reads a target."""
    if (schema.detect_dtype(df[target]) == "numeric"
            and df[target].nunique() > ml.MAX_CLASSES):
        return "regression"
    return "classification"


def _target_options(df):
    ss = _ss()
    prof = ml_eda._profile(ss["ml_df"])
    cands = [c for c in ml.target_candidates(prof, "classification")
             + ml.target_candidates(prof, "regression") if c in df.columns]
    cands = list(dict.fromkeys(cands))
    task = ss.get("ml_w_task", "classification")
    chosen = ss.get(f"ml_w_target_{task}")
    if chosen in cands:
        cands.remove(chosen)
        cands.insert(0, chosen)
    return cands


def _suggest(df, target, key):
    ss = _ss()
    task = _task_for(df, target)
    pool = [c for c in df.columns if c != target
            and not str(c).lower().endswith("_id")]
    with log.timer() as t:
        sig = derive.signals(df, target, task, pool)
        try:
            ideas = derive.suggest_features(df, target, task, sig,
                                            api_key=key)
        except Exception as e:
            log.event(_log, "feature_ideas_failed", level=logging.WARNING,
                      exc_info=True)
            ss[_k("ideas")] = {"error": derive.friendly_error(e)}
            return
    items = []
    for idea in ideas:
        item = dict(idea)
        try:
            values = derive.run_script(idea["code"], df)
            item["signal"] = derive.new_signal(values, df, target, task)
            item["rank"] = int((sig > item["signal"]).sum()) + 1
        except derive.ScriptError as e:
            item["error"] = str(e)
        items.append(item)
    log.event(_log, "feature_ideas", ideas=len(items), ms=t.ms,
              runnable=sum("error" not in i for i in items))
    ss[_k("ideas")] = {"target": target, "task": task, "items": items,
                       "best": float(sig.iloc[0]) if len(sig) else 0.0,
                       "best_name": sig.index[0] if len(sig) else "",
                       "n": len(sig)}


def _load(item):
    """An idea into the script pane, to edit before adding."""
    ss = _ss()
    ss[_k("script")] = item["code"]
    ss[_k("name")] = derive.safe_name(item["name"], set(_working_df().columns))
    ss[_k("note")] = item["rationale"]
    ss[_k("nonce")] = ss.get(_k("nonce"), 0) + 1


def _ideas(df, key):
    ss = _ss()
    with st.container(border=True):
        st.markdown("#### AI feature ideas")
        st.caption("Pick what the model will predict. The assistant reads "
                   "the columns and how strongly each already relates to "
                   "the target, and proposes new features; each is run "
                   "here and scored by its mutual information with the "
                   "target, beside the existing columns.")
        options = _target_options(df)
        if not options:
            st.caption("No column here can be a target.")
            return
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        t_key = _k("idea_target")
        if t_key not in ss or ss[t_key] not in options:
            ss[t_key] = options[0]
        target = c1.selectbox("Target", options, key=t_key)
        if c2.button("Suggest features", type="primary", width="stretch",
                     disabled=not key, key=_k("suggest")):
            with st.spinner("Reading the data and drafting features — "
                            "about half a minute"):
                _suggest(df, target, key)
        got = ss.get(_k("ideas"))
        if not got:
            return
        if got.get("error"):
            st.error(got["error"])
            return
        st.caption(f"For {got['target']} ({got['task']}). Strongest existing "
                   f"column: {got['best_name']}, signal {got['best']:.3f}.")
        for i, item in enumerate(got["items"]):
            left, right = st.columns([4, 1.3])
            with left:
                st.markdown(f"**{item['name']}** — {item['rationale']}")
                st.code(item["code"], language="python")
            with right:
                if item.get("error"):
                    st.caption(f"Does not run here: {item['error']}")
                    continue
                better = item["signal"] > got["best"]
                st.markdown(f"Signal **{item['signal']:.3f}**  \n"
                            f"rank {item['rank']} of {got['n'] + 1}"
                            + (" · beats every column" if better else ""))
                st.button("Edit in script pane", key=_k(f"idea_edit_{i}"),
                          width="stretch", on_click=_load, args=(item,))
                if st.button("Add", key=_k(f"idea_add_{i}"),
                             width="stretch", type="primary"):
                    _stage(item["code"], item["name"])
                    st.rerun()


@st.fragment
@guard.guarded("Explore the columns")
def _explore(df, new_cols):
    """One column's distribution, the new columns first in the picker."""
    st.markdown("#### Explore the columns")
    cols = list(new_cols) + [c for c in df.columns if c not in new_cols]
    st.caption(f"Pick {new_cols[0]} to see what you just built."
               if new_cols else
               "Check a column's shape before building against it.")
    key = _k("explore_col")
    if key not in _ss() or _ss()[key] not in cols:
        _ss()[key] = cols[0]
    col = st.selectbox("Column", cols, key=key)
    s = df[col]
    if schema.detect_dtype(s) == "numeric":
        num = pd.to_numeric(s, errors="coerce")
        fig = go.Figure(go.Histogram(x=num.dropna(), nbinsx=40,
                                     marker_color=ML_PALETTE[0]))
        st.caption(f"Mean {num.mean():,.4g} · median {num.median():,.4g} · "
                   f"min {num.min():,.4g} · max {num.max():,.4g} · "
                   f"{int(num.isna().sum()):,} gaps")
        fig.update_layout(xaxis_title=col, yaxis_title="Rows")
    else:
        counts = s.astype(object).where(s.notna(), "(blank)").astype(str) \
            .value_counts().head(20)
        fig = go.Figure(go.Bar(x=counts.values, y=counts.index,
                               orientation="h", marker_color=ML_PALETTE[0]))
        fig.update_yaxes(autorange="reversed")
        st.caption(f"{s.nunique():,} distinct values · "
                   f"{int(s.isna().sum()):,} gaps"
                   + (" · the 20 commonest shown" if s.nunique() > 20 else ""))
        fig.update_layout(xaxis_title="Rows")
    fig.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10),
                      bargap=0.1)
    st.plotly_chart(fig, width="stretch", key=_k("explore_chart"))


def render():
    ss = _ss()
    df = _working_df()
    base_cols = set(ss["ml_df"].columns)

    st.markdown("## Feature engineering")
    st.caption("Step 3 of 5. Describe a column in plain language. The "
               "assistant writes the script, you edit it, and nothing is "
               "added to your dataset until you confirm. Added columns are "
               "rebuilt in every exported script.")

    key = secrets.resolve("ANTHROPIC_API_KEY")
    guard.section("AI feature ideas", _ideas, df, key)

    # ── prompt ──────────────────────────────────────────────────────────
    # Outside the form: a widget inside one does not rerun until submit, so a
    # recording in there could never be transcribed until after it was sent.
    # It also has to run before the box, which is the only point at which a
    # keyed widget's session_state may be written.
    if key:
        voice_input.capture(PREFIX, done="Transcribed — review and generate")
    with st.form(_k("form"), clear_on_submit=False):
        prompt = st.text_area(
            "Describe the column", height=80, label_visibility="collapsed",
            key=input_keys.input_key(PREFIX),
            placeholder="e.g. claims paid per premium dollar, rounded to 2 "
                        "decimals; or flag customers whose premium rose "
                        "more than 10%")
        sent = st.form_submit_button("Generate script", type="primary",
                                     disabled=not key)
    if not key:
        st.info("Set `ANTHROPIC_API_KEY` (environment variable or "
                "`.streamlit/secrets.toml`) to generate scripts. You can "
                "still write one by hand below.")
    if sent and prompt.strip():
        with st.spinner("Writing the script"):
            _generate(prompt.strip(), key)
        # Fresh recorder for the next instruction, but the prompt stays: this
        # form keeps what was typed so it can be tweaked and re-run.
        voice_input.reset(PREFIX, clear_input=False)

    # ── script pane ─────────────────────────────────────────────────────
    ss.setdefault(_k("script"), "")
    if ss.get(_k("note")):
        st.caption(ss[_k("note")])
    c1, c2 = st.columns([0.68, 0.32])
    with c1:
        code = st_ace(value=ss[_k("script")], language="python",
                      theme="tomorrow_night", height=190, auto_update=False,
                      key=_k(f"ace_{ss.get(_k('nonce'), 0)}"),
                      placeholder="result = df['Total_Claim_Amount'] / "
                                  "df['Annual_Premium']")
    with c2:
        name = st.text_input("Column name",
                             value=ss.get(_k("name"), "New_Column"),
                             key=_k(f"nm_{ss.get(_k('nonce'), 0)}"))
        st.caption("Edit the script, then preview it against your data.")
        preview_clicked = st.button("Preview", width="stretch",
                                    disabled=not (code or "").strip(),
                                    key=_k("preview"))
        add_clicked = st.button("Add column", type="primary", width="stretch",
                                disabled=not (code or "").strip(),
                                key=_k("add"))

    if ss.get(_k("error")):
        st.error(ss[_k("error")])

    if preview_clicked:
        try:
            values = derive.run_script(code, df)
            ss[_k("error")] = None
            info = derive.preview(values, df)
            st.success(f"Runs cleanly — {info['dtype']}, "
                       f"{info['distinct']:,} distinct, "
                       f"{info['missing']:,} missing.")
            preview = pd.DataFrame({name or "preview":
                                    values.head(derive.PREVIEW_ROWS)})
            st.dataframe(preview, column_config=centre_numbers(preview),
                         width="stretch")
        except derive.ScriptError as e:
            st.error(str(e))
    if add_clicked:
        _stage(code, name)
        st.rerun()

    # ── staged columns + confirm ────────────────────────────────────────
    pend = _pending()
    st.divider()
    if pend:
        st.markdown(f"**Ready to add ({len(pend)})**")
        for nm in list(pend):
            r1, r2 = st.columns([0.9, 0.1])
            r1.code(f"{nm}  =  {pend[nm]['code'].splitlines()[-1].strip()}",
                    language="python")
            if r2.button("✕", key=_k(f"rm_{nm}"), help="Remove"):
                del pend[nm]
                st.rerun()
    else:
        st.caption("No new columns waiting to be added.")
    if "ml_run" in ss and pend:
        st.caption("Adding columns clears the trained models: they learned "
                   "from the file as it is now.")
    b1, b2 = st.columns([0.5, 0.5])
    if b1.button(f"Confirm and add {len(pend)} column"
                 f"{'s' if len(pend) != 1 else ''}",
                 type="primary", width="stretch", disabled=not pend,
                 key=_k("confirm")):
        _confirm()
        st.rerun()
    if b2.button("Continue to Configure & train →", width="stretch",
                 key=_k("continue"),
                 help="Columns still waiting to be confirmed are not added."
                 if pend else None):
        ss["page"] = "ml_config"
        scroll.request()
        st.rerun()
    _created()

    # ── data table ──────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Your data")
    new_cols = [c for c in df.columns if c not in base_cols]
    made = [s["into"] for s in ss.get("ml_steps") or []
            if s["kind"] == "script" and s["into"] in df.columns]
    first = new_cols + [c for c in made if c not in new_cols]
    st.caption(f"{len(df):,} rows × {df.shape[1]} columns"
               + (f" · new: {', '.join(first)}" if first else "")
               + f" · showing the first {min(TABLE_ROWS, len(df)):,} rows.")
    show = df.head(TABLE_ROWS)
    if first:                         # surface the new columns first
        show = show[first + [c for c in show.columns if c not in first]]
    st.dataframe(show, column_config=centre_numbers(show), width="stretch",
                 hide_index=True, height=320)

    # ── plot ────────────────────────────────────────────────────────────
    st.divider()
    _explore(df, first)
