"""The assistant on Configure & train: describe the objective, get a
recommended configuration with its reasons, apply it with one click.

The conversation is kept append-only in session state (`ml_a_*`, so a new
file clears it). A proposal changes nothing until Apply, which runs as a
button callback — before the page's widgets are drawn, the only point at
which their values may be set — and every value is checked against the
page's own options; anything that does not fit is listed, not applied.
"""
import json
import logging

import streamlit as st

from components import ml_config
from logic import claude, ml, ml_assistant, ml_recipe
from utils import auth, log, secrets

_log = log.get("assistant_panel")

PREFIX = "ml_a_"
W = ml_config.PREFIX


def _ss():
    return st.session_state


def _options():
    return {"num_missing": list(ml_config.NUM_MISSING),
            "cat_missing": list(ml_config.CAT_MISSING),
            "scale": list(ml_config.SCALING),
            "cat_encoding": list(ml_config.ENCODING),
            "tune_mode": list(ml_config.TUNING),
            "folds": list(ml_config.FOLDS),
            "test_share_percent": ["10 to 40, in steps of 5"]}


def _current():
    """The settings on screen, in a line the assistant reads."""
    ss = _ss()
    task = ss.get(f"{W}task", "classification")
    parts = [f"task {task}"]
    if task != "clustering":
        parts.append(f"target {ss.get(f'{W}target_{task}', '(unset)')}")
    models = ss.get(f"{W}models_{task}") or []
    parts.append("models " + (", ".join(models) or "(none)"))
    for field in ("folds", "test_share", "num_missing", "cat_missing",
                  "scale", "cat_encoding", "tune_mode", "balance"):
        if f"{W}{field}" in ss:
            parts.append(f"{field} {ss[f'{W}{field}']}")
    metric = [f"{k[len(W) + 7:]}={ss[k]}" for k in ss.keys()
              if isinstance(k, str) and k.startswith(f"{W}metric_")]
    if metric:
        parts.append("metric " + ", ".join(metric))
    return "; ".join(parts)


def _system(df, prof):
    """Built once per version of the prepared file."""
    ss = _ss()
    version = ss.get("ml_version", 0)
    if ss.get(f"{PREFIX}system_for") != version:
        ss[f"{PREFIX}system"] = ml_assistant.system_prompt(df, prof,
                                                           _options())
        ss[f"{PREFIX}system_for"] = version
    return ss[f"{PREFIX}system"]


# ── applying a proposal ─────────────────────────────────────────────────
def _match(spec, text):
    """The allowed value `text` names, or a sentinel when none does."""
    want = str(text).strip().lower()
    for v in spec["values"]:
        if want in (str(v).lower(), ml.show_value(spec, v).lower()):
            return v
        try:
            if v is not None and float(want) == float(v):
                return v
        except (TypeError, ValueError):
            pass
    return _NO


_NO = object()


def _assignments(p, df, prof):
    """({widget key: value}, [what was left out and why])."""
    ss = _ss()
    out, notes = {}, []
    task = p.get("task") or ss.get(f"{W}task", "classification")
    if task not in ml.TASKS:
        return {}, [f"Unknown task {task}."]
    out[f"{W}task"] = task
    n_classes, target = None, None
    if task != "clustering":
        target = p.get("target") or ss.get(f"{W}target_{task}")
        cands = ml.target_candidates(prof, task)
        if target not in cands:
            notes.append(f"{target} cannot be the {task} target here.")
            target = None
        else:
            out[f"{W}target_{task}"] = target
        if target and task == "classification":
            if ml.needs_rule(prof, target):
                n_classes = 2
            else:
                classes = sorted(map(str, ml_recipe.label_text(df[target])
                                     .dropna().unique()))
                n_classes = len(classes)
                pos = p.get("positive_class")
                if pos and n_classes == 2:
                    if pos in classes:
                        out[f"{W}positive_{task}_{target}"] = pos
                    else:
                        notes.append(f"{pos} is not a class of {target}.")
    blocked = ml.unavailable(task, len(df))
    avail = [k for k in ml.MODELS[task] if k not in blocked]
    chosen = []
    for m in p.get("models") or []:
        key = m.get("key")
        if key not in avail:
            notes.append(f"Model {key} is not available for {task} here.")
            continue
        chosen.append(key)
        specs = {s["name"]: s for s in ml.MODELS[task][key]["params"]}
        for s in m.get("settings") or []:
            spec = specs.get(s.get("name"))
            value = _match(spec, s.get("value")) if spec else _NO
            if value is _NO:
                notes.append(f"{key} {s.get('name')}={s.get('value')} is "
                             f"not one of its allowed values.")
                continue
            out[f"{W}p_{task}_{key}_{spec['name']}"] = \
                ml_config._to_widget(value)
    if chosen:
        out[f"{W}models_{task}"] = list(dict.fromkeys(chosen))
    if task != "clustering" and p.get("metric"):
        family = ml.metric_family(task, n_classes)
        want = p["metric"].strip().lower()
        hit = next((k for k in ml.SELECTABLE[family]
                    if want in (k, ml.metric_label(family, k).lower())),
                   None)
        if hit:
            out[f"{W}metric_{family}"] = hit
        else:
            notes.append(f"Metric {p['metric']} does not apply to {family}.")
    if p.get("folds"):
        try:
            folds = int(float(p["folds"]))
        except ValueError:
            folds = -1
        if folds in ml_config.FOLDS:
            out[f"{W}folds"] = folds
    if p.get("test_share_percent"):
        try:
            share = int(round(float(p["test_share_percent"]) / 5) * 5)
            out[f"{W}test_share"] = min(max(share, 10), 40)
        except ValueError:
            pass
    if task == "classification" and p.get("balance_classes") in ("yes",
                                                                 "no"):
        out[f"{W}balance"] = p["balance_classes"] == "yes"
    for field, options in (("num_missing", ml_config.NUM_MISSING),
                           ("cat_missing", ml_config.CAT_MISSING),
                           ("scale", ml_config.SCALING),
                           ("cat_encoding", ml_config.ENCODING),
                           ("tune_mode", ml_config.TUNING)):
        value = p.get(field)
        if not value:
            continue
        if value in options and not (field == "cat_encoding"
                                     and task == "clustering"
                                     and value == "target"):
            out[f"{W}{field}"] = value
        else:
            notes.append(f"{field} {value} is not an option here.")
    return out, notes


def _apply(proposal, df, prof):
    """Button callback: runs before the page's widgets are drawn."""
    ss = _ss()
    values, notes = _assignments(proposal, df, prof)
    for k, v in values.items():
        ss[k] = v
    ss[f"{PREFIX}applied"] = notes
    log.event(_log, "ml_assistant_applied", settings=len(values),
              skipped=len(notes))


def _describe(p):
    """The proposal as short lines for the chat."""
    lines = []
    if p.get("task"):
        lines.append(f"Task: {p['task']}"
                     + (f", target {p['target']}" if p.get("target") else "")
                     + (f" (class of interest {p['positive_class']})"
                        if p.get("positive_class") else ""))
    for m in p.get("models") or []:
        s = ", ".join(f"{x['name']}={x['value']}" for x in m["settings"])
        lines.append(f"Model: {m['key']}" + (f" ({s})" if s else ""))
    pairs = [("metric", "Pick the best by"), ("folds", "Folds"),
             ("test_share_percent", "Test share %"),
             ("balance_classes", "Balance classes"),
             ("num_missing", "Missing numbers"),
             ("cat_missing", "Missing categories"), ("scale", "Scaling"),
             ("cat_encoding", "Encoding"), ("tune_mode", "Tuning")]
    for field, label in pairs:
        if p.get(field):
            lines.append(f"{label}: {p[field]}")
    return lines


# ── the panel ───────────────────────────────────────────────────────────
def _send(df, prof, text, key):
    ss = _ss()
    history = ss.setdefault(f"{PREFIX}history", [])
    shown = ss.setdefault(f"{PREFIX}shown", [])
    current = _current()
    try:
        with log.timer() as t:
            got = ml_assistant.turn(_system(df, prof), history, text,
                                    current, api_key=key)
    except Exception as e:
        log.event(_log, "ml_assistant_failed", level=logging.WARNING,
                  exc_info=True)
        ss[f"{PREFIX}error"] = claude.friendly_error(e)
        return
    log.event(_log, "ml_assistant_turn", ms=t.ms,
              proposal=bool(got.get("has_proposal")))
    ss[f"{PREFIX}error"] = None
    # Append-only: the next turn resends exactly these.
    history.append({"role": "user", "content":
                    f"{text}\n\n(Settings on the page now: {current})"})
    history.append({"role": "assistant", "content": json.dumps(got)})
    shown.append({"role": "user", "text": text})
    shown.append({"role": "assistant", "text": got.get("reply", ""),
                  "proposal": got["proposal"] if got.get("has_proposal")
                  else None})
    ss[f"{PREFIX}applied"] = None


def render(df, prof):
    ss = _ss()
    key = secrets.resolve("ANTHROPIC_API_KEY")
    with st.expander("Assistant — describe your objective and get a "
                     "recommended setup", expanded=not ss.get(
                         f"{PREFIX}shown")):
        if auth.denied("ai"):
            return
        if not key:
            st.info("Set `ANTHROPIC_API_KEY` to use the assistant.")
            return
        shown = ss.get(f"{PREFIX}shown") or []
        last_proposal = None
        for i, msg in enumerate(shown):
            with st.chat_message(msg["role"]):
                st.markdown(msg["text"])
                if msg.get("proposal"):
                    st.markdown("\n".join(f"- {line}" for line in
                                          _describe(msg["proposal"])))
                    last_proposal = (i, msg["proposal"])
        if last_proposal:
            st.button("Apply these settings", type="primary",
                      key=f"{PREFIX}apply_{last_proposal[0]}",
                      on_click=_apply, args=(last_proposal[1], df, prof))
            notes = ss.get(f"{PREFIX}applied")
            if notes is not None:
                st.success("Applied: the settings below now follow the "
                           "recommendation. Review them, then train.")
                for n in notes:
                    st.caption(f"Not applied: {n}")
        if ss.get(f"{PREFIX}error"):
            st.error(ss[f"{PREFIX}error"])
        with st.form(f"{PREFIX}form", clear_on_submit=True):
            text = st.text_area(
                "Your objective or question", height=80,
                label_visibility="collapsed",
                placeholder="e.g. Our retention team can call about 10% of "
                            "customers before renewal. Find the ones most "
                            "likely to leave so the calls go where they "
                            "matter.")
            sent = st.form_submit_button("Ask the assistant", type="primary")
        if sent and text.strip():
            with st.spinner("Thinking about the data and your objective"):
                _send(df, prof, text.strip(), key)
            st.rerun()
        if shown and st.button("Start a new conversation",
                               key=f"{PREFIX}clear"):
            for k in ("history", "shown", "applied", "error"):
                ss.pop(f"{PREFIX}{k}", None)
            st.rerun()
