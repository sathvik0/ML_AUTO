"""The project page's Monitoring section: checks of models in production
against newer data, their trend, alerts and retraining
(`logic.monitoring` does the work and the access checks)."""
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import deploy_panel
from logic import monitoring, projects
from utils import alerts, guard, log
from utils.constants import ML_PALETTE

_log = log.get("monitor_panel")

def _chip(status):
    from components import shell
    return shell.status(status or "not run")


def _trend(runs, limits):
    rows = [r for r in reversed(runs) if r["summary"]]
    if len(rows) < 2:
        return
    when = [pd.Timestamp(r["started_at"]) for r in rows]
    fig = go.Figure()
    auc = [r["summary"].get("drift_auc") for r in rows]
    fig.add_trace(go.Scatter(x=when, y=auc, name="Drift AUC",
                             mode="lines+markers",
                             line=dict(color=ML_PALETTE[1])))
    perf = [r["summary"].get("performance") for r in rows]
    if any(perf):
        label = next(p["label"] for p in perf if p)
        fig.add_trace(go.Scatter(
            x=when, y=[p["current"] if p else None for p in perf],
            name=label, mode="lines+markers",
            line=dict(color=ML_PALETTE[0])))
    fig.add_hline(y=limits["drift_auc"], line_dash="dash",
                  line_color="#e34948",
                  annotation_text="drift alert", annotation_position="top left")
    fig.update_layout(height=260, margin=dict(t=20, b=10, l=10, r=10),
                      legend=dict(orientation="h", y=1.15),
                      yaxis_title="Score")
    st.plotly_chart(fig, width="stretch")


def _monitor(p, user, m, editor):
    with st.container(border=True):
        c1, c2 = st.columns([4, 2], vertical_alignment="center")
        when = (f"next {deploy_panel._t(m['next_run_at'])}"
                if m["schedule"] and m["enabled"]
                else "paused" if m["schedule"] else "on demand")
        c1.markdown(f"**{m['name']}** {_chip(m['last_status'])} — "
                    f"{m['model_name']} (production) · "
                    f"{_describe(m['source'])}",
                    unsafe_allow_html=True)
        c2.caption(f"{m['schedule'] or 'manual'} · {when} · "
                   f"{'retrains to staging' if m['retrain'] == 'staging' else 'alerts only'}"
                   + (" · chat webhook" if m["webhook"] else ""))
        if editor:
            b = st.columns(3)
            if b[0].button("Check now", key=f"mon_run_{m['id']}",
                           width="stretch"):
                with st.spinner("Comparing the new rows with the training "
                                "data"):
                    got = monitoring.run_now(user, p["id"], m["id"])
                log.event(_log, "monitor_run_now", status=got["status"])
                show = {"ok": st.success, "warning": st.warning,
                        "alert": st.error}.get(got["status"], st.warning)
                show(f"{got['status'].upper()}: {got['message']}"
                     + (f" — retrained as v{got['retrained']} in staging."
                        if got["retrained"] else ""))
            if m["schedule"] and b[1].button(
                    "Pause" if m["enabled"] else "Resume",
                    key=f"mon_en_{m['id']}", width="stretch"):
                deploy_panel._act("toggle", monitoring.set_enabled, user,
                                  p["id"], m["id"], not m["enabled"])
            if b[2].button("Delete", key=f"mon_del_{m['id']}",
                           width="stretch"):
                deploy_panel._act("delete", monitoring.delete_monitor, user,
                                  p["id"], m["id"])
        runs = monitoring.monitor_runs(user, p["id"], m["id"])
        if not runs:
            st.caption("Not checked yet.")
            return
        last = runs[0]
        s = last["summary"] or {}
        for reason in s.get("reasons") or []:
            st.markdown(f"- {reason}")
        if last["status"] == "failed":
            st.caption(last["message"])
        _trend(runs, {**monitoring.DEFAULTS, **(m["thresholds"] or {})})
        with st.expander(f"History ({len(runs)})"):
            st.dataframe(pd.DataFrame([{
                "When": deploy_panel._t(r["started_at"]),
                "Status": r["status"],
                "Drift AUC": (r["summary"] or {}).get("drift_auc"),
                "Major drift": ", ".join((r["summary"] or {})
                                         .get("major") or []) or "—",
                "Accuracy": ((r["summary"] or {}).get("performance")
                             or {}).get("current"),
                "Alerted": "yes" if r["alerted"] else "",
                "Retrained": "yes" if r["retrained_version_id"] else "",
                "By": r["triggered_by"]} for r in runs]),
                hide_index=True, width="stretch",
                column_config={"Drift AUC": st.column_config.NumberColumn(
                    format="%.3f"), "Accuracy":
                    st.column_config.NumberColumn(format="%.3f")})


WINDOWS = {7: "7 days", 30: "30 days", 90: "90 days"}


def _describe(source):
    kind = source.get("type")
    if kind == "logged":
        return (f"API requests, last {source.get('days', 30)} days"
                + (f", outcomes joined on {source['outcomes']['id']}"
                   if source.get("outcomes") else ""))
    return {"dataset": "the project's dataset", "file": "a stored file",
            "snowflake": "a Snowflake query"}.get(kind, kind)


def _source(p, user):
    """(source spec or None, ready) for a monitor: the requests the model
    API answered, or newer rows from where a batch job reads them."""
    live = st.radio("Newer rows to compare with the training data",
                    ["logged", "rows"], horizontal=True, key="mon_kind",
                    format_func={"logged": "Requests to the model API",
                                 "rows": "A dataset, file or query"}.get)
    if live == "rows":
        got = deploy_panel.source_input(p, user, "mon", "From")
        st.caption("Include the actual outcome column when you have it, "
                   "and accuracy is checked too.")
        return got
    c1, c2 = st.columns([1, 2])
    days = c1.selectbox("Requests from the last", list(WINDOWS),
                        index=1, format_func=WINDOWS.get, key="mon_days")
    c2.caption("The rows other systems sent to the model API (Recent "
               "predictions), so drift is measured on what the model "
               "actually scores.")
    source = {"type": "logged", "days": int(days)}
    if not st.checkbox("Join the actual outcomes as they arrive",
                       key="mon_join"):
        return source, True
    key = st.text_input("ID column, in the requests and the outcomes",
                        key="mon_join_id", placeholder="Customer_ID")
    got, ready = deploy_panel.source_input(
        p, user, "mon_out", "Outcomes: the ID and the target column")
    st.caption("Each ID's latest request is matched to its outcome; "
               "accuracy is measured on the matched rows, against the "
               "model's test score. The outcome column must have the "
               "target's name.")
    if got is None or not key.strip():
        return None, False
    source["outcomes"] = {"source": got, "id": key.strip()}
    return source, ready


def _stored(p, user, source):
    """The spec to save, with any uploaded file stored first."""
    if source["type"] != "logged":
        return deploy_panel.stored_source(p, user, "mon", source)
    if source.get("outcomes"):
        got = deploy_panel.stored_source(p, user, "mon_out",
                                         source["outcomes"]["source"])
        if got is None:
            return None
        source = {**source, "outcomes": {**source["outcomes"],
                                         "source": got}}
    return source


def _new_monitor(p, user, names):
    with st.expander("New monitor"):
        name = st.text_input("Monitor name", key="mon_name",
                             placeholder="e.g. Weekly churn check")
        model = st.selectbox("Model (its production version is checked)",
                             names, key="mon_model")
        source, ready = _source(p, user)
        schedule = deploy_panel.schedule_input("mon")
        st.markdown("**Alert when**")
        d = monitoring.DEFAULTS
        c = st.columns(4)
        limits = {
            "major_columns": c[0].number_input(
                "Columns with major drift ≥", 1, 50,
                int(d["major_columns"]), key="mon_major"),
            "drift_auc": c[1].number_input(
                "Drift AUC ≥", 0.5, 1.0, d["drift_auc"], 0.05,
                key="mon_auc"),
            "metric_drop": c[2].number_input(
                "Score falls by ≥", 0.0, 1.0, d["metric_drop"], 0.01,
                key="mon_drop", help="For ROC AUC, F1, R² and other "
                "higher-is-better metrics."),
            "error_rise": c[3].number_input(
                "Error rises by ≥ (share)", 0.0, 5.0, d["error_rise"], 0.05,
                key="mon_rise", help="For RMSE, MAE, log loss: 0.10 is "
                "10% worse."),
        }
        webhook = st.text_input(
            "Slack or Teams webhook (optional)", key="mon_hook",
            placeholder="https://hooks.slack.com/services/...")
        problem = alerts.webhook_problem(webhook.strip())
        if problem:
            st.caption(problem)
        if os.environ.get("ALERTS_TOPIC_ARN"):
            st.caption("Alerts also go to the deployment's alert topic "
                       "(email and chat subscriptions set by an "
                       "administrator).")
        retrain = st.radio(
            "On an alert", list(monitoring.RETRAIN), key="mon_retrain",
            horizontal=True, format_func={
                "notify": "Alert only",
                "staging": "Alert and retrain into staging"}.get)
        if retrain == "staging":
            st.caption("The model is retrained with its winning settings on "
                       "the training data plus the newly labelled rows, and "
                       "registered as a staging version. Production still "
                       "needs the usual request and approval.")
        if st.button("Create monitor", type="primary", key="mon_go",
                     disabled=not (name.strip() and model and ready)
                     or bool(problem)):
            source = _stored(p, user, source)
            if source is None:
                return
            monitoring.create_monitor(
                user, p["id"], name=name, model_name=model, source=source,
                schedule=schedule, thresholds=limits,
                webhook=webhook.strip(), retrain=retrain)
            log.event(_log, "monitor_created", scheduled=bool(schedule),
                      retrain=retrain)
            st.rerun()


def render(p, user):
    st.markdown("#### Monitoring")
    models = projects.list_models(user, p["id"])
    names = sorted({m["name"] for m in models})
    if not names:
        st.caption("Register a model and put it in production to monitor "
                   "it.")
        return
    editor = projects.can(user, p["id"], "editor")
    monitors = monitoring.list_monitors(user, p["id"])
    if not monitors:
        st.caption("No monitors yet. A monitor compares newer rows with the "
                   "data a production model was trained on — drift, and "
                   "accuracy when outcomes are known — and alerts when "
                   "they move too far.")
    for m in monitors:
        guard.section("Monitor", _monitor, p, user, m, editor)
    if editor:
        _new_monitor(p, user, names)
