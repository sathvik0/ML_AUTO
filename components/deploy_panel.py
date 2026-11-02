"""The project page's deployment sections: model stages and approvals,
API keys, and batch scoring jobs (`logic.deployment` does the work and the
access checks)."""
import os

import pandas as pd
import streamlit as st

from logic import deployment, projects
from logic.projects import AccessDenied
from utils import auth, guard, log
from utils.file_loader import load_file

_log = log.get("deploy_panel")

STAGE_LABEL = {"none": "None", "staging": "Staging",
               "production": "Production", "archived": "Archived"}
SCHEDULES = {"manual": "Only when run by hand", "hourly": "Every hour",
             "daily": "Every day", "weekly": "Every Monday",
             "cron": "Custom (cron)"}


def _t(x):
    return "—" if x is None else f"{pd.Timestamp(x):%d %b %Y %H:%M}"


def _act(what, fn, *args):
    """Run a change; show an access refusal as a warning, then rerun."""
    try:
        fn(*args)
    except AccessDenied as e:
        st.warning(str(e))
        return
    st.rerun()


# ── stages and approvals ────────────────────────────────────────────────
def stages(p, user):
    models = projects.list_models(user, p["id"])
    st.markdown("#### Stages and approvals")
    if not models:
        st.caption("Register a model to move it through staging and "
                   "production.")
        return
    editor = projects.can(user, p["id"], "editor")
    owner = projects.can(user, p["id"], "owner")
    labels = {m["id"]: f"{m['name']} v{m['version']} · "
                       f"{STAGE_LABEL[m['stage']]}" for m in models}
    pick = st.selectbox("Version", list(labels), format_func=labels.get,
                        key="dep_stage_pick")
    m = next(x for x in models if x["id"] == pick)
    if editor:
        c = st.columns(4)
        for col, stage, label in ((c[0], "staging", "Move to staging"),
                                  (c[1], "none", "Back to none"),
                                  (c[2], "archived", "Archive")):
            if col.button(label, key=f"dep_set_{stage}", width="stretch",
                          disabled=m["stage"] == stage):
                _act("stage", deployment.set_stage, user, p["id"], pick,
                     stage)
        if c[3].button("Request production", type="primary",
                       key="dep_req", width="stretch",
                       disabled=m["stage"] == "production"):
            _act("request", deployment.request_production, user, p["id"],
                 pick, st.session_state.get("dep_req_note", ""))
        st.text_input("Note for the approver (optional)",
                      key="dep_req_note")
        st.caption("Production needs approval by a project owner other "
                   "than the requester; approving archives the model's "
                   "previous production version.")
    pending = deployment.transitions(user, p["id"], pending_only=True)
    if pending:
        st.markdown("**Waiting for approval**")
    me = auth.user().get("email", "").lower()
    for t in pending:
        c1, c2, c3 = st.columns([4, 1, 1], vertical_alignment="center")
        c1.markdown(f"{t['model']} v{t['version']} → **production** · "
                    f"asked by {t['requested_by']} {_t(t['requested_at'])}"
                    + (f"  \n_{t['comment']}_" if t["comment"] else ""))
        if t["requested_by"] == me:
            if c3.button("Withdraw", key=f"dep_cancel_{t['id']}",
                         width="stretch"):
                _act("cancel", deployment.cancel, user, p["id"], t["id"])
            c2.caption("Yours: another owner decides.")
        elif owner:
            if c2.button("Approve", type="primary", width="stretch",
                         key=f"dep_ok_{t['id']}"):
                log.event(_log, "model_promotion_approved")
                _act("approve", deployment.approve, user, p["id"], t["id"])
            if c3.button("Reject", width="stretch",
                         key=f"dep_no_{t['id']}"):
                _act("reject", deployment.reject, user, p["id"], t["id"])
    history = deployment.transitions(user, p["id"])
    if history:
        with st.expander("Stage history"):
            st.dataframe(pd.DataFrame([{
                "Model": f"{t['model']} v{t['version']}",
                "To": t["to_stage"], "Status": t["status"],
                "Asked by": t["requested_by"],
                "Decided by": t["decided_by"] or "—",
                "When": _t(t["decided_at"] or t["requested_at"]),
                "Note": t["note"] or t["comment"]} for t in history]),
                hide_index=True, width="stretch")


# ── API keys ────────────────────────────────────────────────────────────
def api_keys(p, user):
    if not projects.can(user, p["id"], "owner"):
        return
    st.markdown("#### Model API")
    base = os.environ.get("APP_URL", "https://<your-app-host>").rstrip("/")
    new = st.session_state.pop("dep_new_key", None)
    if new:
        st.success("New key — copy it now; it is not shown again.")
        st.code(new, language=None)
    keys = deployment.list_keys(user, p["id"])
    for k in keys:
        c1, c2, c3 = st.columns([3, 2.5, 1], vertical_alignment="center")
        c1.markdown(f"**{k['name']}** `{k['prefix']}…`"
                    + (" · revoked" if k["revoked"] else ""))
        c2.caption(f"by {k['created_by']} · last used "
                   f"{_t(k['last_used_at'])}")
        if not k["revoked"] and c3.button("Revoke", width="stretch",
                                          key=f"dep_rv_{k['id']}"):
            _act("revoke", deployment.revoke_key, user, p["id"], k["id"])
    c1, c2 = st.columns([4, 1.2], vertical_alignment="bottom")
    name = c1.text_input("New key for", key="dep_key_name",
                         placeholder="e.g. CRM integration")
    if c2.button("Create key", width="stretch", key="dep_key_go",
                 disabled=not name.strip()):
        key, _ = deployment.create_key(user, p["id"], name)
        st.session_state["dep_new_key"] = key
        log.event(_log, "api_key_created")
        st.rerun()
    names = sorted({m["name"] for m in projects.list_models(user, p["id"])})
    example = names[0] if names else "<model name>"
    st.caption("Keys reach this project's models only. Predictions come "
               "from the production version unless `?stage=staging` or "
               "`?version=N` says otherwise; up to 10,000 rows a call.")
    st.code(f"curl -X POST {base}/api/v1/models/{example}/predict \\\n"
            f"  -H 'Authorization: Bearer mlk_...' \\\n"
            f"  -H 'Content-Type: application/json' \\\n"
            "  -d '{\"rows\": [{\"Age\": 41, \"Region\": \"West\"}]}'",
            language="bash")
    st.caption(f"Full reference: {base}/api/docs")


# ── batch scoring ───────────────────────────────────────────────────────
def _schedule_input():
    kind = st.selectbox("Schedule", list(SCHEDULES), key="dep_job_sched",
                        format_func=SCHEDULES.get)
    if kind == "manual":
        return ""
    if kind == "hourly":
        return "0 * * * *"
    if kind == "cron":
        return st.text_input("Cron (UTC)", value="0 6 * * *",
                             key="dep_job_cron")
    hour = st.slider("Hour (UTC)", 0, 23, 6, key="dep_job_hour")
    return f"0 {hour} * * *" if kind == "daily" else f"0 {hour} * * 1"


def _new_job(p, user, names):
    with st.expander("New batch job"):
        name = st.text_input("Job name", key="dep_job_name",
                             placeholder="e.g. Nightly churn scores")
        c1, c2 = st.columns(2)
        model = c1.selectbox("Model", names, key="dep_job_model")
        stage = c2.selectbox("Version to use", ["production", "staging"],
                             key="dep_job_stage")
        src_kind = st.radio("Rows to score", ["dataset", "file",
                                              "snowflake"],
                            horizontal=True, key="dep_job_src",
                            format_func={"dataset": "The project's dataset",
                                         "file": "Upload a file",
                                         "snowflake": "A Snowflake query"}.get)
        source = None
        if src_kind == "dataset":
            ws = projects.workspace(user, p["id"])
            if ws["dataset_id"]:
                source = {"type": "dataset", "dataset_id": ws["dataset_id"]}
            else:
                st.caption("The project has no dataset yet.")
        elif src_kind == "file":
            up = st.file_uploader("File to score", type=["csv", "xlsx",
                                                         "xls"],
                                  key="dep_job_file")
            if up is not None:
                st.session_state["dep_job_upload"] = up
        else:
            q = st.text_area("Query", key="dep_job_query", height=90,
                             placeholder="SELECT * FROM CUSTOMERS")
            if q.strip():
                source = {"type": "snowflake", "query": q.strip()}
            st.caption("Runs with the deployment's own Snowflake account "
                       "(set by an administrator), not your sign-in.")
        dest_kind = st.radio("Write the scores to", ["storage", "snowflake"],
                             horizontal=True, key="dep_job_dest",
                             format_func={"storage": "A file here, to "
                                          "download",
                                          "snowflake": "A Snowflake table"}
                             .get)
        destination = {"type": "storage"}
        if dest_kind == "snowflake":
            d1, d2, d3 = st.columns(3)
            destination = {"type": "snowflake",
                           "database": d1.text_input("Database",
                                                     key="dep_job_db"),
                           "schema": d2.text_input("Schema",
                                                   key="dep_job_schema"),
                           "table": d3.text_input("Table",
                                                  value="ML_SCORES",
                                                  key="dep_job_table"),
                           "mode": "Replace table"}
        schedule = _schedule_input()
        ready = (name.strip() and model and
                 (source or (src_kind == "file"
                             and st.session_state.get("dep_job_upload")))
                 and (dest_kind == "storage"
                      or all(destination.get(k) for k in
                             ("database", "schema", "table"))))
        if st.button("Create job", type="primary", key="dep_job_go",
                     disabled=not ready):
            if src_kind == "file":
                up = st.session_state.pop("dep_job_upload")
                df = load_file(up)
                if df is None:
                    return
                source = deployment.save_input(user, p["id"], df, up.name)
            deployment.create_job(user, p["id"], name=name,
                                  model_name=model, source=source,
                                  destination=destination,
                                  schedule=schedule, stage=stage)
            log.event(_log, "batch_job_created", scheduled=bool(schedule))
            st.rerun()


def _job(p, user, j, editor):
    when = (f"next {_t(j['next_run_at'])}" if j["schedule"] and j["enabled"]
            else "paused" if j["schedule"] else "on demand")
    with st.container(border=True):
        c1, c2 = st.columns([4, 2], vertical_alignment="center")
        c1.markdown(f"**{j['name']}** — {j['model_name']} "
                    f"({j['stage']}) · {j['source']['type']} → "
                    f"{j['destination']['type']}")
        c2.caption(f"{j['schedule'] or 'manual'} · {when} · last: "
                   f"{j['last_status'] or 'never run'}")
        if editor:
            b = st.columns(3)
            if b[0].button("Run now", key=f"dep_run_{j['id']}",
                           width="stretch"):
                with st.spinner("Scoring"):
                    got = deployment.run_now(user, p["id"], j["id"])
                (st.success if got["status"] == "succeeded"
                 else st.warning)(got["message"])
            if j["schedule"] and b[1].button(
                    "Pause" if j["enabled"] else "Resume",
                    key=f"dep_en_{j['id']}", width="stretch"):
                _act("toggle", deployment.set_enabled, user, p["id"],
                     j["id"], not j["enabled"])
            if b[2].button("Delete", key=f"dep_del_{j['id']}",
                           width="stretch"):
                _act("delete", deployment.delete_job, user, p["id"],
                     j["id"])
        runs = deployment.job_runs(user, p["id"], j["id"], limit=5)
        for r in runs:
            r1, r2 = st.columns([5, 1.2], vertical_alignment="center")
            r1.caption(f"{_t(r['started_at'])} · {r['status']} · "
                       f"{r['message']} · by {r['triggered_by']}")
            if r["has_output"]:
                r2.download_button(
                    "Scores", key=f"dep_dl_{r['id']}", width="stretch",
                    data=lambda rid=r["id"]: deployment.run_output(
                        user, p["id"], rid),
                    file_name=f"{j['name']}_{pd.Timestamp(r['started_at']):%Y%m%d_%H%M}.csv",
                    mime="text/csv")


def batch(p, user):
    st.markdown("#### Batch scoring")
    names = sorted({m["name"] for m in projects.list_models(user, p["id"])})
    if not names:
        st.caption("Register a model first.")
        return
    editor = projects.can(user, p["id"], "editor")
    jobs = deployment.list_jobs(user, p["id"])
    if not jobs:
        st.caption("No jobs yet. A job scores rows with a model's "
                   "production (or staging) version, on demand or on a "
                   "schedule (checked every 15 minutes on AWS).")
    for j in jobs:
        guard.section("Batch job", _job, p, user, j, editor)
    if editor:
        _new_job(p, user, names)
