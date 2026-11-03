"""Background training in the app: submit from Configure & train, watch it
live, cancel it, open its results — and the project page's list of jobs
(`logic.jobs` does the work and the access checks)."""
import pandas as pd
import streamlit as st

from logic import jobs, projects
from logic.projects import AccessDenied
from utils import auth, guard, log, scroll

_log = log.get("jobs_panel")

KEY = "ml_job"            # the job this session is watching

LABEL = {"queued": "⏳ queued", "running": "⚙️ training",
         "succeeded": "✅ done", "failed": "⚠️ failed",
         "cancelled": "✖ cancelled"}


def available():
    """Background training needs the data stored in an open project the
    user may change."""
    from components import project_panel
    ss = st.session_state
    return project_panel.can_edit() and bool(ss.get("proj_dataset_id"))


def submit(config):
    from components import project_panel
    ss = st.session_state
    jid = jobs.submit_training(auth.user(), project_panel.current()["id"],
                               config, ss.get("ml_steps") or [],
                               ss["proj_dataset_id"])
    ss[KEY] = jid
    log.event(_log, "background_training_started",
              models=len(config.get("models") or []))
    st.rerun()


def open_run(project_id, run_id):
    run = projects.load_run(auth.user(), project_id, run_id)
    run["_project_run_id"] = run_id
    st.session_state["ml_run"] = run
    st.session_state["page"] = "ml_results"
    scroll.request()
    st.rerun()


@st.fragment(run_every="2s")
def watch():
    """The job this session started, refreshed every two seconds until
    it ends (the rest of the page is not redrawn)."""
    from components import project_panel
    ss = st.session_state
    jid, p = ss.get(KEY), project_panel.current()
    if not jid or not p:
        return
    try:
        j = jobs.get(auth.user(), p["id"], jid)
    except AccessDenied:
        ss.pop(KEY, None)
        return
    with st.container(border=True):
        st.markdown(f"**Background training** · {LABEL[j['status']]}"
                    + (f" · attempt {j['attempts']}" if j["attempts"] > 1
                       else ""))
        if j["status"] in jobs.ACTIVE:
            st.progress(j["progress"], text=j["progress_text"] or
                        "Waiting for a worker")
            st.caption("It keeps going if you leave this page or close the "
                       "tab; the project page lists it.")
            if j["cancel_requested"]:
                st.caption("Stopping at the next model fit…")
            elif st.button("Cancel", key=f"job_cancel_{jid}"):
                jobs.request_cancel(auth.user(), p["id"], jid)
                st.rerun(scope="fragment")
            return
        if j["status"] == "succeeded":
            st.success(j["message"])
            c1, c2 = st.columns(2)
            if c1.button("Open results", type="primary", width="stretch",
                         key=f"job_open_{jid}"):
                ss.pop(KEY, None)
                open_run(p["id"], j["result"]["run_id"])
            if c2.button("Dismiss", width="stretch", key=f"job_dis_{jid}"):
                ss.pop(KEY, None)
                st.rerun()
            return
        st.warning(j["message"] or j["status"])
        c1, c2 = st.columns(2)
        if c1.button("Retry", width="stretch", key=f"job_retry_{jid}"):
            ss[KEY] = jobs.retry(auth.user(), p["id"], jid)
            st.rerun()
        if c2.button("Dismiss", width="stretch", key=f"job_dis_{jid}"):
            ss.pop(KEY, None)
            st.rerun()


def _when(t):
    return "—" if t is None else f"{pd.Timestamp(t):%d %b %H:%M}"


def project_section(p, user):
    """Recent training jobs on the project page."""
    rows = jobs.list_jobs(user, p["id"], limit=15)
    st.markdown("#### Training jobs")
    if not rows:
        st.caption("Jobs started with **Train in the background** appear "
                   "here.")
        return
    editor = projects.can(user, p["id"], "editor")
    for j in rows:
        c1, c2, c3 = st.columns([4, 2, 1.3], vertical_alignment="center")
        c1.markdown(f"{LABEL[j['status']]} · {j['task'] or ''} "
                    f"{j['target'] or ''} · by {j['created_by']}")
        c2.caption(f"{_when(j['created_at'])}"
                   + (f" · {j['progress']:.0%} {j['progress_text']}"
                      if j["status"] in jobs.ACTIVE else
                      f" · {j['message'][:60]}"))
        if j["status"] == "succeeded":
            if c3.button("Open", key=f"pj_open_{j['id']}", width="stretch"):
                open_run(p["id"], j["result"]["run_id"])
        elif j["status"] in jobs.ACTIVE and editor:
            if c3.button("Cancel", key=f"pj_cancel_{j['id']}",
                         width="stretch"):
                guard.action("Cancelling", "The job could not be "
                             "cancelled.", _cancel, p, j["id"])
        elif editor:
            if c3.button("Retry", key=f"pj_retry_{j['id']}",
                         width="stretch"):
                guard.action("Retrying", "The job could not be retried.",
                             _retry, p, j["id"])


def _cancel(p, jid):
    jobs.request_cancel(auth.user(), p["id"], jid)
    st.rerun()


def _retry(p, jid):
    st.session_state[KEY] = jobs.retry(auth.user(), p["id"], jid)
    st.rerun()
