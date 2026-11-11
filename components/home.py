"""The home page: what you have, what needs a look, and where to start."""
import pandas as pd
import streamlit as st

from components import project_panel, shell
from logic import overview
from utils import auth, guard


def _when(t):
    return "—" if t is None else f"{pd.Timestamp(t):%d %b %Y %H:%M}"


def _open(pid, start_ml, section):
    project_panel.open_project(pid, start_ml)
    shell.go("ml_project", section)


def _attention(data, start_ml):
    st.markdown("#### Needs attention")
    items = [(f"**{a['model']} v{a['version']}** is waiting for approval to "
              f"{a['to_stage']}", f"{a['project']} · asked by "
              f"{a['requested_by']} · {_when(a['requested_at'])}",
              a["project_id"], "models") for a in data["approvals"]]
    items += [(f"Monitor **{m['name']}** on {m['model']}: {m['status']}",
               f"{m['project']} · checked {_when(m['last_run_at'])}",
               m["project_id"], "operations") for m in data["monitors"]]
    if not items:
        st.caption("Nothing right now: no approvals waiting and no monitor "
                   "in warning or alert.")
        return
    for i, (text, detail, pid, section) in enumerate(items[:8]):
        c1, c2 = st.columns([5, 1.2], vertical_alignment="center")
        c1.markdown(f"{text}  \n<span class='home-detail'>{detail}</span>",
                    unsafe_allow_html=True)
        if c2.button("Review", key=f"home_review_{i}", width="stretch"):
            guard.action("Opening", "The project could not be opened.",
                         _open, pid, start_ml, section)
    if len(items) > 8:
        st.caption(f"And {len(items) - 8} more.")


def _score(metric, score):
    return None if score is None else f"{metric or ''} {score:.3f}".strip()


def _models(data):
    st.markdown("#### Models in staging and production")
    if not data["models"]:
        st.caption("None yet. Register a model from Results, then move it "
                   "to staging or request production.")
        return
    shell.table(
        [("Model", "model"), ("Version", "version"), ("Project", "project"),
         ("Stage", "stage"), ("Score", "score", "num"),
         ("Monitor", "monitor")],
        [{"model": m["name"], "version": f"v{m['version']}",
          "project": m["project"], "stage": shell.status(m["stage"]),
          "score": _score(m["metric"], m["score"]),
          "monitor": shell.status(m["monitor"]) if m["monitor"] else None}
         for m in data["models"]])


def _runs(data):
    st.markdown("#### Recent runs")
    if not data["runs"]:
        st.caption("No runs yet. Train a model and it shows here.")
        return
    shell.table(
        [("When", "when"), ("Project", "project"), ("Task", "task"),
         ("Target", "target"), ("Best model", "best"),
         ("Score", "score", "num"), ("By", "by")],
        [{"when": _when(r["created_at"]), "project": r["project"],
          "task": r["task"], "target": r["target"], "best": r["best_model"],
          "score": _score(r["metric"], r["score"]), "by": r["created_by"]}
         for r in data["runs"]])


def render(start_ml, load_sample):
    user = auth.user()
    first = (user.get("name") or "").split(" ")[0]
    head, new, sample = st.columns([5, 1.3, 1.3], vertical_alignment="bottom")
    head.markdown(f"## Welcome{', ' + first if first else ''}")
    head.caption("Your projects, what needs a look, and the latest runs.")
    if new.button("New analysis", type="primary", width="stretch",
                  icon=":material/add:", key="home_new"):
        shell.go("start")
    if load_sample and sample.button("Try sample data", width="stretch",
                                     key="home_sample"):
        load_sample()
    if not project_panel.available():
        st.info("Projects are unavailable right now (the database cannot be "
                "reached). You can still load data and train; the work will "
                "not be saved.")
        return
    data = overview.summary(user)
    cards = st.columns(4)
    cards[0].metric("Projects", len(data["projects"]), border=True)
    cards[1].metric("Models in production", data["production"], border=True)
    cards[2].metric("Approvals waiting", len(data["approvals"]), border=True)
    cards[3].metric("Monitors to check", len(data["monitors"]), border=True)
    with st.container(border=True, key="card_models"):
        guard.section("Models", _models, data)
    left, right = st.columns([1, 1], gap="medium")
    with left:
        with st.container(border=True, key="card_attention"):
            guard.section("Needs attention", _attention, data, start_ml)
        guard.section("Projects", project_panel.landing, start_ml)
    with right, st.container(border=True, key="card_runs"):
        guard.section("Recent runs", _runs, data)
