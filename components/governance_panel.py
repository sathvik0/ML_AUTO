"""Governance in the app: fairness checks and the production policy, a
project's activity log, and the admins' audit log page
(`logic.fairness` and `logic.audit` do the work and the access checks)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from logic import audit, fairness, projects
from logic.projects import AccessDenied
from utils import auth, log
from utils.constants import ML_PALETTE

_log = log.get("governance_panel")

BADGE = {"pass": "pass", "flag": "flagged",
         "insufficient": "too few rows to judge"}


def _chip(status):
    from components import shell
    return shell.status(BADGE[status], shell.TONES.get(status, "idle"))


def _t(x):
    return "—" if x is None else f"{pd.Timestamp(x):%d %b %Y %H:%M}"


def fairness_badge(user, project_id, version_id):
    """The latest fairness result for a version, as a short label."""
    got = fairness.reports(user, project_id, version_id)
    if not got:
        return "no fairness check"
    r = got[0]
    if r["status"] == "flag" and r["accepted_by"]:
        return f"fairness flagged, accepted by {r['accepted_by']}"
    return f"fairness {BADGE[r['status']]}"


# ── fairness ────────────────────────────────────────────────────────────
def _chart(attr, a, task):
    groups = [g for g in a["groups"]]
    names = [str(g["group"]) for g in groups]
    fig = go.Figure()
    if "selection_rate" in groups[0]:
        fig.add_trace(go.Bar(x=names, y=[g["selection_rate"] for g in groups],
                             name="Predicted yes rate",
                             marker_color=ML_PALETTE[0]))
        fig.add_trace(go.Bar(x=names, y=[g["actual_rate"] for g in groups],
                             name="Actual yes rate",
                             marker_color=ML_PALETTE[2]))
        fig.update_layout(barmode="group", yaxis_tickformat=".0%")
    elif "mae" in groups[0]:
        fig.add_trace(go.Bar(x=names, y=[g["mae"] for g in groups],
                             name="MAE", marker_color=ML_PALETTE[0]))
        fig.add_trace(go.Bar(x=names, y=[g["mean_error"] for g in groups],
                             name="Mean error (bias)",
                             marker_color=ML_PALETTE[1]))
        fig.update_layout(barmode="group")
    else:
        fig.add_trace(go.Bar(x=names, y=[g["accuracy"] for g in groups],
                             name="Accuracy", marker_color=ML_PALETTE[0]))
        fig.update_layout(yaxis_tickformat=".0%")
    fig.update_layout(height=260, margin=dict(t=20, b=10, l=10, r=10),
                      legend=dict(orientation="h", y=1.15), xaxis_title=attr)
    st.plotly_chart(fig, width="stretch", key=f"fair_chart_{attr}")


def _report(p, user, r, owner):
    rep = r["report"]
    st.markdown(f"{_chip(r['status'])} {rep['rows']:,} out-of-sample "
                f"rows · by {r['created_by']} {_t(r['created_at'])}",
                unsafe_allow_html=True)
    if r["accepted_by"]:
        st.info(f"Accepted by {r['accepted_by']} {_t(r['accepted_at'])}: "
                f"{r['note']}")
    for attr, a in rep["attributes"].items():
        with st.expander(f"{attr} — {BADGE[a['status']]}",
                         expanded=a["status"] == "flag"):
            for f in a["flags"]:
                st.markdown(f"- {f}")
            metrics = {k: v for k, v in a["metrics"].items()
                       if v is not None}
            if metrics:
                st.caption(" · ".join(f"{k.replace('_', ' ')} {v:.3f}"
                                      for k, v in metrics.items()))
            st.dataframe(pd.DataFrame(a["groups"]), hide_index=True,
                         width="stretch")
            if a["groups"]:
                _chart(attr, a, rep["task"])
            st.caption(f"Groups under {a['min_group']} rows are shown but "
                       f"not judged. A gap can follow a real difference "
                       f"in the actual rates rather than bias in the "
                       f"model — compare the two.")
    if r["status"] == "flag" and not r["accepted_by"] and owner:
        note = st.text_area("Accept with a reason (kept with the report)",
                            key=f"fair_note_{r['id']}", height=70)
        if st.button("Accept flagged report", key=f"fair_ok_{r['id']}"):
            try:
                fairness.accept(user, p["id"], r["id"], note)
                log.event(_log, "fairness_accepted")
                st.rerun()
            except AccessDenied as e:
                st.warning(str(e))


def fairness_section(p, user):
    st.markdown("#### Fairness")
    owner = projects.can(user, p["id"], "owner")
    editor = projects.can(user, p["id"], "editor")
    pol = fairness.policy(user, p["id"])
    if owner:
        want = st.checkbox(
            "Require a passing (or accepted) fairness check before a "
            "version can be requested for production",
            value=pol["require_fairness"], key="fair_policy")
        if want != pol["require_fairness"]:
            fairness.set_policy(user, p["id"], want)
            st.rerun()
    elif pol["require_fairness"]:
        st.caption("This project requires a fairness check before "
                   "production.")
    models = [m for m in projects.list_models(user, p["id"])
              if m["task"] != "clustering"]
    if not models:
        st.caption("Register a model with a target to check it.")
        return
    labels = {m["id"]: f"{m['name']} v{m['version']} ({m['stage']})"
              for m in models}
    vid = st.selectbox("Version", list(labels), format_func=labels.get,
                       key="fair_version")
    if editor:
        try:
            cols = fairness.training_columns(user, p["id"], vid)
        except AccessDenied as e:
            st.caption(str(e))
            cols = []
        if cols:
            key = "fair_attrs"
            if key not in st.session_state or not set(
                    st.session_state[key]) <= set(cols):
                st.session_state[key] = fairness.suggest(cols)[:4]
            attrs = st.multiselect("Sensitive attributes to check", cols,
                                   key=key,
                                   placeholder="e.g. Gender, Age, Region")
            if st.button("Run fairness check", type="primary",
                         key="fair_go", disabled=not attrs):
                with st.spinner("Comparing groups"):
                    try:
                        fairness.check(user, p["id"], vid, attrs)
                        log.event(_log, "fairness_checked",
                                  attributes=len(attrs))
                    except AccessDenied as e:
                        st.warning(str(e))
                st.rerun()
    got = fairness.reports(user, p["id"], vid)
    if not got:
        st.caption("No check yet for this version.")
        return
    _report(p, user, got[0], owner)
    if len(got) > 1:
        with st.expander(f"Earlier checks ({len(got) - 1})"):
            for r in got[1:]:
                st.caption(f"{_t(r['created_at'])} · {BADGE[r['status']]} · "
                           f"{', '.join(r['attributes'])} · by "
                           f"{r['created_by']}")


# ── activity ────────────────────────────────────────────────────────────
def _table(rows):
    return pd.DataFrame([{
        "When": _t(e["at"]), "Who": e["actor"], "Role": e["role"],
        "Action": e["action"], "On": f"{e['target_type']} {e['target_id']}"
        .strip(), "Details": ", ".join(f"{k}={v}" for k, v in
                                       (e["details"] or {}).items())}
        for e in rows])


def activity_section(p, user):
    if not projects.can(user, p["id"], "owner"):
        return
    st.markdown("#### Activity")
    action = st.text_input("Filter by action", key="act_filter",
                           placeholder="e.g. model. or project.share")
    rows = audit.events(user, p["id"], action=action.strip() or None,
                        limit=300)
    if not rows:
        st.caption("Nothing recorded yet.")
        return
    table = _table(rows)
    st.dataframe(table, hide_index=True, width="stretch", height=300)
    st.download_button("Export (CSV)", key="act_csv",
                       data=table.to_csv(index=False).encode("utf-8"),
                       file_name=f"activity_{p['id'][:8]}.csv",
                       mime="text/csv")


# ── the admins' audit log page ──────────────────────────────────────────
def audit_page():
    user = auth.user()
    st.markdown("## Audit log")
    if not auth.can("admin"):
        st.info(auth.why_not("admin"))
        return
    st.caption("Every recorded action across all projects, newest first. "
               "Events are append-only and hash-chained: changing or "
               "removing one breaks the chain from that point.")
    c1, c2, c3 = st.columns([2, 2, 1.2], vertical_alignment="bottom")
    action = c1.text_input("Action starts with", key="aud_action",
                           placeholder="e.g. model.approve")
    actor = c2.text_input("By (email or system:…)", key="aud_actor")
    if c3.button("Verify chain", width="stretch", key="aud_verify"):
        with st.spinner("Checking every event"):
            got = audit.verify()
        log.event(_log, "audit_verified", ok=got["ok"], events=got["events"])
        if got["ok"]:
            n = got["events"]
            st.success(f"Chain intact: {n:,} event{'' if n == 1 else 's'} "
                       f"verified.")
        else:
            st.error(f"Chain broken at event {got['first_bad']}: it was "
                     f"changed or the one before it removed.")
    rows = audit.events(user, action=action.strip() or None,
                        actor=actor.strip() or None, limit=1000)
    if not rows:
        st.caption("No events match.")
        return
    table = _table(rows)
    names = {pr["id"]: pr["name"] for pr in projects.list_projects(user)}
    table.insert(2, "Project", [names.get(e["project_id"], e["project_id"]
                                          or "—") for e in rows])
    st.dataframe(table, hide_index=True, width="stretch", height=480)
    st.download_button("Export (CSV)", key="aud_csv",
                       data=table.to_csv(index=False).encode("utf-8"),
                       file_name="audit_log.csv", mime="text/csv")
