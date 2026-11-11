"""The frame around every page: a sidebar to move between the areas of the
app, and a top bar that says where you are, whether the work is saved, and
who is signed in. The workflow steps inside an analysis are `nav`."""
import html

import streamlit as st

from components import project_panel
from utils import auth, guard, scroll

# page -> what the top bar calls it
TITLES = {None: "Home", "start": "Data", "ml_eda": "Explore & prepare",
          "ml_features": "Feature engineering",
          "ml_config": "Configure & train", "ml_results": "Results",
          "ml_upload": "Use own model", "ml_drift": "Drift analysis",
          "ml_audit": "Audit log"}
# The project page, one section at a time: (key, label, icon).
SECTIONS = (("runs", "Runs", ":material/history:"),
            ("models", "Models", ":material/deployed_code:"),
            ("operations", "Monitoring & batch", ":material/monitoring:"),
            ("access", "Access & activity", ":material/group:"))
ANALYSIS = ("ml_eda", "ml_features", "ml_config", "ml_results", "ml_upload",
            "ml_drift")


# What a status says, as a chip's colour.
TONES = {"production": "good", "ok": "good", "succeeded": "good",
         "approved": "good", "pass": "good", "owner": "good",
         "staging": "info", "running": "info", "queued": "info",
         "editor": "info", "pending": "warn", "warning": "warn",
         "alert": "bad", "failed": "bad", "rejected": "bad", "flag": "bad"}


class Html(str):
    """Markup this module made, which `table` does not escape again."""


def status(text, tone=None):
    """A small coloured label for a status."""
    tone = tone or TONES.get(str(text).lower(), "idle")
    return Html(f"<span class='chip chip-{tone}'>"
                f"{html.escape(str(text).capitalize())}</span>")


def table(columns, rows):
    """A dense table: `columns` are (label, key) or (label, key, "num");
    a cell is text (escaped here) or a `status(...)`."""
    head = "".join(
        f"<th class='{c[2] if len(c) > 2 else ''}'>{html.escape(c[0])}</th>"
        for c in columns)
    body = ""
    for row in rows:
        cells = ""
        for c in columns:
            value = row.get(c[1])
            text = (value if isinstance(value, Html) else
                    "<span class='muted'>—</span>" if value in (None, "")
                    else html.escape(str(value)))
            cells += f"<td class='{c[2] if len(c) > 2 else ''}'>{text}</td>"
        body += f"<tr>{cells}</tr>"
    st.markdown(f"<div class='ent-scroll'><table class='ent-table'><thead>"
                f"<tr>{head}</tr></thead><tbody>{body}</tbody></table></div>",
                unsafe_allow_html=True)


def section():
    """The project page's open section."""
    keys = [k for k, _, _ in SECTIONS]
    chosen = st.session_state.get("proj_section")
    return chosen if chosen in keys else keys[0]


def go(page, project_section=None):
    ss = st.session_state
    ss["page"] = page
    if project_section:
        ss["proj_section"] = project_section
    scroll.request()
    st.rerun()


def _item(label, icon, here, key, disabled=False, help_=None):
    return st.button(label, icon=icon, key=f"shell_{key}", width="stretch",
                     type="primary" if here else "tertiary",
                     disabled=disabled, help=help_) and not here


def _sidebar(current):
    ss = st.session_state
    with st.sidebar:
        st.markdown("<div class='shell-brand'><span class='shell-mark'>"
                    "</span>ML Automation</div>", unsafe_allow_html=True)
        with st.container(key="shell_nav"):
            st.markdown("<div class='shell-group'>Workspace</div>",
                        unsafe_allow_html=True)
            if _item("Home", ":material/home:", current is None, "home"):
                go(None)
            if _item("Data", ":material/upload_file:", current == "start",
                     "data"):
                go("start")
            if _item("Analysis", ":material/query_stats:",
                     current in ANALYSIS, "analysis",
                     disabled="ml_df" not in ss,
                     help_=None if "ml_df" in ss else
                     "Load a file on Data, or open a project, first."):
                go(ss.get("shell_last_analysis") or "ml_eda")
            p = project_panel.current()
            if p:
                st.markdown(f"<div class='shell-group'>Project · "
                            f"{html.escape(p['name'])}</div>",
                            unsafe_allow_html=True)
                for key, label, icon in SECTIONS:
                    here = current == "ml_project" and section() == key
                    if _item(label, icon, here, f"proj_{key}"):
                        go("ml_project", key)
            if auth.can("admin"):
                st.markdown("<div class='shell-group'>Admin</div>",
                            unsafe_allow_html=True)
                if _item("Audit log", ":material/verified_user:",
                         current == "ml_audit", "audit"):
                    go("ml_audit")


def _crumbs(current):
    parts = []
    p = project_panel.current()
    if p:
        parts.append(p["name"])
    if current == "ml_project":
        parts.append(next(label for key, label, _ in SECTIONS
                          if key == section()))
    else:
        if current in ANALYSIS and st.session_state.get("ml_name"):
            parts.append(st.session_state["ml_name"])
        parts.append(TITLES.get(current, "Home"))
    head = "".join(f"<span>{html.escape(str(x))}</span><i>/</i>"
                   for x in parts[:-1])
    return (f"<div class='shell-crumbs'>{head}"
            f"<b>{html.escape(str(parts[-1]))}</b></div>")


def render(current):
    """Sidebar and top bar; `current` is the page being shown (None for
    home)."""
    if current in ANALYSIS:
        st.session_state["shell_last_analysis"] = current
    _sidebar(current)
    with st.container(key="shell_top"):
        crumbs, project, account = st.columns([5, 1.6, 1.6],
                                              vertical_alignment="center")
        crumbs.markdown(_crumbs(current), unsafe_allow_html=True)
        with project:
            guard.section("Project", project_panel.chip)
        with account:
            auth.account_menu()
