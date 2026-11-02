"""The process bar at the top of every page: the steps in order, the page
the user is on highlighted, the steps before it ticked, and the ones that
cannot be opened yet greyed out. It replaces the per-page navigation
buttons."""
import streamlit as st

from utils import auth, guard, scroll

# (page, label). "start" is the start page (no page set).
STEPS = (("start", "Data"), ("ml_eda", "Explore & prepare"),
         ("ml_features", "Feature engineering"), ("ml_config", "Configure & train"),
         ("ml_results", "Results"))
# Reached instead of training: scored on the same file.
BRANCH = ("ml_upload", "Use own model")
# Open whenever a file is loaded: compares it with newer data.
DRIFT = ("ml_drift", "Drift analysis")

_CSS = """
<style>
.st-key-ml_nav { margin-bottom: 0.6rem; }
.st-key-ml_nav [data-testid="stHorizontalBlock"] { gap: 0.35rem; }
.st-key-ml_nav button {
  border-radius: 999px !important; min-height: 2.4rem;
  white-space: nowrap;
}
.st-key-ml_nav button p { font-size: 0.86rem; }
.st-key-ml_nav [data-testid="stBaseButton-secondary"]:not(:disabled) {
  background: #F2F1FC; border-color: #DDDCF5; color: #3C3489;
}
.st-key-ml_nav [data-testid="stBaseButton-primary"] {
  background: #534AB7 !important; border-color: #534AB7 !important;
  box-shadow: 0 2px 10px rgba(83,74,183,0.30);
}
.st-key-ml_nav .nav-or {
  text-align: center; color: #898781; font-size: 0.8rem; padding-top: 0.6rem;
}
</style>
"""


def _available(page):
    ss = st.session_state
    if page == "start":
        return True
    if "ml_df" not in ss:
        return False
    if page == "ml_results":
        return "ml_run" in ss
    if page == "ml_upload":
        from components import ml_config
        return ml_config.upload_allowed() and auth.can("own_model")
    return True


def _go(page):
    ss = st.session_state
    if page == "start":
        # A new file: everything about this one goes.
        from components import ml_config
        ml_config.leave()
    ss["page"] = page
    scroll.request()
    st.rerun()


def render(current):
    """Draw the bar with `current` highlighted ("start" on the start page)."""
    st.markdown(_CSS, unsafe_allow_html=True)
    _, project, account = st.columns([4, 1.5, 1.5])
    with project:
        from components import project_panel
        guard.section("Project", project_panel.chip)
    with account:
        auth.account_menu()
    order = [p for p, _ in STEPS]
    # The branch sits where training would; drift needs only the file.
    at = (order.index(current) if current in order else
          order.index("ml_features") if current == DRIFT[0] else
          0 if current == "ml_project" else
          order.index("ml_config"))
    with st.container(key="ml_nav"):
        cols = st.columns([0.8, 1.2, 1.3, 1.25, 0.85, 0.2, 1.15, 1.15])
        items = list(STEPS) + [None, BRANCH, DRIFT]
        for i, (col, item) in enumerate(zip(cols, items)):
            if item is None:
                col.markdown("<div class='nav-or'>or</div>",
                             unsafe_allow_html=True)
                continue
            page, label = item
            n = order.index(page) + 1 if page in order else None
            done = n is not None and n - 1 < at
            text = (f"✓ {label}" if done else
                    f"{n}. {label}" if n else label)
            here = page == current
            help_ = ("Start over with another file: this file, its steps "
                     "and any trained models are cleared."
                     if page == "start" and current != "start" else
                     auth.why_not("own_model") if page == "ml_upload"
                     else None)
            if col.button(text, key=f"ml_nav_{page}", width="stretch",
                          type="primary" if here else "secondary",
                          disabled=not here and not _available(page),
                          help=help_) and not here:
                _go(page)
