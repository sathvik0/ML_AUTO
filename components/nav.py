"""The steps of an analysis, as a row of tabs under the top bar: the page
the user is on underlined, the steps before it ticked, and the ones that
cannot be opened yet greyed out. The frame around it is `shell`."""
import streamlit as st

from utils import auth, scroll

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
.st-key-ml_nav {
  border-bottom: 1px solid #E3E5EA; margin-bottom: 0.9rem;
}
.st-key-ml_nav [data-testid="stHorizontalBlock"] { gap: 0.2rem; }
.st-key-ml_nav button {
  border-radius: 0 !important; min-height: 2.5rem; white-space: nowrap;
  background: transparent !important; border: none !important;
  border-bottom: 2px solid transparent !important;
  box-shadow: none !important;
  color: #5B6170 !important; font-weight: 500 !important;
}
.st-key-ml_nav button p { font-size: 0.88rem; }
.st-key-ml_nav button:hover:not(:disabled) { color: #1F2430 !important; }
.st-key-ml_nav button:disabled { color: #B4B8C2 !important; }
.st-key-ml_nav [data-testid="stBaseButton-primary"] {
  color: #3C3489 !important; border-bottom-color: #534AB7 !important;
}
.st-key-ml_nav .nav-or {
  border-left: 1px solid #E3E5EA; height: 1.4rem; margin: 0.55rem auto 0;
  width: 1px;
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
    order = [p for p, _ in STEPS]
    # The branch sits where training would; drift needs only the file.
    at = (order.index(current) if current in order else
          order.index("ml_features") if current == DRIFT[0] else
          order.index("ml_config"))
    with st.container(key="ml_nav"):
        cols = st.columns([0.8, 1.2, 1.3, 1.25, 0.85, 0.2, 1.15, 1.15])
        items = list(STEPS) + [None, BRANCH, DRIFT]
        for i, (col, item) in enumerate(zip(cols, items)):
            if item is None:
                col.markdown("<div class='nav-or'></div>",
                             unsafe_allow_html=True)
                continue
            page, label = item
            n = order.index(page) + 1 if page in order else None
            done = n is not None and n - 1 < at
            text = label if done or not n else f"{n}. {label}"
            here = page == current
            help_ = ("Start over with another file: this file, its steps "
                     "and any trained models are cleared."
                     if page == "start" and current != "start" else
                     auth.why_not("own_model") if page == "ml_upload"
                     else None)
            if col.button(text, key=f"ml_nav_{page}", width="stretch",
                          icon=":material/check:" if done else None,
                          type="primary" if here else "tertiary",
                          disabled=not here and not _available(page),
                          help=help_) and not here:
                _go(page)
