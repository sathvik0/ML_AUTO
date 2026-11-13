"""Machine Learning app — Streamlit entry point.

Upload a table, explore and prepare it, then train and compare
classification, regression or clustering models, or evaluate a model of
your own. Every page keeps its state under keys starting ml_.
"""
import importlib.util
import logging
import os
import warnings

# Pandas/plotly deprecation chatter is noise for the people using this app.
warnings.filterwarnings("ignore")
logging.getLogger("streamlit").setLevel(logging.ERROR)

# JSON lines on stdout for mlapp.*. Idempotent: this runs on every rerun, and
# only the first call attaches the handler.
from utils import log

log.setup()
_log = log.get("app")

import streamlit as st

st.set_page_config(page_title="ML Automation", layout="wide",
                   page_icon=":material/query_stats:")

from components import (ml_unavailable, nav, project_panel, shell,
                        snowflake_panel)
from utils import auth, file_loader, frames, guard, scroll
from utils.file_loader import load_file

_CSS_PATH = os.path.join(os.path.dirname(__file__), "assets", "style.css")
with open(_CSS_PATH, encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

ss = st.session_state

# Nothing renders for a request the sign-in service did not vouch for.
if auth.current_user() is None:
    auth.render_signed_out()
    st.stop()
project_panel.signed_in()

ML_PAGES = ("ml_eda", "ml_features", "ml_config", "ml_upload", "ml_results",
            "ml_drift")


def _drop_ml_state():
    for k in [k for k in ss.keys() if isinstance(k, str)
              and k.startswith("ml_")]:
        del ss[k]


def _ml_frame(df):
    """Names as text, and unique: the configure page addresses every column
    by name, and an Excel header can be a number."""
    if all(isinstance(c, str) for c in df.columns) \
            and not df.columns.duplicated().any():
        return df
    return frames.dedupe_columns(df.rename(columns=str))


def _start_ml(df, source, name=None, save=True):
    """`ml_source` is the file as it came, `ml_df` that file with the
    Explore & prepare steps applied (none yet). With a project open, the
    file becomes its dataset (`save=False` when it came from the project)."""
    _drop_ml_state()
    df = _ml_frame(df)
    ss["ml_source"] = df
    ss["ml_df"] = df
    ss["ml_name"] = name or source
    ss["ml_steps"] = []
    ss["ml_version"] = 0
    ss["page"] = "ml_eda"
    log.event(_log, "ml_dataset_loaded", rows=len(df), columns=df.shape[1],
              source=source)
    if save:
        project_panel.on_new_dataset(df, ss["ml_name"], source)
    scroll.request()


def _back_to_start():
    _drop_ml_state()
    ss["page"] = "start"
    st.rerun()


def _missing():
    st.markdown("## Machine Learning")
    st.warning("This app needs scikit-learn, which is not installed here. "
               "Install it with `pip install -r requirements.txt` and restart "
               "the app.")
    if st.button("← Back to start", key="ml_missing_back"):
        _back_to_start()


def _unloadable(exc):
    """scikit-learn is installed but cannot run: older than the pages need,
    or a library they load fails to import. The page names the library and
    the versions; the log keeps the stack."""
    ref = log.incident_id()
    too_old = isinstance(exc, ml_unavailable.TooOld)
    log.event(_log, "ml_unavailable", level=logging.ERROR,
              exc_info=None if too_old else exc, ref=ref,
              reason="too_old" if too_old else "import_failed",
              library=None if too_old else ml_unavailable.culprit(exc),
              **{d.replace("-", "_"): ml_unavailable.version(d)
                 for d in ml_unavailable.LIBRARIES})
    ml_unavailable.render(exc, ref, _back_to_start)


# ── The admins' audit log ───────────────────────────────────────────────
if ss.get("page") == "ml_audit":
    from components import governance_panel
    scroll.apply_if_requested()
    guard.section("Navigation", shell.render, "ml_audit")
    guard.section("Audit log", governance_panel.audit_page)
    st.stop()


# ── The project page: needs a project, not a file ───────────────────────
if ss.get("page") == "ml_project":
    scroll.apply_if_requested()
    guard.section("Navigation", shell.render, "ml_project")
    guard.section("Project", project_panel.render)
    st.stop()


# ── Machine Learning pages ──────────────────────────────────────────────
if ss.get("page") in ML_PAGES and "ml_df" in ss:
    scroll.apply_if_requested()
    if importlib.util.find_spec("sklearn") is None:
        guard.section("Machine Learning", _missing)
        st.stop()
    try:
        ml_config, ml_eda, ml_evaluate, ml_results = \
            ml_unavailable.load_pages()
    except Exception as _ml_exc:
        guard.section("Machine Learning", _unloadable, _ml_exc)
        st.stop()
    from components import drift_page, ml_features
    page = ss["page"]
    if page == "ml_results" and "ml_run" not in ss:
        page = "ml_config"
    guard.section("Navigation", shell.render, page)
    guard.section("Steps", nav.render, page)
    if page == "ml_results":
        guard.section("Model results", ml_results.render)
    elif page == "ml_eda":
        guard.section("Explore & prepare", ml_eda.render)
    elif page == "ml_features":
        guard.section("Feature engineering", ml_features.render)
    elif page == "ml_upload":
        guard.section("Use own model", ml_evaluate.render)
    elif page == "ml_drift":
        guard.section("Drift analysis", drift_page.render)
    else:
        guard.section("Configure & train", ml_config.render)
    project_panel.autosave()
    st.stop()


# ── Home, and the data page: upload a file or try the sample ────────────
SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "sample_data",
                           "insurance_churn.csv")
SALES_PATH = os.path.join(os.path.dirname(__file__), "sample_data",
                          "store_sales.csv")


def _load_sample(path=SAMPLE_PATH):
    _start_ml(file_loader.read_path(path), "sample", os.path.basename(path))
    st.rerun()


def _home():
    from components import home
    shell.render(None)
    home.render(_start_ml,
                _load_sample if os.path.exists(SAMPLE_PATH) else None)


_HOW = """
1. **Data**: upload a file, use the sample, or read from Snowflake.
2. **Explore & prepare**: profile the columns and clean them.
3. **Feature engineering**: add columns, with AI help.
4. **Configure & train**: pick the target and the models.
5. **Results**: compare, explain, export and register.
"""


def _data():
    shell.render("start")
    nav.render("start")
    st.markdown("## Data")
    p = project_panel.current()
    st.caption(f"A file loaded here becomes the dataset of {p['name']}."
               if p else "Load a file to explore it and train models. Save "
               "it as a project (top right) to keep the work.")
    mid, side = st.columns([1.5, 1], gap="medium")
    with side, st.container(border=True, key="card_how"):
        st.markdown("#### How it works")
        st.markdown(_HOW)
    with mid, st.container(border=True, key="card_load"):
        uploaded = st.file_uploader("Upload a .csv or .xlsx / .xls file",
                                    type=["csv", "xlsx", "xls"],
                                    key="main_upload")
        has_sample = os.path.exists(SAMPLE_PATH)
        b_go, b_sample, b_sales = st.columns(3)
        go = b_go.button("Start", type="primary", disabled=uploaded is None,
                         width="stretch")
        use_sample = has_sample and b_sample.button(
            "Try sample data", width="stretch",
            help="Open the bundled sample dataset.")
        use_sales = os.path.exists(SALES_PATH) and b_sales.button(
            "Try a sales forecast", width="stretch", key="sample_sales",
            help="Two years of daily sales for eight stores.")
        if uploaded is None:
            st.caption("Add a file above, then start. Or try a sample: "
                       "10,000 insurance customers with who churned at "
                       "renewal (classification), their annual premium and "
                       "lifetime value (regression), and behaviour that "
                       "falls into customer segments (clustering); or two "
                       "years of daily sales for eight stores, to forecast "
                       "with past values and a time split.")
        with st.expander("Or load the data from Snowflake"):
            st.caption("Read a table or view, or write a query that selects "
                       "only the rows needed from a very large one. A table "
                       "is read whole by default — every row is analysed.")
            # An optional panel: if it breaks, the upload still works.
            sf_df = guard.section("Snowflake", snowflake_panel.source_picker,
                                  "load")
    # Its own guard inside this page's: a dataset that fails to load is
    # reported under the controls, rather than replacing the page.
    guard.action("Loading the dataset", "That dataset could not be loaded.",
                 _launch, go, uploaded, use_sample, sf_df, use_sales)


def _launch(go, uploaded, use_sample, sf_df=None, use_sales=False):
    if go and uploaded is not None:
        df = load_file(uploaded)
        if df is not None:
            _start_ml(df, "upload", uploaded.name)
            st.rerun()
    elif use_sample:
        _load_sample()
    elif use_sales:
        _load_sample(SALES_PATH)
    elif sf_df is not None:
        # The picker only returns a frame on the run its Load button was
        # pressed, so this cannot re-fire on a later interaction.
        _start_ml(sf_df, "snowflake", "Snowflake data")
        st.rerun()


if ss.get("page") == "start":
    guard.section("Data", _data)
else:
    guard.section("Home", _home)
