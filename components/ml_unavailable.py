"""The Machine Learning page when scikit-learn is installed but cannot run.

Installed is not the same as loadable: a Python with an old scikit-learn,
or with a dependency it cannot use, starts the app, and the pages then fail
with a bare "ImportError": messages are withheld on screen. Two ways to get there: a scikit-learn from
before 1.0, which has no HistGradientBoostingClassifier to import, or a
threadpoolctl from before 3.0 under a current one; installing into an
existing Python keeps any version that meets the minimum, and an install
that fails part way changes nothing at all.

So a scikit-learn below the floor is stopped before anything imports it,
the import is tried here, and a failure names the library it came from, the
versions installed against the ones the app is tested with
(requirements.lock.txt, when present), and the commands that install those. Nothing is
read from the exception's message: the library comes from the traceback's
file paths and the module an ImportError names, and only names are shown.
"""
import importlib.metadata as metadata
import re
import sys
from pathlib import Path

import streamlit as st

# Distribution name → how the page writes it. The four the pages import, in
# the order they depend on each other.
LIBRARIES = {"scikit-learn": "scikit-learn", "scipy": "SciPy",
             "numpy": "NumPy", "joblib": "joblib"}
# requirements.txt's floor: partial dependence takes custom values from 1.7.
SKLEARN_MIN = (1, 7)
# What the pinned versions install on: NumPy 2.4 and SciPy 1.17 need 3.11.
PYTHON_MIN = (3, 11)
LOCK = Path(__file__).resolve().parent.parent / "requirements.lock.txt"

# The top-level package of a file inside site-packages (dist-packages on
# Debian), on either separator.
_SITE = re.compile(r"[\\/](?:site|dist)-packages[\\/]([^\\/]+)")


class TooOld(Exception):
    """scikit-learn older than the pages need."""


def _norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def version(dist):
    """A distribution's installed version, read from its metadata, or None.
    Never imports it: importing is what fails."""
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def tested():
    """The pinned versions of requirements.lock.txt, by distribution name;
    empty when the file is not there."""
    try:
        text = LOCK.read_text(encoding="utf-8")
    except OSError:
        return {}
    pins = {}
    for line in text.splitlines():
        m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([^\s;#]+)",
                     line)
        if m:
            pins[_norm(m.group(1))] = m.group(2)
    return pins


def release(text):
    """(major, minor) of a version string, or None."""
    m = re.match(r"\s*(\d+)\.(\d+)", text or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def load_pages():
    """The four Machine Learning pages. Raises TooOld for a scikit-learn
    below SKLEARN_MIN, and whatever importing them raises."""
    have = release(version("scikit-learn"))
    if have is not None and have < SKLEARN_MIN:
        raise TooOld()
    from components import ml_config, ml_eda, ml_evaluate, ml_results
    return ml_config, ml_eda, ml_evaluate, ml_results


def culprit(exc):
    """The distribution the failure came from: the package of the deepest
    frame inside site-packages, across the whole chain. None when no frame
    is in an installed library."""
    found, seen, cur = None, set(), exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        tb, deepest = cur.__traceback__, None
        while tb is not None:
            m = _SITE.search(tb.tb_frame.f_code.co_filename)
            if m:
                deepest = m.group(1)
            tb = tb.tb_next
        # The last exception raised is the one that reached us; a cause it
        # was raised from only answers when the last has no library frame.
        found = found or deepest
        cur = cur.__cause__ or cur.__context__
    return None if found is None else _distribution(found)


def named(exc):
    """The distribution an ImportError was importing from ("cannot import
    name 'ThreadpoolController' from 'threadpoolctl'" names threadpoolctl),
    or None."""
    name = getattr(exc, "name", None) if isinstance(exc, ImportError) \
        else None
    return _distribution(name) if name else None


def _distribution(module):
    """The installed distribution a module comes from: sklearn is
    scikit-learn. The package name itself when nothing claims it."""
    package = module.split(".")[0]
    try:
        dists = metadata.packages_distributions().get(package) or []
    except Exception:
        dists = []
    return _norm(dists[0]) if dists else _norm(package)


def label(dist):
    return LIBRARIES.get(dist, dist)


def installed_line(extras=()):
    """Python, the four libraries and any other one the failure involves, as
    "Python 3.12.4, scikit-learn 1.9.1, SciPy 1.11.4 (tested with 1.17.1)"."""
    pins = tested()
    names = list(LIBRARIES)
    names += [d for d in dict.fromkeys(extras) if d and d not in names]
    parts = [f"Python {sys.version.split()[0]}"]
    for dist in names:
        have, pin = version(dist), pins.get(dist)
        if have is None:
            parts.append(f"{label(dist)} not installed")
        elif pin and pin != have:
            parts.append(f"{label(dist)} {have} (tested with {pin})")
        else:
            parts.append(f"{label(dist)} {have}")
    return ", ".join(parts)


def commands():
    """A new environment with the tested versions, for this machine. macOS
    takes requirements.txt: the lock pins the CPU XGBoost wheel, which has
    no macOS build."""
    py = r".venv\Scripts\python" if sys.platform == "win32" \
        else ".venv/bin/python"
    req = "requirements.txt" if sys.platform == "darwin" \
        else "requirements.lock.txt"
    return (f"python -m venv .venv\n"
            f"{py} -m pip install -r {req}\n"
            f"{py} -m streamlit run app.py")


def render(exc, ref, on_back):
    """The page in place of Machine Learning: what could not load, the
    versions, and how to install the tested ones."""
    st.markdown("## Machine Learning")
    if isinstance(exc, TooOld):
        lead = (f"**Machine Learning needs scikit-learn "
                f"{SKLEARN_MIN[0]}.{SKLEARN_MIN[1]} or later**, and "
                f"{version('scikit-learn')} is installed here.")
        extras = ()
    else:
        extras = (culprit(exc), named(exc))
        what = label(extras[0]) if extras[0] else "its libraries"
        lead = (f"**Machine Learning could not load {what} here** "
                f"({type(exc).__name__}). A library left over from an "
                f"earlier install is the usual cause: installing into an "
                f"existing Python keeps any version that meets the minimum.")
    st.warning(f"{lead}\n\nInstalled: {installed_line(extras)}.\n\n"
               f"Reference {ref}.")
    st.markdown("The tested versions install into a new environment. From "
                "the app's folder:")
    st.code(commands(), language=None)
    if sys.version_info < PYTHON_MIN:
        st.caption(f"They need Python {PYTHON_MIN[0]}.{PYTHON_MIN[1]} or "
                   f"later, and this is {sys.version.split()[0]}.")
    if st.button("← Back to start", key="ml_unavailable_back"):
        on_back()
