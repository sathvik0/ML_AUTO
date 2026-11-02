"""One failing section must not take the page down.

The dashboard draws its sections one after another, so before this an
exception in one stopped every section below it and put a traceback — with
server file paths — in front of whoever was looking. Now each section is drawn
through `section()`: if it raises, its place shows one line with a reference,
the traceback goes to the log under the same reference, and the next section
draws as normal.

    utils.guard.section("Tier KPI Summary", tier_summary.render, result)

**A section that is an `st.fragment` also needs `@guarded` inside the
fragment.** Touching a widget in a fragment re-runs the fragment function on
its own; the call site in app.py does not run at all, so its guard cannot
catch that failure. Order matters — the fragment decorator goes on top:

    @st.fragment
    @guard.guarded("Pivot Table")
    def render(df): ...

`st.rerun()` and `st.stop()` raise BaseException subclasses, so
`except Exception` lets them through — a guarded section can still rerun the
app or stop the script.
"""
import functools
import logging

import streamlit as st

from utils import log

_log = log.get("guard")


def section(label, fn, *args, **kwargs):
    """Draw `fn(*args, **kwargs)`; if it raises, say so in its place.

    On failure the notice takes the section's place: what it drew before
    failing is hidden, rather than left as half a table above the warning.
    Returns whatever `fn` returned, or None if it failed.
    """
    return _run(label, f"{label} could not be drawn.", fn, args, kwargs)


def action(label, notice, fn, *args, **kwargs):
    """`section` for a step that does work rather than draw, such as loading
    an upload: `notice` says what failed ("That dataset could not be loaded.")
    where "could not be drawn" would not."""
    return _run(label, notice, fn, args, kwargs)


def _run(label, notice, fn, args, kwargs):
    # A plain container, never an st.empty() slot. A slot goes out as "empty"
    # and then as the container that replaces it; when a flush falls between
    # the two, the browser draws the empty slot and everything inside is torn
    # down and rebuilt — every expander closes, every tab resets. With the
    # split forced, the Snowflake panel closed after Connect and after every
    # picker change — the symptom reported from a Windows machine.
    holder = st.container()
    try:
        with log.timer() as t, holder:
            out = fn(*args, **kwargs)
    except Exception:
        ref = log.incident_id()
        log.event(_log, "section_failed", level=logging.ERROR, exc_info=True,
                  section=label, ref=ref)
        _notice(holder, f"{notice} Reference {ref}.", ref)
        return None
    log.event(_log, "section_drawn", level=logging.DEBUG, section=label,
              ms=t.ms)
    return out


def _notice(holder, text, ref):
    """The notice, in place of whatever the section drew before it failed.

    That output cannot be removed — only an st.empty() slot can be replaced,
    and the slot is what rebuilt the page — so it is hidden instead: the
    notice sits in a container keyed by this failure's reference, and a style
    rule hides every sibling of it. Two forms: Streamlit 1.63 wraps a nested
    container in a layout div, and the first form covers a version that does
    not. A browser without :has() shows the partial output above the notice,
    nothing worse.
    """
    key = f"guard_failed_{ref}"
    block = '[data-testid="stVerticalBlock"]'
    with holder:
        # Style-only HTML goes to Streamlit's event container: no space taken.
        st.html(f"<style>{block}:has(> .st-key-{key}) > :not(.st-key-{key}),"
                f"{block}:has(> * > .st-key-{key}) > "
                f":not(:has(> .st-key-{key})) {{ display: none !important; }}"
                f"</style>")
        with st.container(key=key):
            st.warning(text)


def guarded(label):
    """Decorator form of `section`, for use inside an `st.fragment`."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return section(label, fn, *args, **kwargs)
        return wrapper
    return deco
