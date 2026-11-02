"""Put the main pane back at the top when the app moves to another page.

Streamlit keeps `stMain`'s scroll position across a rerun, which is right for
a filter edit and wrong for a page change: the buttons that move between pages
sit at the bottom of the page you are leaving, so the next one opens most of
the way down. Measured on the sample data, Continue landed the dashboard at
8,102 of 10,995 px.

Streamlit 1.58 has no scroll API, so this reaches the host document from a
zero-height iframe — the standard route, and the iframe is same-origin so
`window.parent` is reachable.
"""
import streamlit as st

FLAG = "scroll_to_top"
CONTAINER_KEY = "scroll_helper"   # collapsed to zero height in style.css

_SCRIPT = """
<script>
(() => {
  const win = window.parent, doc = win.document;
  const main = () => doc.querySelector('[data-testid="stMain"]')
                  || doc.scrollingElement;
  // Streamlit restores its own scroll position only once the page has
  // finished rendering, which on the dashboard is seconds after this mounts —
  // a couple of frames is not enough. Hold the top until the page settles,
  // but let go the instant the user scrolls, or this would fight them.
  let stop = false;
  const release = () => { stop = true; };
  ["wheel", "touchstart", "keydown", "mousedown"].forEach(
    e => win.addEventListener(e, release, {once: true, passive: true}));
  const deadline = Date.now() + 3000;
  const tick = () => {
    if (stop || Date.now() > deadline) return;
    const m = main();
    if (m && m.scrollTop !== 0) m.scrollTop = 0;
    win.requestAnimationFrame(tick);
  };
  tick();
})();
</script>
"""


def request():
    """Ask for a scroll to top on the next run. Safe to call from a callback."""
    st.session_state[FLAG] = True


def apply_if_requested():
    """Emit the scroll once, if one was requested. Call early in a page."""
    if st.session_state.pop(FLAG, False):
        # st.iframe, not st.components.v1.html — that is deprecated and due
        # for removal. A raw HTML string is auto-detected and embedded the
        # same way, so window.parent stays reachable.
        #
        # height cannot be 0 here (st.iframe rejects it) and "content" falls
        # back to the 150px iframe default, which is a blank gap at the top of
        # the page. So it is 1px inside a keyed container that style.css
        # collapses — clipped rather than display:none, which would risk the
        # script not running.
        with st.container(key=CONTAINER_KEY):
            st.iframe(_SCRIPT, height=1)
