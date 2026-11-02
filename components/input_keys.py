"""Message-box keys that can actually be cleared.

Streamlit cannot blank a text widget in place. `clear_on_submit` keeps blanking
it on the *frontend* after the first submit, so anything written into
session_state on a later run is held server-side and never shown; and writing
`""` into a live widget loses to the frontend value, which is written back on
the next run. The only reliable clear is a **new key** — a new widget is empty
by construction.

So every box that must clear after a send is keyed by a nonce this module hands
out, and the caller steps the nonce instead of trying to clear the widget. The
prefix keeps each box's keys apart.
"""
import streamlit as st


def _nonce(prefix):
    return st.session_state.get(f"{prefix}_input_nonce", 0)


def input_key(prefix):
    """Key for the text box this prefix owns."""
    return f"{prefix}_input_{_nonce(prefix)}"


def reset(prefix):
    """Hand the next run a fresh, empty box under a new key."""
    st.session_state[f"{prefix}_input_nonce"] = _nonce(prefix) + 1
