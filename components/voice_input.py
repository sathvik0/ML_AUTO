"""Shared voice capture: record, transcribe, drop the text into a text box.

Used by Feature engineering. The caller passes a `prefix` that keys its
session state, so a second recorder could never consume this one's clip.

Never auto-sends. The transcript lands in the box and the user presses the
button — a misheard "4000" as "400" must not fire an instruction on its own.
"""
import hashlib

import streamlit as st

from components import input_keys
from logic import transcribe


def _nonce(prefix, which):
    return st.session_state.get(f"{prefix}_{which}_nonce", 0)


def reset(prefix, clear_input=True):
    """Hand the next turn a fresh recorder, and optionally an empty box.

    A new key is a new widget, empty by construction — the only reliable way
    to clear one (see `components/input_keys.py`, which owns that half and is
    what the box itself is keyed by). This adds the recorder side: a new mic
    widget, and the per-clip state dropped with the old one.

    `clear_input=False` keeps what is typed — the column builder deliberately
    leaves the prompt in place so it can be tweaked and re-run.
    """
    st.session_state[f"{prefix}_mic_nonce"] = _nonce(prefix, "mic") + 1
    if clear_input:
        input_keys.reset(prefix)
    # These belong to the recorder going away; carrying them over would mute a
    # re-recorded instruction as a duplicate.
    for suffix in ("seen", "error", "done"):
        st.session_state.pop(f"{prefix}_mic_{suffix}", None)


def capture(prefix, label="Speak your instruction",
            done="Transcribed — review and send", show_status=True):
    """Render the recorder; a transcript lands in `input_keys.input_key(prefix)`.

    Call this *before* the box renders: session_state for a keyed widget may
    only be set ahead of the widget itself. It must also sit outside any
    `st.form` — a widget inside a form does not rerun until submit, so a
    recording in there could never be transcribed until after it was sent.

    `show_status=False` keeps the stock success and warning boxes out of the
    way for a caller that renders the same facts itself (the voice console puts
    them in its status pill). The state is still recorded, so `status(prefix)`
    reports it either way.

    Returns True when this run produced a new transcript.
    """
    if not transcribe.available():
        return False
    # Full width: a half-width recorder floating on one side reads as a stray
    # control rather than part of the composer.
    clip = st.audio_input(label, key=f"{prefix}_mic_{_nonce(prefix, 'mic')}",
                          label_visibility="collapsed")
    if clip is None:
        return False

    raw = clip.getvalue()
    # Streamlit hands back the same clip on every rerun; transcribe each
    # recording once, not once per widget interaction.
    digest = hashlib.md5(raw).hexdigest()
    if st.session_state.get(f"{prefix}_mic_seen") == digest:
        # Already transcribed; say so on every later rerun. The app-wide loader
        # runs for the length of the API call and looks like any other rerun,
        # so without this the only signal that the work finished is text
        # appearing in a box the user may not be watching.
        if show_status:
            if st.session_state.get(f"{prefix}_mic_error"):
                st.warning(st.session_state[f"{prefix}_mic_error"])
            elif st.session_state.get(f"{prefix}_mic_done"):
                st.success(done, icon="✅")
        return False

    st.session_state[f"{prefix}_mic_seen"] = digest
    with st.spinner("Transcribing"):
        text, err = transcribe.transcribe(raw, getattr(clip, "name", None))
    st.session_state[f"{prefix}_mic_error"] = err
    st.session_state[f"{prefix}_mic_done"] = not err
    if err:
        if show_status:
            st.warning(err)
        return False
    st.session_state[input_keys.input_key(prefix)] = text
    if show_status:
        st.success(done, icon="✅")
    return True


def status(prefix):
    """Where this recorder stands: ("ready" | "done" | "error", message).

    Returned rather than rendered so a caller can put it in its own status
    pill. Recording is deliberately not one of these states: the browser hands
    over a clip only when the user stops, so the server never sees a run in
    between. The DOM does know — Streamlit swaps the record button for a stop
    button — which is why the recording look is CSS.
    """
    err = st.session_state.get(f"{prefix}_mic_error")
    if err:
        return "error", err
    if st.session_state.get(f"{prefix}_mic_done"):
        return "done", "Transcribed"
    return "ready", "Ready"


def unavailable_note():
    """One line explaining why no recorder is shown, or None when it is."""
    if transcribe.available():
        return None
    return transcribe.unavailable_reason()
