"""Voice input for Feature engineering: record in the browser, transcribe server-side.

Anthropic has no speech-to-text, so this is a second vendor and its own key.
Kept in a separate module with its own resolution and its own failure handling:
voice is a convenience, so a missing package, a missing key or a bad response
must degrade to a message beside the chat box, never break the assistant.

Capture is `st.audio_input`, which is native to Streamlit — no custom component,
no JS build, and it works through a corporate proxy like any other widget.
"""
import importlib
import io
import logging

from utils import log, secrets

_log = log.get("transcribe")

# gpt-4o-mini-transcribe over whisper-1: measured 0.9-1.1s against 1.4-2.5s
# on the same 3s clip, and cheaper. Latency is the whole user experience here —
# the app-wide loader runs for the length of this call.
MODEL = "gpt-4o-mini-transcribe"
MAX_BYTES = 24 * 1024 * 1024     # the API's own limit is 25 MB
MIN_BYTES = 2000                 # anything smaller is a mis-click, not speech


def resolve_api_key():
    return secrets.resolve("OPENAI_API_KEY")


def package_installed() -> bool:
    try:
        importlib.import_module("openai")
        return True
    except ImportError:
        return False


def available() -> bool:
    """Whether voice input can be offered at all."""
    return package_installed() and bool(resolve_api_key())


def unavailable_reason() -> str:
    if not package_installed():
        return ("Voice input needs the `openai` package — "
                "`pip install openai`, then restart the app.")
    return ("Voice input needs `OPENAI_API_KEY` (environment variable or "
            "`.streamlit/secrets.toml`). Speech-to-text is a separate service "
            "from the assistant's own key.")


def friendly_error(e) -> str:
    try:
        import openai
        if isinstance(e, openai.AuthenticationError):
            return "The OPENAI_API_KEY was rejected — check it is current."
        if isinstance(e, openai.RateLimitError):
            return "Transcription is rate limited — wait a moment and retry."
        if isinstance(e, (openai.APITimeoutError, openai.APIConnectionError)):
            return "Could not reach the transcription service. Type instead."
    except Exception:
        pass
    return f"Could not transcribe that: {e}"


# Logged as a kind, never as the sentence: the last one quotes the API's
# own error text.
_ERROR_KINDS = (
    ("That recording was empty", "empty"),
    ("That recording is", "too_large"),
    ("Voice input needs", "unavailable"),
    ("The OPENAI_API_KEY was rejected", "credentials"),
    ("Transcription is rate limited", "rate_limited"),
    ("Could not reach the transcription service", "unreachable"),
    ("Nothing recognisable", "no_speech"),
)


def _error_kind(error) -> str:
    return next((kind for prefix, kind in _ERROR_KINDS
                 if str(error).startswith(prefix)), "api_error")


def transcribe(audio_bytes, filename="speech.wav", api_key=None, client=None):
    """Return (text, error). Exactly one is meaningful; never raises.

    `client` is injectable so the path is testable without calling the API.
    """
    with log.timer() as t:
        text, error = _transcribe(audio_bytes, filename, api_key, client)
    # The size and how long it took — never the words.
    log.event(_log, "voice_transcribed",
              level=logging.INFO if error is None else logging.WARNING,
              ms=t.ms, bytes=len(audio_bytes or b""), ok=error is None,
              chars=len(text or ""),
              error_kind=None if error is None else _error_kind(error))
    return text, error


def _transcribe(audio_bytes, filename, api_key, client):
    if not audio_bytes or len(audio_bytes) < MIN_BYTES:
        return "", "That recording was empty — hold the button while speaking."
    if len(audio_bytes) > MAX_BYTES:
        return "", (f"That recording is {len(audio_bytes) / 1e6:.0f} MB, over "
                    f"the {MAX_BYTES // 1024 // 1024} MB limit. Record a "
                    f"shorter instruction.")
    if client is None:
        if not package_installed():
            return "", unavailable_reason()
        key = api_key or resolve_api_key()
        if not key:
            return "", unavailable_reason()
        import openai
        client = openai.OpenAI(api_key=key, timeout=60)

    buf = io.BytesIO(audio_bytes)
    buf.name = filename or "speech.wav"   # the API infers format from the name
    try:
        resp = client.audio.transcriptions.create(model=MODEL, file=buf)
    except Exception as e:
        log.event(_log, "voice_api_failed", level=logging.WARNING,
                  exc_info=True)
        return "", friendly_error(e)
    text = (getattr(resp, "text", "") or "").strip()
    if not text:
        return "", "Nothing recognisable in that recording."
    return text, None
