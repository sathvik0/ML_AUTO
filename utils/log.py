"""Structured logging: one JSON object per line.

Why JSON on stdout: a container platform ships stdout to its log store
(CloudWatch, Cloud Logging, ...) with no agent, and those parse JSON on their
own — `event`, `session` and `ref` become fields to filter on.

    {"ts": "2026-10-08T12:01:02.345Z", "level": "INFO", "logger": "mlapp.ml",
     "event": "ml_trained", "session": "3f9a1c2e", "models": 4, "seconds": 12.3}

What is never logged: credentials, the uploaded data, record identifiers or
names, file names, and the text of anything a person typed or said (prompts,
transcripts, scripts, column names they chose). Events carry counts, timings
and outcomes. `redact` scrubs key-shaped strings as a safety net, not as the
policy.

**An exception's message is withheld.** Its type and the stack — file, line,
function and the line of code — are logged; the message is not, because a
message quotes whatever it was handed (`could not convert string to float:
'Dr Smith'`, `Numeric value 'x' is not recognized`) and no filter can tell a
cell value from a column name. Set ``LOG_EXCEPTION_MESSAGES=1`` to keep them
when debugging locally; never in a deployment that sees real data.

Streamlit logs an uncaught error itself, as text with the full message. That
record is re-emitted here as an ``uncaught_exception`` event under the same
rule, so nothing reaches the console with a message attached.

Every app logger is a child of ``mlapp``, which has no output until `setup()`
runs: the test suites import these modules and stay quiet, and attach their
own handler when they want to read what was logged.
"""
import functools
import json
import logging
import os
import re
import sys
import time
import traceback
import uuid

ROOT = "mlapp"

logging.getLogger(ROOT).addHandler(logging.NullHandler())
logging.getLogger(ROOT).propagate = False

_HANDLER_NAME = "mlapp-json"
# Reserved by the formatter; an event field of the same name is dropped
# rather than allowed to overwrite it.
_RESERVED = ("ts", "level", "logger", "event", "session", "msg", "exc",
             "exc_type")

_SECRETS = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),            # Anthropic keys
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),             # OpenAI keys
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?"
               r"-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(r"(?i)\b(password|passwd|pwd|passphrase|token|secret|api_key)"
               r"(\s*[=:]\s*)(['\"]?)[^\s,;'\"]+"),
)


def get(name: str) -> logging.Logger:
    """The logger for one module: ``mlapp.<name>``."""
    return logging.getLogger(f"{ROOT}.{name}")


def redact(text) -> str:
    text = str(text)
    for pat in _SECRETS:
        if pat.groups >= 3:
            text = pat.sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}[redacted]",
                           text)
        else:
            text = pat.sub("[redacted]", text)
    return text


def session_id():
    """A short id for the browser session this run belongs to, or None.

    None outside a script run — a download callable runs on a worker thread
    with no context, so it captures the id at render time and passes it on.
    """
    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import \
            get_script_run_ctx
        ctx = get_script_run_ctx(suppress_warning=True)
        return ctx.session_id[:8] if ctx is not None else None
    except Exception:
        return None


def incident_id() -> str:
    """Six characters a person can read out over the phone."""
    return uuid.uuid4().hex[:6].upper()


class _Context(logging.Filter):
    """Stamps the session and scrubs secrets. Must never raise: a filter that
    throws loses the record and prints a logging error instead."""

    def filter(self, record):
        try:
            if getattr(record, "session", None) is None:
                record.session = session_id()
            record.msg = redact(record.getMessage())
            record.args = None
        except Exception:
            pass
        return True


def _keep_messages() -> bool:
    return os.environ.get("LOG_EXCEPTION_MESSAGES", "").lower() in (
        "1", "true", "yes")


def format_exception(exc_info) -> str:
    """A traceback with every exception's message withheld.

    The whole chain is walked (``raise ... from``, and an error raised while
    handling another), because a cause's message can quote data as easily as
    the last one's. Notes added with ``add_note`` are dropped with them.
    """
    if _keep_messages():
        return "".join(traceback.format_exception(*exc_info))
    chain, seen, cur = [], set(), exc_info[1]
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        chain.append(cur)
        cur = (cur.__cause__ if cur.__cause__ is not None
               else None if cur.__suppress_context__ else cur.__context__)
    parts = []
    for i, exc in enumerate(reversed(chain)):
        if i:
            parts.append("\nThe exception above led to this one:\n\n")
        parts.append("Traceback (most recent call last):\n")
        parts.extend(traceback.format_tb(exc.__traceback__))
        parts.append(f"{_type_name(exc)} (message withheld)\n")
    return "".join(parts)


def _type_name(exc) -> str:
    kind = type(exc)
    module = "" if kind.__module__ == "builtins" else f"{kind.__module__}."
    return module + kind.__qualname__


class JsonFormatter(logging.Formatter):
    def format(self, record):
        name = getattr(record, "event", None)
        out = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            # A stable name to filter on; a plain logger call is just "log".
            "event": name or "log",
            "session": getattr(record, "session", None),
        }
        msg = record.getMessage()
        if msg != name:
            out["msg"] = msg
        for key, value in (getattr(record, "fields", None) or {}).items():
            if key not in _RESERVED:
                out[key] = value
        if record.exc_info and record.exc_info[1] is not None:
            out["exc_type"] = _type_name(record.exc_info[1])
            out["exc"] = redact(format_exception(record.exc_info))
        return json.dumps(out, default=str, ensure_ascii=False)


class _DropConnectionReset(logging.Filter):
    """Windows' "[WinError 10054] An existing connection was forcibly closed".

    When a browser tab sleeps or closes, asyncio's proactor transport logs the
    reset at ERROR from `_call_connection_lost`. The session is already gone
    and nothing failed, but after the app sat idle it filled the console.
    """

    def filter(self, record):
        exc = record.exc_info[1] if record.exc_info else None
        if isinstance(exc, ConnectionResetError):
            return False
        try:
            text = record.getMessage()
        except Exception:
            return True
        return "WinError 10054" not in text


class _RouteUncaught(logging.Filter):
    """Streamlit's own record of an uncaught error, re-emitted as ours.

    Streamlit logs it as text through ``streamlit.error_util``, message and
    all — a callback or a line of app.py outside every guard ends up there.
    The record is dropped and logged again here, as an ``uncaught_exception``
    event with a reference and the message withheld. If Streamlit renames the
    logger or the text, this stops matching and its own line comes back —
    louder, never quieter.
    """

    def filter(self, record):
        try:
            if record.exc_info and record.getMessage().startswith(
                    "Uncaught app"):
                event(get("uncaught"), "uncaught_exception",
                      level=logging.ERROR, exc_info=record.exc_info,
                      ref=incident_id())
                return False
        except Exception:
            pass
        return True


def setup(stream=None, level=None) -> logging.Logger:
    """Send ``mlapp.*`` to `stream` (stdout by default) as JSON lines.

    Idempotent — app.py calls it on every rerun — and the first call wins, so
    a test suite can call it first with its own stream and read the lines.
    The level comes from ``LOG_LEVEL`` (default INFO); DEBUG adds per-section
    render timings.
    """
    root = logging.getLogger(ROOT)
    if not any(h.get_name() == _HANDLER_NAME for h in root.handlers):
        handler = logging.StreamHandler(stream or sys.stdout)
        handler.set_name(_HANDLER_NAME)
        handler.setFormatter(JsonFormatter())
        handler.addFilter(_Context())
        root.addHandler(handler)
        root.setLevel((level or os.environ.get("LOG_LEVEL") or "INFO").upper())
        for name in ("asyncio", "tornado.application", "tornado.general"):
            other = logging.getLogger(name)
            if not any(isinstance(f, _DropConnectionReset) for f in other.filters):
                other.addFilter(_DropConnectionReset())
        uncaught = logging.getLogger("streamlit.error_util")
        if not any(isinstance(f, _RouteUncaught) for f in uncaught.filters):
            uncaught.addFilter(_RouteUncaught())
    return root


def event(logger: logging.Logger, event_name: str, /, *, level=logging.INFO,
          session=None, exc_info=False, **fields):
    """Log one named event with its fields as top-level JSON keys.

    The first two parameters are positional-only so that any field name is
    free — a refinement's `name=` collided with this function's own and
    crashed the dialog that creates it.
    """
    extra = {"event": event_name, "fields": fields}
    if session is not None:
        extra["session"] = session
    logger.log(level, event_name, extra=extra, exc_info=exc_info)


def logged_download(logger: logging.Logger, name: str, **fields):
    """Decorate a deferred download so its outcome is logged.

    `st.download_button(data=<callable>)` runs the callable on a worker thread
    with no script context, so the session id is captured here, at render
    time, while there is one. A failure is logged with its traceback and
    re-raised — the download still fails, but now somebody can see why.
    """
    sid = session_id()

    def deco(fn):
        @functools.wraps(fn)
        def run():
            try:
                with timer() as t:
                    data = fn()
            except Exception:
                event(logger, "download_failed", level=logging.ERROR,
                      exc_info=True, session=sid, download=name, **fields)
                raise
            event(logger, "download_built", session=sid, download=name,
                  bytes=len(data), ms=t.ms, **fields)
            return data
        return run
    return deco


class timer:
    """``with log.timer() as t: ...`` then ``t.ms`` — wall time, rounded."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        self.ms = 0
        return self

    def __exit__(self, *exc):
        self.ms = round((time.perf_counter() - self._t0) * 1000)
        return False
