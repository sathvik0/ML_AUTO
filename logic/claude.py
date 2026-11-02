"""One structured call to Claude, shared by the app's AI features.

Every answer is held to a JSON schema by structured outputs, so it parses.
A request the model declines is retried on a fallback model by the API
itself (server-side fallbacks); one the whole chain declines raises
`AIError`, as does an answer that is not the JSON asked for.
"""
import json

MODEL = "claude-opus-5-5"


class AIError(ValueError):
    """A failed call, in words the page can show."""


def ask(system, messages, schema, *, api_key=None, client=None,
        model=MODEL, effort="medium", max_tokens=16000) -> dict:
    """The parsed JSON answer to `messages` (a Messages API list)."""
    if client is None:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key, timeout=180)
    resp = client.beta.messages.create(
        model=model, max_tokens=max_tokens, system=system,
        output_config={"effort": effort,
                       "format": {"type": "json_schema", "schema": schema}},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        messages=messages,
    )
    if resp.stop_reason == "refusal":
        raise AIError("The assistant declined that request. Try rewording "
                      "it.")
    if resp.stop_reason == "max_tokens":
        raise AIError("The answer was cut off before it finished. Try a "
                      "narrower request.")
    text = next((b.text for b in resp.content if b.type == "text"), "")
    try:
        return json.loads(text)
    except ValueError:
        raise AIError("The assistant did not return a usable answer. Try "
                      "again.") from None


def friendly_error(e) -> str:
    """A failed call in words the page can show."""
    if isinstance(e, AIError):
        return str(e)
    try:
        import anthropic
        if isinstance(e, anthropic.AuthenticationError):
            return ("Your API key was rejected — check `ANTHROPIC_API_KEY` "
                    "(it may be wrong or revoked).")
        if isinstance(e, anthropic.RateLimitError):
            return "Rate limited by the API — wait a moment and try again."
        if isinstance(e, anthropic.APITimeoutError):
            return "That took too long and timed out. Try again or simplify."
        if isinstance(e, anthropic.APIConnectionError):
            return "Unable to reach the API — check your connection and retry."
        if isinstance(e, anthropic.APIStatusError) and \
                getattr(e, "status_code", 0) in (500, 529):
            return "The API is overloaded right now — try again shortly."
    except Exception:
        pass
    return f"Something went wrong: {e}"
