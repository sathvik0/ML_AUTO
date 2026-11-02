"""API keys: environment variable → st.secrets → the project's
.streamlit/secrets.toml read directly (so it works whichever directory
Streamlit was launched from). None when the key is set nowhere."""
import os


def resolve(name):
    key = os.environ.get(name)
    if key:
        return key
    try:
        import streamlit as st
        key = st.secrets.get(name)
        if key:
            return key
    except Exception:
        pass
    try:
        import tomllib
        root = os.path.dirname(os.path.dirname(__file__))
        with open(os.path.join(root, ".streamlit", "secrets.toml"), "rb") as f:
            return tomllib.load(f).get(name)
    except Exception:
        return None
