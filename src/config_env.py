"""Environment variables with optional Streamlit Cloud secrets."""

from __future__ import annotations

import os


def get_config(key: str, default: str = "") -> str:
    """Read config from Streamlit secrets (cloud) or os.environ / .env."""
    try:
        import streamlit as st

        if key in st.secrets:
            val = st.secrets[key]
            return str(val) if val is not None else default
    except Exception:
        pass
    return os.getenv(key, default)
