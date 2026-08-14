"""
config.py — Centralised secret + env-var loader.

Works locally (reads .env) AND on Streamlit Community Cloud
(reads st.secrets which the dashboard injects).

Import this module ONCE at the top of main.py BEFORE any other
local import.  It pushes every secret into os.environ so the rest
of the codebase can keep using os.getenv() unchanged.
"""

import os
import streamlit as st
from dotenv import load_dotenv

# 1. Load .env file if it exists (local dev)
load_dotenv()

# 2. Push Streamlit Cloud secrets into os.environ
#    so that os.getenv() and libraries that read env-vars
#    (LangChain, NVIDIA SDKs) all just work.
_REQUIRED_KEYS = [
    "NVIDIA_API_KEY",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_PROJECT",
]

for key in _REQUIRED_KEYS:
    # st.secrets takes priority over .env
    try:
        val = st.secrets[key]
        if val:
            os.environ[key] = str(val)
    except (KeyError, FileNotFoundError):
        pass  # Fall back to whatever load_dotenv() put there