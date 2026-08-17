"""
auth.py — Login gate for VibeTalk.

Checks name + password against auth/users.json (bcrypt-hashed, never
plaintext — safe to commit even to a public repo). There is no in-app way
to add, remove, or change anyone's credentials; that's only possible via
auth/manage_users.py, run directly by whoever has filesystem access to
the repo.
"""

import json
import os

import bcrypt
import streamlit as st

_USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")


def _load_users() -> dict:
    with open(_USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _verify(username: str, password: str):
    users = _load_users()
    record = users.get(username.strip().lower())
    if not record:
        return None
    if bcrypt.checkpw(password.encode("utf-8"), record["password_hash"].encode("utf-8")):
        return record["display_name"]
    return None


def require_login() -> None:
    """Blocks the rest of the script (via st.stop()) until a valid
    name + password is submitted. Call this before rendering anything else."""
    if st.session_state.get("auth_user"):
        return

    st.title("🎓 VibeTalk")
    st.subheader("Sign in")

    with st.form("login_form"):
        username = st.text_input("Name")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        display_name = _verify(username, password)
        if display_name:
            st.session_state.auth_user = display_name
            st.rerun()
        else:
            st.error("Name or password is incorrect.")

    st.stop()


def render_logout_control() -> None:
    """Small sidebar control showing who's logged in, with a log-out button."""
    with st.sidebar:
        st.caption(f"Signed in as **{st.session_state.get('auth_user')}**")
        if st.button("Log out", key="auth_logout"):
            del st.session_state["auth_user"]
            st.rerun()
        st.markdown("---")
