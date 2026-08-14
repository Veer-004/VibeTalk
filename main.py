"""
main.py — VibeTalk entry point.
Run with:  streamlit run main.py
"""

import config  # noqa: F401  — must be first to load secrets

import streamlit as st

st.set_page_config(
    page_title="VibeTalk",
    page_icon="🎓",
    layout="centered",
)

from modules import vibe_talk_arena_app

ARENA = "🎭 VibeTalk Arena"

PAGES = {
    ARENA: vibe_talk_arena_app.render,
}


def render_landing():
    st.title("🎓 VibeTalk")
    st.subheader("Welcome! 👋")
    st.write(
        "Practice English speaking. Pick a module from the sidebar on the left to begin."
    )
    st.markdown("---")

    st.markdown("### 🎭 VibeTalk Arena")
    st.write(
        "Have a free, natural chat in simple English. Every session is a fresh "
        "**roleplay**, **debate**, or **discussion**. Talk by voice or text, and get a "
        "friendly coach review at the end."
    )

    st.markdown("---")
    st.info("👈 Choose a module from the sidebar to start.")


with st.sidebar:
    st.title("🎓 VibeTalk")
    choice = st.radio(
        "Choose a module",
        list(PAGES.keys()),
        index=None,
        key="nav_choice",
    )
    if choice == ARENA:
        if st.button("🔄 Start New Conversation", key="sidebar_new_conv"):
            vibe_talk_arena_app.reset_conversation()
            st.rerun()
    st.markdown("---")

if choice is None:
    render_landing()
else:
    PAGES[choice]()
