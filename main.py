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
from modules import vibe_talk_express_app

ARENA = "🎭 VibeTalk Arena"
EXPRESS = "🗣️ VibeTalk Express"

PAGES = {
    ARENA: vibe_talk_arena_app.render,
    EXPRESS: vibe_talk_express_app.render,
}


def render_landing():
    st.title("🎓 VibeTalk")
    st.subheader("Welcome! 👋")
    st.write(
        "Practice English in two fun ways. Pick a module from the sidebar on the left to begin."
    )
    st.markdown("---")

    st.markdown("### 🎭 VibeTalk Arena")
    st.write(
        "Have a free, natural chat in simple English. Every session is a fresh "
        "**roleplay**, **debate**, or **discussion**. Talk by voice or text, and get a "
        "friendly coach review at the end."
    )

    st.markdown("### 🗣️ VibeTalk Express")
    st.write(
        "Build sentences step by step. The coach speaks a short line, then gives you a "
        "**Hindi sentence to translate** into English. Answer by voice or text across "
        "8-10 turns, then get a warm written review of how you did."
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
    st.markdown("---")

if choice is None:
    render_landing()
else:
    PAGES[choice]()
