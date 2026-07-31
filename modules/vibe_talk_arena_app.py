"""
VibeTalk Arena page.
Wrapped as render() so it can live inside a multi-page app.
Session-state keys are prefixed with `cb_` to avoid clashing with the coach page.
"""

import io
import os
import asyncio
import re
import streamlit as st
from langchain_core.messages import HumanMessage
from groq import Groq
import edge_tts

from engines.vibe_talk_arena_bot import start_app, turn_app, ChatState

_groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Characters/symbols we never want the voice to read out loud.
_STRIP_CHARS = '_:;"“”‘’`*#|~<>[]{}()'


def _clean_for_speech(text: str) -> str:
    """Remove emojis, underscores, quotes, colons, semicolons and other
    symbols so the TTS voice reads clean, natural sentences."""
    if not text:
        return ""

    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001FAFF"
        "\U00002600-\U000027BF"
        "\U0001F1E6-\U0001F1FF"
        "\U00002190-\U000021FF"
        "\U00002B00-\U00002BFF"
        "\U0000FE00-\U0000FE0F"
        "\U00002500-\U000025FF"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(" ", text)
    text = "".join(ch for ch in text if ch not in _STRIP_CHARS)
    text = re.sub(r"[-–—]{2,}", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _transcribe_audio(audio_bytes: bytes) -> str:
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "voice.wav"
    result = _groq_client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-large-v3-turbo",
        language="en",
        temperature=0.0,
    )
    return result.text.strip()


async def _tts(text: str) -> bytes:
    communicate = edge_tts.Communicate(text, "en-US-JennyNeural")
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    buf.seek(0)
    return buf.read()


def _text_to_speech(text: str):
    clean = _clean_for_speech(text)
    if not clean:
        return None
    try:
        return asyncio.run(_tts(clean))
    except Exception:
        return None


def _init_state():
    defaults = {
        "cb_messages": [],
        "cb_conversation_ended": False,
        "cb_topic_started": False,
        "cb_final_review": "",
        "cb_last_bot_audio": None,
        "cb_mic_key": 0,
        "cb_pending_user_text": None,
        "cb_scenario_data": {},
        "cb_conversation_type": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _get_state() -> ChatState:
    return {
        "messages": st.session_state.cb_messages,
        "exit_score": 0,
        "conversation_ended": False,
        "final_review": "",
        "topic_started": st.session_state.cb_topic_started,
        "scenario_data": st.session_state.cb_scenario_data,
        "conversation_type": st.session_state.cb_conversation_type,
    }


def _reset_conversation():
    st.session_state.cb_messages = []
    st.session_state.cb_conversation_ended = False
    st.session_state.cb_topic_started = False
    st.session_state.cb_final_review = ""
    st.session_state.cb_last_bot_audio = None
    st.session_state.cb_mic_key = 0
    st.session_state.cb_pending_user_text = None
    st.session_state.cb_scenario_data = {}
    st.session_state.cb_conversation_type = ""


def render():
    _init_state()

    st.title("🎭 VibeTalk Arena")
    st.caption("Simple English practice • Debate • Discussion • Roleplay")

    with st.sidebar:
        st.header("Arena Settings")
        voice_mode = st.toggle("🎤 Voice Input", value=True, key="cb_voice_mode")
        auto_speak = st.toggle("🔊 Speak bot replies", value=True, key="cb_auto_speak")
        st.markdown("---")
        st.info(
            "**Voice tips**\n\n"
            "1. Click the microphone\n"
            "2. Speak clearly\n"
            "3. Click **Stop** → message is sent automatically\n"
            "4. Wait for the bot"
        )
        st.markdown("---")
        with st.expander("🔍 Debug", expanded=False):
            st.write("topic_started:", st.session_state.cb_topic_started)
            st.write("conversation_ended:", st.session_state.cb_conversation_ended)
            st.write("conversation_type:", st.session_state.cb_conversation_type)
            st.write("messages count:", len(st.session_state.cb_messages))
            st.write("pending_user_text:", st.session_state.cb_pending_user_text)

    # 1. Process pending user message
    if st.session_state.cb_pending_user_text and not st.session_state.cb_conversation_ended:
        user_text = st.session_state.cb_pending_user_text
        st.session_state.cb_pending_user_text = None

        st.session_state.cb_messages.append(HumanMessage(content=user_text))

        with st.spinner("Thinking…"):
            result = turn_app.invoke(_get_state(), config={"run_name": "Arena-Turn"})

        st.session_state.cb_messages = result["messages"]
        st.session_state.cb_conversation_ended = result.get("conversation_ended", False)
        st.session_state.cb_final_review = result.get("final_review", "")

        if auto_speak and result["messages"]:
            st.session_state.cb_last_bot_audio = _text_to_speech(
                result["messages"][-1].content
            )
        st.rerun()

    # 2. Start a new topic only once
    if not st.session_state.cb_topic_started and not st.session_state.cb_conversation_ended:
        with st.spinner("Starting a fun topic..."):
            result = start_app.invoke(_get_state(), config={"run_name": "Arena-Start"})
            st.session_state.cb_messages = result["messages"]
            st.session_state.cb_topic_started = True
            st.session_state.cb_scenario_data = result.get("scenario_data", {})
            st.session_state.cb_conversation_type = result.get("conversation_type", "")
            if auto_speak and result["messages"]:
                st.session_state.cb_last_bot_audio = _text_to_speech(
                    result["messages"][-1].content
                )
            st.rerun()

    # 3. Show conversation
    for msg in st.session_state.cb_messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    if st.session_state.cb_last_bot_audio and auto_speak:
        st.audio(st.session_state.cb_last_bot_audio, format="audio/mp3", autoplay=True)
        st.session_state.cb_last_bot_audio = None

    # 4. Input area
    if st.session_state.cb_conversation_ended:
        st.success("Chat finished! Check the review above.")
        if st.button("🔄 New conversation", type="primary", key="cb_new_conv"):
            _reset_conversation()
            st.rerun()
    else:
        if voice_mode:
            st.markdown("**🎤 Voice Message** (stop recording → auto send)")
            audio = st.audio_input(
                "Click mic → Speak → Stop",
                key=f"cb_mic_{st.session_state.cb_mic_key}",
            )
            if audio is not None:
                with st.spinner("Listening..."):
                    try:
                        text = _transcribe_audio(audio.getvalue())
                        if text:
                            st.session_state.cb_pending_user_text = text
                            st.session_state.cb_mic_key += 1
                            st.rerun()
                    except Exception as e:
                        st.error(f"Could not understand. Try again. ({e})")

        typed = st.chat_input("Or type your reply here…")
        if typed:
            st.session_state.cb_pending_user_text = typed
            st.rerun()