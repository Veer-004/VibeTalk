"""
VibeTalk Express page.
Immersive conversation-style UI — mirrors the Arena layout.

Flow:
  - First message: bot's English sentence + Hindi translation (TTS speaks English).
    Below it, the Hindi prompt the user must translate.
  - Every turn after: user speaks/types English → bot replies in character →
    next Hindi prompt appears.
  - No turn numbers, no scenario headers, no Submit button.
  - At the end: warm coach-style review in a chat bubble.

Session-state keys are prefixed with `sc_` to avoid clashing with Arena.
"""

import asyncio
import io
import re
import time

import edge_tts
import streamlit as st
from langchain_core.messages import HumanMessage

from engines.vibe_talk_express_bot import start_app, turn_app, ExpressState
from modules.nvidia_asr import transcribe_audio

# Characters/symbols we never want the voice to read out loud.
_STRIP_CHARS = '_:;\u201c\u201d\u201e\u2018\u2019\u201a`*#|~<>[]{}()'


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
    text = re.sub(r"[-\u2013\u2014]{2,}", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


async def _tts(text: str) -> bytes:
    communicate = edge_tts.Communicate(text, "en-US-JennyNeural")
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    buf.seek(0)
    return buf.read()


def _text_to_speech(text: str, retries: int = 3):
    """edge-tts occasionally drops the connection to Microsoft's speech
    service (transient network errors), so retry a few times before
    giving up rather than silently returning no audio."""
    clean = _clean_for_speech(text)
    if not clean:
        return None
    for attempt in range(retries):
        try:
            return asyncio.run(_tts(clean))
        except Exception:
            if attempt < retries - 1:
                time.sleep(0.5)
    return None


# ------------------------------------------------------------------
# Session helpers
# ------------------------------------------------------------------
def _init_state():
    defaults = {
        "sc_messages": [],
        "sc_conversation_ended": False,
        "sc_topic_started": False,
        "sc_final_review": "",
        "sc_last_bot_audio": None,
        "sc_mic_key": 0,
        "sc_pending_user_text": None,
        "sc_user_turns": 0,
        # Express-specific extras stored alongside the chat
        "sc_current_user_hindi": "",
        "sc_user_hindi_prompts": [],
        "sc_user_answers": [],
        "sc_turn_scores": [],
        "sc_scenario_id": None,
        "sc_scenario_text": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _get_state() -> ExpressState:
    return {
        "messages": st.session_state.sc_messages,
        "exit_score": 0,
        "conversation_ended": False,
        "final_review": "",
        "topic_started": st.session_state.sc_topic_started,
        "current_user_hindi": st.session_state.sc_current_user_hindi,
        "user_hindi_prompts": st.session_state.sc_user_hindi_prompts,
        "user_answers": st.session_state.sc_user_answers,
        "turn_scores": st.session_state.sc_turn_scores,
        "scenario_id": st.session_state.sc_scenario_id or 0,
        "scenario_text": st.session_state.sc_scenario_text,
    }


def _reset_conversation():
    st.session_state.sc_messages = []
    st.session_state.sc_conversation_ended = False
    st.session_state.sc_topic_started = False
    st.session_state.sc_final_review = ""
    st.session_state.sc_last_bot_audio = None
    st.session_state.sc_mic_key = 0
    st.session_state.sc_pending_user_text = None
    st.session_state.sc_user_turns = 0
    st.session_state.sc_current_user_hindi = ""
    st.session_state.sc_user_hindi_prompts = []
    st.session_state.sc_user_answers = []
    st.session_state.sc_turn_scores = []
    st.session_state.sc_scenario_id = None
    st.session_state.sc_scenario_text = ""


def _sync_from_result(result: dict):
    """Pull relevant fields from a graph result into session state."""
    st.session_state.sc_messages = result["messages"]
    st.session_state.sc_conversation_ended = result.get("conversation_ended", False)
    st.session_state.sc_final_review = result.get("final_review", "")
    st.session_state.sc_current_user_hindi = result.get("current_user_hindi", "")
    st.session_state.sc_user_hindi_prompts = result.get("user_hindi_prompts", st.session_state.sc_user_hindi_prompts)
    st.session_state.sc_user_answers = result.get("user_answers", st.session_state.sc_user_answers)
    st.session_state.sc_turn_scores = result.get("turn_scores", st.session_state.sc_turn_scores)
    if result.get("scenario_id"):
        st.session_state.sc_scenario_id = result["scenario_id"]
    if result.get("scenario_text"):
        st.session_state.sc_scenario_text = result["scenario_text"]


# ------------------------------------------------------------------
# Main render
# ------------------------------------------------------------------
def render():
    _init_state()

    st.title("\U0001F5E3\uFE0F VibeTalk Express")
    st.caption("Immersive Hindi → English conversation practice • A1 Beginner")

    # ---- Sidebar ----
    with st.sidebar:
        st.header("Express Settings")
        voice_mode = st.toggle("\U0001F3A4 Voice Input", value=True, key="sc_voice_mode")
        auto_speak = st.toggle("\U0001F50A Speak bot replies", value=True, key="sc_auto_speak")
        st.markdown("---")
        st.info(
            "**How it works**\n\n"
            "1. The bot speaks as a character\n"
            "2. You see a Hindi sentence\n"
            "3. Translate it to English (speak or type)\n"
            "4. The conversation continues naturally"
        )
        st.markdown("---")
        with st.expander("\U0001F50D Debug", expanded=False):
            st.write("topic_started:", st.session_state.sc_topic_started)
            st.write("conversation_ended:", st.session_state.sc_conversation_ended)
            st.write("user_turns:", st.session_state.sc_user_turns)
            st.write("messages count:", len(st.session_state.sc_messages))
            st.write("current_hindi:", st.session_state.sc_current_user_hindi)
            st.write("answers:", len(st.session_state.sc_user_answers))

    # ---- 1. Process pending user message ----
    if st.session_state.sc_pending_user_text and not st.session_state.sc_conversation_ended:
        user_text = st.session_state.sc_pending_user_text
        st.session_state.sc_pending_user_text = None

        st.session_state.sc_messages.append(HumanMessage(content=user_text))
        st.session_state.sc_user_turns += 1

        with st.spinner("Thinking\u2026"):
            result = turn_app.invoke(_get_state(), config={"run_name": "Express-Turn"})

        _sync_from_result(result)

        if auto_speak:
            if st.session_state.sc_conversation_ended:
                st.session_state.sc_last_bot_audio = _text_to_speech(
                    st.session_state.sc_final_review
                )
            elif result["messages"]:
                # Speak only the bot's English reply (last AI message)
                last_ai = result["messages"][-1]
                if isinstance(last_ai, HumanMessage):
                    # Shouldn't happen, but safeguard
                    st.session_state.sc_last_bot_audio = None
                else:
                    st.session_state.sc_last_bot_audio = _text_to_speech(last_ai.content)
        st.rerun()

    # ---- 2. Start a new topic only once ----
    if not st.session_state.sc_topic_started and not st.session_state.sc_conversation_ended:
        with st.spinner("Setting the scene\u2026"):
            result = start_app.invoke(_get_state(), config={"run_name": "Express-Start"})
            _sync_from_result(result)
            st.session_state.sc_topic_started = True

            if auto_speak and result["messages"]:
                # For the first message, speak only the English part
                # The first message format is: "English sentence\n\n Hindi sentence"
                first_content = result["messages"][-1].content
                english_part = first_content.split("\n\n")[0] if "\n\n" in first_content else first_content
                st.session_state.sc_last_bot_audio = _text_to_speech(english_part)
            st.rerun()

    # ---- 3. Show conversation (Arena-style chat bubbles). The final review
    # is rendered separately below, so skip it here to avoid showing it twice.
    chat_messages = st.session_state.sc_messages
    if st.session_state.sc_conversation_ended and chat_messages:
        chat_messages = chat_messages[:-1]

    for msg in chat_messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    # Play queued audio
    if st.session_state.sc_last_bot_audio and auto_speak:
        st.audio(st.session_state.sc_last_bot_audio, format="audio/mp3", autoplay=True)
        st.session_state.sc_last_bot_audio = None

    # ---- 4. Show Hindi prompt (if conversation is ongoing) ----
    if not st.session_state.sc_conversation_ended and st.session_state.sc_current_user_hindi:
        st.markdown("---")
        with st.chat_message("assistant"):
            st.markdown(f"\U0001F5E3\uFE0F **Translate this:**\n\n{st.session_state.sc_current_user_hindi}")

    # ---- 5. Input area ----
    if st.session_state.sc_conversation_ended:
        st.markdown("---")
        st.markdown("### \U0001F4CB Your Assessment")
        st.info(st.session_state.sc_final_review)
        st.success("Conversation finished! Check your assessment above.")
        if st.button("\U0001F504 New conversation", type="primary", key="sc_new_conv"):
            _reset_conversation()
            st.rerun()
    else:
        if voice_mode:
            st.markdown("**\U0001F3A4 Voice Message** (stop recording → auto send)")
            audio = st.audio_input(
                "Click mic → Speak → Stop",
                key=f"sc_mic_{st.session_state.sc_mic_key}",
            )
            if audio is not None:
                with st.spinner("Listening..."):
                    try:
                        text = transcribe_audio(audio.getvalue())
                        if text:
                            st.session_state.sc_pending_user_text = text
                            st.session_state.sc_mic_key += 1
                            st.rerun()
                    except Exception as e:
                        st.error(f"Could not understand. Try again. ({e})")

        typed = st.chat_input("Type your English translation here\u2026")
        if typed:
            st.session_state.sc_pending_user_text = typed
            st.rerun()
