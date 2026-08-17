"""
VibeTalk Arena page.
Wrapped as render() so it can live inside a multi-page app.
Session-state keys are prefixed with `cb_` to avoid clashing with the coach page.
"""

import io
import asyncio
import re
import time
import uuid
import streamlit as st
from langchain_core.messages import HumanMessage
import edge_tts

from engines.vibe_talk_arena_bot import start_app, turn_app, ChatState
from modules.nvidia_asr import transcribe_audio
from modules import chat_history_db

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


_TYPE_ICONS = {"roleplay": "🎭", "debate": "📢", "discussion": "💭"}


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
        "cb_viewing_history_id": None,
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


def _history_label(row: dict) -> str:
    icon = _TYPE_ICONS.get(row.get("conversation_type", ""), "💬")
    title = row.get("title") or "Conversation"
    if len(title) > 32:
        title = title[:31].rstrip() + "…"
    suffix = "" if row.get("ended") else " (unfinished)"
    return f"{icon} {title}{suffix}"


def _current_conversation_label() -> str:
    icon = _TYPE_ICONS.get(st.session_state.cb_conversation_type, "💬")
    scenario = st.session_state.cb_scenario_data or {}
    title = scenario.get("title") or scenario.get("topic") or "Conversation"
    if len(title) > 28:
        title = title[:27].rstrip() + "…"
    return f"🟢 {icon} {title} (current)"


def _archive_current_conversation():
    """Save the current conversation to MySQL (capped at the last 30 rows,
    oldest pruned automatically) before it's cleared, whether it finished
    naturally or is being abandoned mid-way by starting a new one."""
    if not st.session_state.cb_topic_started or not st.session_state.cb_messages:
        return
    scenario = st.session_state.cb_scenario_data or {}
    title = scenario.get("title") or scenario.get("topic") or "Conversation"
    serialized_messages = [
        {
            "role": "user" if isinstance(m, HumanMessage) else "assistant",
            "content": m.content,
        }
        for m in st.session_state.cb_messages
    ]
    try:
        chat_history_db.save_chat(st.session_state.auth_user, {
            "id": str(uuid.uuid4()),
            "conversation_type": st.session_state.cb_conversation_type,
            "title": title,
            "messages": serialized_messages,
            "final_review": st.session_state.cb_final_review,
            "ended": st.session_state.cb_conversation_ended,
        })
    except Exception:
        # A DB hiccup shouldn't block starting a new conversation.
        st.toast("⚠️ Couldn't save this chat to history (MySQL unreachable).", icon="⚠️")


def reset_conversation():
    _archive_current_conversation()
    st.session_state.cb_messages = []
    st.session_state.cb_conversation_ended = False
    st.session_state.cb_topic_started = False
    st.session_state.cb_final_review = ""
    st.session_state.cb_last_bot_audio = None
    st.session_state.cb_mic_key = 0
    st.session_state.cb_pending_user_text = None
    st.session_state.cb_scenario_data = {}
    st.session_state.cb_conversation_type = ""
    st.session_state.cb_viewing_history_id = None


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
        st.subheader("📜 History")

        has_current = st.session_state.cb_topic_started and st.session_state.cb_messages
        if has_current:
            is_viewing_current = st.session_state.cb_viewing_history_id is None
            if st.button(
                _current_conversation_label(),
                key="cb_hist_current",
                type="primary" if is_viewing_current else "secondary",
                use_container_width=True,
            ):
                st.session_state.cb_viewing_history_id = None
                st.rerun()

        try:
            history_rows = chat_history_db.list_chats(st.session_state.auth_user)
            history_error = False
        except Exception:
            history_rows = []
            history_error = True

        if history_error:
            st.caption("⚠️ Chat history unavailable (MySQL unreachable).")
        elif not history_rows and not has_current:
            st.caption("Finished or abandoned chats will show up here, saved in MySQL.")
        else:
            for row in history_rows:
                is_active = st.session_state.cb_viewing_history_id == row["id"]
                if st.button(
                    _history_label(row),
                    key=f"cb_hist_{row['id']}",
                    type="primary" if is_active else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.cb_viewing_history_id = row["id"]
                    st.rerun()
        st.markdown("---")
        with st.expander("🔍 Debug", expanded=False):
            st.write("topic_started:", st.session_state.cb_topic_started)
            st.write("conversation_ended:", st.session_state.cb_conversation_ended)
            st.write("conversation_type:", st.session_state.cb_conversation_type)
            st.write("messages count:", len(st.session_state.cb_messages))
            st.write("pending_user_text:", st.session_state.cb_pending_user_text)

    # 0. Viewing a saved past conversation (read-only) instead of the live chat
    if st.session_state.cb_viewing_history_id:
        try:
            entry = chat_history_db.get_chat(
                st.session_state.cb_viewing_history_id, st.session_state.auth_user
            )
        except Exception:
            st.error("⚠️ Couldn't load that saved conversation (MySQL unreachable).")
            if st.button("← Back to current chat"):
                st.session_state.cb_viewing_history_id = None
                st.rerun()
            return
        if entry is None:
            st.session_state.cb_viewing_history_id = None
            st.rerun()
        st.info("📜 Viewing a saved conversation (from MySQL).")
        if st.button("← Back to current chat"):
            st.session_state.cb_viewing_history_id = None
            st.rerun()

        hist_messages = entry["messages"]
        if entry.get("ended") and hist_messages:
            hist_messages = hist_messages[:-1]
        for msg in hist_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if entry.get("ended"):
            st.markdown("---")
            st.markdown("### 📋 Assessment")
            st.info(entry.get("final_review", ""))
        return

    # 1. Process pending user message
    if st.session_state.cb_pending_user_text and not st.session_state.cb_conversation_ended:
        user_text = st.session_state.cb_pending_user_text
        st.session_state.cb_pending_user_text = None

        st.session_state.cb_messages.append(HumanMessage(content=user_text))

        try:
            with st.spinner("Thinking…"):
                result = turn_app.invoke(_get_state(), config={"run_name": "Arena-Turn"})
        except Exception as e:
            st.error(f"🤖 AI engine error: {e}")
            st.stop()

        st.session_state.cb_messages = result["messages"]
        st.session_state.cb_conversation_ended = result.get("conversation_ended", False)
        st.session_state.cb_final_review = result.get("final_review", "")

        if auto_speak:
            if st.session_state.cb_conversation_ended:
                speak_text = st.session_state.cb_final_review
            else:
                speak_text = result["messages"][-1].content if result["messages"] else ""
            if speak_text:
                st.session_state.cb_last_bot_audio = _text_to_speech(speak_text)
        st.rerun()

    # 2. Start a new topic only once
    if not st.session_state.cb_topic_started and not st.session_state.cb_conversation_ended:
        try:
            with st.spinner("Starting a fun topic..."):
                result = start_app.invoke(_get_state(), config={"run_name": "Arena-Start"})
        except Exception as e:
            st.error(f"🤖 AI engine error: {e}")
            st.stop()
        st.session_state.cb_messages = result["messages"]
        st.session_state.cb_topic_started = True
        st.session_state.cb_scenario_data = result.get("scenario_data", {})
        st.session_state.cb_conversation_type = result.get("conversation_type", "")
        if auto_speak and result["messages"]:
            st.session_state.cb_last_bot_audio = _text_to_speech(
                result["messages"][-1].content
            )
        st.rerun()

    # 3. Show conversation (the final review is rendered separately below,
    # so skip it here to avoid showing it twice).
    chat_messages = st.session_state.cb_messages
    if st.session_state.cb_conversation_ended and chat_messages:
        chat_messages = chat_messages[:-1]

    for msg in chat_messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    if st.session_state.cb_last_bot_audio and auto_speak:
        st.audio(st.session_state.cb_last_bot_audio, format="audio/mp3", autoplay=True)
        st.session_state.cb_last_bot_audio = None

    # 4. Input area
    if st.session_state.cb_conversation_ended:
        st.markdown("---")
        st.markdown("### 📋 Your Assessment")
        st.info(st.session_state.cb_final_review)
        st.success("Chat finished! Check your assessment above.")
        if st.button("🔄 New conversation", type="primary", key="cb_new_conv"):
            reset_conversation()
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
                        text = transcribe_audio(audio.getvalue())
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