"""
vibe_talk_express_bot.py
VibeTalk Express — immersive Hindi→English conversation practice (LangGraph).

Architecture mirrors vibe_talk_arena_bot.py (two compiled graphs):
  start_app  — picks a scenario, generates the first immersive message.
  turn_app   — exit check → (chat | farewell+review).

Conversation flow:
  1. First message: bot's English sentence + Hindi translation + TTS.
     Also provides the Hindi prompt the user must translate.
  2. Every subsequent turn:
       Bot Hindi prompt → User speaks English → Bot replies in character →
       next Hindi prompt → …
  3. Bot stays 100 % in character (waiter, doctor, officer …).
     Evaluation is silent; feedback appears only in the final review.
"""

from __future__ import annotations

import json
import os
import random
import re
from typing import Annotated, Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
TRANSLATION_PATH = os.path.join(DATA_DIR, "translation.json")

# Turn limits (soft = bot starts wrapping up, force = hard stop)
SOFT_ENDING_TURN = 9
FORCE_ENDING_TURN = 12
EXIT_SCORE_THRESHOLD = 8

# ------------------------------------------------------------------
# LLM Setup
# ------------------------------------------------------------------
NVIDIA_MODEL = "meta/llama-3.1-8b-instruct"

llm = ChatNVIDIA(
    model=NVIDIA_MODEL,
    temperature=0.7,
    api_key=os.getenv("NVIDIA_API_KEY"),
)

# Raw OpenAI-compatible client for the JSON-mode evaluation call below,
# pointed at NVIDIA's NIM endpoint instead of OpenAI's.
nvidia_raw = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
)


# ------------------------------------------------------------------
# State Definition  (Arena-style with add_messages)
# ------------------------------------------------------------------
class ExpressState(TypedDict):
    messages: Annotated[list, add_messages]
    scenario_id: int
    scenario_text: str
    topic_started: bool

    # Express-specific
    current_user_hindi: str       # Hindi prompt the user must translate next
    user_hindi_prompts: list      # All Hindi prompts given (for final review)
    user_answers: list            # All user English answers (for final review)
    turn_scores: list             # Internal evaluation scores per turn

    # Ending
    exit_score: int
    conversation_ended: bool
    final_review: str


# ------------------------------------------------------------------
# JSON Helpers
# ------------------------------------------------------------------
def load_scenario(scenario_id: int) -> str:
    try:
        with open(TRANSLATION_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for s in data["scenarios"]:
            if s["id"] == scenario_id:
                return s["scenario"]
    except Exception:
        return "You are at a cafe ordering coffee."
    return "You are at a cafe ordering coffee."


# ------------------------------------------------------------------
# Exit detection  (mirrors arena_bot)
# ------------------------------------------------------------------
EXIT_KEYWORDS = (
    "bye", "goodbye", "see you", "see ya", "quit", "stop",
    "i have to go", "gotta go", "gtg", "that's all", "thats all",
    "end chat", "i'm done", "im done",
)


def looks_like_exit(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in EXIT_KEYWORDS)


def parse_score(text: str) -> int:
    score = 0
    for word in text.split():
        cleaned = word.strip(".,!:\"'")
        if cleaned.isdigit():
            score = min(int(cleaned), 10)
    return score


def count_user_turns(messages) -> int:
    return sum(1 for m in messages if isinstance(m, HumanMessage))


def get_last_user_message(state: ExpressState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.content
    return ""


# ------------------------------------------------------------------
# Internal evaluation  (silent — never shown to user during chat)
# ------------------------------------------------------------------
def evaluate_answer(hindi: str, user_answer: str, bot_context: str = "") -> Dict[str, Any]:
    if not user_answer.strip():
        return {
            "score": 0, "grammar": 0, "meaning": 0, "fluency": 0, "vocabulary": 0,
            "feedback": "No answer.", "ideal": "",
        }

    try:
        prompt = f"""You are an A1 English teacher.
Bot said: "{bot_context}"
Hindi the student had to translate: "{hindi}"
Student said: "{user_answer}"

Reply ONLY with valid JSON:
{{
  "score": <0-100>,
  "grammar": <0-100>,
  "meaning": <0-100>,
  "fluency": <0-100>,
  "vocabulary": <0-100>,
  "feedback": "<one short sentence>",
  "ideal": "<one natural A1 English sentence that translates the Hindi>"
}}"""
        resp = nvidia_raw.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return {
                "score": int(data.get("score", 50)),
                "grammar": int(data.get("grammar", 50)),
                "meaning": int(data.get("meaning", 50)),
                "fluency": int(data.get("fluency", 50)),
                "vocabulary": int(data.get("vocabulary", 50)),
                "feedback": str(data.get("feedback", "Good effort!")),
                "ideal": str(data.get("ideal", "")),
            }
    except Exception:
        pass

    return {
        "score": 70, "grammar": 70, "meaning": 70, "fluency": 70, "vocabulary": 70,
        "feedback": "Good try!", "ideal": user_answer,
    }


# ------------------------------------------------------------------
# START GRAPH — First message
# ------------------------------------------------------------------
FIRST_TURN_PROMPT = """You are creating the opening of an immersive A1 English conversation.

SCENARIO (inspiration only — do NOT copy word-for-word):
{scenario}

You must fully become a character inside this scenario (waiter, doctor, officer,
shopkeeper, stranger, etc.). The learner is the other person in the scene.

Reply with ONLY a valid JSON object, no markdown, no extra text:
{{
  "bot_english": "<your first in-character English sentence, 1-2 short sentences max>",
  "bot_hindi": "<accurate Hindi translation of bot_english>",
  "user_hindi": "<a short Hindi sentence the user should translate as their reply>"
}}

RULES:
- Start the scene naturally, as if it is already happening.
- NEVER introduce yourself with "I am a …" or copy the scenario text.
- Keep English very simple (A1 beginner level, max 12 words per sentence).
- Be creative. Make every opening feel fresh and alive.
- The user_hindi must be a natural reply to what the bot just said.
- Never start romantic, flirting, dating, or marriage topics.
"""


def start_topic_node(state: ExpressState) -> dict:
    sid = random.randint(1, 100)
    scenario_text = load_scenario(sid)

    prompt = FIRST_TURN_PROMPT.format(scenario=scenario_text)

    bot_english = ""
    bot_hindi = ""
    user_hindi = ""

    try:
        resp = llm.invoke([SystemMessage(content=prompt)])
        raw = resp.content.strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            bot_english = str(data.get("bot_english", "")).strip().replace('\n', ' ')
            bot_hindi = str(data.get("bot_hindi", "")).strip().replace('\n', ' ')
            user_hindi = str(data.get("user_hindi", "")).strip().replace('\n', ' ')
    except Exception:
        pass

    # Fallback
    if not bot_english or not bot_hindi or not user_hindi:
        bot_english = "Welcome! What can I help you with today?"
        bot_hindi = "स्वागत है! आज मैं आपकी क्या मदद कर सकता हूँ?"
        user_hindi = "मुझे एक कॉफी चाहिए।"

    # The first message includes both English and Hindi translation
    first_display = f"{bot_english}\n\n{bot_hindi}"
    ai_message = AIMessage(content=first_display)

    return {
        "messages": [ai_message],
        "scenario_id": sid,
        "scenario_text": scenario_text,
        "topic_started": True,
        "conversation_ended": False,
        "final_review": "",
        "exit_score": 0,
        "current_user_hindi": user_hindi,
        "user_hindi_prompts": [],
        "user_answers": [],
        "turn_scores": [],
    }


# ------------------------------------------------------------------
# TURN GRAPH — Exit check → Chat or Farewell
# ------------------------------------------------------------------
EXIT_SYSTEM_PROMPT = (
    "You are analyzing ONE message from an English-practice chat.\n"
    "Decide if the user is trying to END the conversation (saying goodbye, "
    "asking to stop/quit, or clearly signaling they want to leave).\n"
    "Reply with ONLY a single digit 0-10. No words, no explanation, no punctuation.\n\n"
    "0-3 = the user is continuing the conversation normally (agreeing, disagreeing, "
    "sharing an opinion, asking a question, making small talk, etc.)\n"
    "7-10 = the user is clearly saying goodbye / bye / stop / quit / see you / "
    "I have to go / let's end this\n\n"
    "If the message does NOT explicitly signal leaving, you MUST answer 0.\n\n"
    "Examples:\n"
    "Message: \"But robots are consistent every single time.\" -> 0\n"
    "Message: \"I have to go now, bye!\" -> 9\n"
    "Message: \"That's a great point, I agree.\" -> 0\n"
    "Message: \"Can we stop here for today?\" -> 8\n"
)


def exit_check_node(state: ExpressState) -> dict:
    user_text = get_last_user_message(state)
    if not user_text:
        return {"exit_score": 0}
    if looks_like_exit(user_text):
        return {"exit_score": 10}
    response = llm.invoke(
        [SystemMessage(content=EXIT_SYSTEM_PROMPT), HumanMessage(content=user_text)]
    )
    return {"exit_score": parse_score(response.content)}


# ------------------------------------------------------------------
# Chat node — in-character reply + internal eval + next Hindi prompt
# ------------------------------------------------------------------
CHAT_SYSTEM_PROMPT = """You are a real person inside this scenario, talking naturally.

SCENARIO: {scenario}

You are fully in character. The learner is the other person in the scene.
You must reply to what they just said, continuing the conversation naturally.

ALSO generate the next Hindi sentence the user should translate as their reply.

Reply with ONLY a valid JSON object, no markdown, no extra text:
{{
  "bot_english": "<your in-character English reply, 1-2 short sentences max>",
  "user_hindi": "<a short Hindi sentence the user should translate as their reply>"
}}

RULES:
- React to what the user said. Every reply must logically follow the previous message.
- Stay 100% in character. Never mention English, learning, grammar, or practice.
- Use only simple A1-A2 English (short sentences, basic words).
- Use contractions (I'm, don't, that's…).
- Keep replies to 1-2 short sentences maximum.
- The user_hindi must be a natural continuation of the conversation.
- Never start romantic, flirting, dating, or marriage topics.
- If the user says something romantic, politely deflect and return to the scenario.

Previous conversation (for continuity):
{history}
"""

SOFT_ENDING_ADDITION = """
The conversation has been going for a while.
Gently start wrapping up without forcing an abrupt stop.
Still stay in character. Do not introduce brand-new topics.
"""


def chat_node(state: ExpressState) -> dict:
    user_text = get_last_user_message(state)
    user_turns = count_user_turns(state["messages"])

    # --- Internal evaluation (silent) ---
    current_hindi = state.get("current_user_hindi", "")
    new_scores = list(state.get("turn_scores", []))
    new_hindi_prompts = list(state.get("user_hindi_prompts", []))
    new_user_answers = list(state.get("user_answers", []))

    if user_text and current_hindi:
        # Find the last bot message for context
        bot_context = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                bot_context = msg.content
                break

        eval_result = evaluate_answer(current_hindi, user_text, bot_context)
        new_scores.append({
            "grammar": float(eval_result.get("grammar", 50)),
            "meaning": float(eval_result.get("meaning", 50)),
            "fluency": float(eval_result.get("fluency", 50)),
            "vocabulary": float(eval_result.get("vocabulary", 50)),
            "overall": float(eval_result.get("score", 50)),
            "hindi_prompt": current_hindi,
            "ideal": eval_result.get("ideal", ""),
        })
        new_hindi_prompts.append(current_hindi)
        new_user_answers.append(user_text)

    # --- Build history for the LLM ---
    hist_lines = []
    for m in state["messages"][-10:]:
        if isinstance(m, AIMessage):
            hist_lines.append(f"Bot: {m.content}")
        elif isinstance(m, HumanMessage):
            hist_lines.append(f"User: {m.content}")
    history_text = "\n".join(hist_lines) if hist_lines else "(start of conversation)"

    prompt = CHAT_SYSTEM_PROMPT.format(
        scenario=state.get("scenario_text", ""),
        history=history_text,
    )
    if user_turns >= SOFT_ENDING_TURN:
        prompt += SOFT_ENDING_ADDITION

    # --- Generate reply ---
    bot_english = ""
    user_hindi = ""

    try:
        resp = llm.invoke([SystemMessage(content=prompt)])
        raw = resp.content.strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            bot_english = str(data.get("bot_english", "")).strip().replace('\n', ' ')
            user_hindi = str(data.get("user_hindi", "")).strip().replace('\n', ' ')
    except Exception:
        pass

    # Fallback
    if not bot_english:
        bot_english = "I see. Tell me more."
    if not user_hindi:
        user_hindi = "मुझे और बताइए।"

    ai_message = AIMessage(content=bot_english)

    return {
        "messages": [ai_message],
        "current_user_hindi": user_hindi,
        "turn_scores": new_scores,
        "user_hindi_prompts": new_hindi_prompts,
        "user_answers": new_user_answers,
    }


# ------------------------------------------------------------------
# Farewell + Review node
# ------------------------------------------------------------------
FAREWELL_REVIEW_PROMPT = """The English translation practice conversation has ended.
Write a detailed, warm, friendly review for a beginner (A1) learner who was
translating Hindi sentences into English during an immersive roleplay conversation.

You will be given each turn as:
  Hindi prompt: "<the Hindi the learner had to translate>"
  Learner said: "<their English answer>"

Write the review in this exact structure:

1. A one-line warm opening acknowledging the scenario they practiced in.
2. Grammar / word corrections (MAXIMUM 3, only real mistakes that matter).
   Use exactly this format for each:
     ❌ You said: "…"
     ✅ Better: "…"
     (one short reason — e.g. a better word, a grammar fix)
3. **Strengths** – list 3 real, specific things the learner did well.
4. **Tips for next time** – exactly 3 short practical tips.
5. **Vocabulary** – 3-5 useful words or phrases from the conversation with Hindi meanings.
6. **Overall Level** – estimate their CEFR level (A1, A2, B1, etc.) in one sentence.
7. One encouraging closing sentence.

RULES:
- Sound like a supportive coach, never an examiner.
- Never invent mistakes. If an answer was fine, do not correct it.
- Keep it warm, short, and easy to read.
- Do NOT show any numerical scores.
- When a sentence ends with an emoji, finish the sentence with a full stop FIRST,
  then put the emoji after it. Example: "Great job. 🎉" (never "Great job 🎉").
"""


def farewell_and_review_node(state: ExpressState) -> dict:
    """Generate detailed coach-style review at conversation end."""
    user_text = get_last_user_message(state)

    # Do one final eval if the last user message hasn't been evaluated yet
    current_hindi = state.get("current_user_hindi", "")
    new_scores = list(state.get("turn_scores", []))
    new_hindi_prompts = list(state.get("user_hindi_prompts", []))
    new_user_answers = list(state.get("user_answers", []))

    if user_text and current_hindi and len(new_user_answers) < count_user_turns(state["messages"]):
        bot_context = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                bot_context = msg.content
                break
        eval_result = evaluate_answer(current_hindi, user_text, bot_context)
        new_scores.append({
            "grammar": float(eval_result.get("grammar", 50)),
            "meaning": float(eval_result.get("meaning", 50)),
            "fluency": float(eval_result.get("fluency", 50)),
            "vocabulary": float(eval_result.get("vocabulary", 50)),
            "overall": float(eval_result.get("score", 50)),
            "hindi_prompt": current_hindi,
            "ideal": eval_result.get("ideal", ""),
        })
        new_hindi_prompts.append(current_hindi)
        new_user_answers.append(user_text)

    # Build transcript for the reviewer
    review_text = ""
    if new_user_answers:
        lines = []
        for i, ans in enumerate(new_user_answers):
            hindi = new_hindi_prompts[i] if i < len(new_hindi_prompts) else ""
            lines.append(f'Hindi prompt: "{hindi}"')
            lines.append(f'Learner said: "{ans}"')
        transcript = "\n".join(lines)

        try:
            resp = llm.invoke([
                SystemMessage(content=FAREWELL_REVIEW_PROMPT),
                HumanMessage(content=transcript),
            ])
            review_text = resp.content.strip()
        except Exception:
            pass

    if not review_text:
        review_text = (
            "Nice work finishing the conversation. 🎉\n\n"
            "**Strengths**\n"
            "- You completed every turn.\n"
            "- You kept your sentences simple and clear.\n"
            "- You stayed with the story to the end.\n\n"
            "**Tips for next time**\n"
            "- Read your answer out loud before sending.\n"
            "- Try one new word each session.\n"
            "- Keep practicing a little every day.\n\n"
            "You're improving. Keep it up. 💪"
        )

    review_message = AIMessage(content=review_text)

    return {
        "messages": [review_message],
        "conversation_ended": True,
        "final_review": review_text,
        "turn_scores": new_scores,
        "user_hindi_prompts": new_hindi_prompts,
        "user_answers": new_user_answers,
    }


# ------------------------------------------------------------------
# Routing
# ------------------------------------------------------------------
def route_turn(state: ExpressState) -> str:
    user_turns = count_user_turns(state["messages"])
    if state["exit_score"] >= EXIT_SCORE_THRESHOLD:
        return "farewell"
    if user_turns >= FORCE_ENDING_TURN:
        return "farewell"
    return "chat"


# ------------------------------------------------------------------
# Build the two graphs  (same pattern as arena_bot)
# ------------------------------------------------------------------
def build_start_graph():
    g = StateGraph(ExpressState)
    g.add_node("start_topic", start_topic_node)
    g.add_edge(START, "start_topic")
    g.add_edge("start_topic", END)
    return g.compile()


def build_turn_graph():
    g = StateGraph(ExpressState)
    g.add_node("exit_check", exit_check_node)
    g.add_node("chat", chat_node)
    g.add_node("farewell", farewell_and_review_node)
    g.add_edge(START, "exit_check")
    g.add_conditional_edges(
        "exit_check", route_turn, {"farewell": "farewell", "chat": "chat"}
    )
    g.add_edge("farewell", END)
    g.add_edge("chat", END)
    return g.compile()


start_app = build_start_graph()
turn_app = build_turn_graph()
