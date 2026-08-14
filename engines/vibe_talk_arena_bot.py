"""
vibe_talk_arena_bot.py
VibeTalk Arena — free English practice engine (roleplay / debate / discussion).
Fast + beginner-friendly. Human-like chat + soft ending after long conversations.
"""

from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
import random
import json
import os
import re

load_dotenv()

llm = ChatNVIDIA(
    model="meta/llama-3.1-8b-instruct",
    temperature=0.75,
    api_key=os.getenv("NVIDIA_API_KEY"),
)

# A separate, near-zero-temperature instance for classification (exit-intent
# scoring). Reusing the creative `llm` (temperature=0.75) for a "reply with
# only a digit" task made the score noisy enough to randomly misfire on
# ordinary replies and end conversations early.
classifier_llm = ChatNVIDIA(
    model="meta/llama-3.1-8b-instruct",
    temperature=0.0,
    api_key=os.getenv("NVIDIA_API_KEY"),
)

SOFT_ENDING_TURN = 7
FORCE_ENDING_TURN = 11
EXIT_SCORE_THRESHOLD = 8

# Engines live in src/vibetalk/engines/, JSON data lives in src/vibetalk/data/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    exit_score: int
    conversation_ended: bool
    final_review: str
    topic_started: bool
    scenario_data: dict
    conversation_type: str


def get_last_user_message(state: ChatState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.content
    return ""


def parse_score(text: str) -> int:
    score = 0
    for word in text.split():
        cleaned = word.strip(".,!:\"'")
        if cleaned.isdigit():
            score = min(int(cleaned), 10)
    return score


def count_user_turns(messages) -> int:
    return sum(1 for m in messages if isinstance(m, HumanMessage))


# Lines that are decorative/structural and must NOT get a forced period.
_SKIP_PUNCT_PREFIXES = (
    "━", "📋", "📖", "👤", "📍", "💡", "📢", "💭", "📝",
    "❓", "☕", "✨", "🗣️", "🎭", "🎬", "•", "-",
)
_SENTENCE_END = re.compile(r'[.!?]["\')\]]?\s*$')


def ensure_terminal_punctuation(text: str) -> str:
    """Guarantee each real sentence line ends with sentence punctuation so the
    TTS voice pauses correctly and doesn't run everything together once emojis
    are stripped."""
    lines = []
    for line in text.split("\n"):
        stripped = line.rstrip()
        if stripped and not stripped.startswith(_SKIP_PUNCT_PREFIXES):
            if not _SENTENCE_END.search(stripped):
                stripped += "."
        lines.append(stripped)
    return "\n".join(lines)


EXIT_KEYWORDS = (
    "bye", "goodbye", "see you", "see ya", "quit", "stop",
    "i have to go", "gotta go", "gtg", "that's all", "thats all",
    "end chat", "i'm done", "im done",
)

# Word-boundary matching, not plain substring: a naive `in` check made "see
# you" match inside "see your" (as in "I see your point, but I disagree"),
# "quit" match inside "quite", and "stop" match inside "stopped"/"stopping"
# — all common in normal debate/roleplay replies, causing false "user wants
# to end the chat" hits and premature endings.
_EXIT_KEYWORD_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in EXIT_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def looks_like_exit(text: str) -> bool:
    return bool(_EXIT_KEYWORD_PATTERN.search(text))


def load_scenario(conversation_type: int) -> dict:
    type_map = {1: "roleplay.json", 2: "debate.json", 3: "discussion.json"}
    type_names = {1: "roleplay", 2: "debate", 3: "discussion"}

    filename = type_map.get(conversation_type)
    if not filename:
        raise ValueError(f"Invalid conversation type: {conversation_type}")

    filepath = os.path.join(DATA_DIR, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

    if isinstance(data, list):
        scenarios = data
    elif isinstance(data, dict) and "scenarios" in data:
        scenarios = data["scenarios"]
    else:
        raise ValueError(f"Invalid JSON format in {filename}")

    if not scenarios:
        raise ValueError(f"No scenarios found in {filename}")

    selected = random.choice(scenarios)
    selected["_type"] = type_names[conversation_type]
    selected["_type_id"] = conversation_type
    return selected


def get_conversation_intro(conversation_type: str, scenario: dict) -> str:
    if conversation_type == "roleplay":
        title = scenario.get("title", "")
        character = scenario.get("character", "")
        user_role = scenario.get("user_role", "")
        setting = scenario.get("setting", "")
        situation = scenario.get("situation", "")
        intro = "🎭 **Roleplay**\n━━━━━━━━━━━━━━━━━━━━━\n"
        if title:
            intro += f"📖 **Scenario:** {title}.\n"
        if user_role:
            intro += f"🎬 **Your Role:** {user_role}.\n"
        if character:
            intro += f"👤 **You're talking to:** {character}.\n"
        if setting:
            intro += f"📍 **Setting:** {setting}.\n"
        if situation:
            intro += f"💡 **Situation:** {situation}.\n"
        intro += "\nLet's begin!"
    elif conversation_type == "debate":
        topic = scenario.get("topic", "")
        your_side = scenario.get("your_side", "")
        context = scenario.get("context", "")
        intro = "📢 **Debate**\n━━━━━━━━━━━━━━━━━━━━━\n"
        intro += f"**Topic:** {topic}.\n"
        if your_side:
            intro += f"**Your side:** {your_side}.\n"
            intro += f"**AI:** The AI will take the opposite side.\n"
        if context:
            intro += f"**Context:** {context}.\n"
        intro += "\nLet's debate!"
    else:
        topic = scenario.get("topic", "")
        question = scenario.get("question", "")
        context = scenario.get("context", "")
        intro = "💭 **Discussion**\n━━━━━━━━━━━━━━━━━━━━━\n"
        intro += f"**Topic:** {topic}.\n"
        if question:
            intro += f"**Question:** {question}.\n"
        if context:
            intro += f"**Context:** {context}.\n"
        intro += "\nLet's chat!"
    return intro


def get_conversation_type_prompt(scenario: dict) -> str:
    conversation_type = scenario.get("_type", "conversation")
    if conversation_type == "roleplay":
        return get_roleplay_prompt(scenario)
    elif conversation_type == "debate":
        return get_debate_prompt(scenario)
    else:
        return get_discussion_prompt(scenario)


def get_roleplay_prompt(scenario: dict) -> str:
    user_role = scenario.get("user_role", "") or "the other person in the scene"
    return f"""
You are starting a roleplay with an English learner (A1-A2 level).

SCENARIO INSPIRATION (do NOT copy word-for-word):
Title: {scenario.get('title','')}
Description: {scenario.get('description','')}
YOUR character: {scenario.get('character','')}
THE USER's character: {user_role}
Setting: {scenario.get('setting','')}
Situation: {scenario.get('situation','')}
Goal: {scenario.get('goal','')}
Tone: {scenario.get('tone','friendly')}

STRICT RULES:
- You play ONLY your own character ({scenario.get('character','')}). NEVER speak or act as the user.
- The user is "{user_role}". Treat them as a separate person and address them in that role.
- Use only simple A1-A2 English (short sentences, basic words).
- NEVER introduce yourself with "I am a ..." or copy the JSON text.
- Start the scene naturally, as if it is already happening.
- Stay 100% in character.
- Keep the opening to 1-3 short sentences.
- Be creative and make every opening feel fresh.
- Never start romantic, flirting, dating, or marriage topics.
- End EVERY sentence with a full stop, question mark, or exclamation mark. Never end a sentence with only an emoji or with no punctuation.
"""


def get_debate_prompt(scenario: dict) -> str:
    return f"""
You are starting a friendly debate with an English learner (A1-A2 level).

TOPIC INSPIRATION (do NOT copy word-for-word):
Topic: {scenario.get('topic','')}
Description: {scenario.get('description','')}
Context: {scenario.get('context','')}

THE USER WILL ARGUE THIS: "{scenario.get('your_side','')}"
YOUR JOB: Take the OPPOSITE side. Disagree with the user in a friendly way and defend your own view.

STRICT RULES:
- Use only simple A1-A2 English.
- Present your opinion naturally, like chatting with a friend.
- Never say "I believe that..." or "My position is...".
- Keep the opening to 1-3 short sentences.
- Stay friendly and curious, but clearly disagree with the user's side.
- Never start romantic, flirting, dating, or marriage topics.
- End EVERY sentence with a full stop, question mark, or exclamation mark. Never end a sentence with only an emoji or with no punctuation.
"""


def get_discussion_prompt(scenario: dict) -> str:
    return f"""
You are starting a casual chat with an English learner (A1-A2 level).

TOPIC INSPIRATION (do NOT copy word-for-word):
Topic: {scenario.get('topic','')}
Description: {scenario.get('description','')}
Context: {scenario.get('context','')}
Question: {scenario.get('question','')}

STRICT RULES:
- Use only simple A1-A2 English.
- Start like a real friend, not a teacher.
- Never say "Let's discuss..." or "Today we will talk about...".
- Share a short thought first, then invite the other person.
- Keep the opening to 1-3 short sentences.
- Never start romantic, flirting, dating, or marriage topics.
- End EVERY sentence with a full stop, question mark, or exclamation mark. Never end a sentence with only an emoji or with no punctuation.
"""


def start_topic_node(state: ChatState) -> dict:
    conversation_type = random.randint(1, 3)
    scenario = load_scenario(conversation_type)
    conv_type_name = scenario.get("_type", "conversation")

    intro_message = get_conversation_intro(conv_type_name, scenario)
    system_content = get_conversation_type_prompt(scenario)

    response = llm.invoke([SystemMessage(content=system_content)])
    clean_reply = ensure_terminal_punctuation(response.content)

    first_message = f"{intro_message}\n\n━━━━━━━━━━━━━━━━━━━━━\n\n{clean_reply}"
    ai_message = AIMessage(content=first_message)

    return {
        "messages": [ai_message],
        "topic_started": True,
        "conversation_ended": False,
        "final_review": "",
        "exit_score": 0,
        "scenario_data": scenario,
        "conversation_type": conv_type_name,
    }


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


def _classify_exit_score(user_text: str) -> int:
    response = classifier_llm.invoke(
        [SystemMessage(content=EXIT_SYSTEM_PROMPT), HumanMessage(content=user_text)]
    )
    return parse_score(response.content)


def exit_check_node(state: ChatState) -> dict:
    user_text = get_last_user_message(state)
    if not user_text:
        return {"exit_score": 0}
    if looks_like_exit(user_text):
        return {"exit_score": 10}

    # Even at temperature=0 the hosted model occasionally misfires a high
    # score on an ordinary reply (hosted-inference nondeterminism). Only
    # trust a high score once a second, independent call confirms it, so a
    # single noisy reading can't end the conversation early.
    first_score = _classify_exit_score(user_text)
    if first_score < EXIT_SCORE_THRESHOLD:
        return {"exit_score": first_score}
    second_score = _classify_exit_score(user_text)
    return {"exit_score": min(first_score, second_score)}


FAREWELL_AND_REVIEW_PROMPT = """
The conversation has ended. Write a short, friendly English review of the
USER's English (the "user" role messages) — never review or correct your
own ("assistant" role) messages, since you are the coach, not the student.

1. Detect the style (casual chat, roleplay, debate, etc.).
2. Only correct things the USER wrote that actually hurt understanding
   (max 3). Never produce a correction block for something you (the
   assistant) said.
   Use EXACTLY this format, one block per correction, nothing extra on these lines:
     SAID: <exactly what the user wrote>
     BETTER: <the improved version>
     WHY: <one short reason>
3. List 3-5 real strengths the user showed (be specific).
4. Give exactly 3 practical tips for the next chat.
5. Sound like a supportive coach, never an examiner.
6. Never invent mistakes. Never lecture about slang or short answers if the style was casual/roleplay.
7. End with one encouraging sentence.

Keep the whole review warm, short, and helpful.
"""


def format_review(raw: str) -> str:
    """Attach the ❌/✅ markers in code so the model can never swap them.
    The model only emits plain SAID / BETTER / WHY lines."""
    out = []
    for line in raw.split("\n"):
        s = line.strip()
        upper = s.upper()
        if upper.startswith("SAID:"):
            out.append("❌ You said:" + s[len("SAID:"):])
        elif upper.startswith("BETTER:"):
            out.append("✅ Better:" + s[len("BETTER:"):])
        elif upper.startswith("WHY:"):
            out.append("   (" + s[len("WHY:"):].strip() + ")")
        else:
            out.append(line)
    return "\n".join(out)


_INTRO_SEPARATOR = "\n\n━━━━━━━━━━━━━━━━━━━━━\n\n"


def _build_review_transcript(messages: list) -> list:
    """Strip the decorative scenario-header block (topic/character/setting
    labels) from the very first bot message before handing the history to
    the review LLM. Left in, a small model can mistake that formatted intro
    text for something the user said, producing bogus corrections."""
    cleaned = []
    first_ai_seen = False
    for m in messages:
        if isinstance(m, AIMessage) and not first_ai_seen:
            first_ai_seen = True
            content = m.content
            if _INTRO_SEPARATOR in content:
                content = content.split(_INTRO_SEPARATOR, 1)[1]
            cleaned.append(AIMessage(content=content))
        else:
            cleaned.append(m)
    return cleaned


_REVIEW_NOW_INSTRUCTION = (
    "The conversation above has ended. Write the structured review now, "
    "covering strengths and tips exactly as instructed above. Do not just "
    "reply with a casual goodbye or farewell message — write the full review."
)

FALLBACK_REVIEW = (
    "Great job finishing the conversation!\n\n"
    "**Strengths:**\n"
    "- You stayed engaged and kept the conversation going.\n"
    "- You expressed your opinions clearly.\n"
    "- You responded thoughtfully to the topic.\n\n"
    "**Tips for next time:**\n"
    "1. Try adding a few more details to your answers.\n"
    "2. Ask follow-up questions to keep the conversation flowing.\n"
    "3. Don't be afraid to disagree and explain why.\n\n"
    "Keep practicing, you're doing great!"
)


def _looks_like_real_review(text: str) -> bool:
    """A real review is substantial and mentions strengths/tips. Guards
    against the model just echoing a casual goodbye instead of writing the
    review — which happens when the user's last message was itself a
    farewell, since a small model tends to keep that casual tone going."""
    lower = text.lower()
    return len(text.split()) >= 40 and ("strength" in lower or "tip" in lower)


def farewell_and_review_node(state: ChatState) -> dict:
    transcript = _build_review_transcript(state["messages"])
    review_messages = (
        [SystemMessage(content=FAREWELL_AND_REVIEW_PROMPT)]
        + transcript
        + [HumanMessage(content=_REVIEW_NOW_INSTRUCTION)]
    )

    response = llm.invoke(review_messages)
    reviewed = format_review(ensure_terminal_punctuation(response.content))

    if not _looks_like_real_review(reviewed):
        retry_response = llm.invoke(
            review_messages
            + [AIMessage(content=response.content)]
            + [HumanMessage(content="That was not a review. Write the full "
                                     "structured review with Strengths and "
                                     "Tips sections now.")]
        )
        reviewed = format_review(ensure_terminal_punctuation(retry_response.content))

    if not _looks_like_real_review(reviewed):
        reviewed = FALLBACK_REVIEW

    return {
        "messages": [AIMessage(content=reviewed)],
        "conversation_ended": True,
        "final_review": reviewed,
    }


CHAT_SYSTEM_PROMPT = """
You are a real person talking to a friend who is practising English (A1-A2 level).

Core rules:
- Use only simple English: short sentences, basic vocabulary.
- Stay completely inside the current topic or roleplay.
- React first (emotion, agreement, short story, joke) before asking anything.
- Roughly 60% share / react, 40% ask.
- Keep every reply 1-4 short sentences.
- Use contractions (I'm, don't, that's...).
- Never mention English, learning, grammar, or that this is practice.
- If this is a roleplay, stay in YOUR character only. Never speak or act for the user's character.
- If this is a debate, keep defending your side and disagree in a friendly way.
- Show curiosity and share small personal opinions or stories.
- Do NOT ask a question in every reply.
- End EVERY sentence with a full stop, question mark, or exclamation mark. Never end a sentence with only an emoji or with no punctuation.

STRICT SAFETY:
- Never engage with romantic, flirting, dating, marriage, or "I love you / marry me" talk.
- If the user starts that, politely deflect in one short friendly sentence and return to the original topic.
- Never continue or play along with romantic requests.
"""

SOFT_ENDING_PROMPT = """
The conversation has been going for a while.
Gently start wrapping up without forcing an abrupt stop.
Still stay in character / topic. Do not introduce brand-new topics.
"""


def chat_node(state: ChatState) -> dict:
    user_turns = count_user_turns(state["messages"])
    scenario = state.get("scenario_data", {})
    conv_type = state.get("conversation_type", "")
    scenario_context = ""

    if scenario:
        if conv_type == "roleplay":
            parts = []
            if scenario.get("character"):
                parts.append(f"You are {scenario['character']}")
            if scenario.get("user_role"):
                parts.append(
                    f"The user is {scenario['user_role']}. Address them as that person; never speak for them."
                )
            if scenario.get("setting"):
                parts.append(f"Setting: {scenario['setting']}")
            if scenario.get("situation"):
                parts.append(f"Situation: {scenario['situation']}")
            if scenario.get("goal"):
                parts.append(f"Your goal: {scenario['goal']}")
            if parts:
                scenario_context = "\n\nRoleplay context:\n- " + "\n- ".join(parts)
        elif conv_type == "debate":
            parts = []
            if scenario.get("topic"):
                parts.append(f"Topic: {scenario['topic']}")
            if scenario.get("your_side"):
                parts.append(
                    f"The user argues: {scenario['your_side']}. You take the OPPOSITE side and disagree with them."
                )
            if scenario.get("context"):
                parts.append(f"Context: {scenario['context']}")
            if parts:
                scenario_context = "\n\nDebate context:\n- " + "\n- ".join(parts)
        else:
            parts = []
            if scenario.get("topic"):
                parts.append(f"Topic: {scenario['topic']}")
            if scenario.get("context"):
                parts.append(f"Context: {scenario['context']}")
            if scenario.get("question"):
                parts.append(f"Question: {scenario['question']}")
            if parts:
                scenario_context = "\n\nDiscussion context:\n- " + "\n- ".join(parts)

    system_content = CHAT_SYSTEM_PROMPT + scenario_context
    if user_turns >= SOFT_ENDING_TURN:
        system_content += "\n\n" + SOFT_ENDING_PROMPT

    response = llm.invoke([SystemMessage(content=system_content)] + state["messages"])
    response.content = ensure_terminal_punctuation(response.content)
    return {"messages": [response]}


def route_turn(state: ChatState) -> str:
    user_turns = count_user_turns(state["messages"])
    if state["exit_score"] >= EXIT_SCORE_THRESHOLD:
        return "farewell"
    if user_turns >= FORCE_ENDING_TURN:
        return "farewell"
    return "chat"


def build_start_graph():
    g = StateGraph(ChatState)
    g.add_node("start_topic", start_topic_node)
    g.add_edge(START, "start_topic")
    g.add_edge("start_topic", END)
    return g.compile()


def build_turn_graph():
    g = StateGraph(ChatState)
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