# VibeTalk

English speaking practice with two modules (LangGraph + Groq).

- **🎭 VibeTalk Arena** — free chat: roleplay / debate / discussion.
- **🗣️ VibeTalk Express** — Hindi→English translation practice.

## Structure

```
vibetalk/
├── main.py              # run this
├── requirements.txt
├── .env                 # you add: GROQ_API_KEY=your_key
├── engines/             # backend (LangGraph)
│   ├── vibe_talk_arena_bot.py
│   └── vibe_talk_express_bot.py
├── pages/               # frontend (Streamlit)
│   ├── vibe_talk_arena_app.py
│   └── vibe_talk_express_app.py
└── data/                # scenario JSON
    ├── roleplay.json
    ├── debate.json
    ├── discussion.json
    └── translation.json
```

## Run

```bash
pip install -r requirements.txt
# create a .env file containing:  GROQ_API_KEY=your_key_here
streamlit run main.py
```

That's it. The sidebar lets you pick a module.

> Note: `engines/vibe_talk_arena_bot.py` was rebuilt from an earlier version.
> If your real Arena engine changed since, drop yours in and make sure
> `load_scenario` reads JSON from the `data/` folder
> (`os.path.join(os.path.dirname(__file__), "..", "data", filename)`).
