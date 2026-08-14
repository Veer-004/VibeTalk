# VibeTalk

English speaking practice (LangGraph + NVIDIA API for chat, NVIDIA Parakeet for voice input).

- **🎭 VibeTalk Arena** — free chat: roleplay / debate / discussion.

## Structure

```
vibetalk/
├── main.py              # run this
├── requirements.txt
├── .env                 # you add: NVIDIA_API_KEY=your_key
├── engines/             # backend (LangGraph)
│   └── vibe_talk_arena_bot.py
├── pages/               # frontend (Streamlit)
│   └── vibe_talk_arena_app.py
└── data/                # scenario JSON
    ├── roleplay.json
    ├── debate.json
    └── discussion.json
```

## Run

```bash
pip install -r requirements.txt
# create a .env file containing:  NVIDIA_API_KEY=your_key_here
streamlit run main.py
```

That's it. The sidebar lets you pick a module.

## Chat history (MySQL)

Arena saves the last 30 conversations (with their assessments) to MySQL, shown
in the sidebar and viewable read-only after the fact. One-time local setup via
Docker:

```bash
docker volume create vibetalk_mysql_data
docker run -d --name vibetalk-mysql \
  -e MYSQL_ROOT_PASSWORD=<pick_a_password> \
  -e MYSQL_DATABASE=vibetalk \
  -e MYSQL_USER=vibetalk_app \
  -e MYSQL_PASSWORD=<pick_a_password> \
  -p 3306:3306 \
  -v vibetalk_mysql_data:/var/lib/mysql \
  mysql:8.0

# once the container is healthy:
docker exec -i vibetalk-mysql mysql -uvibetalk_app -p<password> vibetalk < db/schema.sql
```

Then add to `.env`:

```
DB_HOST = "127.0.0.1"
DB_PORT = "3306"
DB_NAME = "vibetalk"
DB_USER = "vibetalk_app"
DB_PASSWORD = "<same password as above>"
```

If MySQL isn't running, the app still works — history just shows
"unavailable" instead of crashing the page.

> Note: `engines/vibe_talk_arena_bot.py` was rebuilt from an earlier version.
> If your real Arena engine changed since, drop yours in and make sure
> `load_scenario` reads JSON from the `data/` folder
> (`os.path.join(os.path.dirname(__file__), "..", "data", filename)`).
