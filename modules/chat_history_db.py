"""
chat_history_db.py — MySQL-backed storage for VibeTalk Arena chat history.

Every row belongs to exactly one logged-in user (the `username` column).
All reads and writes are scoped to a username, so each of the 5 accounts
only ever sees its own history — never another user's. Keeps only the
most recent MAX_HISTORY conversations *per user* (older ones for that
same user are pruned automatically on every save; it never touches other
users' rows). Messages are stored as JSON
([{"role": "user"|"assistant", "content": "..."}]) rather than LangChain
message objects, since those aren't natively SQL-serializable.
"""

import json
import logging
import os
import threading
from functools import wraps

import pymysql
import streamlit as st

MAX_HISTORY = 30

_log = logging.getLogger(__name__)

# ── Robust connection management ──────────────────────────────────
# A single cached pymysql connection + ping(reconnect=True) silently
# breaks after MySQL's wait_timeout (default 8 h).  Instead we keep
# the connection in a module-level variable behind a lock and rebuild
# it whenever we detect a "server has gone away"-class error.

_conn: pymysql.connections.Connection | None = None
_lock = threading.Lock()

# OperationalError codes that mean "connection is dead, make a new one"
_RETRIABLE_CODES = {2003, 2006, 2013}


def _new_connection() -> pymysql.connections.Connection:
    """Create a fresh pymysql connection from env-vars."""
    return pymysql.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        autocommit=True,
        charset="utf8mb4",
        connect_timeout=10,
        read_timeout=30,
        write_timeout=30,
    )


def _connection() -> pymysql.connections.Connection:
    """Return a live connection, creating or replacing a dead one."""
    global _conn
    with _lock:
        if _conn is None:
            _conn = _new_connection()
            return _conn
        try:
            _conn.ping(reconnect=False)   # test only, don't let it half-reconnect
            return _conn
        except Exception:
            # ping failed → connection is dead, build a fresh one
            try:
                _conn.close()
            except Exception:
                pass
            _conn = _new_connection()
            return _conn


def _with_retry(func):
    """Decorator: if a DB function fails with a retriable OperationalError,
    drop the cached connection and retry once with a fresh one."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        global _conn
        try:
            return func(*args, **kwargs)
        except pymysql.OperationalError as exc:
            code = exc.args[0] if exc.args else None
            if code not in _RETRIABLE_CODES:
                raise
            _log.warning("MySQL gone (code %s), reconnecting…", code)
            with _lock:
                try:
                    if _conn is not None:
                        _conn.close()
                except Exception:
                    pass
                _conn = None            # force _connection() to rebuild
            return func(*args, **kwargs)  # one retry with a fresh connection
    return wrapper


@_with_retry
def save_chat(username: str, entry: dict) -> None:
    """Insert a chat for `username` and prune that same user down to their
    MAX_HISTORY most recent rows. Never touches other users' rows."""
    conn = _connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chat_history
                (id, username, conversation_type, title, messages_json, final_review, ended)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                entry["id"],
                username,
                entry.get("conversation_type", ""),
                entry.get("title", "Conversation"),
                json.dumps(entry["messages"]),
                entry.get("final_review", ""),
                1 if entry.get("ended") else 0,
            ),
        )
        cur.execute(
            """
            DELETE FROM chat_history
            WHERE username = %s AND id NOT IN (
                SELECT id FROM (
                    SELECT id FROM chat_history
                    WHERE username = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                ) AS keep_ids
            )
            """,
            (username, username, MAX_HISTORY),
        )


@_with_retry
def list_chats(username: str) -> list:
    """This user's most recent MAX_HISTORY chats, newest first (summary
    fields only) — never includes another user's chats."""
    conn = _connection()
    with conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute(
            """
            SELECT id, conversation_type, title, ended, created_at
            FROM chat_history
            WHERE username = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (username, MAX_HISTORY),
        )
        return cur.fetchall()


@_with_retry
def get_chat(chat_id: str, username: str):
    """Fetch a chat by id, but only if it belongs to `username` — so one
    user can never load another user's saved conversation, even by
    guessing/reusing an id."""
    conn = _connection()
    with conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute(
            "SELECT * FROM chat_history WHERE id = %s AND username = %s",
            (chat_id, username),
        )
        row = cur.fetchone()
    if row is None:
        return None
    row["messages"] = json.loads(row["messages_json"])
    return row
