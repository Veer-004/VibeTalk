"""
chat_history_db.py — MySQL-backed storage for VibeTalk Arena chat history.

Keeps only the most recent MAX_HISTORY conversations (older ones are
pruned automatically on every save). Messages are stored as JSON
([{"role": "user"|"assistant", "content": "..."}]) rather than LangChain
message objects, since those aren't natively SQL-serializable.
"""

import json
import os

import pymysql
import streamlit as st

MAX_HISTORY = 30


@st.cache_resource
def _get_connection():
    return pymysql.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        autocommit=True,
        charset="utf8mb4",
    )


def _connection():
    conn = _get_connection()
    conn.ping(reconnect=True)
    return conn


def save_chat(entry: dict) -> None:
    """Insert a chat and prune down to the MAX_HISTORY most recent rows."""
    conn = _connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chat_history
                (id, conversation_type, title, messages_json, final_review, ended)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                entry["id"],
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
            WHERE id NOT IN (
                SELECT id FROM (
                    SELECT id FROM chat_history
                    ORDER BY created_at DESC
                    LIMIT %s
                ) AS keep_ids
            )
            """,
            (MAX_HISTORY,),
        )


def list_chats() -> list:
    """Most recent MAX_HISTORY chats, newest first (summary fields only)."""
    conn = _connection()
    with conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute(
            """
            SELECT id, conversation_type, title, ended, created_at
            FROM chat_history
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (MAX_HISTORY,),
        )
        return cur.fetchall()


def get_chat(chat_id: str):
    conn = _connection()
    with conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute("SELECT * FROM chat_history WHERE id = %s", (chat_id,))
        row = cur.fetchone()
    if row is None:
        return None
    row["messages"] = json.loads(row["messages_json"])
    return row
