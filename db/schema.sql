-- VibeTalk Arena chat history schema.
-- Applied automatically by the mysql:8.0 container the first time it starts
-- (MYSQL_DATABASE=vibetalk creates the database; this table is created once
-- manually — see README.md for the docker run + setup commands).

CREATE TABLE IF NOT EXISTS chat_history (
    id VARCHAR(36) PRIMARY KEY,
    username VARCHAR(64) NOT NULL,
    conversation_type VARCHAR(32),
    title VARCHAR(255),
    messages_json LONGTEXT NOT NULL,
    final_review TEXT,
    ended TINYINT(1) NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_chat_history_username_created ON chat_history (username, created_at);
