import sqlite3
from pathlib import Path

DB_PATH = Path("/data/chat_history.db")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            role      TEXT NOT NULL,
            content   TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def load_history() -> list:
    with _connect() as conn:
        rows = conn.execute("SELECT role, content FROM messages ORDER BY id").fetchall()
    return [{"role": row[0], "content": row[1]} for row in rows]


def save_message(role: str, content: str) -> None:
    with _connect() as conn:
        conn.execute("INSERT INTO messages (role, content) VALUES (?, ?)", (role, content))
        conn.commit()
