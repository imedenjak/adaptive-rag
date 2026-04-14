import psycopg2
import psycopg2.extras

from app.config import DATABASE_URL


def _connect() -> psycopg2.extensions.connection:
    conn = psycopg2.connect(DATABASE_URL)
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id         SERIAL PRIMARY KEY,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
        """)
    conn.commit()
    return conn


def load_history() -> list:
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT role, content FROM messages ORDER BY id")
            rows = cur.fetchall()
    return [{"role": row[0], "content": row[1]} for row in rows]


def save_message(role: str, content: str) -> None:
    conn = _connect()
    with conn.cursor() as cur:
        cur.execute("INSERT INTO messages (role, content) VALUES (%s, %s)", (role, content))
    conn.commit()
    conn.close()
