"""SQLite persistence for Askwise: conversations, messages and PDF documents."""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "askwise.db"
MAX_MEMORY_TURNS = 8  # conversational turns kept as context (question+answer pairs)

_thread_local = threading.local()


def _conn() -> sqlite3.Connection:
    conn = getattr(_thread_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        _thread_local.conn = conn
    return conn


def init_db() -> None:
    conn = _conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL DEFAULT 'New conversation',
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role TEXT NOT NULL CHECK (role IN ('user','assistant')),
            content TEXT NOT NULL,
            mode TEXT,
            sources_json TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            filename TEXT NOT NULL,
            text TEXT NOT NULL,
            uploaded_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
        """
    )
    conn.commit()


def create_conversation(title: str = "New conversation") -> int:
    with _conn() as conn:
        return int(conn.execute("INSERT INTO conversations (title) VALUES (?)", (title,)).lastrowid or 0)


def list_conversations() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.title, c.created_at,
                   (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count
            FROM conversations c ORDER BY c.id DESC LIMIT 100
            """
        ).fetchall()
    return [
        {"id": r[0], "title": r[1], "createdAt": r[2], "messageCount": r[3]} for r in rows
    ]


def delete_conversation(conversation_id: int) -> bool:
    with _conn() as conn:
        cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.execute("DELETE FROM documents WHERE conversation_id = ?", (conversation_id,))
        return cur.rowcount > 0


def add_message(conversation_id: int, role: str, content: str, mode: str | None = None,
                sources: list[dict] | None = None) -> int:
    import json

    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO messages (conversation_id, role, content, mode, sources_json) VALUES (?,?,?,?,?)",
            (conversation_id, role, content, mode,
             json.dumps(sources) if sources is not None else None),
        )
        if role == "user":
            title = content.strip()[:60]
            conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ? AND title = 'New conversation'",
                (title, conversation_id),
            )
        return int(cur.lastrowid or 0)


def conversation_messages(conversation_id: int) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT role, content, mode, sources_json FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
    import json

    out = []
    for role, content, mode, sources_json in rows:
        out.append({
            "role": role,
            "content": content,
            "mode": mode,
            "sources": json.loads(sources_json) if sources_json else None,
        })
    return out


def save_document(conversation_id: int | None, filename: str, text: str) -> int:
    with _conn() as conn:
        return int(conn.execute(
            "INSERT INTO documents (conversation_id, filename, text) VALUES (?,?,?)",
            (conversation_id, filename, text),
        ).lastrowid or 0)


def latest_documents(conversation_id: int | None, limit: int = 3) -> list[dict]:
    """Most recently uploaded documents, optionally scoped to a conversation."""
    with _conn() as conn:
        if conversation_id is not None:
            rows = conn.execute(
                "SELECT id, filename, text FROM documents WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, filename, text FROM documents ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
    return [{"id": r[0], "filename": r[1], "text": r[2]} for r in rows]


def memory_history(conversation_id: int) -> list[tuple[str, str]]:
    """Last MAX_MEMORY_TURNS user/assistant exchanges for prompt context."""
    msgs = conversation_messages(conversation_id)
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(msgs):
        if msgs[i]["role"] == "user" and i + 1 < len(msgs) and msgs[i + 1]["role"] == "assistant":
            pairs.append((msgs[i]["content"], msgs[i + 1]["content"]))
            i += 2
        else:
            i += 1
    return pairs[-MAX_MEMORY_TURNS:]
