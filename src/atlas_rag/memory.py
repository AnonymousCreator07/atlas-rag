"""Small persistent SQLite conversation-memory store."""

import sqlite3
from pathlib import Path


class ConversationMemory:
    def __init__(self, path: Path, max_turns: int = 8) -> None:
        self.path = path
        self.max_turns = max_turns
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS messages (
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            db.commit()

    def get(self, session_id: str) -> list[dict[str, str]]:
        with sqlite3.connect(self.path) as db:
            rows = db.execute(
                """SELECT role, content FROM messages
                   WHERE session_id = ? ORDER BY rowid DESC LIMIT ?""",
                (session_id, self.max_turns * 2),
            ).fetchall()
        return [{"role": role, "content": content} for role, content in reversed(rows)]

    def add(self, session_id: str, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("role must be user or assistant")
        with sqlite3.connect(self.path) as db:
            db.execute(
                "INSERT INTO messages(session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content),
            )
            db.execute(
                """DELETE FROM messages
                   WHERE session_id = ? AND rowid NOT IN (
                     SELECT rowid FROM messages WHERE session_id = ?
                     ORDER BY rowid DESC LIMIT ?
                   )""",
                (session_id, session_id, self.max_turns * 2),
            )
            db.commit()

    def clear(self, session_id: str) -> None:
        with sqlite3.connect(self.path) as db:
            db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            db.commit()
