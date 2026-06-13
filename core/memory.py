"""Persistent memory — SQLite, stdlib only.

Three tables:
    sessions  — one row per conversation
    messages  — full transcript (history survives restarts, --resume)
    facts     — long-term knowledge the agent saves via memory tools

Fact recall is keyword-overlap scoring: deterministic, zero RAM, no
embedding model required. Swap in embeddings later if one is present —
the interface (`recall`) won't change.
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
import time
from typing import Optional

from .config import DATA_DIR

DB_PATH = os.path.join(DATA_DIR, "memory.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_ts REAL NOT NULL,
    title TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    ts REAL NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    content TEXT NOT NULL,
    topic TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    due_ts REAL NOT NULL,
    repeat_seconds REAL,
    prompt TEXT NOT NULL,
    enabled INTEGER DEFAULT 1
);
"""

_WORD = re.compile(r"[a-zA-Z0-9_]{3,}")


class Memory:
    def __init__(self, db_path: str = DB_PATH) -> None:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        self.session_id: int = 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ── Sessions / messages ──────────────────────────────────────

    def new_session(self) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO sessions (started_ts) VALUES (?)", (time.time(),))
            self._conn.commit()
            self.session_id = cur.lastrowid or 0
        return self.session_id

    def resume_last_session(self) -> Optional[int]:
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM sessions ORDER BY id DESC LIMIT 1").fetchone()
        if row:
            self.session_id = row[0]
            return self.session_id
        return None

    def add_message(self, role: str, content: str) -> None:
        if not self.session_id:
            self.new_session()
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages (session_id, ts, role, content) VALUES (?,?,?,?)",
                (self.session_id, time.time(), role, content))
            self._conn.commit()

    def recent_messages(self, limit: int, max_chars: int = 4000) -> list[dict]:
        """Last `limit` messages of the current session, oldest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM messages WHERE session_id=? "
                "ORDER BY id DESC LIMIT ?", (self.session_id, limit)).fetchall()
        out = []
        for role, content in reversed(rows):
            if len(content) > max_chars:
                content = content[:max_chars] + "\n...[truncated]"
            out.append({"role": role, "content": content})
        return out

    # ── Facts (long-term) ────────────────────────────────────────

    def remember(self, content: str, topic: str = "") -> int:
        content = content.strip()
        with self._lock:
            dup = self._conn.execute(
                "SELECT id FROM facts WHERE content=?", (content,)).fetchone()
            if dup:
                return dup[0]
            cur = self._conn.execute(
                "INSERT INTO facts (ts, content, topic) VALUES (?,?,?)",
                (time.time(), content, topic.strip()))
            self._conn.commit()
            return cur.lastrowid or 0

    def forget(self, fact_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM facts WHERE id=?", (fact_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def all_facts(self) -> list[tuple[int, str, str]]:
        with self._lock:
            return self._conn.execute(
                "SELECT id, content, topic FROM facts ORDER BY id").fetchall()

    def recall(self, query: str, limit: int = 8) -> list[str]:
        """Facts ranked by keyword overlap with the query."""
        facts = self.all_facts()
        if not facts:
            return []
        q_words = set(w.lower() for w in _WORD.findall(query))
        scored = []
        for _id, content, topic in facts:
            f_words = set(w.lower() for w in _WORD.findall(content + " " + topic))
            score = len(q_words & f_words)
            scored.append((score, content))
        scored.sort(key=lambda x: -x[0])
        # Always include a few recent facts even with zero overlap so the
        # model knows the user's standing context (name, role, ...).
        hits = [c for s, c in scored if s > 0][:limit]
        if len(hits) < limit:
            for _id, content, _topic in reversed(facts):
                if content not in hits:
                    hits.append(content)
                if len(hits) >= limit:
                    break
        return hits[:limit]

    # ── Scheduled tasks (used by core.scheduler) ─────────────────

    def add_task(self, due_ts: float, prompt: str, repeat_seconds: Optional[float]) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO tasks (due_ts, repeat_seconds, prompt) VALUES (?,?,?)",
                (due_ts, repeat_seconds, prompt))
            self._conn.commit()
            return cur.lastrowid or 0

    def update_task_due(self, task_id: int, due_ts: float) -> None:
        with self._lock:
            self._conn.execute("UPDATE tasks SET due_ts=? WHERE id=?", (due_ts, task_id))
            self._conn.commit()

    def disable_task(self, task_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("UPDATE tasks SET enabled=0 WHERE id=?", (task_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def enabled_tasks(self) -> list[tuple[int, float, Optional[float], str]]:
        with self._lock:
            return self._conn.execute(
                "SELECT id, due_ts, repeat_seconds, prompt FROM tasks WHERE enabled=1"
            ).fetchall()
