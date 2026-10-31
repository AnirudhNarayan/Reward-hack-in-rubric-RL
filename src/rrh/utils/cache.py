"""SQLite cache for LLM responses, so re-running an analysis never re-pays for a call."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading


class ResponseCache:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=60)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
        self._lock = threading.Lock()

    @staticmethod
    def _key(parts) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()

    def get(self, parts) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT v FROM cache WHERE k=?", (self._key(parts),)).fetchone()
        return row[0] if row else None

    def put(self, parts, value: str) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (self._key(parts), value))
            self._conn.commit()
