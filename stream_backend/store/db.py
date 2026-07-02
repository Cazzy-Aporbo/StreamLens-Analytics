from __future__ import annotations

import os
import sqlite3
from pathlib import Path


def _coerce_busy_timeout_ms() -> int:
    raw = os.getenv("STREAM_SQLITE_BUSY_TIMEOUT_MS", "").strip()
    if not raw:
        return 30_000
    try:
        value = int(raw)
    except ValueError:
        return 30_000
    return max(1_000, min(value, 120_000))


def connect_sqlite(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    busy_timeout_ms = _coerce_busy_timeout_ms()
    conn = sqlite3.connect(str(path), timeout=busy_timeout_ms / 1000)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms};")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA wal_autocheckpoint=1000;")
    return conn
