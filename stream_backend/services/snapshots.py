from __future__ import annotations

from stream_backend.store import (
    connect_sqlite,
    ensure_schema,
    list_runs,
    load_latest_snapshot,
    verify_run_chain,
)


def load_latest_runtime_snapshot(sqlite_path):
    conn = connect_sqlite(sqlite_path)
    try:
        ensure_schema(conn)
        return load_latest_snapshot(conn)
    finally:
        conn.close()


def load_runtime_ledger(sqlite_path, *, limit: int = 12):
    conn = connect_sqlite(sqlite_path)
    try:
        ensure_schema(conn)
        rows = list_runs(conn)
        verification = verify_run_chain(rows)
        return {
            "status": verification["status"],
            "ok": verification["ok"],
            "run_count": verification["run_count"],
            "verifiable_run_count": verification["verifiable_run_count"],
            "skipped_run_count": verification["skipped_run_count"],
            "latest_chain_hash": verification["latest_chain_hash"],
            "broken_links": verification["broken_links"],
            "items": rows[:limit],
        }
    finally:
        conn.close()
