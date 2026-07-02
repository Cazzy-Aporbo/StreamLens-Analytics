"""SQLite persistence helpers for the public Stream backend."""

from .db import connect_sqlite
from .events import count_events, insert_events, list_events, prune_events
from .queries import latest_payload_names
from .runs import insert_run, list_runs, latest_run_row, verify_run_chain
from .schema import ensure_schema
from .snapshots import load_latest_snapshot, save_snapshot

__all__ = [
    "count_events",
    "connect_sqlite",
    "ensure_schema",
    "insert_run",
    "insert_events",
    "list_events",
    "latest_payload_names",
    "latest_run_row",
    "list_runs",
    "load_latest_snapshot",
    "prune_events",
    "save_snapshot",
    "verify_run_chain",
]
