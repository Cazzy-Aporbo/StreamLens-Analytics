from __future__ import annotations

import sqlite3


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS runtime_runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            sample_size INTEGER NOT NULL,
            synthetic_rows INTEGER NOT NULL DEFAULT 0,
            music_rows INTEGER NOT NULL DEFAULT 0,
            music_index_rows INTEGER NOT NULL DEFAULT 0,
            payload_hash TEXT NOT NULL,
            previous_chain_hash TEXT NOT NULL DEFAULT '',
            chain_hash TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS runtime_payloads (
            run_id TEXT NOT NULL,
            payload_name TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            PRIMARY KEY (run_id, payload_name),
            FOREIGN KEY (run_id) REFERENCES runtime_runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS runtime_artifacts (
            run_id TEXT NOT NULL,
            artifact_name TEXT NOT NULL,
            artifact_kind TEXT NOT NULL,
            artifact_path TEXT NOT NULL,
            freshness_label TEXT NOT NULL,
            row_count INTEGER NOT NULL DEFAULT 0,
            note TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (run_id, artifact_name),
            FOREIGN KEY (run_id) REFERENCES runtime_runs(run_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_runtime_runs_created_at
        ON runtime_runs(created_at DESC);

        CREATE TABLE IF NOT EXISTS runtime_events (
            event_id TEXT PRIMARY KEY,
            observed_at TEXT NOT NULL,
            ingested_at TEXT NOT NULL,
            platform TEXT NOT NULL,
            market TEXT NOT NULL,
            track_id TEXT NOT NULL,
            track_title TEXT NOT NULL,
            artist_id TEXT NOT NULL,
            artist_name TEXT NOT NULL,
            genre TEXT NOT NULL,
            language TEXT NOT NULL,
            label_tier TEXT NOT NULL,
            independent_flag INTEGER NOT NULL DEFAULT 0,
            editorial_flag INTEGER NOT NULL DEFAULT 0,
            exposure_share REAL NOT NULL DEFAULT 0.0,
            recommendation_share REAL NOT NULL DEFAULT 0.0,
            completion_rate REAL NOT NULL DEFAULT 0.0,
            skip_rate REAL NOT NULL DEFAULT 0.0,
            save_rate REAL NOT NULL DEFAULT 0.0,
            payload_hash TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT ''
        );

        CREATE INDEX IF NOT EXISTS idx_runtime_events_observed_at
        ON runtime_events(observed_at DESC);

        CREATE INDEX IF NOT EXISTS idx_runtime_events_platform_market
        ON runtime_events(platform, market);
        """
    )
    runtime_run_columns = _table_columns(conn, "runtime_runs")
    if "previous_chain_hash" not in runtime_run_columns:
        conn.execute(
            "ALTER TABLE runtime_runs ADD COLUMN previous_chain_hash TEXT NOT NULL DEFAULT ''"
        )
    if "chain_hash" not in runtime_run_columns:
        conn.execute(
            "ALTER TABLE runtime_runs ADD COLUMN chain_hash TEXT NOT NULL DEFAULT ''"
        )
    conn.commit()
