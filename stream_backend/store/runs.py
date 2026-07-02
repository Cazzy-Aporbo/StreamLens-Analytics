from __future__ import annotations

import sqlite3

from stream_backend.utils import stable_hash


def build_chain_hash(
    *,
    run_id: str,
    created_at: str,
    payload_hash: str,
    previous_chain_hash: str,
) -> str:
    return stable_hash(
        {
            "run_id": run_id,
            "created_at": created_at,
            "payload_hash": payload_hash,
            "previous_chain_hash": previous_chain_hash,
        }
    )


def latest_run_row(conn: sqlite3.Connection) -> dict | None:
    row = conn.execute(
        """
        SELECT
            run_id,
            created_at,
            sample_size,
            synthetic_rows,
            music_rows,
            music_index_rows,
            payload_hash,
            previous_chain_hash,
            chain_hash
        FROM runtime_runs
        ORDER BY created_at DESC, run_id DESC
        LIMIT 1
        """
    ).fetchone()
    return dict(row) if row is not None else None


def insert_run(
    conn: sqlite3.Connection,
    run_id: str,
    created_at: str,
    sample_size: int,
    synthetic_rows: int,
    music_rows: int,
    music_index_rows: int,
    payload_hash: str,
    previous_chain_hash: str,
    chain_hash: str,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO runtime_runs (
            run_id,
            created_at,
            sample_size,
            synthetic_rows,
            music_rows,
            music_index_rows,
            payload_hash,
            previous_chain_hash,
            chain_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            created_at,
            sample_size,
            synthetic_rows,
            music_rows,
            music_index_rows,
            payload_hash,
            previous_chain_hash,
            chain_hash,
        ),
    )


def list_runs(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT
            run_id,
            created_at,
            sample_size,
            synthetic_rows,
            music_rows,
            music_index_rows,
            payload_hash,
            previous_chain_hash,
            chain_hash
        FROM runtime_runs
        ORDER BY created_at DESC, run_id DESC
        """
    ).fetchall()
    return [dict(row) for row in rows]


def verify_run_chain(rows: list[dict]) -> dict:
    if not rows:
        return {
            "status": "empty",
            "ok": True,
            "run_count": 0,
            "verifiable_run_count": 0,
            "skipped_run_count": 0,
            "latest_chain_hash": "",
            "broken_links": [],
        }

    ordered = sorted(rows, key=lambda row: (str(row.get("created_at") or ""), str(row.get("run_id") or "")))
    eligible = [row for row in ordered if str(row.get("chain_hash") or "").strip()]
    broken_links: list[dict] = []
    previous_chain_hash = ""
    for row in eligible:
        expected_chain_hash = build_chain_hash(
            run_id=str(row.get("run_id") or ""),
            created_at=str(row.get("created_at") or ""),
            payload_hash=str(row.get("payload_hash") or ""),
            previous_chain_hash=previous_chain_hash,
        )
        row_previous = str(row.get("previous_chain_hash") or "")
        row_chain = str(row.get("chain_hash") or "")
        if row_previous != previous_chain_hash or row_chain != expected_chain_hash:
            broken_links.append(
                {
                    "run_id": str(row.get("run_id") or ""),
                    "created_at": str(row.get("created_at") or ""),
                    "expected_previous_chain_hash": previous_chain_hash,
                    "recorded_previous_chain_hash": row_previous,
                    "expected_chain_hash": expected_chain_hash,
                    "recorded_chain_hash": row_chain,
                }
            )
        previous_chain_hash = row_chain

    skipped_run_count = len(ordered) - len(eligible)
    if broken_links:
        status = "broken"
    elif skipped_run_count:
        status = "mixed"
    else:
        status = "verified"

    return {
        "status": status,
        "ok": not broken_links,
        "run_count": len(ordered),
        "verifiable_run_count": len(eligible),
        "skipped_run_count": skipped_run_count,
        "latest_chain_hash": previous_chain_hash,
        "broken_links": broken_links,
    }
