from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping


EVENT_COLUMNS = (
    "event_id",
    "observed_at",
    "ingested_at",
    "platform",
    "market",
    "track_id",
    "track_title",
    "artist_id",
    "artist_name",
    "genre",
    "language",
    "label_tier",
    "independent_flag",
    "editorial_flag",
    "exposure_share",
    "recommendation_share",
    "completion_rate",
    "skip_rate",
    "save_rate",
    "payload_hash",
    "source",
)


def insert_events(conn: sqlite3.Connection, events: list[Mapping[str, Any]]) -> dict[str, int]:
    inserted = 0
    skipped = 0
    for event in events:
        cursor = conn.execute(
            f"""
            INSERT OR IGNORE INTO runtime_events ({", ".join(EVENT_COLUMNS)})
            VALUES ({", ".join("?" for _ in EVENT_COLUMNS)})
            """,
            tuple(event.get(column) for column in EVENT_COLUMNS),
        )
        if cursor.rowcount:
            inserted += 1
        else:
            skipped += 1
    conn.commit()
    return {"inserted": inserted, "skipped": skipped}


FILTERABLE_EVENT_COLUMNS = {
    "platform",
    "market",
    "genre",
    "language",
    "label_tier",
    "source",
}


def _event_filters_sql(filters: Mapping[str, Any] | None) -> tuple[str, list[Any]]:
    if not filters:
        return "", []
    clauses: list[str] = []
    params: list[Any] = []
    for key, value in filters.items():
        if key not in FILTERABLE_EVENT_COLUMNS:
            continue
        normalized = str(value or "").strip()
        if not normalized:
            continue
        clauses.append(f"{key} = ?")
        params.append(normalized)
    if not clauses:
        return "", []
    return "WHERE " + " AND ".join(clauses), params


def list_events(
    conn: sqlite3.Connection,
    *,
    limit: int = 250,
    filters: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    where_sql, params = _event_filters_sql(filters)
    rows = conn.execute(
        f"""
        SELECT
            event_id,
            observed_at,
            ingested_at,
            platform,
            market,
            track_id,
            track_title,
            artist_id,
            artist_name,
            genre,
            language,
            label_tier,
            independent_flag,
            editorial_flag,
            exposure_share,
            recommendation_share,
            completion_rate,
            skip_rate,
            save_rate,
            payload_hash,
            source
        FROM runtime_events
        {where_sql}
        ORDER BY observed_at DESC, event_id DESC
        LIMIT ?
        """,
        [*params, int(limit)],
    ).fetchall()
    return [dict(row) for row in rows]


def count_events(conn: sqlite3.Connection, *, filters: Mapping[str, Any] | None = None) -> int:
    where_sql, params = _event_filters_sql(filters)
    query = f"SELECT COUNT(*) FROM runtime_events {where_sql}"
    return int(conn.execute(query, params).fetchone()[0])


def prune_events(
    conn: sqlite3.Connection,
    *,
    max_rows: int,
    max_age_days: int,
) -> dict[str, int]:
    deleted = 0
    if max_age_days > 0:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
        cursor = conn.execute(
            "DELETE FROM runtime_events WHERE observed_at < ?",
            (cutoff,),
        )
        deleted += max(0, int(cursor.rowcount or 0))

    total = count_events(conn)
    overflow = max(0, total - int(max_rows))
    if overflow > 0:
        cursor = conn.execute(
            """
            DELETE FROM runtime_events
            WHERE event_id IN (
                SELECT event_id
                FROM runtime_events
                ORDER BY observed_at ASC, event_id ASC
                LIMIT ?
            )
            """,
            (overflow,),
        )
        deleted += max(0, int(cursor.rowcount or 0))

    conn.commit()
    return {"deleted": deleted, "remaining": count_events(conn)}
