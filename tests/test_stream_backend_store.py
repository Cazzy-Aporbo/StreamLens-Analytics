from __future__ import annotations

from stream_backend.orchestration.build import build_runtime_snapshot
from stream_backend.store import connect_sqlite, ensure_schema, load_latest_snapshot, save_snapshot
from stream_backend.utils import utc_now_iso
from tests.runtime_fixtures import music_index, music_report, representation_results


def test_store_round_trip_snapshot(tmp_path):
    generated_at = utc_now_iso()
    payloads = {
        "representation": representation_results(800),
        "music": music_report(),
        "music_index": music_index(),
        "data_engineering": {
            "synthetic_contract": {"dataset_id": "synthetic_character_fact", "quality_checks": [], "row_count": 2},
            "music_contract": {"dataset_id": "music_song_fact_enriched", "quality_checks": [], "row_count": 2},
        },
        "frontend_state": {
            "generated_at": generated_at,
            "sample_size": 800,
            "hero": {"signals": []},
            "roles": {},
            "narratives": {},
            "comparatives": {},
            "note_signals": {},
            "governance": {},
            "freshness": {},
        },
        "orchestration": {"runtime": "stream_backend"},
        "comparatives": {"synthetic_vs_public": []},
    }
    snapshot = build_runtime_snapshot(tmp_path, "run-1", generated_at, 800, payloads)
    conn = connect_sqlite(tmp_path / "runtime.sqlite3")
    try:
        ensure_schema(conn)
        save_snapshot(conn, snapshot)
        latest = load_latest_snapshot(conn)
    finally:
        conn.close()

    assert latest is not None
    assert latest["run_id"] == "run-1"
    assert latest["previous_chain_hash"] == ""
    assert len(latest["chain_hash"]) == 64
    assert "frontend_state" in latest
    assert "artifacts" in latest


def test_store_chain_links_snapshots(tmp_path):
    payloads = {
        "representation": representation_results(400),
        "music": music_report(),
        "music_index": music_index(),
        "data_engineering": {
            "synthetic_contract": {"dataset_id": "synthetic_character_fact", "quality_checks": [], "row_count": 2},
            "music_contract": {"dataset_id": "music_song_fact_enriched", "quality_checks": [], "row_count": 2},
        },
        "frontend_state": {
            "generated_at": "2026-07-02T01:00:00.000+00:00",
            "sample_size": 400,
            "hero": {"signals": []},
            "roles": {},
            "narratives": {},
            "comparatives": {},
            "note_signals": {},
            "governance": {},
            "freshness": {},
        },
        "orchestration": {"runtime": "stream_backend"},
        "comparatives": {"synthetic_vs_public": []},
    }
    snapshot_one = build_runtime_snapshot(
        tmp_path,
        "run-1",
        "2026-07-02T01:00:00.000+00:00",
        400,
        payloads,
    )
    snapshot_two = build_runtime_snapshot(
        tmp_path,
        "run-2",
        "2026-07-02T01:00:01.000+00:00",
        400,
        payloads,
    )
    conn = connect_sqlite(tmp_path / "runtime.sqlite3")
    try:
        ensure_schema(conn)
        save_snapshot(conn, snapshot_one)
        first_latest = load_latest_snapshot(conn)
        save_snapshot(conn, snapshot_two)
        second_latest = load_latest_snapshot(conn)
    finally:
        conn.close()

    assert first_latest is not None
    assert second_latest is not None
    assert first_latest["chain_hash"]
    assert second_latest["previous_chain_hash"] == first_latest["chain_hash"]
    assert second_latest["chain_hash"] != first_latest["chain_hash"]
