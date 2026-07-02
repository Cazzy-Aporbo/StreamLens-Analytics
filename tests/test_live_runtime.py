from __future__ import annotations

from stream_backend.services.live_runtime import (
    demo_runtime_events,
    ingest_runtime_events,
    list_runtime_events,
    load_live_metrics,
    runtime_event_contract,
)
from stream_backend.config import RuntimeConfig


def test_live_runtime_demo_window_is_structured(tmp_path):
    surface = load_live_metrics(tmp_path / "runtime.sqlite3")
    assert surface["source"] in {"live", "demonstration"}
    assert surface["pressure_band"] in {"open", "watch", "narrow", "compressed"}
    assert len(surface["axes"]) >= 4
    assert len(surface["watchpoints"]) >= 3


def test_live_runtime_ingest_is_idempotent(tmp_path):
    sqlite_path = tmp_path / "runtime.sqlite3"
    events = demo_runtime_events()

    first = ingest_runtime_events(sqlite_path, events)
    second = ingest_runtime_events(sqlite_path, events)
    latest = list_runtime_events(sqlite_path, limit=20)
    surface = load_live_metrics(sqlite_path)

    assert first["inserted"] == len(events)
    assert second["skipped"] == len(events)
    assert len(latest) == len(events)
    assert surface["source"] == "live"
    assert surface["event_count"] == len(events)


def test_runtime_event_contract_is_bounded(tmp_path):
    config = RuntimeConfig.from_base_dir(tmp_path)
    contract = runtime_event_contract(config)

    assert contract["batch_limits"]["max_events"] == config.event_ingest_batch_limit
    assert contract["retention"]["max_rows"] == config.event_retention_max_rows
    assert len(contract["fields"]) >= 10
    assert contract["sample_event"]["event_id"].startswith("demo-")
