from __future__ import annotations

from stream_backend.services.enterprise_bus import EnterpriseBusConfig, publish_runtime_batch


def test_enterprise_bus_is_honest_when_unconfigured():
    config = EnterpriseBusConfig()
    result = publish_runtime_batch(
        events=[{"event_id": "evt-1", "platform": "spotify"}],
        metrics={"pressure_band": "watch", "event_count": 1},
        config=config,
    )

    assert result["mode"] == "disabled"
    assert result["receipts"] == []
    assert result["configured_sinks"]["pandaproxy_enabled"] is False
    assert result["configured_sinks"]["clickhouse_enabled"] is False
