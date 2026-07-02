from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping
from urllib.parse import quote
from urllib import error, request


@dataclass(frozen=True)
class EnterpriseBusConfig:
    pandaproxy_url: str = ""
    pandaproxy_topic: str = "stream.runtime.events"
    clickhouse_url: str = ""
    clickhouse_table: str = "stream_runtime_metrics"
    timeout_seconds: float = 3.0

    @classmethod
    def from_env(cls) -> "EnterpriseBusConfig":
        timeout_raw = os.getenv("STREAM_BUS_TIMEOUT_SECONDS", "").strip()
        try:
            timeout_seconds = float(timeout_raw) if timeout_raw else 3.0
        except ValueError:
            timeout_seconds = 3.0
        return cls(
            pandaproxy_url=os.getenv("STREAM_PANDAPROXY_URL", "").strip().rstrip("/"),
            pandaproxy_topic=os.getenv("STREAM_PANDAPROXY_TOPIC", "stream.runtime.events").strip(),
            clickhouse_url=os.getenv("STREAM_CLICKHOUSE_URL", "").strip().rstrip("/"),
            clickhouse_table=os.getenv("STREAM_CLICKHOUSE_TABLE", "stream_runtime_metrics").strip(),
            timeout_seconds=timeout_seconds,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "pandaproxy_enabled": bool(self.pandaproxy_url and self.pandaproxy_topic),
            "clickhouse_enabled": bool(self.clickhouse_url and self.clickhouse_table),
            "pandaproxy_topic": self.pandaproxy_topic,
            "clickhouse_table": self.clickhouse_table,
            "timeout_seconds": self.timeout_seconds,
        }


def _post_json(url: str, payload: Any, *, timeout_seconds: float, headers: Mapping[str, str] | None = None) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", **dict(headers or {})},
    )
    with request.urlopen(req, timeout=timeout_seconds) as response:  # noqa: S310 - controlled env endpoints
        text = response.read().decode("utf-8", errors="replace")
        return {
            "status_code": getattr(response, "status", 200),
            "body_preview": text[:240],
        }


def publish_runtime_batch(
    events: list[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    *,
    filters: Mapping[str, Any] | None = None,
    config: EnterpriseBusConfig | None = None,
) -> dict[str, Any]:
    bus = config or EnterpriseBusConfig.from_env()
    receipts: list[dict[str, Any]] = []
    if not bus.describe()["pandaproxy_enabled"] and not bus.describe()["clickhouse_enabled"]:
        return {
            "mode": "disabled",
            "receipts": [],
            "configured_sinks": bus.describe(),
        }

    if bus.pandaproxy_url and bus.pandaproxy_topic:
        topic_url = f"{bus.pandaproxy_url}/topics/{bus.pandaproxy_topic}"
        payload = {
            "records": [
                {"value": dict(event)}
                for event in events
            ]
        }
        try:
            receipts.append(
                {
                    "sink": "pandaproxy",
                    "url": topic_url,
                    **_post_json(
                        topic_url,
                        payload,
                        timeout_seconds=bus.timeout_seconds,
                    ),
                }
            )
        except (error.URLError, TimeoutError, OSError) as exc:
            receipts.append(
                {
                    "sink": "pandaproxy",
                    "url": topic_url,
                    "error": str(exc),
                }
            )

    if bus.clickhouse_url and bus.clickhouse_table:
        query = (
            f"INSERT INTO {bus.clickhouse_table} "
            "(generated_at, pressure_band, pressure_score, event_count, filters_json, metrics_json) "
            "FORMAT JSONEachRow"
        )
        row = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pressure_band": metrics.get("pressure_band"),
            "pressure_score": metrics.get("pressure_score"),
            "event_count": metrics.get("event_count"),
            "filters_json": json.dumps(filters or {}, ensure_ascii=False),
            "metrics_json": json.dumps(
                {
                    "metrics": metrics.get("metrics", {}),
                    "signals": metrics.get("signals", {}),
                    "summary": metrics.get("summary", ""),
                },
                ensure_ascii=False,
            ),
        }
        clickhouse_url = f"{bus.clickhouse_url}/?query={quote(query, safe='')}"
        try:
            receipts.append(
                {
                    "sink": "clickhouse",
                    "url": clickhouse_url.split("?", 1)[0],
                    **_post_json(
                        clickhouse_url,
                        row,
                        timeout_seconds=bus.timeout_seconds,
                        headers={"Content-Type": "application/json"},
                    ),
                }
            )
        except (error.URLError, TimeoutError, OSError) as exc:
            receipts.append(
                {
                    "sink": "clickhouse",
                    "url": clickhouse_url.split("?", 1)[0],
                    "error": str(exc),
                }
            )

    mode = "published" if receipts and not any("error" in receipt for receipt in receipts) else "partial"
    return {
        "mode": mode,
        "receipts": receipts,
        "configured_sinks": bus.describe(),
    }
