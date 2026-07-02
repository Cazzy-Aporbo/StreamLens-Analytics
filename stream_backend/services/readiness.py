from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


def _status_label(level: str) -> tuple[str, str]:
    labels = {
        "pass": (
            "ready_for_local_use",
            "The repository is in a stable enough state for local analysis, public review, and careful contribution.",
        ),
        "warn": (
            "ready_with_visible_limits",
            "The repository runs cleanly, though some claims still need broader coverage, fresher evidence, or stronger operating proof.",
        ),
        "fail": (
            "not_ready",
            "The local environment or the public evidence lane is incomplete enough that repair should come before reliance.",
        ),
    }
    return labels.get(level, labels["warn"])


def build_readiness_surface(
    *,
    doctor_report: Mapping[str, Any],
    catalog: Mapping[str, Any],
    data_engineering: Mapping[str, Any],
    streaming_readiness: Mapping[str, Any],
    api_route_count: int | None = None,
    live_event_count: int | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    counts = dict(doctor_report.get("counts") or {})
    overall = str(doctor_report.get("overall") or "warn")
    label, summary = _status_label(overall)
    checks_by_name = {
        str(row.get("name")): row for row in (doctor_report.get("checks") or [])
    }
    operating_model = dict(data_engineering.get("operating_model") or {})
    coverage = (
        dict((data_engineering.get("music_quality") or {}).get("coverage") or {})
        if isinstance(data_engineering, Mapping)
        else {}
    )
    warnings = [
        {
            "name": row.get("name", "unknown"),
            "detail": row.get("detail", ""),
        }
        for row in doctor_report.get("checks", [])
        if row.get("level") == "warn"
    ][:5]

    return {
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "overall": {
            "level": overall,
            "label": label,
            "summary": summary,
            "doctor_pass_count": int(counts.get("pass", 0) or 0),
            "doctor_warn_count": int(counts.get("warn", 0) or 0),
            "doctor_fail_count": int(counts.get("fail", 0) or 0),
        },
        "proof_points": {
            "api_route_count": int(api_route_count or 0),
            "entrypoint_count": len(catalog.get("entrypoints") or []),
            "module_count": len(catalog.get("modules") or []),
            "public_asset_count": len(catalog.get("public_assets") or []),
            "dataset_count": int(operating_model.get("dataset_count", 0) or 0),
            "quality_gate_count": int(operating_model.get("quality_gate_count", 0) or 0),
            "tracked_rows": int(operating_model.get("tracked_rows", 0) or 0),
            "live_event_count": int(live_event_count or 0),
            "genre_known_share": round(float(coverage.get("genre_known_share", 0.0) or 0.0), 3),
            "publication_year_share": round(
                float(coverage.get("publication_year_explicit_share", 0.0) or 0.0)
                + float(coverage.get("publication_year_inferred_share", 0.0) or 0.0),
                3,
            ),
            "write_endpoint_mode": str(
                (checks_by_name.get("write-endpoint-auth") or {}).get("mode")
                or "open_local_default"
            ),
            "local_ledger_mode": "hash_linked"
            if (checks_by_name.get("sqlite-ledger-schema") or {}).get("level") == "pass"
            else "shape_incomplete",
        },
        "quickstart": [
            {
                "title": "Install and preflight",
                "command": "pip install -r requirements-dev.txt && python -m stream_backend.cli.doctor",
                "why": "Checks imports, files, SQLite writeability, public music health, and built artifacts before a local run is treated as dependable.",
            },
            {
                "title": "Run the live API",
                "command": "python app.py",
                "why": "Serves the browser, JSON endpoints, and docs from the same local repository state.",
            },
            {
                "title": "Bake the static surface",
                "command": "python build_static.py",
                "why": "Regenerates the committed JSON artifacts so the browser can run without the live API.",
            },
            {
                "title": "Seed the live event window",
                "command": "make seed-live-demo",
                "why": "Loads the bounded demonstration event batch so the rolling window, contract lane, and WebSocket feed have immediate data to inspect.",
            },
        ],
        "adoption_paths": [
            {
                "id": "research_strategy",
                "title": "Research and strategy teams",
                "best_for": "Inspecting bias, attention concentration, and public narrative pressure without private platform access.",
                "start_with": "Overview → Findings → Methods",
                "proof_lane": "/api/system/readiness",
            },
            {
                "id": "engineering_data",
                "title": "Engineering and data teams",
                "best_for": "Auditing contracts, build parity, static artifacts, runtime shape, and the current boundary between batch analytics and real streaming.",
                "start_with": "Methods → /api/runtime/events/contract → /api/system/data-engineering → /api/system/streaming-readiness",
                "proof_lane": "/api/system/catalog",
            },
            {
                "id": "music_editorial",
                "title": "Music, editorial, and market readers",
                "best_for": "Reading public reach, release concentration, channel exposure, and role-based consequences from a separate real-data lane.",
                "start_with": "Real Music → Decision Lab → Intelligence",
                "proof_lane": "/api/music/quality",
            },
        ],
        "commercial_posture": {
            "ready_now": [
                "Internal research reviews",
                "Editorial and catalog strategy sessions",
                "Methods teaching and reproducibility checks",
                "Public contribution and benchmarking",
            ],
            "best_fit_today": (
                "A public analysis and experimentation surface with a working backend spine. The full streaming control plane remains a later step."
            ),
            "not_claiming_yet": [
                "Exactly-once streaming guarantees",
                "Distributed checkpoint recovery",
                "Multi-tenant production hardening",
                "Managed identity, rate limiting, and operational SLOs",
            ],
            "next_engineering_moves": (streaming_readiness.get("roadmap") or {}).get("quick_wins", [])[:3],
        },
        "friction_log": warnings,
    }
