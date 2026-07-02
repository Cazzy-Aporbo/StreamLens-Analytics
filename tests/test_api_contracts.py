from __future__ import annotations

from stream_backend.services.api_contracts import build_api_contract_surface
from stream_backend.services.runtime_invariants import evaluate_live_metric_invariants
from stream_backend.services.live_runtime import build_live_metrics_surface, demo_runtime_events
from app import app


def test_api_contract_surface_indexes_routes_and_write_policy():
    surface = build_api_contract_surface(app.openapi())

    assert surface["route_count"] >= 40
    assert surface["schema_count"] >= 1
    assert surface["write_surface_policy"]["header"] == "X-API-Key"
    assert "openapi.stream.json" in surface["static_contracts"]

    route_keys = {(row["method"], row["path"]) for row in surface["routes"]}
    assert ("GET", "/api/system/model-registry") in route_keys
    assert ("GET", "/api/system/api-contracts") in route_keys
    assert ("POST", "/api/runtime/events") in route_keys


def test_live_metric_invariants_allow_demo_window_with_visible_boundaries():
    metrics = build_live_metrics_surface(demo_runtime_events(), source="test")
    invariants = metrics["invariants"]

    assert invariants["action"] in {"allow", "review", "quarantine"}
    assert len(invariants["invariants"]) >= 5
    assert all({"id", "observed", "threshold", "passed", "severity"} <= set(row) for row in invariants["invariants"])


def test_live_metric_invariants_quarantine_concentrated_window():
    surface = {
        "event_count": 20,
        "metrics": {
            "top_label_share": 0.95,
            "language_entropy": 0.2,
            "genre_entropy": 0.3,
            "skip_rate": 0.61,
            "recommendation_volatility": 0.2,
        },
    }
    invariants = evaluate_live_metric_invariants(surface)

    assert invariants["action"] == "quarantine"
    failed_ids = {row["id"] for row in invariants["failed"]}
    assert "label_concentration_cap" in failed_ids
    assert "skip_rate_ceiling" in failed_ids
