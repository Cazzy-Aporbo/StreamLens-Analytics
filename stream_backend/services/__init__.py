"""Service facades for app.py, build_static.py, and CLI entrypoints."""

from .bias_dynamics import build_bias_dynamics_surface
from .catalog import build_runtime_catalog
from .enterprise_bus import EnterpriseBusConfig, publish_runtime_batch
from .frontend import build_frontend_state_payload
from .integrations import build_integration_surface
from .live_runtime import (
    build_live_metrics_surface,
    demo_runtime_events,
    load_live_metrics,
    normalize_runtime_filters,
    runtime_event_contract,
)
from .model_registry import build_model_registry_surface
from .readiness import build_readiness_surface
from .runtime_invariants import evaluate_live_metric_invariants
from .runtime import StreamRuntime
from .snapshots import load_latest_runtime_snapshot

__all__ = [
    "build_bias_dynamics_surface",
    "build_frontend_state_payload",
    "build_integration_surface",
    "build_live_metrics_surface",
    "build_model_registry_surface",
    "build_runtime_catalog",
    "build_readiness_surface",
    "demo_runtime_events",
    "EnterpriseBusConfig",
    "load_latest_runtime_snapshot",
    "load_live_metrics",
    "normalize_runtime_filters",
    "publish_runtime_batch",
    "runtime_event_contract",
    "evaluate_live_metric_invariants",
    "StreamRuntime",
]
