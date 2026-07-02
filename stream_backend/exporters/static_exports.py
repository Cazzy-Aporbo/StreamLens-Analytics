from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_runtime_json_exports(base_dir: Path, snapshot_payload: Mapping[str, Any]) -> list[Path]:
    system_dir = base_dir / "data" / "system"
    outputs = {
        system_dir / "frontend-state.json": snapshot_payload["frontend_state"],
        system_dir / "governance.json": snapshot_payload["frontend_state"]["governance"],
        system_dir / "readiness.json": snapshot_payload.get("readiness", {}),
        system_dir / "integrations.json": snapshot_payload.get("integrations", {}),
        system_dir / "bias-dynamics.json": snapshot_payload.get("bias_dynamics", {}),
        system_dir / "live-contract.json": snapshot_payload.get("live_contract", {}),
        system_dir / "live-metrics.json": snapshot_payload.get("live_metrics", {}),
        system_dir / "runtime-review.json": snapshot_payload.get("runtime_review", {}),
        system_dir / "runtime-drift.json": snapshot_payload.get("runtime_drift", {}),
        system_dir / "media-insurability.json": snapshot_payload.get("media_insurability", {}),
        system_dir / "critical-spine.json": snapshot_payload["critical_spine"],
        system_dir / "comparatives.json": snapshot_payload["comparatives"],
        system_dir / "runtime.json": snapshot_payload,
        system_dir / "orchestration.json": snapshot_payload["orchestration"],
        system_dir / "contracts.json": snapshot_payload["data_engineering"],
    }
    written: list[Path] = []
    for path, payload in outputs.items():
        _write_json(path, payload)
        written.append(path)
    return written
