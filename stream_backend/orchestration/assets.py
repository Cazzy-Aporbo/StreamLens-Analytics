from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from stream_backend.models import RuntimeArtifact


def build_runtime_artifacts(
    base_dir: Path,
    frontend_state: Mapping[str, Any],
    representation: Mapping[str, Any],
    music_index: Mapping[str, Any],
) -> list[RuntimeArtifact]:
    system_dir = base_dir / "data" / "system"
    return [
        RuntimeArtifact(
            name="frontend_state",
            kind="json",
            path=str((system_dir / "frontend-state.json").relative_to(base_dir)),
            freshness_label=frontend_state.get("generated_at", ""),
            row_count=0,
            note="Generated UI state for the public dashboard.",
        ),
        RuntimeArtifact(
            name="critical_spine",
            kind="json",
            path=str((system_dir / "critical-spine.json").relative_to(base_dir)),
            freshness_label=frontend_state.get("generated_at", ""),
            row_count=0,
            note="Per-visual guidance explaining purpose, limits, and next checks.",
        ),
        RuntimeArtifact(
            name="streaming_readiness",
            kind="json",
            path=str((system_dir / "streaming-readiness.json").relative_to(base_dir)),
            freshness_label=frontend_state.get("generated_at", ""),
            row_count=0,
            note="Current streaming posture, operating boundary, and missing guarantees.",
        ),
        RuntimeArtifact(
            name="governance_surface",
            kind="json",
            path=str((system_dir / "governance.json").relative_to(base_dir)),
            freshness_label=frontend_state.get("generated_at", ""),
            row_count=0,
            note="Public governance contract for fairness, review posture, and claim boundaries.",
        ),
        RuntimeArtifact(
            name="readiness_surface",
            kind="json",
            path=str((system_dir / "readiness.json").relative_to(base_dir)),
            freshness_label=frontend_state.get("generated_at", ""),
            row_count=0,
            note="Operational readiness surface for onboarding, bounded commercial use, and contribution paths.",
        ),
        RuntimeArtifact(
            name="integration_surface",
            kind="json",
            path=str((system_dir / "integrations.json").relative_to(base_dir)),
            freshness_label=frontend_state.get("generated_at", ""),
            row_count=0,
            note="Actual integration paths the public repo supports today.",
        ),
        RuntimeArtifact(
            name="bias_dynamics_surface",
            kind="json",
            path=str((system_dir / "bias-dynamics.json").relative_to(base_dir)),
            freshness_label=frontend_state.get("generated_at", ""),
            row_count=0,
            note="Structural bias surface combining representation balance, network isolation, and public drift signals.",
        ),
        RuntimeArtifact(
            name="live_contract_surface",
            kind="json",
            path=str((system_dir / "live-contract.json").relative_to(base_dir)),
            freshness_label=frontend_state.get("generated_at", ""),
            row_count=0,
            note="Ingress contract, query filters, and retention posture for the local event window.",
        ),
        RuntimeArtifact(
            name="live_metrics_surface",
            kind="json",
            path=str((system_dir / "live-metrics.json").relative_to(base_dir)),
            freshness_label=frontend_state.get("generated_at", ""),
            row_count=0,
            note="Rolling event-window metrics for the active market surface.",
        ),
        RuntimeArtifact(
            name="runtime_review_surface",
            kind="json",
            path=str((system_dir / "runtime-review.json").relative_to(base_dir)),
            freshness_label=frontend_state.get("generated_at", ""),
            row_count=0,
            note="Prompt-plus-draft review example for the public guard lane.",
        ),
        RuntimeArtifact(
            name="runtime_drift_surface",
            kind="json",
            path=str((system_dir / "runtime-drift.json").relative_to(base_dir)),
            freshness_label=frontend_state.get("generated_at", ""),
            row_count=0,
            note="Text-path drift probe for repeated output windows.",
        ),
        RuntimeArtifact(
            name="media_insurability_surface",
            kind="json",
            path=str((system_dir / "media-insurability.json").relative_to(base_dir)),
            freshness_label=frontend_state.get("generated_at", ""),
            row_count=0,
            note="Bounded liability translation for the public media lab.",
        ),
        RuntimeArtifact(
            name="runtime_surface",
            kind="markdown",
            path=str((system_dir / "runtime_surface.md").relative_to(base_dir)),
            freshness_label=frontend_state.get("generated_at", ""),
            row_count=0,
            note="Human-readable runtime summary.",
        ),
        RuntimeArtifact(
            name="representation_results",
            kind="json",
            path="analysis_results.json",
            freshness_label=frontend_state.get("generated_at", ""),
            row_count=int(representation.get("row_count", 0)),
            note="Synthetic representation export used by the API and static build.",
        ),
        RuntimeArtifact(
            name="music_index",
            kind="json",
            path="music_index.json",
            freshness_label=frontend_state.get("generated_at", ""),
            row_count=int(music_index.get("summary", {}).get("catalog_song_count", 0)),
            note="Catalog + notation export for the music intelligence lane.",
        ),
    ]
