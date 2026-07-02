from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


def _utc_day() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _round(value: Any, digits: int = 4) -> float | None:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _top_features(predictability: Mapping[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    rows = list(predictability.get("features") or [])
    return [
        {
            "feature": row.get("feature"),
            "label": row.get("label"),
            "importance": _round(row.get("importance"), 4),
        }
        for row in rows[:limit]
    ]


def build_model_registry_surface(music_report: Mapping[str, Any]) -> dict[str, Any]:
    """Describe executable models and analytical methods without overstating scope."""
    predictability = music_report.get("predictability") or {}
    archetypes = music_report.get("archetypes") or {}
    power_law = music_report.get("power_law") or {}
    network = music_report.get("network") or {}
    quality = music_report.get("quality") or {}
    model_rigor = quality.get("model_rigor") or {}
    decision_lab = music_report.get("decision_lab") or {}

    clusters = archetypes.get("clusters") or []
    strongest_experiment = ((decision_lab.get("experiments") or {}).get("strongest") or {})

    return {
        "generated_at": _utc_day(),
        "scope": {
            "public_repo_only": True,
            "private_loopchii_runtime": False,
            "purpose": (
                "Make the executable analytical methods visible enough for review, "
                "testing, and contribution."
            ),
        },
        "trained_models": [
            {
                "id": "music_view_predictability_ensemble",
                "status": "executed",
                "implementation": "music_pipeline.predictability_analysis",
                "model_family": "supervised regression",
                "estimators": ["RandomForestRegressor", "GradientBoostingRegressor"],
                "target": model_rigor.get("target", "log1p(view_count)"),
                "validation": {
                    "method": "5-fold cross-validation with out-of-fold ensemble prediction",
                    "folds": int(model_rigor.get("cv_folds") or 5),
                    "ensemble_r2": _round(predictability.get("ensemble_r2"), 3),
                    "per_model_r2": predictability.get("per_model_r2") or {},
                    "leakage_guard": model_rigor.get("blocked_from_prediction") or [],
                },
                "current_signal": {
                    "predictable_pct": _round(predictability.get("predictable_pct"), 1),
                    "luck_pct": _round(predictability.get("luck_pct"), 1),
                    "top_features": _top_features(predictability),
                },
                "claim_boundary": (
                    "This estimates how much public upload metadata explains log views "
                    "inside the committed corpus. It does not predict culture by itself."
                ),
            },
            {
                "id": "music_release_archetypes",
                "status": "executed",
                "implementation": "music_pipeline.archetype_analysis",
                "model_family": "unsupervised clustering",
                "estimators": ["KMeans"],
                "validation": {
                    "cluster_count": int(archetypes.get("k") or len(clusters) or 0),
                    "seeded": True,
                    "feature_space": [
                        "view_count",
                        "virality_coefficient",
                        "channel_follower_count",
                        "duration_min",
                        "tag_count",
                    ],
                },
                "current_signal": {
                    "clusters": [
                        {
                            "name": row.get("name"),
                            "count": int(row.get("count") or 0),
                            "avg_views": _round(row.get("avg_views"), 2),
                        }
                        for row in clusters
                    ],
                },
                "claim_boundary": (
                    "Clusters are descriptive release patterns. They are not artist labels "
                    "and should not be used as identity judgments."
                ),
            },
        ],
        "statistical_methods": [
            {
                "id": "attention_power_law",
                "status": "executed",
                "implementation": "music_pipeline.power_law_analysis",
                "method": "maximum-likelihood tail fit with KS distance and bootstrap interval",
                "current_signal": {
                    "alpha": _round(power_law.get("alpha"), 3),
                    "xmin": _round(power_law.get("xmin"), 2),
                    "ks_distance": _round(power_law.get("ks_distance"), 4),
                    "bootstrap_resamples": int(model_rigor.get("bootstrap_power_law") or 500),
                    "alpha_ci": power_law.get("alpha_ci") or {},
                },
                "why_it_matters": (
                    "A concentrated attention field changes what fairness and discovery "
                    "mean. It shows whether a small set of releases is carrying most of "
                    "the public signal."
                ),
            },
            {
                "id": "tag_network_pressure",
                "status": "executed",
                "implementation": "music_pipeline.network_analysis",
                "method": "co-occurrence graph with weighted edges and community labels",
                "current_signal": {
                    "nodes": int(network.get("n_nodes") or len(network.get("nodes") or [])),
                    "edges": int(network.get("n_edges") or len(network.get("edges") or [])),
                    "density": _round(network.get("density"), 4),
                },
                "why_it_matters": (
                    "Discovery is not only a song-level question. Tags, labels, language, "
                    "and scenes form routes that can widen or narrow cultural movement."
                ),
            },
            {
                "id": "controlled_decision_lab",
                "status": "executed",
                "implementation": "music_decision_lab.build_decision_lab",
                "method": "cohort drift, controlled comparison, and counterfactual pressure checks",
                "current_signal": {
                    "strongest_experiment": strongest_experiment.get("label"),
                    "effect": strongest_experiment.get("effect_label"),
                },
                "why_it_matters": (
                    "It helps separate a visible chart from a decision someone might make "
                    "with it. Weak evidence stays weak."
                ),
            },
        ],
        "deterministic_probes": [
            {
                "id": "runtime_drift_probe",
                "status": "executed",
                "implementation": "runtime_drift.analyze_runtime_drift",
                "method": "text-window checks for repetition, leakage markers, and unstable refusal behavior",
                "claim_boundary": "This is a local text-path probe, not hidden model telemetry.",
            },
            {
                "id": "local_event_window",
                "status": "executed",
                "implementation": "stream_backend.services.live_runtime",
                "method": "bounded event contract, SQLite persistence, filters, retention, and rolling metrics",
                "claim_boundary": "Useful for local inspection and contract design, not a distributed event bus by itself.",
            },
        ],
        "compiled_kernels": [
            {
                "id": "loopchii_wasm_metric_primitives",
                "status": "tested",
                "implementation": "loopchii-wasm-core/src/lib.rs",
                "language": "Rust",
                "methods": [
                    "shannon_entropy",
                    "normalized_entropy",
                    "concentration_hhi",
                    "top_share",
                    "weighted_mean",
                    "chi_square_gof",
                ],
                "verification": "cargo test --manifest-path loopchii-wasm-core/Cargo.toml",
                "claim_boundary": "Compiled primitives are available, but the full pipeline still runs through Python today.",
            }
        ],
        "not_claimed": [
            "production stream processing",
            "exactly-once delivery",
            "hosted compliance automation",
            "non-public runtime behavior",
            "cultural causality from metadata alone",
        ],
        "proof_commands": [
            "python3 run_analysis.py --output-dir exports",
            "python3 -m stream_backend.cli.doctor",
            "python3 -m pytest -q tests",
            "cargo test --manifest-path loopchii-wasm-core/Cargo.toml",
        ],
    }
