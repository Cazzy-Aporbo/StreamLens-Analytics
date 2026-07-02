from __future__ import annotations

from typing import Any, Mapping


def _metric(metrics: Mapping[str, Any], key: str) -> float:
    try:
        return float(metrics.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _row(
    *,
    invariant_id: str,
    label: str,
    observed: float,
    operator: str,
    threshold: float,
    severity: str,
    passed: bool,
    note: str,
) -> dict[str, Any]:
    return {
        "id": invariant_id,
        "label": label,
        "observed": round(observed, 4),
        "operator": operator,
        "threshold": threshold,
        "severity": severity,
        "passed": bool(passed),
        "note": note,
    }


def evaluate_live_metric_invariants(metrics_surface: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate deterministic review boundaries over a live metric surface."""
    metrics = metrics_surface.get("metrics") or {}
    event_count = int(metrics_surface.get("event_count") or 0)
    rows = [
        _row(
            invariant_id="minimum_window_size",
            label="Minimum event window",
            observed=float(event_count),
            operator=">=",
            threshold=10,
            severity="review",
            passed=event_count >= 10,
            note="Small windows can be inspected, but should not carry broad market claims.",
        ),
        _row(
            invariant_id="label_concentration_cap",
            label="Label concentration cap",
            observed=_metric(metrics, "top_label_share"),
            operator="<=",
            threshold=0.72,
            severity="quarantine",
            passed=_metric(metrics, "top_label_share") <= 0.72,
            note="A single label tier should not dominate the exposure window without review.",
        ),
        _row(
            invariant_id="language_balance_floor",
            label="Language balance floor",
            observed=_metric(metrics, "language_entropy"),
            operator=">=",
            threshold=0.45,
            severity="review",
            passed=_metric(metrics, "language_entropy") >= 0.45,
            note="A narrow language corridor changes the meaning of discovery breadth.",
        ),
        _row(
            invariant_id="genre_breadth_floor",
            label="Genre breadth floor",
            observed=_metric(metrics, "genre_entropy"),
            operator=">=",
            threshold=0.5,
            severity="review",
            passed=_metric(metrics, "genre_entropy") >= 0.5,
            note="Low genre entropy can hide a corridor behind a large event count.",
        ),
        _row(
            invariant_id="skip_rate_ceiling",
            label="Skip-rate ceiling",
            observed=_metric(metrics, "skip_rate"),
            operator="<=",
            threshold=0.42,
            severity="quarantine",
            passed=_metric(metrics, "skip_rate") <= 0.42,
            note="High skip pressure should stop automated positive claims about fit.",
        ),
        _row(
            invariant_id="recommendation_volatility_ceiling",
            label="Recommendation volatility ceiling",
            observed=_metric(metrics, "recommendation_volatility"),
            operator="<=",
            threshold=0.16,
            severity="review",
            passed=_metric(metrics, "recommendation_volatility") <= 0.16,
            note="Volatile recommendation share needs a second look before it is treated as stable signal.",
        ),
    ]

    failed = [row for row in rows if not row["passed"]]
    has_quarantine = any(row["severity"] == "quarantine" for row in failed)
    has_review = bool(failed)
    action = "allow"
    if has_review:
        action = "review"
    if has_quarantine:
        action = "quarantine"

    return {
        "action": action,
        "passed": not failed,
        "failed_count": len(failed),
        "invariants": rows,
        "failed": failed,
        "policy": {
            "allow": "No invariant failed.",
            "review": "At least one soft boundary failed. Publish with caveat or inspect the window.",
            "quarantine": "A hard boundary failed. Do not automate the downstream claim.",
        },
    }
