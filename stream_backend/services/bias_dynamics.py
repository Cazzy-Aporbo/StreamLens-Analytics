from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return low
    return max(low, min(high, number))


def _average(values: list[float]) -> float:
    cleaned = [value for value in values if value is not None]
    if not cleaned:
        return 0.0
    return sum(cleaned) / len(cleaned)


def _ratio_label(raw: str) -> str:
    return str(raw or "").replace("_", " ").replace(">", "over ").replace("<", "under ")


def _score_label(value: float) -> str:
    return f"{round(_clamp(value) * 100):.0f}/100"


def _band(score: float) -> str:
    if score < 0.28:
        return "clear"
    if score < 0.45:
        return "watch"
    if score < 0.65:
        return "elevated"
    return "concentrated"


def _intersection_extremes(intersectionality: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for key, payload in (intersectionality or {}).items():
        try:
            ratio = float((payload or {}).get("ratio", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "label": _ratio_label(key),
                "ratio": ratio,
                "representation": float((payload or {}).get("representation", 0.0) or 0.0),
                "screen_time": float((payload or {}).get("screen_time", 0.0) or 0.0),
            }
        )
    if not rows:
        return {"overrepresented": None, "underrepresented": None}
    ordered = sorted(rows, key=lambda item: item["ratio"])
    return {
        "underrepresented": ordered[0],
        "overrepresented": ordered[-1],
    }


def build_bias_dynamics_surface(
    *,
    representation_results: Mapping[str, Any],
    music_report: Mapping[str, Any],
    generated_at: str | None = None,
) -> dict[str, Any]:
    overall = dict(representation_results.get("overall_metrics") or {})
    network = dict(representation_results.get("network_metrics") or {})
    advanced = dict(representation_results.get("advanced_metrics") or {})
    effect_sizes = dict(advanced.get("effect_sizes") or {})

    diversity_index = _clamp(overall.get("diversity_index", 0.0))
    gender_parity = _clamp(overall.get("gender_parity", 0.0))
    representation_balance = _average([diversity_index, gender_parity])

    gender_homophily = abs(float(network.get("gender_homophily", 0.0) or 0.0))
    racial_homophily = abs(float(network.get("racial_homophily", 0.0) or 0.0))
    effect_pressure = _average(
        [
            _clamp((payload or {}).get("cramers_v", 0.0))
            for payload in effect_sizes.values()
            if isinstance(payload, Mapping)
        ]
    )
    structural_isolation = _clamp((effect_pressure * 0.68) + (_average([gender_homophily, racial_homophily]) * 0.32))

    decision_lab = dict(music_report.get("decision_lab") or {})
    drift = dict(decision_lab.get("drift") or {})
    trust = dict(decision_lab.get("trust") or {})
    living_media = dict(music_report.get("living_media") or {})
    summary_index = dict(living_media.get("summary_index") or {})
    bias_field = list(living_media.get("bias_field") or [])

    trust_score = _clamp(trust.get("score", 0.0))
    engineered_pressure = _clamp(summary_index.get("engineered_pressure", 0.0))
    corridor_balance = _clamp(summary_index.get("corridor_balance", 0.5))
    bias_resistance = _clamp(summary_index.get("bias_resistance", 0.5))

    drift_highlights = list(drift.get("highlights") or [])
    strongest_drift = max(
        drift_highlights,
        key=lambda item: float(item.get("salience", 0.0) or 0.0),
        default={},
    )
    drift_peak = _clamp(float(strongest_drift.get("salience", 0.0) or 0.0) / 100.0)
    reinforcement_pressure = _clamp(
        (engineered_pressure * 0.48)
        + ((1 - corridor_balance) * 0.27)
        + ((1 - bias_resistance) * 0.15)
        + (drift_peak * 0.10)
    )

    bias_score = _clamp(
        ((1 - representation_balance) * 0.23)
        + (structural_isolation * 0.34)
        + (reinforcement_pressure * 0.31)
        + ((1 - trust_score) * 0.12)
    )
    posture_band = _band(bias_score)

    field_signal = max(
        bias_field,
        key=lambda item: float(item.get("value", 0.0) or 0.0),
        default={},
    )
    extremes = _intersection_extremes(dict(overall.get("intersectionality") or {}))

    headline = "Bias is easier to miss when counts look tidy."
    summary = (
        "This surface reads bias as movement: who stays central, which corridors narrow, and whether the public lane is covered well enough to support a claim."
    )
    if posture_band == "elevated":
        summary = (
            "The field is still mixed on the surface, yet the structural signals are pulling toward narrower routes than the topline counts suggest."
        )
    elif posture_band == "concentrated":
        summary = (
            "Several signals are converging: representation is uneven, narrative paths are sticky, and the public attention lane is tightening around a small corridor."
        )

    dimensions = [
        {
            "id": "representation_balance",
            "label": "Representation balance",
            "value": round(representation_balance, 4),
            "value_label": _score_label(representation_balance),
            "reading": "Counts still matter. They simply do not settle the structural question on their own.",
            "evidence": "Diversity index and parity read.",
        },
        {
            "id": "structural_isolation",
            "label": "Structural isolation",
            "value": round(structural_isolation, 4),
            "value_label": _score_label(structural_isolation),
            "reading": "Homophily and dialogue effect sizes show whether groups remain in narrow narrative corridors.",
            "evidence": "Network homophily and Cramer's V effect sizes.",
        },
        {
            "id": "reinforcement_pressure",
            "label": "Reinforcement pressure",
            "value": round(reinforcement_pressure, 4),
            "value_label": _score_label(reinforcement_pressure),
            "reading": "Public attention drift shows whether discovery is opening or repeatedly returning to the same route.",
            "evidence": "Decision-lab drift and living-media pressure signals.",
        },
        {
            "id": "evidence_discipline",
            "label": "Evidence discipline",
            "value": round(trust_score, 4),
            "value_label": _score_label(trust_score),
            "reading": "Coverage matters. A careful weak claim is worth more than a confident overreach.",
            "evidence": "Trust score, year coverage, and source balance.",
        },
    ]

    watchpoints = [
        (
            f"Strongest visible drift: {strongest_drift.get('label', 'no major drift recorded')} "
            f"({strongest_drift.get('delta_label', 'stable')})."
            if strongest_drift
            else "No standout drift window is available yet."
        ),
        (
            f"Most overrepresented intersection in the modeled lane: {extremes['overrepresented']['label']} "
            f"at {extremes['overrepresented']['ratio']:.2f}x baseline."
            if extremes["overrepresented"]
            else "Intersectional extremes are not populated yet."
        ),
        (
            f"Most constrained field signal: {field_signal.get('label', 'field signal pending')} "
            f"({field_signal.get('value_label', '—')})."
            if field_signal
            else "Living-media field signals are still loading."
        ),
    ]

    questions = list(trust.get("questions") or [])
    if not questions:
        questions = [
            "Would this pattern survive if the thinner cohorts were removed from the claim?",
            "Is the system broadening discovery, or simply repeating what scale has already made easy to find?",
            "What changes first when the field begins to narrow: counts, structure, or attention flow?",
        ]

    contribution_paths = [
        "Bring in richer note or score coverage so public attention can be read alongside musical structure.",
        "Extend the drift lane with longer rolling windows before calling a shift durable.",
        "Add more real platform cohorts so structural isolation can be compared across contexts, not only within one field.",
    ]
    method_choices = [
        {
            "measure": "Homophily plus dialogue effect sizes",
            "why": "Counts can look balanced while roles still stay segregated. Structural measures reveal whether people are being kept in narrower narrative lanes.",
        },
        {
            "measure": "Public drift windows",
            "why": "A static snapshot cannot show whether discovery is opening or tightening. Drift windows make reinforcement pressure visible through time.",
        },
        {
            "measure": "Trust-weighted interpretation",
            "why": "Coverage limits are part of the result. A strong-looking signal should carry less force when year, language, or source support is thin.",
        },
    ]
    claim_boundaries = [
        "This surface joins a modeled representation lane with a public music lane. It is meant to show pressure and direction, not to flatten both into one dataset.",
        "It does not infer private platform intent. It reads the structural signals that are already visible in the available evidence.",
        "Where coverage is thin, the score should open a question rather than close an argument.",
    ]

    return {
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "headline": headline,
        "posture_band": posture_band,
        "bias_score": round(bias_score, 4),
        "summary": summary,
        "dimensions": dimensions,
        "watchpoints": watchpoints,
        "questions": questions[:4],
        "contribution_paths": contribution_paths,
        "method_choices": method_choices,
        "claim_boundaries": claim_boundaries,
        "extremes": extremes,
        "signals": {
            "diversity_index": round(diversity_index, 4),
            "gender_parity": round(gender_parity, 4),
            "network_density": round(_clamp(network.get("density", 0.0)), 4),
            "gender_homophily": round(gender_homophily, 4),
            "racial_homophily": round(racial_homophily, 4),
            "effect_pressure": round(effect_pressure, 4),
            "trust_score": round(trust_score, 4),
            "drift_peak": round(drift_peak, 4),
            "engineered_pressure": round(engineered_pressure, 4),
            "corridor_balance": round(corridor_balance, 4),
            "bias_resistance": round(bias_resistance, 4),
        },
        "proof_lanes": [
            "/api/metrics/network",
            "/api/metrics/advanced",
            "/api/music/decision-lab",
            "/api/music/living-media",
        ],
    }
