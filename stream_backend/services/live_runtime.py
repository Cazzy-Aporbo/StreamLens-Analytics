from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import pstdev
from typing import Any, Mapping

from stream_backend.config import RuntimeConfig
from stream_backend.store import (
    connect_sqlite,
    count_events,
    ensure_schema,
    insert_events,
    list_events,
    prune_events,
)
from stream_backend.utils import stable_hash
from stream_backend.services.runtime_invariants import evaluate_live_metric_invariants


FILTER_KEYS = ("platform", "market", "genre", "language", "label_tier", "source")


def _resolve_runtime_config(sqlite_path: str | Path, config: RuntimeConfig | None) -> RuntimeConfig:
    if config is not None:
        return config
    path = Path(sqlite_path).resolve()
    parents = list(path.parents)
    if len(parents) >= 4:
        return RuntimeConfig.from_base_dir(parents[3])
    return RuntimeConfig.from_base_dir(path.parent)


def normalize_runtime_filters(filters: Mapping[str, Any] | None = None) -> dict[str, str]:
    normalized: dict[str, str] = {}
    if not filters:
        return normalized
    for key in FILTER_KEYS:
        value = str(filters.get(key) or "").strip()
        if value:
            normalized[key] = value
    return normalized


def runtime_event_contract(config: RuntimeConfig) -> dict[str, Any]:
    sample = demo_runtime_events()[0]
    return {
        "headline": "Local ingress contract for the runtime event window.",
        "summary": "The public event lane accepts bounded batches, keeps duplicate ids from inflating the window, and prunes the oldest rows before the table grows quietly out of proportion.",
        "batch_limits": {
            "min_events": 1,
            "max_events": config.event_ingest_batch_limit,
        },
        "retention": {
            "max_rows": config.event_retention_max_rows,
            "max_age_days": config.event_retention_days,
        },
        "filters": [
            {
                "name": key,
                "note": f"Optional query filter for {key.replace('_', ' ')}-specific window reads.",
            }
            for key in FILTER_KEYS
        ],
        "fields": [
            {"name": "event_id", "type": "string", "required": True, "note": "Stable unique identifier for idempotent ingress."},
            {"name": "observed_at", "type": "iso8601 datetime", "required": True, "note": "Observed event time used for ordering and retention."},
            {"name": "platform", "type": "string", "required": True, "note": "Source platform for the event."},
            {"name": "market", "type": "string", "required": True, "note": "Market or territory code for the event."},
            {"name": "track_id", "type": "string", "required": True, "note": "Track identifier within the source lane."},
            {"name": "track_title", "type": "string", "required": True, "note": "Readable title for inspection."},
            {"name": "artist_id", "type": "string", "required": True, "note": "Stable artist identifier."},
            {"name": "artist_name", "type": "string", "required": True, "note": "Readable artist name."},
            {"name": "genre", "type": "string", "required": True, "note": "Genre label used in weighted breadth reads."},
            {"name": "language", "type": "string", "required": True, "note": "Language label for cross-market concentration reads."},
            {"name": "label_tier", "type": "string", "required": True, "note": "Label grouping such as major or independent."},
            {"name": "independent_flag", "type": "boolean", "required": False, "note": "Marks independent participation in the current window."},
            {"name": "editorial_flag", "type": "boolean", "required": False, "note": "Marks editorial lift in the current window."},
            {"name": "exposure_share", "type": "float [0,1]", "required": True, "note": "Share of exposure carried by the event."},
            {"name": "recommendation_share", "type": "float [0,1]", "required": True, "note": "Share of recommendation placement carried by the event."},
            {"name": "completion_rate", "type": "float [0,1]", "required": True, "note": "Completion response for the event."},
            {"name": "skip_rate", "type": "float [0,1]", "required": True, "note": "Skip response for the event."},
            {"name": "save_rate", "type": "float [0,1]", "required": True, "note": "Save response for the event."},
            {"name": "source", "type": "string", "required": False, "note": "Ingress source label for later inspection."},
        ],
        "sample_event": sample,
    }


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return low
    return max(low, min(high, number))


def _weighted_mean(rows: list[Mapping[str, Any]], key: str, weight_key: str = "exposure_share") -> float:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        weight = max(0.0, float(row.get(weight_key, 0.0) or 0.0))
        numerator += float(row.get(key, 0.0) or 0.0) * weight
        denominator += weight
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _weighted_distribution(rows: list[Mapping[str, Any]], key: str, weight_key: str = "exposure_share") -> dict[str, float]:
    totals: defaultdict[str, float] = defaultdict(float)
    for row in rows:
        label = str(row.get(key) or "unknown").strip() or "unknown"
        totals[label] += max(0.0, float(row.get(weight_key, 0.0) or 0.0))
    grand_total = sum(totals.values())
    if grand_total <= 0:
        return {}
    return {label: value / grand_total for label, value in sorted(totals.items(), key=lambda item: (-item[1], item[0]))}


def _normalized_entropy(distribution: Mapping[str, float]) -> float:
    import math

    probs = [value for value in distribution.values() if value > 0]
    if len(probs) <= 1:
        return 0.0
    entropy = -sum(prob * math.log2(prob) for prob in probs)
    return entropy / math.log2(len(probs))


def _top_share(distribution: Mapping[str, float]) -> float:
    if not distribution:
        return 0.0
    return max(distribution.values())


def _hhi(distribution: Mapping[str, float]) -> float:
    return sum(value * value for value in distribution.values())


def _band(score: float) -> str:
    if score < 0.26:
        return "open"
    if score < 0.45:
        return "watch"
    if score < 0.62:
        return "narrow"
    return "compressed"


def _score_label(value: float) -> str:
    return f"{round(_clamp(value) * 100):.0f}/100"


def demo_runtime_events(now: datetime | None = None) -> list[dict[str, Any]]:
    anchor = now or datetime.now(timezone.utc)
    specs = [
        ("spotify", "US", "en", "pop", "major", 0, 1, 0.19, 0.22, 0.63, 0.21, 0.08, "Nova Bloom", "nova-bloom", "Glass Hearts", "glass-hearts"),
        ("spotify", "PH", "tl", "rnb", "independent", 1, 0, 0.12, 0.11, 0.71, 0.14, 0.10, "Luna Cielo", "luna-cielo", "Harbor Skin", "harbor-skin"),
        ("youtube", "BR", "pt", "funk", "independent", 1, 0, 0.09, 0.08, 0.76, 0.11, 0.09, "Mar Azul", "mar-azul", "Rua Clara", "rua-clara"),
        ("spotify", "GB", "en", "alternative", "major", 0, 1, 0.11, 0.12, 0.67, 0.18, 0.07, "North Field", "north-field", "Glass Harbor", "glass-harbor"),
        ("youtube", "MX", "es", "regional", "independent", 1, 0, 0.08, 0.07, 0.74, 0.13, 0.08, "Sol Dorado", "sol-dorado", "Carta Viva", "carta-viva"),
        ("spotify", "KR", "ko", "pop", "major", 0, 1, 0.14, 0.16, 0.61, 0.24, 0.06, "Violet Circuit", "violet-circuit", "Silver Noon", "silver-noon"),
        ("youtube", "NG", "en", "afrobeats", "independent", 1, 0, 0.07, 0.06, 0.79, 0.09, 0.11, "Kairo Sun", "kairo-sun", "River Wire", "river-wire"),
        ("spotify", "JP", "ja", "city-pop", "major", 0, 1, 0.06, 0.07, 0.73, 0.12, 0.09, "Sora Wren", "sora-wren", "Night Relay", "night-relay"),
        ("youtube", "FR", "fr", "electronic", "independent", 1, 0, 0.05, 0.04, 0.77, 0.10, 0.10, "Ciel Moderne", "ciel-moderne", "Pulse Verre", "pulse-verre"),
        ("spotify", "IN", "hi", "film", "major", 0, 1, 0.04, 0.03, 0.82, 0.08, 0.11, "Asha Vale", "asha-vale", "Monsoon Thread", "monsoon-thread"),
        ("youtube", "ZA", "zu", "gqom", "independent", 1, 0, 0.03, 0.02, 0.81, 0.07, 0.12, "Uma Tide", "uma-tide", "City Ember", "city-ember"),
        ("spotify", "CA", "en", "folk", "independent", 1, 0, 0.02, 0.02, 0.84, 0.05, 0.14, "Rowan Vale", "rowan-vale", "Mosslight", "mosslight"),
    ]
    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        (
            platform,
            market,
            language,
            genre,
            label_tier,
            independent_flag,
            editorial_flag,
            exposure_share,
            recommendation_share,
            completion_rate,
            skip_rate,
            save_rate,
            artist_name,
            artist_id,
            track_title,
            track_id,
        ) = spec
        observed_at = (anchor - timedelta(minutes=(len(specs) - index) * 4)).isoformat()
        base = {
            "event_id": f"demo-{track_id}-{index + 1}",
            "observed_at": observed_at,
            "platform": platform,
            "market": market,
            "track_id": track_id,
            "track_title": track_title,
            "artist_id": artist_id,
            "artist_name": artist_name,
            "genre": genre,
            "language": language,
            "label_tier": label_tier,
            "independent_flag": int(independent_flag),
            "editorial_flag": int(editorial_flag),
            "exposure_share": exposure_share,
            "recommendation_share": recommendation_share,
            "completion_rate": completion_rate,
            "skip_rate": skip_rate,
            "save_rate": save_rate,
            "source": "demo_runtime_seed",
        }
        normalized = normalize_runtime_event(base)
        rows.append(normalized)
    return rows


def normalize_runtime_event(event: Mapping[str, Any], *, ingested_at: str | None = None) -> dict[str, Any]:
    observed_at = str(event.get("observed_at") or datetime.now(timezone.utc).isoformat())
    normalized = {
        "event_id": str(event.get("event_id") or ""),
        "observed_at": observed_at,
        "ingested_at": ingested_at or datetime.now(timezone.utc).isoformat(),
        "platform": str(event.get("platform") or "unknown"),
        "market": str(event.get("market") or "global"),
        "track_id": str(event.get("track_id") or ""),
        "track_title": str(event.get("track_title") or "untitled"),
        "artist_id": str(event.get("artist_id") or ""),
        "artist_name": str(event.get("artist_name") or "unknown artist"),
        "genre": str(event.get("genre") or "unknown"),
        "language": str(event.get("language") or "unknown"),
        "label_tier": str(event.get("label_tier") or "unknown"),
        "independent_flag": 1 if bool(event.get("independent_flag")) else 0,
        "editorial_flag": 1 if bool(event.get("editorial_flag")) else 0,
        "exposure_share": round(_clamp(event.get("exposure_share", 0.0)), 6),
        "recommendation_share": round(_clamp(event.get("recommendation_share", 0.0)), 6),
        "completion_rate": round(_clamp(event.get("completion_rate", 0.0)), 6),
        "skip_rate": round(_clamp(event.get("skip_rate", 0.0)), 6),
        "save_rate": round(_clamp(event.get("save_rate", 0.0)), 6),
        "source": str(event.get("source") or "api_ingest"),
    }
    normalized["payload_hash"] = stable_hash(
        {
            key: normalized[key]
            for key in normalized
            if key not in {"ingested_at", "payload_hash"}
        }
    )
    return normalized


def ingest_runtime_events(
    sqlite_path,
    events: list[Mapping[str, Any]],
    *,
    config: RuntimeConfig | None = None,
) -> dict[str, Any]:
    normalized = [normalize_runtime_event(event) for event in events]
    runtime_config = _resolve_runtime_config(sqlite_path, config)
    conn = connect_sqlite(sqlite_path)
    try:
        ensure_schema(conn)
        result = insert_events(conn, normalized)
        pruning = prune_events(
            conn,
            max_rows=runtime_config.event_retention_max_rows,
            max_age_days=runtime_config.event_retention_days,
        )
        return {
            "inserted": result["inserted"],
            "skipped": result["skipped"],
            "pruned": pruning["deleted"],
            "event_count": pruning["remaining"],
        }
    finally:
        conn.close()


def list_runtime_events(
    sqlite_path,
    *,
    limit: int = 25,
    filters: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    conn = connect_sqlite(sqlite_path)
    try:
        ensure_schema(conn)
        return list_events(conn, limit=limit, filters=normalize_runtime_filters(filters))
    finally:
        conn.close()


def build_live_metrics_surface(
    rows: list[Mapping[str, Any]],
    *,
    source: str,
    generated_at: str | None = None,
    filters: Mapping[str, Any] | None = None,
    retention: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    distribution_genre = _weighted_distribution(rows, "genre")
    distribution_language = _weighted_distribution(rows, "language")
    distribution_label = _weighted_distribution(rows, "label_tier")
    distribution_market = _weighted_distribution(rows, "market")

    genre_entropy = _normalized_entropy(distribution_genre)
    language_entropy = _normalized_entropy(distribution_language)
    label_hhi = _hhi(distribution_label)
    top_language_share = _top_share(distribution_language)
    top_label_share = _top_share(distribution_label)
    independent_share = _weighted_mean(rows, "independent_flag")
    editorial_share = _weighted_mean(rows, "editorial_flag")
    completion_rate = _weighted_mean(rows, "completion_rate")
    skip_rate = _weighted_mean(rows, "skip_rate")
    save_rate = _weighted_mean(rows, "save_rate")
    recommendation_volatility = pstdev(
        [float(row.get("recommendation_share", 0.0) or 0.0) for row in rows]
    ) if len(rows) > 1 else 0.0

    pressure_score = _clamp(
        ((1 - genre_entropy) * 0.24)
        + ((1 - language_entropy) * 0.14)
        + (top_label_share * 0.18)
        + (editorial_share * 0.14)
        + ((1 - independent_share) * 0.15)
        + (skip_rate * 0.08)
        + (_clamp(recommendation_volatility * 6) * 0.07)
    )
    pressure_band = _band(pressure_score)

    most_recent = max(rows, key=lambda row: str(row.get("observed_at") or ""), default={})
    oldest = min(rows, key=lambda row: str(row.get("observed_at") or ""), default={})
    top_market = next(iter(distribution_market.items()), ("unknown", 0.0))
    top_label = next(iter(distribution_label.items()), ("unknown", 0.0))
    top_language = next(iter(distribution_language.items()), ("unknown", 0.0))
    top_genre = next(iter(distribution_genre.items()), ("unknown", 0.0))

    summary = (
        "This window joins exposure, recommendation share, and listener response so the stream can be read as a living operating surface rather than a static report."
    )
    if pressure_band in {"narrow", "compressed"}:
        summary = (
            "The current window is leaning into a smaller corridor than its topline variety suggests. Label weight, editorial lift, and response patterns should be read together."
        )

    watchpoints = [
        f"Top language corridor: {top_language[0]} at {round(top_language[1] * 100):.0f}% of weighted exposure.",
        f"Top label share: {top_label[0]} at {round(top_label[1] * 100):.0f}% of weighted exposure.",
        f"Independent exposure: {round(independent_share * 100):.0f}% with average save rate at {round(save_rate * 100):.0f}%.",
    ]

    axes = [
        {
            "id": "discovery_breadth",
            "label": "Discovery breadth",
            "value": round(genre_entropy, 4),
            "value_label": _score_label(genre_entropy),
            "note": "Weighted genre entropy across the current event window.",
        },
        {
            "id": "language_balance",
            "label": "Language balance",
            "value": round(language_entropy, 4),
            "value_label": _score_label(language_entropy),
            "note": "A broader language spread usually leaves more room for cross-market discovery.",
        },
        {
            "id": "independent_lift",
            "label": "Independent lift",
            "value": round(independent_share, 4),
            "value_label": _score_label(independent_share),
            "note": "Weighted exposure going to independent releases in the current window.",
        },
        {
            "id": "audience_followthrough",
            "label": "Audience follow-through",
            "value": round(completion_rate, 4),
            "value_label": _score_label(completion_rate),
            "note": "Completion is carried as a weighted average rather than a raw mean so larger exposure lanes count proportionally.",
        },
    ]

    method_choices = [
        {
            "measure": "Exposure-weighted distributions",
            "why": "Raw event counts flatten the field. Weighted shares preserve the actual force each item had in the stream.",
        },
        {
            "measure": "Entropy plus concentration",
            "why": "Breadth and dominance answer different questions. Both are needed before a market read becomes persuasive.",
        },
        {
            "measure": "Idempotent event ingress",
            "why": "Duplicate events should not quietly inflate the window. Stable event ids keep the live surface reproducible.",
        },
    ]

    checks_and_balances = [
        "If the live table is empty, the public surface falls back to a clearly marked demonstration window rather than fabricating live certainty.",
        "The current window is weighted by exposure share, not by whichever cohort generated the most rows.",
        "The result is a live diagnostic surface. It should guide review and intervention, not replace judgment about cause or motive.",
    ]

    payload = {
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "source": source,
        "event_count": len(rows),
        "filters": normalize_runtime_filters(filters),
        "window": {
            "start": oldest.get("observed_at"),
            "end": most_recent.get("observed_at"),
        },
        "headline": "A live operating window for market pressure.",
        "pressure_band": pressure_band,
        "pressure_score": round(pressure_score, 4),
        "summary": summary,
        "axes": axes,
        "metrics": {
            "unique_tracks": len({str(row.get("track_id") or "") for row in rows}),
            "unique_artists": len({str(row.get("artist_id") or "") for row in rows}),
            "genre_entropy": round(genre_entropy, 4),
            "language_entropy": round(language_entropy, 4),
            "label_hhi": round(label_hhi, 4),
            "top_language_share": round(top_language_share, 4),
            "top_label_share": round(top_label_share, 4),
            "independent_share": round(independent_share, 4),
            "editorial_share": round(editorial_share, 4),
            "completion_rate": round(completion_rate, 4),
            "skip_rate": round(skip_rate, 4),
            "save_rate": round(save_rate, 4),
            "recommendation_volatility": round(recommendation_volatility, 4),
        },
        "distributions": {
            "genres": [{"label": key, "share": round(value, 4)} for key, value in list(distribution_genre.items())[:6]],
            "languages": [{"label": key, "share": round(value, 4)} for key, value in list(distribution_language.items())[:6]],
            "labels": [{"label": key, "share": round(value, 4)} for key, value in list(distribution_label.items())[:6]],
            "markets": [{"label": key, "share": round(value, 4)} for key, value in list(distribution_market.items())[:6]],
        },
        "watchpoints": watchpoints,
        "method_choices": method_choices,
        "checks_and_balances": checks_and_balances,
        "retention": retention or {},
        "recent_events": rows[:8],
        "signals": {
            "top_market": {"label": top_market[0], "share": round(top_market[1], 4)},
            "top_genre": {"label": top_genre[0], "share": round(top_genre[1], 4)},
            "top_language": {"label": top_language[0], "share": round(top_language[1], 4)},
            "top_label": {"label": top_label[0], "share": round(top_label[1], 4)},
        },
    }
    payload["invariants"] = evaluate_live_metric_invariants(payload)
    return payload


def load_live_metrics(
    sqlite_path,
    *,
    limit: int = 250,
    filters: Mapping[str, Any] | None = None,
    config: RuntimeConfig | None = None,
) -> dict[str, Any]:
    runtime_config = _resolve_runtime_config(sqlite_path, config)
    conn = connect_sqlite(sqlite_path)
    try:
        ensure_schema(conn)
        normalized_filters = normalize_runtime_filters(filters)
        rows = list_events(conn, limit=limit, filters=normalized_filters)
        retention = {
            "max_rows": runtime_config.event_retention_max_rows,
            "max_age_days": runtime_config.event_retention_days,
            "matching_event_count": count_events(conn, filters=normalized_filters),
        }
        if rows:
            return build_live_metrics_surface(
                rows,
                source="live",
                filters=normalized_filters,
                retention=retention,
            )
        return build_live_metrics_surface(
            demo_runtime_events(),
            source="demonstration",
            filters=normalized_filters,
            retention=retention,
        )
    finally:
        conn.close()
