"""Tests for the lightweight runtime drift surface."""

from runtime_drift import analyze_runtime_drift, public_runtime_drift_snapshot, sample_runtime_turns


def test_runtime_drift_analysis_returns_bounded_score():
    report = analyze_runtime_drift(sample_runtime_turns())
    assert 0 <= report["drift_score"] <= 1
    assert report["drift_band"] in {"clear", "watch", "elevated", "severe"}
    assert report["signals"]["turn_count"] == len(sample_runtime_turns())


def test_runtime_drift_snapshot_contains_turns_and_analysis():
    snapshot = public_runtime_drift_snapshot()
    assert len(snapshot["turns"]) >= 4
    assert "analysis" in snapshot
