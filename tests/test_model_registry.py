from __future__ import annotations

from stream_backend.services.model_registry import build_model_registry_surface
import music_pipeline


def test_model_registry_names_executable_models_and_boundaries():
    registry = build_model_registry_surface(music_pipeline.full_report())

    trained_ids = {item["id"] for item in registry["trained_models"]}
    method_ids = {item["id"] for item in registry["statistical_methods"]}
    probe_ids = {item["id"] for item in registry["deterministic_probes"]}

    assert "music_view_predictability_ensemble" in trained_ids
    assert "music_release_archetypes" in trained_ids
    assert "attention_power_law" in method_ids
    assert "controlled_decision_lab" in method_ids
    assert "local_event_window" in probe_ids
    assert registry["scope"]["public_repo_only"] is True
    assert registry["scope"]["private_loopchii_runtime"] is False
    assert "production stream processing" in registry["not_claimed"]
    assert "non-public runtime behavior" in registry["not_claimed"]
    assert any("pytest" in command for command in registry["proof_commands"])
