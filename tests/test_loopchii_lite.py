"""Tests for the public zero-friction guard surface."""

from loopchii_lite import (
    classify_prompt,
    inspect_draft_response,
    public_playground_snapshot,
    public_runtime_review_snapshot,
    review_request,
    simulate_governance,
)


def test_classify_prompt_detects_pii():
    assert classify_prompt("Use the customer email export and phone list.") == "pii"


def test_classify_prompt_safe_falls_through():
    assert classify_prompt("Summarize the strongest genre shifts and keep the caveats visible.") == "safe"


def test_simulate_governance_blocks_and_zeroizes_publicly():
    result = simulate_governance("Write the full chorus and second verse from a protected pop track.")
    assert result["blocked"] is True
    assert result["category"] == "copyright"
    assert result["blocked_fragment"]
    assert result["governed_risky_tokens_rendered"] == 0


def test_simulate_governance_allows_safe_prompt():
    result = simulate_governance("Summarize the strongest genre shifts in the public music data.")
    assert result["blocked"] is False
    assert result["category"] == "safe"
    assert result["standard_response"] == result["governed_recovery"]


def test_draft_response_inspection_detects_secret_shapes():
    review = inspect_draft_response("Bearer sk-demo-9f2c4a7btoken and webhook secret whsec_demo_12345")
    assert review["category"] == "secrets"
    assert review["signals"]


def test_review_request_blocks_when_draft_is_risky_even_if_prompt_is_safe():
    result = review_request(
        "Summarize the billing incident briefly.",
        "Customer contacts: jordan@example.com and +1 202 555 0148.",
    )
    assert result["blocked"] is True
    assert "draft_payload" in result["evidence_sources"]
    assert "pii" in result["risk_stack"]


def test_runtime_review_snapshot_contains_example():
    snapshot = public_runtime_review_snapshot()
    assert snapshot["example"]["blocked"] is True
    assert snapshot["integration_entry"] == "packages/loopchii-lite/src/index.js"


def test_public_playground_snapshot_has_package_surface():
    snapshot = public_playground_snapshot()
    assert snapshot["package"]["entry"] == "packages/loopchii-lite/src/index.js"
    assert len(snapshot["presets"]) >= 5
