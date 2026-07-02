from __future__ import annotations

from pathlib import Path

from stream_backend.cli.prepare_pages_site import prepare_pages_site


def test_prepare_pages_site_copies_expected_public_surface(tmp_path):
    base_dir = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "_site"

    copied = prepare_pages_site(base_dir, output_dir)

    assert "index.html" in copied
    assert "data" in copied
    assert (output_dir / "index.html").exists()
    assert (output_dir / "data" / "system" / "api-contracts.json").exists()
    assert (output_dir / "data" / "system" / "model-registry.json").exists()
    assert (output_dir / "data" / "system" / "live-contract.json").exists()
