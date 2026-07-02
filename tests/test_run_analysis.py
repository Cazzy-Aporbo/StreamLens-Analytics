from __future__ import annotations

import json

from run_analysis import build_public_music_exports


def test_public_music_export_runner_writes_reproducible_artifacts(tmp_path):
    summary = build_public_music_exports(tmp_path, include_music_index=False)

    assert summary["api_key_required"] is False
    assert summary["song_rows"] >= 100
    assert summary["genre_known_share"] > 0.6
    assert summary["year_known_share"] > 0.0

    report_path = tmp_path / "stream_music_report.json"
    quality_path = tmp_path / "stream_music_quality.json"
    songs_path = tmp_path / "stream_music_songs.csv"
    assert report_path.exists()
    assert quality_path.exists()
    assert songs_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "overview" in report
    assert "songs" in report
