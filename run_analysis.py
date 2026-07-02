#!/usr/bin/env python3
"""
Stream public analysis runner.

This is the shortest proof path in the repository: committed public data goes
in, reproducible artifacts come out, and no API key is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"joblib\.externals\.loky\.backend\.context",
)

import music_intelligence
import music_pipeline


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "exports"


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Any, pretty: bool) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2 if pretty else None, sort_keys=False),
        encoding="utf-8",
    )


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_public_music_exports(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    pretty: bool = True,
    include_music_index: bool = True,
) -> dict[str, Any]:
    """Run the public music pipeline and write inspectable artifacts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    report = music_pipeline.full_report()
    quality = report["quality"]
    songs = list(report["songs"])

    report_path = out / "stream_music_report.json"
    quality_path = out / "stream_music_quality.json"
    songs_path = out / "stream_music_songs.csv"

    _write_json(report_path, report, pretty)
    _write_json(quality_path, quality, pretty)
    _write_rows_csv(songs_path, songs)

    written = {
        "report": str(report_path),
        "quality": str(quality_path),
        "songs": str(songs_path),
    }

    music_index_summary: dict[str, Any] | None = None
    if include_music_index:
        index_package = music_intelligence.build_music_data_package(
            repo_root=BASE_DIR,
            enable_live_enrichment=False,
        )
        index_json = out / "music_index.json"
        index_csv = out / "music_index.csv"
        index_report = out / "music_analysis_report.md"
        music_intelligence.write_package(index_package, index_json, index_csv, index_report)
        music_index_summary = dict(index_package["summary"])
        written.update(
            {
                "music_index": str(index_json),
                "music_index_csv": str(index_csv),
                "music_index_report": str(index_report),
            }
        )

    coverage = quality.get("coverage") or {}
    overview = report.get("overview") or {}
    summary = {
        "song_rows": int(len(songs)),
        "total_views": int(overview.get("total_views") or 0),
        "genre_known_share": float(coverage.get("genre_known_share") or 0.0),
        "year_known_share": round(
            float(coverage.get("publication_year_explicit_share") or 0.0)
            + float(coverage.get("publication_year_inferred_share") or 0.0),
            4,
        ),
        "source": "committed public music data",
        "api_key_required": False,
        "written": written,
    }
    if music_index_summary:
        summary["notation_link_rate"] = float(music_index_summary.get("notation_link_rate") or 0.0)
        summary["discovered_music_files"] = int(music_index_summary.get("discovered_music_files") or 0)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the no-key public music analysis path and export the results."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--compact", action="store_true", help="Write compact JSON instead of indented JSON.")
    parser.add_argument(
        "--skip-music-index",
        action="store_true",
        help="Skip the notation-aware music index if you only need the core public data report.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = build_public_music_exports(
        args.output_dir,
        pretty=not args.compact,
        include_music_index=not args.skip_music_index,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
