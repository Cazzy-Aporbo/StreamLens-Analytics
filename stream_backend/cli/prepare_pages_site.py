from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SITE_FILES = (
    ".nojekyll",
    "index.html",
    "StreamLen_processors.html",
    "streaming-bias-index.html",
    "manifest.webmanifest",
    "service-worker.js",
    "openapi.stream.json",
    "music_index.json",
    "music_index.csv",
    "music_analysis_report.md",
)

SITE_DIRS = (
    "assets",
    "data",
)


def prepare_pages_site(base_dir: Path, output_dir: Path) -> list[str]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for relative in SITE_FILES:
        src = base_dir / relative
        if not src.exists():
            raise FileNotFoundError(f"Required Pages file is missing: {relative}")
        dst = output_dir / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(relative)

    for relative in SITE_DIRS:
        src = base_dir / relative
        if not src.exists():
            raise FileNotFoundError(f"Required Pages directory is missing: {relative}")
        dst = output_dir / relative
        shutil.copytree(src, dst)
        copied.append(relative)

    return copied


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a lean static site directory for GitHub Pages deployment."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root containing the built static files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("_site"),
        help="Destination directory for the Pages artifact.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    copied = prepare_pages_site(args.base_dir.resolve(), args.output_dir.resolve())
    print(f"prepared_pages_site files={len(copied)} output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
