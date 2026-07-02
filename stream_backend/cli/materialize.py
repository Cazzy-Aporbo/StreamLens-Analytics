from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize the Stream backend runtime snapshot.")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--no-sqlite", action="store_true", help="Skip SQLite persistence and only refresh JSON/markdown outputs.")
    parser.add_argument("--json", action="store_true", help="Emit the materialization result as JSON.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    from app import runtime_service

    payload = runtime_service.materialize(
        sample_size=args.sample_size,
        persist_sqlite=not args.no_sqlite,
    )
    summary = {
        "run_id": payload["run_id"],
        "created_at": payload["created_at"],
        "sample_size": payload["sample_size"],
        "artifact_count": len(payload.get("artifacts") or []),
        "sqlite_persisted": not args.no_sqlite,
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("Stream materialize")
        for key, value in summary.items():
            print(f"{key}={value}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
