# Quickstart

This is the shortest way to see what Stream does today.

## 1. Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
```

## 2. Check the local environment

```bash
python3 -m stream_backend.cli.doctor
```

The doctor checks imports, committed public data, SQLite writability, static
artifacts, genre coverage, year coverage, and the decision-lab path.

## 3. Run the no-key public analysis

```bash
python3 run_analysis.py --output-dir exports
```

This command writes:

- `exports/stream_music_report.json`
- `exports/stream_music_quality.json`
- `exports/stream_music_songs.csv`
- `exports/music_index.json`
- `exports/music_index.csv`
- `exports/music_analysis_report.md`

The command uses committed public music datasets only. It does not require a
YouTube key, a private LOOPCHii service, or a hosted database.

## 4. Serve the API

```bash
python3 app.py
```

Open:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/api/music/quality`
- `http://127.0.0.1:8000/api/system/readiness`
- `http://127.0.0.1:8000/api/runtime/events/contract`

## 5. Refresh static artifacts

```bash
python3 build_static.py
```

The browser preview reads these static artifacts when the live API is not
running. This keeps the GitHub Pages-style surface inspectable without asking
contributors to operate a backend.

## 6. Optional event window

```bash
python3 -m stream_backend.cli.materialize --json
make seed-live-demo
```

The live event lane is bounded and local. It is useful for testing filters,
rolling metrics, WebSocket behavior, and event-contract changes. It is not a
replacement for Kafka, Flink, or a hosted production stream.
