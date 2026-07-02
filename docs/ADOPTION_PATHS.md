# Adoption Paths

Stream is easiest to adopt when the entry point matches the reader's job.

## Researchers and data scientists

Start with the public music lane.

```bash
python3 run_analysis.py --output-dir exports
```

Use the generated quality report before interpreting the charts. The most
important habit is to ask what the current corpus can support, what it cannot
support yet, and whether a metric survives when missing years, unlabeled
genres, and concentration effects are kept visible.

Good first contributions:

- improve source notes for a committed public dataset
- add a validation check for year, genre, language, or platform coverage
- add a notebook or markdown walkthrough that reproduces one metric from raw rows
- correct a conclusion that is too strong for the available evidence

## Engineers and maintainers

Start with the backend spine.

```bash
python3 -m stream_backend.cli.doctor
python3 -m stream_backend.cli.materialize --json
pytest -q
```

The useful engineering surface is the alignment between:

- `stream_backend/`, the shared API and static-export spine
- `build_static.py`, the static artifact builder
- `app.py`, the FastAPI surface
- `data/system/*.json`, the browser-readable contracts
- `loopchii-wasm-core/`, the compiled metric primitives

Good first contributions:

- add a test before changing an endpoint response
- make an artifact contract more explicit
- improve replay or materialization behavior
- reduce duplicated math between frontend and backend
- strengthen the doctor so failures are caught before a contributor opens the UI

## Production evaluators

Start with the readiness boundary.

```bash
python3 benchmarks/analytics_scale.py --rows 100000
python3 -m stream_backend.cli.doctor --json
```

The repository is not presented as a production stream processor. It is a
reproducible media-analysis engine with a local event window, public API
surface, optional event-bus sink, static export parity, and a growing compiled
metric lane.

Evaluate it as:

- a public analytical method
- a local inspection harness
- a prototype for media-governance event contracts
- a contribution surface for better metrics, data hygiene, and review tooling

Do not evaluate it as:

- a drop-in Kafka or Flink replacement
- a hosted compliance platform
- a complete production runtime
- a private LOOPCHii architecture disclosure

That distinction is deliberate. The open-source work should be useful without
pretending to contain private systems it does not contain.
