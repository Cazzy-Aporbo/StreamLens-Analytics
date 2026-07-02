# Repository Structure

Stream keeps one public repository so the evidence is easy to inspect, but the
layers are intentionally separated.

## Analysis layer

- `music_pipeline.py`
- `music_decision_lab.py`
- `music_theory.py`
- `music_intelligence.py`
- `streamlens_processor.py`
- `advanced_metrics.py`
- `media_liability_lab.py`

This is where rows become metrics. Changes here should usually include tests
and a note about what claim changed.

## Backend layer

- `app.py`
- `stream_backend/`
- `build_static.py`
- `run_analysis.py`

This is where API routes, static artifacts, local materialization, event
contracts, and adoption commands live. The frontend should read from this layer
instead of rebuilding analysis logic in the browser.

Contract surfaces:

- `GET /api/system/api-contracts`
- `GET /api/system/model-registry`
- `GET /api/runtime/events/contract`
- `GET /api/runtime/invariants/live`

## Compiled metric layer

- `loopchii-wasm-core/`

This is the Rust lane for small, inspectable metric kernels. It should grow
where profiling shows a real need, not where a README wants a stronger word.

## Frontend layer

- `index.html`
- `assets/`
- `manifest.webmanifest`
- `service-worker.js`
- `data/**/*.json`

The browser surface is a public explanation layer. It should make the work
legible, not invent capability.

## Verification layer

- `tests/`
- `benchmarks/`
- `stream_backend/cli/doctor.py`
- `docs/PRODUCTION_READINESS.md`
- `docs/MODEL_REGISTRY.md`

This is where the repository earns its claims. If a capability matters, it
should have a command, a test, or a readiness note.
