# Roadmap

This roadmap keeps present capability separate from future work.

## Now

- Public music analysis with committed source files
- Synthetic representation lane for method teaching and testing
- FastAPI surface for music, governance, runtime, and readiness endpoints
- Static artifact generation for GitHub Pages-style inspection
- SQLite-backed materialization ledger
- Local runtime event window with bounded ingestion
- Optional Redpanda and ClickHouse publishing hooks
- Rust metric primitives for selected statistical calculations
- Environment doctor, tests, and OpenAPI export

## Next

- Versioned replay command for event windows
- More explicit stage timing in materialization outputs
- Stronger schema compatibility tests
- Frontend removal of duplicated backend math
- Benchmark table for 10K, 100K, and 1M row workloads
- More public datasets with clear license and provenance notes
- Better examples for adding a new metric without changing the UI first

## Later

- Durable event-log integration with replay proofs
- Checkpointed processing and interrupted-run recovery
- Real schema registry support
- Observability bundle for metrics, traces, logs, and freshness alerts
- More compiled metric kernels where profiling proves they matter
- Cloud deployment templates with explicit security boundaries

The repository should only move an item from future work to present capability
after the code, tests, and documentation all support it.
