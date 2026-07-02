# Production Readiness

Stream is a working public analysis repository. It is not yet a production
streaming system.

That sentence is part of the engineering posture. It protects readers from
overclaiming, and it gives contributors a clear map for what would have to be
true before the repository could be used in heavier operational settings.

## What works today

- Public music datasets can be loaded, cleaned, measured, and exported without
  an API key.
- Static artifacts and live API routes are generated from the same backend
  spine.
- The environment doctor checks required files, imports, SQLite readiness,
  public music quality, and generated artifacts.
- The local runtime ledger uses hash-linked rows for materialization history.
- The live event lane has a versioned ingress contract, bounded batch limits,
  filterable fields, retention controls, and local replay through SQLite.
- The optional enterprise profile can publish to Redpanda PandaProxy and
  ClickHouse when those services are configured.
- The Rust crate contains tested metric primitives for entropy, concentration,
  weighted mean, and chi-square calculations.

## What is intentionally not claimed

- Exactly-once processing
- Global event ordering
- Distributed checkpoint recovery
- Multi-tenant access control
- Hosted observability
- Production incident response
- Full schema registry integration
- Sub-millisecond performance guarantees across arbitrary workloads

Those capabilities require infrastructure, load testing, security review, and
operational discipline beyond this repository's current public scope.

## What a production transition would require

### Ingest and replay

- durable append-only event log
- partitioning policy by platform, market, and entity id
- replay command with deterministic output comparison
- dead-letter lane for invalid or incomplete events

### State and recovery

- checkpointed materializations
- interrupted-run recovery tests
- retention policy verified under load
- schema migration tests for persisted data

### Processing

- bounded memory profile under large event windows
- streaming aggregations instead of batch-only recomputation
- clear backpressure behavior
- compatibility tests for each event-contract version

### Operations

- metrics for latency, queue depth, dropped events, and export freshness
- structured logs with correlation ids
- alert thresholds tied to user-visible degradation
- deployment templates that do not assume one developer laptop

## Current adoption recommendation

Use Stream today for public media analysis, reproducible research, local API
inspection, contributor education, and event-contract prototyping.

Use Kafka, Redpanda, Flink, Spark, ClickHouse, or cloud-native equivalents for
production streaming infrastructure until Stream earns those guarantees
through code, tests, benchmarks, and failure drills.
