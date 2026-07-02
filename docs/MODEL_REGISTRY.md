# Model Registry

Stream now exposes a public model and method registry at:

- `GET /api/system/model-registry`
- `data/system/model-registry.json`

The registry separates trained models, statistical methods, deterministic
probes, and compiled kernels. That distinction is important. A useful repo does
not need to call every calculation a model.

## Trained models

The music lane trains and evaluates a view-predictability ensemble over the
committed public corpus.

- Implementation: `music_pipeline.predictability_analysis`
- Estimators: `RandomForestRegressor`, `GradientBoostingRegressor`
- Target: `log1p(view_count)`
- Validation: 5-fold cross-validation with out-of-fold ensemble prediction
- Leakage guard: `virality_coefficient` is excluded because it is derived from
  views

The music lane also runs KMeans clustering to describe release archetypes.
Those clusters are descriptive. They are not identity labels and should not be
used as artist judgments.

## Statistical methods

The registry also names the non-trained analytical methods:

- power-law tail fit over view counts
- attention inequality metrics
- tag co-occurrence network analysis
- controlled decision-lab comparisons

These methods are there to make the shape of attention easier to inspect. They
do not turn metadata into cultural certainty.

## Deterministic probes

The runtime drift and local event-window surfaces are deterministic checks. They
are useful for local inspection, API testing, and event-contract design. They do
not claim hidden access to model internals.

## Compiled kernels

`loopchii-wasm-core` contains Rust metric primitives for entropy,
concentration, weighted means, top-share calculations, and chi-square
goodness-of-fit.

Run:

```bash
cargo test --manifest-path loopchii-wasm-core/Cargo.toml
```

## Verification path

```bash
python3 run_analysis.py --output-dir exports
python3 -m stream_backend.cli.doctor
python3 -m pytest -q tests
cargo test --manifest-path loopchii-wasm-core/Cargo.toml
```

If a future model is added, it should appear in the registry with its
implementation path, validation method, leakage guard, and claim boundary.
