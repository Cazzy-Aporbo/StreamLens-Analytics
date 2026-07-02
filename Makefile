PYTHON ?= python3

.PHONY: doctor test api static materialize analysis pages-site docker-build docker-up enterprise-up cargo-test seed-live-demo

doctor:
	$(PYTHON) -m stream_backend.cli.doctor

test:
	$(PYTHON) -m pytest -q tests

api:
	$(PYTHON) app.py

static:
	$(PYTHON) build_static.py

materialize:
	$(PYTHON) -m stream_backend.cli.materialize --json

analysis:
	$(PYTHON) run_analysis.py --output-dir exports

pages-site:
	$(PYTHON) -m stream_backend.cli.prepare_pages_site --output-dir _site

docker-build:
	docker build -t loopchii-stream .

docker-up:
	docker compose up --build

enterprise-up:
	docker compose --profile enterprise up --build

cargo-test:
	cargo test --manifest-path loopchii-wasm-core/Cargo.toml

seed-live-demo:
	$(PYTHON) -c "import app; print(app.runtime_events_demo_seed())"
