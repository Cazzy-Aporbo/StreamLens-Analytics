# API Contracts

Stream exposes its API contract in three forms:

- Swagger UI: `/docs`
- Live OpenAPI: `/openapi.json`
- Static OpenAPI: `openapi.stream.json`

The compact route index is available at:

- `GET /api/system/api-contracts`
- `data/system/api-contracts.json`

## Contract surfaces

| Surface | Purpose |
|---|---|
| `openapi.stream.json` | Full OpenAPI schema generated from FastAPI |
| `data/system/api-contracts.json` | Compact route, schema, and write-surface index |
| `data/system/live-contract.json` | Runtime event ingress contract |
| `data/system/model-registry.json` | Executable model and method registry |
| `data/system/streaming-readiness.json` | Current production-readiness boundary |

## Write surfaces

Write routes can require `X-API-Key` when `STREAM_API_KEY` is set. Local
development stays open by default so contributors can run the repo without a
private secret.

Shared or hosted environments should set:

```bash
STREAM_API_KEY=<strong local secret>
```

## Contract checks

```bash
python3 build_static.py
python3 -m stream_backend.cli.doctor
python3 -m pytest -q tests/test_api.py tests/test_api_contracts.py
```

The contract is generated from the application. If a route changes, regenerate
the static artifacts before publishing the Pages view.
