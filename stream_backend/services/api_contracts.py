from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _schema_ref(payload: Mapping[str, Any] | None) -> str | None:
    if not payload:
        return None
    ref = payload.get("$ref")
    if isinstance(ref, str):
        return ref.rsplit("/", 1)[-1]
    return None


def _request_schema(operation: Mapping[str, Any]) -> str | None:
    body = operation.get("requestBody") or {}
    content = body.get("content") or {}
    json_body = content.get("application/json") or {}
    return _schema_ref(json_body.get("schema") or {})


def _response_codes(operation: Mapping[str, Any]) -> list[str]:
    responses = operation.get("responses") or {}
    return sorted(str(code) for code in responses)


def build_api_contract_surface(openapi: Mapping[str, Any]) -> dict[str, Any]:
    """Build a compact API contract index from FastAPI's OpenAPI schema."""
    paths = openapi.get("paths") or {}
    rows: list[dict[str, Any]] = []
    write_methods = {"post", "put", "patch", "delete"}
    for path, methods in sorted(paths.items()):
        if not isinstance(methods, Mapping):
            continue
        for method, operation in sorted(methods.items()):
            if method.lower() not in {"get", "post", "put", "patch", "delete"}:
                continue
            operation = operation or {}
            rows.append(
                {
                    "path": path,
                    "method": method.upper(),
                    "operation_id": operation.get("operationId"),
                    "summary": operation.get("summary") or "",
                    "tags": operation.get("tags") or [],
                    "request_schema": _request_schema(operation),
                    "response_codes": _response_codes(operation),
                    "write_surface": method.lower() in write_methods,
                }
            )

    write_routes = [row for row in rows if row["write_surface"]]
    read_routes = [row for row in rows if not row["write_surface"]]
    schema_names = sorted((openapi.get("components") or {}).get("schemas") or {})
    static_contracts = [
        "openapi.stream.json",
        "data/system/api-contracts.json",
        "data/system/model-registry.json",
        "data/system/live-contract.json",
        "data/system/streaming-readiness.json",
    ]

    return {
        "generated_at": _now(),
        "source": "FastAPI OpenAPI schema",
        "version": openapi.get("info", {}).get("version"),
        "title": openapi.get("info", {}).get("title"),
        "route_count": len(rows),
        "read_route_count": len(read_routes),
        "write_route_count": len(write_routes),
        "schema_count": len(schema_names),
        "static_contracts": static_contracts,
        "write_surface_policy": {
            "environment_key": "STREAM_API_KEY",
            "header": "X-API-Key",
            "local_default": "open when unset",
            "shared_environment": "set STREAM_API_KEY before exposing write routes",
        },
        "routes": rows,
        "schemas": schema_names,
        "docs": {
            "swagger_ui": "/docs",
            "openapi_live": "/openapi.json",
            "openapi_static": "openapi.stream.json",
            "api_surface": "docs/API_SURFACE.md",
            "production_readiness": "docs/PRODUCTION_READINESS.md",
            "model_registry": "docs/MODEL_REGISTRY.md",
        },
    }
