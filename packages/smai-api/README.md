# smai-api

FastAPI implementation of the SMAI v2 HTTP API contract per `designs/smai/11-api.md`. Implements the shared contract published in `smai-api-spec` (Pydantic models + URL constants + error taxonomy per DEC-037); the parameterizable conformance suite in `smai-api-conformance` certifies this implementation in CI.

This package is consumed by the `smai ui` verb (Task 4.L1, forthcoming) which boots a `Runtime` and serves `make_api_app(runtime)` via `uvicorn`. Yaarp v2's hosted-product backend ships an independent FastAPI codebase against the same contract package — the two implementations stay in sync because both run the conformance suite.

## What it implements

Every endpoint in `11-api.md` §4 except the SSE events endpoint:

- **Proposals** — `POST/GET /proposals`, `GET /proposals/{id}`, `POST /proposals/{id}/approve`, `POST /proposals/{id}/reject`.
- **Papers** — `POST/GET /papers`, `GET /papers/{arxiv_id}`, `POST /papers/{arxiv_id}/promote-partial`.
- **Experiments** — `POST /experiments/compile`, `POST /experiments` (the `smai run` adapter).
- **Comparison Groups** — list, detail, status, agent-status (composite read), entries, evaluation, artifact list + fetch (streaming or 302 to presigned URL).
- **Runs** — list (with filters), detail.
- **System** — version, config (secrets redacted), plugins, verify, dashboard, migrate-status, health.

The SSE events endpoint (`GET /events`) and its `EventBroker` machinery ship in Task 4.K2; this implementation deliberately does not register that route.

## Auth + Host validation

Per `11-api.md` §7:

- **Host validation** is on by default. Requests with a `Host:` header outside the allowlist (`127.0.0.1`, `localhost`, `[::1]`, plus any configured `api.host`) get `421 HOST_REJECTED`. The `httpx.AsyncClient(base_url="http://test")` ASGI form sets `Host: test`; that synthetic value is also accepted so the test transport works without the SPA having to disable Host validation in test environments.
- **Bearer token** is opt-in (`api.auth.enabled: true` in `smai.yaml`). When on, every request must carry `Authorization: Bearer <token>` matching the file at `api.auth.token_path` (default `~/.smai/api-token`, mode `0600`, auto-generated via `secrets.token_urlsafe(32)` on first launch). Mismatch → `403 FORBIDDEN`.

## Construction

```python
from smai_cli.runtime import Runtime
from smai_api import make_api_app

async with Runtime.start_in_band(config) as runtime:
    app = make_api_app(runtime)
    # uvicorn.run(app, host="127.0.0.1", port=...)
```

`make_api_app(runtime, *, auth_config=None)` returns a configured `FastAPI`. The optional `auth_config` argument lets callers (the `smai ui` verb) flip on bearer-token mode and point at a token file.

## Errors

Every non-2xx response carries the `ErrorEnvelope` shape from `smai_api_spec.errors` — `{"error": {"code", "message", "issues"?, "retryable"?}}`. Translation happens in central FastAPI exception handlers; route handlers raise typed exceptions from `smai_cli.runtime` (e.g. `CGNotFoundError`) and the handlers map them to the right `(status, code)` per `11-api.md` §6.2.

## Testing

```bash
uv run pytest packages/smai-api/tests
```

The test suite subclasses `APIConformanceBase` against an in-memory `SqliteStore` + `LocalFsStore` runtime, plus router-specific edge-case tests (malformed bodies, redact-secrets behavior on `/system/config`, Host-rejection envelope shape). The events tests are explicitly skipped — Task 4.K2 implements the events endpoint and removes those skips.
