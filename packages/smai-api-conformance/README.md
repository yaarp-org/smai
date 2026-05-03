# smai-api-conformance

Parameterizable pytest suite that asserts an HTTP API implementing
[`smai-api-spec`](../smai-api-spec/) conforms to the contract defined in
[`designs/smai/11-api.md`](../../) §10. Per
[DEC-037](../../README.md#decision-log) this is the parity-enforcement
mechanism between SMAI's OSS `smai-api` (Phase 4 task K1) and the Yaarp v2
hosted-product API — both implementations subclass the base, override the
`client` fixture, and run the inherited tests in CI.

The pattern mirrors the four plugin-Protocol conformance bases at
`smai_core.plugins.conformance.*`: a single subclass per implementation,
one factory-style fixture override, and the contract suite runs.

## Opt-in pattern

```python
# In packages/smai-api/tests/test_conformance.py:
import pytest
from httpx import ASGITransport, AsyncClient
from smai_api import build_api_app
from smai_api_conformance import APIConformanceBase
from smai_cli.runtime import Runtime

class TestSmaiApiConformance(APIConformanceBase):
    @pytest.fixture
    async def client(self) -> AsyncClient:
        runtime = await Runtime.start_in_band(test_config(), run_worker=False)
        app = build_api_app(runtime)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            yield c
```

Then `uv run pytest packages/smai-api/tests` runs the inherited contract
suite against the FastAPI app via ASGI transport (no socket, no port).
Yaarp v2's hosted API does the same with its own app constructor.

## Scope boundary

| In scope (this suite) | Out of scope (lives in implementation tests) |
|---|---|
| Status-code semantics (200 / 202 / 302 / 400 / 404 / 409 / 421 / 403) | State-machine lifecycle correctness (e.g. `proposal_submitted → designing → designed`) |
| Response-body shape (parses cleanly into spec Pydantic models) | Worker behavior (the test that "approving fires the engine") |
| Error envelope shape on every non-2xx | Plugin-substrate behavior (Postgres-vs-SQLite differences) |
| Pagination round-trip (`next_cursor` is acceptable as `?cursor=`) | Performance (latency, throughput, connection pooling) |
| RPC verb idempotency (re-firing on terminal-state returns 409) | Implementation-specific behavior (Yaarp tenant scoping, SMAI structured logging) |
| SSE delivery of `state_change` within a bounded delay | Yaarp-only endpoints (`/api/v1/orgs/`, `/api/v1/billing/`, `/api/v1/audit/`) |
| Auth posture (Host validation; bearer mode when enabled) | Real LLM / database I/O |

The shorthand: this suite tests the **shape and protocol semantics** of
the API. It does NOT test that the underlying engine drives state
machines correctly — that is implementation-specific behavior, covered
by `packages/smai-api/tests/integration/` (Task 4.K1) on the SMAI side
and the Yaarp v2 backend's own integration suite on the closed side.

## Configuration knobs

`APIConformanceBase` exposes a small set of class attributes that
subclasses override to adapt the base to their deployment shape:

| Attribute | Default | Purpose |
|---|---|---|
| `sse_event_timeout_seconds` | `5.0` | Bounded delay for the SSE state-change tests (per `11-api.md` §10.2 / §13 OQ12). Increase if your worker→API channel is slower in test. |
| `auth_mode` | `"disabled"` | `"disabled"` skips bearer-required tests; `"bearer"` runs them and demands a `bearer_token` fixture. |
| `bearer_token` (fixture) | unimplemented | Override only when `auth_mode == "bearer"` — return the bearer token to use in the `Authorization: Bearer <token>` header. |

## Self-test

`tests/test_self_conformance.py` subclasses `APIConformanceBase` with a
fixture that returns an `AsyncClient(transport=httpx.MockTransport(...))`.
The mock transport returns spec-conformant canned responses for every
endpoint the suite tests. If the suite passes against the mock, the
suite is internally coherent — proves the test methods are well-formed
and can run end-to-end without a real implementation.

The mock responses live at `src/smai_api_conformance/_4_j2_mock_responses.py`
and are minimal but spec-conformant (correct status codes, valid envelope
shapes, payloads that parse into the Pydantic models exported from
`smai_api_spec`). They are **for self-test only** — real implementations
do not import them.

## Dependencies

Runtime: `smai-api-spec`, `httpx`, `pytest`, `pytest-asyncio`. No
FastAPI, no orchestrator, no `smai-core`. The conformance suite is
test-shaped, not application-shaped.

## What's NOT in here

- The API implementation. Lives in `smai-api` (OSS, Task 4.K1) and
  Yaarp v2's hosted backend (closed).
- The SSE worker→API event channel. Lives in `smai-events` (Task 4.K2).
- Lifecycle / state-machine correctness tests. Live in each
  implementation's own integration test suite.
