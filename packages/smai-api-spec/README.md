# smai-api-spec

The shared API contract for SMAI v2 — pure Pydantic request/response models, URL
path constants, and the error taxonomy. Per
[DEC-037](../../README.md#decision-log), this package is depended on by both
SMAI's OSS HTTP API (`smai-api`, planned for Phase 4 task K1) and the Yaarp v2
hosted-product HTTP API. Parity between the two implementations is enforced by
the sibling `smai-api-conformance` pytest suite (Phase 4 task J2).

## What's in here

| Module | Contents |
|---|---|
| `paths` | URL constants for every `v1` endpoint (single source of truth for route registration). |
| `pagination` | `CursorPage[T]` + the common `?cursor=` / `?limit=` query model. |
| `_common` | `BaseAuditedResponse` mixin (`created_at` / `updated_at` / `version` / `last_error`); state Literals duplicated from `smai-orchestrator`. |
| `errors` | `APIError`, `ErrorEnvelope`, `ValidationIssue`, the `ErrorCode` catalog. |
| `events` | `StateChangeEvent` and `WorkerHeartbeatEvent` SSE payload types. |
| `proposals` / `papers` / `experiments` / `comparison_groups` / `entries` / `runs` / `system` | Per-resource request and response models matching `11-api.md` §4. |

The state Literals (`ProposalState`, `PaperState`, `CGState`, `EntryState`,
`RunState`) are intentionally **duplicated** from
`smai_orchestrator.entities.tracking`. Importing the orchestrator would break
this package's deps-light contract (see "Dependencies" below). The
`tests/test_state_literal_parity.py` suite imports both sides and asserts they
are identical — silent drift breaks the tests.

## Dependencies

Runtime: **`pydantic>=2.5` only**. No FastAPI, no `httpx`, no orchestrator, no
`smai-core`. The package is meant to be cheap for any consumer (CLI tools,
remote SDK clients, codegen pipelines) to depend on.

## Versioning

Semver, applied to the contract surface:

| Bump | Triggers |
|---|---|
| Major | Renaming or removing a model field; removing an endpoint URL constant; changing an error `code` value; tightening a Literal. |
| Minor | Adding a new endpoint constant; adding a new optional model field; adding a new entry to a `Literal` whose call sites tolerate widening; adding a new error code. |
| Patch | Docstring fixes, comment fixes, internal refactors that leave the JSON Schema bit-identical. |

Both SMAI's `smai-api` and Yaarp v2's hosted-API codebase pin a compatible
range (e.g. `smai-api-spec>=0.3,<0.4`). Major bumps are coordinated; minor /
patch are not.

## Snapshot-test discipline

`tests/test_schema_stability.py` snapshots `Model.model_json_schema()` for every
public model into `tests/snapshots/<model>.json`. Any change to a model's wire
shape — even an apparently-cosmetic field rename — produces a snapshot diff and
forces a deliberate accept (`-p no:cacheprovider --snapshot-update`-equivalent
in this hand-rolled snapshot suite: regenerate `tests/snapshots/` and re-commit).

The snapshot mechanism is the contract-stability surface that lets Yaarp v2's
API depend on this package without coordinating every release: the snapshots in
git are the on-disk record of "what shape the wire was at version X".

## What's NOT in here

- Handler code. Lives in `smai-api` (OSS) and Yaarp v2's repo (closed).
- Authentication. Lives in each implementation. The contract dictates only that
  auth failures map to `403` with `code: "FORBIDDEN"`.
- The conformance test suite. Lives in `smai-api-conformance` (Phase 4 task J2).
- The SSE worker→API channel mechanics (in-process pub/sub vs Postgres
  `LISTEN/NOTIFY`). Lives in `smai-events` (Phase 4 task K2). This package
  carries only the SSE event payload types.
