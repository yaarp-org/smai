# Declarative retry bookkeeping for dispatched states

Round 10 lifted per-state retry bookkeeping out of dispatch-handler
bodies and per-state manual terminal-gate edges, into a single
declarative primitive on `DispatchAction`:

```python
from smai_orchestrator.engine.types import DispatchAction, RetryPolicy, StateDef

StateDef(
    name="designing",
    on_entry_dispatch=DispatchAction(
        name="proposal.dispatch_planner",
        handler=planner_handler,
        pool="proposal_pipeline",
        handle_field="planner_job_handle",
        retry_policy=RetryPolicy(
            attempt_counter_field="design_attempt",
            max_attempts=max_design_attempts,
            on_exhaustion_target_state="failed",
            on_exhaustion_reason="design retry budget exhausted; terminal",
        ),
    ),
)
```

## What the engine derives from one `RetryPolicy`

1. **Counter bump on step-1 CAS.** `engine.run_dispatch` includes
   `fields={policy.attempt_counter_field: current + 1}` in the same
   write-first CAS that transitions the entity into the dispatch state.
   No separate write; no version race against the handler's own
   `transition_*_state` calls.
2. **Synthesized retry-exhausted terminal edge.** When the handler
   fails, `engine._handle_dispatch_failure` synthesizes a
   `<dispatch_state> -> <on_exhaustion_target_state>` `EdgeDef` whose
   gate predicate is `record.<attempt_counter_field> >= max_attempts`.
   The edge is **declaration-first-equivalent** — it pre-empts any
   manual `dispatch_time` edges from the dispatch state, matching the
   round-8 ordering rule ("retry-exhausted terminal pre-empts retry-
   one-more-time").

The synthesized edge name is
`"<from> → <to> (synthesized retry-exhausted terminal)"` and the
`gate_outcome_reason` written to `transition_log` is exactly
`policy.on_exhaustion_reason`. Audit queries that match on the pre-
round-10 reason wording (e.g.
`WHERE gate_outcome_reason LIKE '%retry budget exhausted%'`) keep
matching post-round-10.

## The contract for spec authors

Spec authors writing a new dispatched state with a retry budget MUST:

- Declare a `RetryPolicy` on the `DispatchAction`.
- Ensure the entity record carries `<attempt_counter_field>` as an
  `int = 0` Pydantic field, persisted in `migrations/metadata.py`.
- NOT manually bump `<attempt_counter_field>` in the dispatch handler
  body (the engine does it on step-1's CAS — a manual bump on top
  double-counts and breaks the retry budget).
- NOT manually declare a `*_failed (retry exhausted)` `EdgeDef` (the
  engine synthesizes it — a manual edge alongside would double-evaluate
  and break the round-8 declaration-order rule).
- Cap (`max_attempts`) defaults to a number passed in via the spec's
  `build_*_spec` factory so deployments can tune via `smai.yaml`.

A dispatched state with no retry budget leaves `retry_policy=None`
(the explicit opt-out shape). The engine then bumps nothing and
synthesizes nothing; a dispatch failure forward-rolls-back to
`edge.from_state` indefinitely (which is the right shape when the
state has no counter on the record).

## What the engine does NOT cover

- Gate-body counter bumps that happen on non-dispatch-failure paths
  (e.g., `cg_execution.py`'s `_make_gate_review_fail_with_retry` bumps
  `EntryRecord.implementation_attempt` when routing an entry from
  `implemented` back to `pending` for a re-implementation). The engine
  only bumps on dispatch entry; recovery-path bumps in gate bodies
  remain the spec author's concern.
- Backoff timing. `RetryPolicy` carries the budget and the terminal
  destination; deployment-tunable backoff knobs (delay, multiplier)
  live in the separate `RetryBackoffConfig` named-config map on
  `EngineConfig.retry_policies` (which spec authors consult from gate
  bodies if they need backoff-aware retry logic).

## Per-spec inventory at round 11

| Spec | State | counter | max | terminal | source |
|---|---|---|---|---|---|
| proposal | `designing` | `design_attempt` | `max_design_attempts` (1) | `failed` | engine-synthesized |
| proposal | `registered` | `registration_attempt` | `max_registration_attempts` (2) | `failed` | engine-synthesized |
| cg_execution | `implementing` | `implementation_phase_attempt` | `max_implementation_phase_attempts` (2) | `implementation_failed` | engine-synthesized (round-10 net-new) |
| cg_entries | `implementing` | `entry_dispatch_attempt` | `max_entry_dispatch_attempts` (2) | `implementation_failed` | engine-synthesized (round-11 net-new) |
| run_record | `submitted` | `run_attempt` | `max_run_attempts` (3) | `failed` | engine-synthesized (round-10 net-new) |
| paper_ingestion | `screening` | `screening_attempt` | `max_screening_attempts` (1) | `failed` | engine-synthesized |
| paper_ingestion | `planning` | `planning_attempt` | `max_planning_attempts` (1) | `failed` | engine-synthesized |

Dispatched states with `retry_policy=None`:

- `proposal.proposal_submitted` — not a dispatched state.
- `cg_execution.implemented` — idempotent fanout dispatch; failure
  retried indefinitely by design (manifest-hash write is idempotent).
- `cg_execution.running` — creates `RunRecord`s; the per-run sub-spec
  handles run-level retries.
- `cg_execution.evaluating` — writes evaluation artifacts to a
  predictable key; idempotent.
- `paper_ingestion.fetching` — no counter on `PaperRecord` for fetch
  attempts; `08` §5.2 explicitly defers fetch retries to v2. Add a
  `fetch_attempt` field + a `RetryPolicy` if a deployment needs it.
- `paper_ingestion.registered` — terminal state; transitioning *out*
  of a terminal would be unusual. Registration handler failures
  forward-roll-back to `planning`.

## How to add a new dispatched state with a retry budget

1. Add `<name>_attempt: int = 0` to the entity's Pydantic record under
   `smai_orchestrator.entities.tracking.<entity>`.
2. Add an Alembic migration following the round-9/round-10 pattern
   (offline mode renders ADD COLUMN unconditionally; online mode
   inspects and skips). Revision id MUST fit in 32 chars
   (`alembic_version.version_num` is varchar(32)).
3. Add the column to the canonical SQLAlchemy declaration in
   `smai_orchestrator.migrations.metadata`.
4. Declare a `RetryPolicy(...)` on the `DispatchAction` in the spec
   factory.
5. Write a smoke test in `tests/specs/test_retry_counters.py` (or the
   per-spec module) that asserts (a) the policy is declared on the
   spec's `StateDef` and (b) a clean round trip leaves the counter at
   1 (the engine bump fires on first entry).

The engine-level synthesis tests in
`tests/engine/test_retry_policy_synthesis.py` cover the cross-
spec behavior generically — you don't need to re-derive those
guarantees per spec.
