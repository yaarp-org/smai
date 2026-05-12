"""Three-phase worker poll cycle.

Per ``designs/smai/05-orchestrator.md`` §3 (worker loop and concurrency).
The engine's runtime is one or more worker processes — each running a
poll loop on a configurable cadence — driving entities through their
pipeline-spec lifecycle. Per `05` §3.5 / §6, C2 ships single-worker
default (``EngineConfig.worker_count=1``); multi-worker leasing is
Task 3.G1.

Per Task 4.K2 (`12-ui-process.md` §6.4): each loop also fires
:meth:`EventChannel.fire_heartbeat` on
:attr:`EngineConfig.event_channel` after every cycle — feeding the SSE
``worker_heartbeat`` event the SPA uses for its "last cycle at"
indicator. The heartbeat fires AFTER any ``on_cycle_complete``
callback so callers that want to observe stats before the wire signal
goes out can do so. Failures inside ``fire_heartbeat`` are logged-and-
swallowed so a broken event channel cannot kill the loop.

The poll cycle has the v1 three-phase shape:

* **Phase 1 — complete.** For every entity in an in-progress state with
  a non-null ``job_handle``, call :class:`Compute.status` and fire the
  matching ``fires_on=job_succeeded`` / ``fires_on=job_failed`` edge
  (or detect orphan-grace expiry and reset). Delegated per-entity to
  :func:`smai_orchestrator.engine.phase1_step`.
* **Phase 2 — discover.** For each non-terminal source state with a
  registered :class:`SchedulingQueryRef`, call the spec's query to
  collect candidate records.
* **Phase 3 — dispatch.** For each candidate (in pool-priority order),
  call :func:`smai_orchestrator.engine.drive_entity_phase3`. Pool slot
  computation runs once per cycle ahead of phase 3 so the worker can
  drain higher-priority pools first while respecting the
  ``available = limit − count_with_in_flight_jobs`` budget.

Phase 1 is the only phase that ever blocks on :class:`Compute`; phases
2 and 3 are pure :class:`MetadataStore` reads / CAS writes / dispatch
handler invocations. Cycle cadence is :attr:`EngineConfig.poll_interval_seconds`,
slept against :attr:`EngineConfig.time_provider` (so tests inject a
fake clock and drive the loop synchronously).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from uuid import uuid4

from smai_core.plugins import (
    ArtifactStore,
    Compute,
    LlmProvider,
    MetadataStore,
)

from smai_orchestrator.checkpointer.types import Checkpointer
from smai_orchestrator.engine._metadata_ops import (
    StateDrivenRecord,
    entity_id_for,
)
from smai_orchestrator.engine.config import EngineConfig
from smai_orchestrator.engine.phase1 import (
    CostRecordHandler,
    Phase1Outcome,
    phase1_step,
)
from smai_orchestrator.engine.state_machine import (
    DriveOutcome,
    drive_entity_phase3,
)
from smai_orchestrator.engine.types import (
    EngineSpec,
    SchedulingQueryRef,
)
from smai_orchestrator.worker.concurrency import (
    PoolSlot,
    compute_pool_slots,
)

# Re-export :class:`SchedulingQueryRef` so the brief's expected
# location (``worker/loop.py``) carries the spec-author surface alias.
SchedulingQueryFn = Callable[[MetadataStore], Awaitable[list[StateDrivenRecord]]]

_log = logging.getLogger(__name__)


@dataclass
class WorkerCycleStats:
    """Per-cycle counters for telemetry / test introspection (`05` §3).

    The worker doesn't yet write cycle statistics to the
    :class:`MetadataStore` — `05` §9 #5 leaves that as an open
    question. C2 surfaces the counters in-memory (returned from
    :func:`run_worker_cycle`) so tests can assert on cycle behavior
    without a separate read path; a future task (likely 3.G3) decides
    the persistence shape.
    """

    phase1_inspected: int = 0
    phase1_advanced: int = 0
    phase1_orphan_reset: int = 0
    phase1_running: int = 0
    phase1_conflict: int = 0
    phase1_other: int = 0

    phase2_candidates: int = 0

    phase3_advanced: int = 0
    phase3_no_match: int = 0
    phase3_conflict: int = 0
    phase3_dispatch_failed: int = 0
    phase3_skipped_pool_full: int = 0
    phase3_lease_held: int = 0
    """Per-cycle count of phase-3 candidates skipped because another
    worker holds the per-entity lease (`05` §3.5 / Task 3.G1). Distinct
    from :attr:`phase3_conflict` (CAS race lost): ``lease_held`` is the
    pre-acquire skip case where the dispatch handler never fires;
    ``conflict`` is the post-acquire CAS race lost (orphan reclaim,
    peer-reset). Single-worker deployments stay at 0 here."""

    pool_slots: list[PoolSlot] = field(default_factory=list[PoolSlot])


async def _gather_phase1_records(
    *,
    spec: EngineSpec,
    metadata_store: MetadataStore,
) -> list[StateDrivenRecord]:
    """Phase-1 candidate set: every in-flight record across all
    registered :attr:`EngineSpec.phase1_queries` (`05` §3.1)."""
    out: list[StateDrivenRecord] = []
    for state_name, query_ref in spec.phase1_queries.items():
        if state_name not in {s.name for s in spec.states}:
            raise LookupError(
                f"phase1_queries declares state {state_name!r} but the spec "
                f"has no matching StateDef"
            )
        records = await query_ref.fn(metadata_store)
        out.extend(records)
    return out


async def _gather_phase2_records(
    *,
    spec: EngineSpec,
    metadata_store: MetadataStore,
) -> list[StateDrivenRecord]:
    """Phase-2 candidate set: every ready record across all registered
    :attr:`EngineSpec.phase2_queries` (`05` §3.2)."""
    out: list[StateDrivenRecord] = []
    for state_name, query_ref in spec.phase2_queries.items():
        state_def = spec.state_def(state_name)
        if state_def.is_terminal:
            # Terminal states never have phase-2 outgoing transitions
            # — the spec author's mistake; raise loudly rather than
            # silently swallow records that can never advance.
            raise ValueError(
                f"phase2_queries[{state_name!r}] points at a terminal "
                f"state; phase-2 discovery never fires from terminals"
            )
        records = await query_ref.fn(metadata_store)
        out.extend(records)
    return out


async def _drive_phase1_for_records(
    *,
    spec: EngineSpec,
    metadata_store: MetadataStore,
    artifact_store: ArtifactStore,
    compute: Compute,
    config: EngineConfig,
    records: list[StateDrivenRecord],
    cost_handler: CostRecordHandler | None,
    stats: WorkerCycleStats,
) -> None:
    """Run :func:`phase1_step` against every gathered record (`05` §3.1).

    Per-record outcomes are folded into ``stats``; the worker keeps
    iterating after a per-record conflict / failure so the next
    record's phase-1 step still fires.
    """
    for record in records:
        stats.phase1_inspected += 1
        try:
            outcome: Phase1Outcome = await phase1_step(
                spec=spec,
                metadata_store=metadata_store,
                artifact_store=artifact_store,
                compute=compute,
                config=config,
                record=record,
                cost_handler=cost_handler,
            )
        except Exception:  # noqa: BLE001 — log + continue per `05` §3
            _log.exception(
                "phase1_step raised for entity %s in state %s; continuing",
                entity_id_for(record),
                record.state,
            )
            stats.phase1_other += 1
            continue
        match outcome.status:
            case "advanced":
                stats.phase1_advanced += 1
            case "orphan_reset":
                stats.phase1_orphan_reset += 1
            case "orphan_no_grace" | "running":
                stats.phase1_running += 1
            case "conflict":
                stats.phase1_conflict += 1
            case "terminated_no_match":
                stats.phase1_other += 1


async def _drive_phase3_for_records(  # noqa: PLR0913
    *,
    spec: EngineSpec,
    metadata_store: MetadataStore,
    artifact_store: ArtifactStore,
    compute: Compute,
    llm_providers: dict[str, LlmProvider] | None,
    config: EngineConfig,
    records: list[StateDrivenRecord],
    pool_slots: list[PoolSlot],
    stats: WorkerCycleStats,
    worker_id: str,
) -> None:
    """Drive phase 3 over the gathered candidate set in pool-priority
    order, respecting per-pool slot budgets (`05` §3.3 / §3.4).

    Records whose target dispatch belongs to a pool that has no
    available slots are skipped this cycle (counted in
    ``stats.phase3_skipped_pool_full``); the next cycle re-examines
    them once in-flight jobs drain.

    ``llm_providers`` is the per-task-role map (Task 2.A1's
    carry-forward; the agent loop instantiates one provider per role
    and routes by the dispatch handler's role). The worker stays
    role-agnostic — passes through whatever the spec's dispatch
    handler asks for via :class:`DispatchContext.llm`. C2 chooses one
    representative provider for the :class:`DispatchContext` (the
    only role-specific routing is inside the handler); a future task
    (3.G3 or similar) elaborates the routing if needed.
    """
    # Choose a representative LLM provider for ``DispatchContext.llm``.
    # The spec's dispatch handler is responsible for further role-
    # specific routing; the engine just hands one through. C4 / 3.G3
    # may revisit this when multi-role dispatch lands.
    representative_llm: LlmProvider | None = None
    if llm_providers:
        representative_llm = next(iter(llm_providers.values()))

    # Group records by their target-pool (resolved via the state's
    # on_entry_dispatch). Records whose target state has no dispatch
    # action are inline / sentinel — the engine still calls
    # drive_entity_phase3 for those, but they don't consume pool slots.
    state_to_pool = spec.state_to_pool()

    available_per_pool: dict[str, int] = {slot.name: slot.available for slot in pool_slots}

    # Iterate in declaration order over the records; the priority
    # ordering is enforced by `pool_slots` (returned in descending
    # priority). For records that DON'T map to a pool (e.g., inline
    # transitions with no on-entry dispatch on the target state), we
    # process them unconditionally — those don't consume slots.
    #
    # For records whose target state's dispatch action is in pool P:
    # - If available_per_pool[P] > 0, dispatch and decrement.
    # - Else skip; next cycle.

    # Per-record pool resolution: walk the record's outgoing edges to
    # find the target state; look up the target state's pool. If no
    # outgoing edge fires (gate rule says no), we don't know the pool
    # ahead of time — but since no dispatch happens, no slot is
    # consumed. So we resolve the pool *after* drive_entity_phase3
    # returns. C2's slot accounting is therefore "best-effort" — we
    # check pool availability before driving, and decrement after.
    # This matches v1's behavior (`execution_orchestration.md` §6).
    for record in records:
        target_pool = _resolve_target_pool_for_record(
            record=record, spec=spec, state_to_pool=state_to_pool
        )
        if target_pool is not None and available_per_pool.get(target_pool, 0) <= 0:
            stats.phase3_skipped_pool_full += 1
            continue

        try:
            outcome: DriveOutcome = await drive_entity_phase3(
                spec=spec,
                metadata_store=metadata_store,
                artifact_store=artifact_store,
                compute=compute,
                llm=representative_llm,
                config=config,
                record=record,
                worker_id=worker_id,
            )
        except Exception:  # noqa: BLE001 — log + continue per `05` §3
            _log.exception(
                "drive_entity_phase3 raised for entity %s in state %s; continuing",
                entity_id_for(record),
                record.state,
            )
            stats.phase3_dispatch_failed += 1
            continue

        match outcome.status:
            case "advanced":
                stats.phase3_advanced += 1
                # An external-dispatch fired and consumed a pool slot.
                if target_pool is not None and outcome.dispatch_outcome is not None:
                    handler_outcome = outcome.dispatch_outcome.handler_outcome
                    if handler_outcome.submitted_handles:
                        available_per_pool[target_pool] = available_per_pool.get(target_pool, 0) - 1
            case "no_match":
                stats.phase3_no_match += 1
            case "conflict":
                stats.phase3_conflict += 1
            case "dispatch_failed_rolled_back":
                stats.phase3_dispatch_failed += 1
            case "lease_held":
                stats.phase3_lease_held += 1


def _resolve_target_pool_for_record(
    *,
    record: StateDrivenRecord,
    spec: EngineSpec,
    state_to_pool: dict[str, str],
) -> str | None:
    """Best-effort: project the pool of the dispatch action that *would*
    fire for ``record`` (`05` §3.3 / §3.4).

    The actual outgoing edge is decided by gate rules at dispatch
    time; the worker can't know which edge wins ahead of evaluating
    them. As a slot-accounting heuristic, we look at the unique
    outgoing ``dispatch_time`` edge — if there is exactly one, its
    target state's pool is the answer. Multiple edges (an entity has
    branching gates) → conservative fallback: return ``None`` (no
    slot accounting), which means the dispatch may exceed the pool's
    declared limit by one entity per cycle in worst case. Acceptable
    for C2's single-worker default; multi-worker / fair-scheduling
    elaborations (Task 3.G1 / 3.G2) revisit.
    """
    candidates = spec.edges_from(record.state, fires_on="dispatch_time")
    if len(candidates) != 1:
        return None
    target_state = candidates[0].target_state
    return state_to_pool.get(target_state)


async def run_worker_cycle(  # noqa: PLR0913
    *,
    spec: EngineSpec,
    metadata_store: MetadataStore,
    artifact_store: ArtifactStore,
    compute: Compute,
    llm_providers: dict[str, LlmProvider] | None,
    config: EngineConfig,
    cost_handler: CostRecordHandler | None = None,
    worker_id: str | None = None,
) -> WorkerCycleStats:
    """Run one full poll cycle (phase 1 → phase 2 → phase 3) (`05` §3).

    Returns per-cycle stats for telemetry / test inspection. Exceptions
    raised by individual record drives are logged and folded into the
    stats; the cycle keeps running so a single bad entity doesn't stall
    the worker.

    ``worker_id`` is the deployment-stable identity threaded into
    :meth:`MetadataStore.acquire_lease` as ``lease_holder_id`` for the
    phase-3 lease wrapper (`05` §3.5 / DEC-035 #2). ``None`` (the test
    default) auto-generates ``f"worker-{uuid4()}"`` per cycle so unit
    tests that drive ``run_worker_cycle`` directly don't need to invent
    one; production callers (:func:`run_worker_loop`,
    :class:`smai_cli.runtime.Runtime`) pin a stable id.
    """
    stats = WorkerCycleStats()
    resolved_worker_id = worker_id if worker_id is not None else f"worker-{uuid4().hex[:8]}"

    # ---- Phase 1: complete ---------------------------------------------------
    phase1_records = await _gather_phase1_records(spec=spec, metadata_store=metadata_store)
    await _drive_phase1_for_records(
        spec=spec,
        metadata_store=metadata_store,
        artifact_store=artifact_store,
        compute=compute,
        config=config,
        records=phase1_records,
        cost_handler=cost_handler,
        stats=stats,
    )

    # ---- Phase 2: discover ---------------------------------------------------
    phase2_records = await _gather_phase2_records(spec=spec, metadata_store=metadata_store)
    stats.phase2_candidates = len(phase2_records)

    # ---- Phase 3: dispatch ---------------------------------------------------
    pool_slots = await compute_pool_slots(
        pools=spec.pools,
        state_to_pool=spec.state_to_pool(),
        entity_kind=spec.entity_kind,
        metadata_store=metadata_store,
        config=config,
    )
    stats.pool_slots = pool_slots

    await _drive_phase3_for_records(
        spec=spec,
        metadata_store=metadata_store,
        artifact_store=artifact_store,
        compute=compute,
        llm_providers=llm_providers,
        config=config,
        records=phase2_records,
        pool_slots=pool_slots,
        stats=stats,
        worker_id=resolved_worker_id,
    )

    return stats


async def run_worker_loop(  # noqa: PLR0913
    *,
    spec: EngineSpec,
    metadata_store: MetadataStore,
    artifact_store: ArtifactStore,
    compute: Compute,
    llm_providers: dict[str, LlmProvider] | None,
    config: EngineConfig,
    checkpointer: Checkpointer | None = None,  # noqa: ARG001 — handlers consume; loop only relays
    shutdown_event: asyncio.Event,
    cost_handler: CostRecordHandler | None = None,
    on_cycle_complete: Callable[[WorkerCycleStats], Awaitable[None]] | None = None,
    worker_id: str | None = None,
) -> None:
    """Long-running worker loop driving entities through ``spec`` (`05` §3).

    Loops on :attr:`EngineConfig.poll_interval_seconds`-cadence cycles
    until ``shutdown_event`` is set. Each cycle runs phase 1 → phase 2
    → phase 3 via :func:`run_worker_cycle`; ``on_cycle_complete`` (if
    provided) fires after each cycle with the per-cycle stats.

    The ``checkpointer`` parameter is not consumed by the loop itself —
    dispatch handlers receive it via :class:`DispatchContext` and
    decide per-handler whether to memoize via
    :func:`smai_orchestrator.checkpointer.memoized`. The loop simply
    propagates it to dispatch contexts (today the engine's
    :func:`run_dispatch` constructs the :class:`DispatchContext` with
    ``checkpointer=None`` — wiring the worker's checkpointer through
    is a one-line follow-up; surfaced as a C4 / 3.G3 carry-forward
    rather than re-shaping the engine surface mid-task).

    Per `05` §3.5 / §6, ``worker_count`` is a deployment concern — N=2
    means two worker processes running this loop in parallel, each with
    a distinct ``worker_id``. Task 3.G1's lease wrapper inside
    :func:`drive_entity_phase3` enforces correctness across them via
    per-entity :meth:`MetadataStore.acquire_lease`. Single-process
    deployments (``smai dev``) leave ``worker_id=None`` and let the
    cycle generate a stable per-process id.
    """
    resolved_worker_id = worker_id if worker_id is not None else f"worker-{uuid4().hex[:8]}"
    cycle_id = 0
    cycles_processed = 0
    while not shutdown_event.is_set():
        cycle_start = config.time_provider()
        stats = await run_worker_cycle(
            spec=spec,
            metadata_store=metadata_store,
            artifact_store=artifact_store,
            compute=compute,
            llm_providers=llm_providers,
            config=config,
            cost_handler=cost_handler,
            worker_id=resolved_worker_id,
        )
        cycle_id += 1
        cycles_processed += 1
        # Per round-6 item 6: one INFO line per cycle that actually moved
        # something — idle cycles stay quiet so a steady-state worker
        # doesn't spam the log.
        if (
            stats.phase1_advanced
            or stats.phase1_orphan_reset
            or stats.phase3_advanced
            or stats.phase3_dispatch_failed
        ):
            _log.info(
                "worker %s cycle %d: phase1_advanced=%d orphan_reset=%d "
                "phase3_advanced=%d dispatch_failed=%d candidates=%d",
                resolved_worker_id,
                cycle_id,
                stats.phase1_advanced,
                stats.phase1_orphan_reset,
                stats.phase3_advanced,
                stats.phase3_dispatch_failed,
                stats.phase2_candidates,
            )
        if on_cycle_complete is not None:
            await on_cycle_complete(stats)
        # Per Task 4.K2 (`12` §6.4): fire the worker_heartbeat SSE
        # event after each cycle so the SPA's "last cycle at" indicator
        # advances. Default :class:`NullEventChannel` makes this a no-op
        # for deployments without a paired API consumer.
        try:
            await config.event_channel.fire_heartbeat(
                cycle_id=cycle_id,
                cycles_processed=cycles_processed,
            )
        except Exception:  # noqa: BLE001 — event channel must not break the loop
            _log.exception("event_channel.fire_heartbeat raised on cycle %d; ignoring", cycle_id)

        elapsed = config.time_provider() - cycle_start
        sleep_for = max(0.0, config.poll_interval_seconds - elapsed)
        if sleep_for == 0.0:
            # Yield to the event loop so the shutdown event can be observed
            # even when the cycle takes longer than the poll interval.
            await asyncio.sleep(0)
            continue
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=sleep_for)
        except TimeoutError:
            # Normal cadence — sleep elapsed without shutdown signal.
            continue


__all__ = [
    "SchedulingQueryFn",
    "SchedulingQueryRef",
    "WorkerCycleStats",
    "run_worker_cycle",
    "run_worker_loop",
]
