"""Write-first dispatch ordering.

Per ``designs/smai/05-orchestrator.md`` §1.4. When a phase-3 edge fires
into a state with an ``on_entry_dispatch`` handler, the engine performs
a three-step write-first sequence:

1. CAS the entity's state to the target via :class:`MetadataStore`'s
   ``transition_*_state`` (the new state and ``version + 1`` are set in
   the same UPDATE; the predicate is ``version = expected_version``).
   On :class:`ConflictError` another worker won the race; bail.
2. Submit the external job via the handler (which calls
   :class:`Compute.submit`); receive a :class:`JobHandle`.
3. CAS-record the handle on the entity's configured handle field
   (``DispatchAction.handle_field``).

If step 2 raises (or returns :class:`DispatchOutcome` with a non-None
``error``), the engine forward-rolls-back the entity's state — a
version-incrementing CAS write that resets the state field to the
edge's ``from_state`` (preserving version monotonicity per `05` §1.4).

If the worker crashes between steps 1 and 2, phase-1 orphan detection
(:mod:`phase1`) finds the entity in an in-progress state with a null
handle past ``orphan_grace_seconds`` and resets it.

For inline dispatches (``handle_field=None``) the same shape holds:
step 1 transitions, step 2 runs the handler synchronously, step 3 is
a no-op (no handle to record). Failures in step 2 still trigger the
forward-rolled-back path.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict
from smai_core.plugins import (
    ArtifactStore,
    Compute,
    LeaseToken,
    LlmProvider,
    MetadataStore,
)
from smai_core.plugins.metadata_store import ConflictError, LeaseLostError

from smai_orchestrator.engine._metadata_ops import (
    StateDrivenRecord,
    get_entity,
    transition_state,
)
from smai_orchestrator.engine.config import EngineConfig
from smai_orchestrator.engine.types import (
    DispatchContext,
    DispatchOutcome,
    EdgeDef,
    EngineSpec,
    StateDef,
)

if TYPE_CHECKING:
    from smai_orchestrator.engine.state_machine import DriveOutcome

_log = logging.getLogger(__name__)


class DispatchOutcomeWire(BaseModel):
    """Telemetry payload for a single phase-3 dispatch attempt.

    The engine returns one of these as part of :class:`DriveOutcome`
    when a transition fires (`05` §3.3); it captures the handler's
    own :class:`DispatchOutcome` plus the post-dispatch record so
    callers / tests can assert end-state without a separate read.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    handler_outcome: DispatchOutcome
    final_record: StateDrivenRecord


def _no_handle_field_inline(action_handle_field: str | None) -> bool:
    """Return ``True`` when the action declares no handle field.

    Inline dispatches (single LLM call, no external compute) leave
    :attr:`DispatchAction.handle_field` ``None``. The engine's step 3
    (record-handle CAS) is a no-op in that case; the handler's own
    side effects are the dispatch result, and the transition is
    considered complete after the handler returns successfully.
    """
    return action_handle_field is None


async def run_dispatch(  # noqa: PLR0913
    *,
    spec: EngineSpec,
    metadata_store: MetadataStore,
    artifact_store: ArtifactStore,
    compute: Compute,
    llm: LlmProvider | None,
    config: EngineConfig,
    edge: EdgeDef,
    target_state: StateDef,
    entity_id: str,
    expected_version: int,
    worker_id: str | None = None,
    gate_outcome_reason: str | None = None,
) -> DriveOutcome:
    """Execute one full write-first dispatch sequence (`05` §1.4).

    Returns :class:`DriveOutcome`-shaped status the caller (the
    :func:`drive_entity_phase3` driver in :mod:`state_machine`) wraps
    into the consolidated phase-3 result.

    ``worker_id`` is threaded into the ``transition_log`` audit row so
    multi-worker contention is coherent in audit queries; ``None`` is
    the test default. ``gate_outcome_reason`` is the
    :attr:`GateOutcome.reason` of the edge that fired (when the caller
    has it), recorded on the step-1 audit row.
    """
    # Lazy import to break the import cycle with :mod:`state_machine`,
    # which imports :func:`run_dispatch` at module-import time.
    from smai_orchestrator.engine.state_machine import DriveOutcome

    # ---- Step 1: CAS state to the target ---------------------------------
    try:
        post_transition_record = await transition_state(
            metadata_store,
            spec.entity_kind,
            entity_id,
            expected_version,
            target_state.name,
            fields={},
            event_channel=config.event_channel,
            from_state=edge.from_state,
            edge_name=edge.name,
            worker_id=worker_id,
            gate_outcome_reason=gate_outcome_reason,
        )
    except ConflictError:
        return DriveOutcome(status="conflict", fired_edge=edge)

    # Two paths: terminal-or-no-dispatch vs. real dispatch action.
    action = target_state.on_entry_dispatch
    if action is None:
        return DriveOutcome(
            status="advanced",
            fired_edge=edge,
            dispatch_outcome=DispatchOutcomeWire(
                handler_outcome=DispatchOutcome(),
                final_record=post_transition_record,
            ),
        )

    # ---- Step 2: run the handler (which may call Compute.submit) --------
    dispatch_ctx = DispatchContext(
        entity_kind=spec.entity_kind,
        entity_id=entity_id,
        entity_state=target_state.name,
        entity_version=post_transition_record.version,
        metadata_store=metadata_store,
        artifact_store=artifact_store,
        compute=compute,
        llm=llm,
        config=config,
        checkpointer=None,
    )

    handler_outcome: DispatchOutcome
    handler_failed = False
    handler_error: str | None = None
    try:
        handler_outcome = await action.handler(dispatch_ctx)
        if handler_outcome.error is not None:
            handler_failed = True
            handler_error = handler_outcome.error
    except Exception as exc:  # noqa: BLE001 — we report and roll back
        handler_failed = True
        handler_error = repr(exc)
        handler_outcome = DispatchOutcome(error=handler_error)

    if handler_failed:
        rollback_status = await _forward_rollback(
            spec=spec,
            metadata_store=metadata_store,
            config=config,
            edge=edge,
            entity_id=entity_id,
            current_version=post_transition_record.version,
            worker_id=worker_id,
        )
        return DriveOutcome(
            status=rollback_status,
            fired_edge=edge,
            error=handler_error,
        )

    # ---- Step 3: persist the JobHandle on the entity ---------------------
    # Inline dispatches skip step 3 (no handle to record).
    if _no_handle_field_inline(action.handle_field):
        return DriveOutcome(
            status="advanced",
            fired_edge=edge,
            dispatch_outcome=DispatchOutcomeWire(
                handler_outcome=handler_outcome,
                final_record=post_transition_record,
            ),
        )

    if not handler_outcome.submitted_handles:
        # External-dispatch handler returned no handle. Treat as a
        # handler error: the engine's invariant is "external dispatches
        # produce a handle for phase-1 polling"; absent that, the entity
        # is half-transitioned and would otherwise be stuck.
        rollback_status = await _forward_rollback(
            spec=spec,
            metadata_store=metadata_store,
            config=config,
            edge=edge,
            entity_id=entity_id,
            current_version=post_transition_record.version,
            worker_id=worker_id,
        )
        return DriveOutcome(
            status=rollback_status,
            fired_edge=edge,
            error=(
                f"dispatch handler {action.name!r} declared "
                f"handle_field={action.handle_field!r} but returned no JobHandle"
            ),
        )

    handle = handler_outcome.submitted_handles[0]
    assert action.handle_field is not None  # narrowing for pyright; checked above
    # Re-read before the step-3 handle write so that any field-only
    # writes the handler itself performed (e.g. bumping a retry counter
    # via ``transition_*_state`` with the state preserved — the
    # ``cg_execution.py`` / ``proposal.py`` / ``run_record.py`` pattern)
    # don't make step 3's CAS spuriously conflict. A *state* change by a
    # peer between here and the re-read still surfaces as ``conflict``
    # (the state guard on ``transition_state`` catches it).
    handle_expected_version = post_transition_record.version
    refreshed = await get_entity(metadata_store, spec.entity_kind, entity_id)
    if refreshed is not None and refreshed.state == target_state.name:
        handle_expected_version = refreshed.version
    try:
        post_handle_record = await transition_state(
            metadata_store,
            spec.entity_kind,
            entity_id,
            handle_expected_version,
            target_state.name,
            fields={action.handle_field: handle.model_dump(mode="python")},
        )
    except ConflictError:
        # Step 3's CAS lost — another worker observed the no-handle
        # in-progress entity and acted (orphan reclaim, manual
        # intervention). The handler already submitted the job; we
        # surface a ``conflict`` status so the worker bails. The next
        # phase-1 cycle will reconcile.
        return DriveOutcome(
            status="conflict",
            fired_edge=edge,
            dispatch_outcome=DispatchOutcomeWire(
                handler_outcome=handler_outcome,
                final_record=post_transition_record,
            ),
        )

    return DriveOutcome(
        status="advanced",
        fired_edge=edge,
        dispatch_outcome=DispatchOutcomeWire(
            handler_outcome=handler_outcome,
            final_record=post_handle_record,
        ),
    )


async def _forward_rollback(
    *,
    spec: EngineSpec,
    metadata_store: MetadataStore,
    config: EngineConfig,
    edge: EdgeDef,
    entity_id: str,
    current_version: int,
    worker_id: str | None = None,
) -> Literal["dispatch_failed_rolled_back", "conflict"]:
    """Forward-roll-back the entity's state to ``edge.from_state`` (`05` §1.4).

    Per the spec: a *version-incrementing* forward write that resets
    the state field — NOT a version decrement. Version monotonicity is
    a load-bearing invariant of CAS discipline; rolling the version
    backwards would expose every concurrent reader to ABA shapes.

    On CAS conflict (someone else has already moved the entity since
    step 1), surface ``"conflict"`` so the caller drops the result;
    the next cycle's phase-1 polling reconciles.
    """
    try:
        await transition_state(
            metadata_store,
            spec.entity_kind,
            entity_id,
            current_version,
            edge.from_state,
            fields={},
            event_channel=config.event_channel,
            from_state=edge.target_state,
            edge_name=f"{edge.name}:rollback",
            worker_id=worker_id,
        )
    except ConflictError:
        return "conflict"
    return "dispatch_failed_rolled_back"


async def reset_orphan(  # noqa: PLR0913
    *,
    spec: EngineSpec,
    metadata_store: MetadataStore,
    config: EngineConfig,
    edge_from_state: str,
    entity_id: str,
    current_version: int,
    in_progress_state: str,
    worker_id: str | None = None,
) -> StateDrivenRecord | None:
    """Reset an orphaned in-progress entity to ``edge_from_state``.

    Used by phase-1 orphan detection (`05` §3.1) when an entity has
    been in an in-progress state past ``orphan_grace_seconds`` with a
    null ``job_handle`` (the worker crashed between steps 1 and 2 of
    write-first ordering). Returns the post-rollback record on success
    or ``None`` on CAS conflict (another worker already reset it; this
    worker proceeds).

    ``in_progress_state`` is the state the orphan currently sits in;
    threaded through so the fire-on-transition hook (per Task 4.K2 /
    `12` §6.4) emits ``state_change`` with the correct ``from``.
    """
    try:
        return await transition_state(
            metadata_store,
            spec.entity_kind,
            entity_id,
            current_version,
            edge_from_state,
            fields={},
            event_channel=config.event_channel,
            from_state=in_progress_state,
            edge_name="(orphan-reset)",
            worker_id=worker_id,
        )
    except ConflictError:
        return None


async def _heartbeat_loop(
    *,
    metadata_store: MetadataStore,
    token_ref: list[LeaseToken],
    config: EngineConfig,
) -> None:
    """Periodically extend the lease until cancelled or
    :class:`LeaseLostError` fires (`05` §3.5 / `07` §5.6.7).

    The current lease token lives in the single-element ``token_ref``
    list (a mutable container so :func:`run_dispatch_with_lease`'s
    ``finally`` block sees the latest token for ``release_lease``).
    Each successful extension swaps in the fresh token (new ``nonce``
    and ``expires_at``).

    Raises :class:`LeaseLostError` if the lease has already been
    expired-and-reacquired by another worker — callers handle the
    raise by cancelling the in-flight dispatch and surfacing
    ``status="conflict"``.
    """
    while True:
        await asyncio.sleep(config.extend_lease_interval_seconds)
        new_token = await metadata_store.extend_lease(
            token_ref[0],
            additional_seconds=config.lease_seconds,
        )
        token_ref[0] = new_token


async def run_dispatch_with_lease(  # noqa: PLR0913
    *,
    spec: EngineSpec,
    metadata_store: MetadataStore,
    artifact_store: ArtifactStore,
    compute: Compute,
    llm: LlmProvider | None,
    config: EngineConfig,
    edge: EdgeDef,
    target_state: StateDef,
    entity_id: str,
    expected_version: int,
    worker_id: str,
    gate_outcome_reason: str | None = None,
) -> DriveOutcome:
    """Lease-wrapped phase-3 dispatch (`05` §3.5 / DEC-035 #2).

    Acquires a per-entity lease via :meth:`MetadataStore.acquire_lease`
    around the existing :func:`run_dispatch` body so concurrent workers
    cannot both fire the dispatch handler for the same entity. Spawns
    a background heartbeat that calls :meth:`MetadataStore.extend_lease`
    every :attr:`EngineConfig.extend_lease_interval_seconds` while the
    dispatch is in flight; cancels the heartbeat cleanly on dispatch
    completion and on :class:`LeaseLostError` (where the dispatch is
    aborted via :meth:`asyncio.Task.cancel`).

    Outcomes:

    * ``acquire_lease`` returns ``None`` (another worker holds the
      lease and it has not yet expired): return
      ``DriveOutcome(status="lease_held")`` without firing the handler.
      The next worker cycle re-examines the entity once the lease
      expires.
    * Heartbeat raises :class:`LeaseLostError` mid-dispatch (the lease
      expired and was reacquired by another worker): cancel the
      in-flight dispatch task and return
      ``DriveOutcome(status="conflict", error="lease_lost")``. The new
      lease holder will reconcile from whatever state the cancelled
      dispatch left behind (orphan-grace reset for null-handle in-progress
      entities; phase-1 polling for handle-recorded entities).
    * Otherwise: return whatever :func:`run_dispatch` returned. The
      ``finally`` block always calls :meth:`MetadataStore.release_lease`
      with the latest token; release is idempotent (a stale token is a
      silent no-op per `07` §5.6.7).

    The lease wrapper is the canonical multi-worker correctness
    primitive per DEC-035 #2; orphan-detection (:func:`reset_orphan`)
    remains the gap-filler for crash-between-steps cases (worker dies
    after ``acquire_lease`` succeeds but before lease expiry — the
    expiry window is the recovery deadline).
    """
    # Lazy import to break the import cycle with :mod:`state_machine`,
    # which imports :func:`run_dispatch_with_lease` at module-import
    # time.
    from smai_orchestrator.engine.state_machine import DriveOutcome

    token = await metadata_store.acquire_lease(
        spec.entity_kind,
        entity_id,
        config.lease_seconds,
        worker_id,
    )
    if token is None:
        _log.debug(
            "phase-3 lease held by another worker for %s/%s; skipping this cycle",
            spec.entity_kind,
            entity_id,
        )
        return DriveOutcome(status="lease_held", fired_edge=edge)

    token_ref: list[LeaseToken] = [token]
    heartbeat = asyncio.create_task(
        _heartbeat_loop(
            metadata_store=metadata_store,
            token_ref=token_ref,
            config=config,
        )
    )
    dispatch_task = asyncio.create_task(
        run_dispatch(
            spec=spec,
            metadata_store=metadata_store,
            artifact_store=artifact_store,
            compute=compute,
            llm=llm,
            config=config,
            edge=edge,
            target_state=target_state,
            entity_id=entity_id,
            expected_version=expected_version,
            worker_id=worker_id,
            gate_outcome_reason=gate_outcome_reason,
        )
    )

    try:
        done, _pending = await asyncio.wait(
            {dispatch_task, heartbeat},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if dispatch_task in done:
            # Normal path: dispatch finished first; tear down the
            # heartbeat task and return the dispatch's outcome.
            heartbeat.cancel()
            try:
                await heartbeat
            except (asyncio.CancelledError, LeaseLostError):
                # Cancellation is the expected exit; LeaseLostError
                # raised during teardown is irrelevant — dispatch
                # already finished, so we don't need to abort.
                pass
            except Exception:  # noqa: BLE001 — log + swallow per `05` §3
                _log.exception(
                    "lease heartbeat raised unexpected error during teardown for %s/%s",
                    spec.entity_kind,
                    entity_id,
                )
            return await dispatch_task

        # Heartbeat finished first — almost certainly raised
        # LeaseLostError. Cancel the in-flight dispatch and surface
        # ``status="conflict"``.
        dispatch_task.cancel()
        try:
            await dispatch_task
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001 — best-effort
            _log.exception(
                "in-flight dispatch raised during cancellation for %s/%s",
                spec.entity_kind,
                entity_id,
            )

        try:
            await heartbeat
        except LeaseLostError:
            _log.info(
                "lease lost mid-dispatch for %s/%s; aborting dispatch and surfacing conflict",
                spec.entity_kind,
                entity_id,
            )
            return DriveOutcome(
                status="conflict",
                fired_edge=edge,
                error="lease_lost",
            )
        # Unreachable: the heartbeat loop is an infinite loop that only
        # exits via cancellation or LeaseLostError. If it returned
        # cleanly something is very wrong; treat as conflict.
        return DriveOutcome(
            status="conflict",
            fired_edge=edge,
            error="heartbeat_exited_unexpectedly",
        )
    finally:
        # Best-effort lease release; idempotent per `07` §5.6.7 (stale
        # token is a silent no-op). Use the latest token in case the
        # heartbeat extended.
        try:
            await metadata_store.release_lease(token_ref[0])
        except Exception:  # noqa: BLE001 — release-on-teardown must not raise
            _log.exception(
                "release_lease raised during dispatch teardown for %s/%s",
                spec.entity_kind,
                entity_id,
            )


__all__ = [
    "DispatchOutcomeWire",
    "reset_orphan",
    "run_dispatch",
    "run_dispatch_with_lease",
]
