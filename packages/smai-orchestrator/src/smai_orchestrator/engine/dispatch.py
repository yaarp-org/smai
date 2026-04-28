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

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict
from smai_core.plugins import (
    ArtifactStore,
    Compute,
    LlmProvider,
    MetadataStore,
)
from smai_core.plugins.metadata_store import ConflictError

from smai_orchestrator.engine._metadata_ops import (
    StateDrivenRecord,
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
) -> DriveOutcome:
    """Execute one full write-first dispatch sequence (`05` §1.4).

    Returns :class:`DriveOutcome`-shaped status the caller (the
    :func:`drive_entity_phase3` driver in :mod:`state_machine`) wraps
    into the consolidated phase-3 result.
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
            edge=edge,
            entity_id=entity_id,
            current_version=post_transition_record.version,
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
            edge=edge,
            entity_id=entity_id,
            current_version=post_transition_record.version,
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
    try:
        post_handle_record = await transition_state(
            metadata_store,
            spec.entity_kind,
            entity_id,
            post_transition_record.version,
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
    edge: EdgeDef,
    entity_id: str,
    current_version: int,
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
        )
    except ConflictError:
        return "conflict"
    return "dispatch_failed_rolled_back"


async def reset_orphan(
    *,
    spec: EngineSpec,
    metadata_store: MetadataStore,
    edge_from_state: str,
    entity_id: str,
    current_version: int,
) -> StateDrivenRecord | None:
    """Reset an orphaned in-progress entity to ``edge_from_state``.

    Used by phase-1 orphan detection (`05` §3.1) when an entity has
    been in an in-progress state past ``orphan_grace_seconds`` with a
    null ``job_handle`` (the worker crashed between steps 1 and 2 of
    write-first ordering). Returns the post-rollback record on success
    or ``None`` on CAS conflict (another worker already reset it; this
    worker proceeds).
    """
    try:
        return await transition_state(
            metadata_store,
            spec.entity_kind,
            entity_id,
            current_version,
            edge_from_state,
            fields={},
        )
    except ConflictError:
        return None


__all__ = [
    "DispatchOutcomeWire",
    "reset_orphan",
    "run_dispatch",
]
