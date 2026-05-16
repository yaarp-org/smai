"""Round-6 friction A: a dispatch handler returning an error must not
leave the entity silently parked in its dispatch state.

The pre-round-6 bug: the planner dispatch returned
``DispatchOutcome(error=...)`` after bumping ``design_attempt`` (which
bumps the row version). The engine's forward-rollback CAS was keyed on
the *pre-handler* version, so it spuriously conflicted, and the
proposal stayed wedged in ``designing`` with a null handle — invisible
to the phase-1 scheduling query that filters on a non-null handle, so
never re-discovered, never re-dispatched, never terminated.

The round-6 fix (:func:`smai_orchestrator.engine.dispatch._handle_dispatch_failure`):
re-read for a fresh version, then either advance to a spec-declared
error-handling edge (the ``*_failed`` retry-exhausted terminal) or
forward-roll-back to ``edge.from_state`` — recording ``last_error`` in
both cases, and writing a ``transition_log`` row for each step.

Round-10 (declarative retry bookkeeping): the synthesized retry-
exhausted edge replaces what used to be a manually-declared
``*_failed (retry exhausted)`` EdgeDef. Tests below declare a
:class:`RetryPolicy` on the dispatch action instead of a manual gate
factory; the engine synthesizes the terminal during
:func:`_handle_dispatch_failure` and the failing handler no longer
needs to bump the counter itself (the engine does that on step-1's
CAS).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import pytest
from smai_orchestrator.engine import (
    DispatchAction,
    DispatchContext,
    DispatchOutcome,
    EdgeDef,
    EngineConfig,
    EngineSpec,
    RetryPolicy,
    StateDef,
    drive_entity_phase3,
)
from smai_orchestrator.engine.types import GateContext, GateOutcome
from smai_orchestrator.entities.tracking import ProposalRecord
from sqlalchemy import text

# === Spec mirroring the proposal pipeline's designing-state shape ===========


def _failing_planner_handler(error: str) -> Any:
    """A dispatch handler that returns a ``DispatchOutcome`` with an
    error and no handle. Round-10: the handler no longer bumps the
    retry counter itself — the engine does that on step-1's CAS via
    the :class:`RetryPolicy` on the dispatch action."""

    async def _handler(ctx: DispatchContext) -> DispatchOutcome:
        del ctx
        return DispatchOutcome(submitted_handles=[], error=error)

    return _handler


def _spec(*, handler: Any, max_design_attempts: int) -> EngineSpec:
    """Build a minimal proposal-shaped EngineSpec wired with a round-10
    :class:`RetryPolicy` on ``designing``. The engine synthesizes the
    retry-exhausted terminal off ``designing`` from the policy — no
    manual ``designing → failed`` EdgeDef is declared here, matching the
    real proposal spec after the round-10 migration."""

    async def _gate_enter(ctx: GateContext) -> GateOutcome:
        del ctx
        return GateOutcome(advance=True, reason="enter designing")

    return EngineSpec(
        entity_kind="proposal",
        initial_state="proposal_submitted",
        states=[
            StateDef(name="proposal_submitted"),
            StateDef(
                name="designing",
                on_entry_dispatch=DispatchAction(
                    name="proposal.dispatch_planner",
                    handler=handler,
                    pool="proposal_pipeline",
                    handle_field="planner_job_handle",
                    retry_policy=RetryPolicy(
                        attempt_counter_field="design_attempt",
                        max_attempts=max_design_attempts,
                        on_exhaustion_target_state="failed",
                        on_exhaustion_reason="design retry budget exhausted; terminal",
                    ),
                ),
            ),
            StateDef(name="failed", is_terminal=True),
        ],
        edges=[
            EdgeDef(
                name="proposal.proposal_submitted → designing",
                from_state="proposal_submitted",
                target_state="designing",
                gate_rule=_gate_enter,
                fires_on="dispatch_time",
            ),
            # No manual ``designing → failed`` edge — round 10's engine
            # synthesis covers it from the RetryPolicy above.
        ],
    )


async def _seed_proposal(store: Any) -> ProposalRecord:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    proposal = ProposalRecord(
        id="prop_wedge_test",
        submission_kind="novel_technique",
        state="proposal_submitted",
        created_at=now,
        updated_at=now,
    )
    return await store.create_proposal(proposal)


async def _transition_log_rows(store: Any, entity_id: str) -> list[dict[str, Any]]:
    async with await store.transaction() as tx:
        conn: Any = tx.connection
        result = await conn.execute(
            text(
                "SELECT from_state, to_state, edge_name FROM transition_log "
                "WHERE entity_kind = :k AND entity_id = :i ORDER BY id"
            ),
            {"k": "proposal", "i": entity_id},
        )
        return [dict(r) for r in result.mappings().all()]


# === Tests ===================================================================


async def test_handler_error_does_not_leave_entity_parked(
    sqlite_store, fake_compute, fake_artifact_store
) -> None:
    """One drive cycle: enter ``designing`` (engine bumps the counter
    to 1 on step-1 CAS) → handler returns an error → engine evaluates
    the synthesized retry-exhausted edge (count=1, cap=2 → does NOT
    fire) → engine rolls the proposal back to ``proposal_submitted``
    with ``last_error`` set, and a rollback row lands in
    ``transition_log``."""
    proposal = await _seed_proposal(sqlite_store)
    spec = _spec(
        handler=_failing_planner_handler("planner did not finalize"),
        max_design_attempts=2,
    )

    outcome = await drive_entity_phase3(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        llm=None,
        config=EngineConfig(),
        record=proposal,
    )
    assert outcome.status == "dispatch_failed_rolled_back"
    assert outcome.error == "planner did not finalize"

    final = await sqlite_store.get_proposal("prop_wedge_test")
    assert final is not None
    # NOT wedged in designing — rolled back to be re-dispatched.
    assert final.state == "proposal_submitted"
    assert final.last_error == "planner did not finalize"
    assert final.design_attempt == 1  # engine bumped it on step-1 CAS
    assert final.planner_job_handle is None

    rows = await _transition_log_rows(sqlite_store, "prop_wedge_test")
    pairs = [(r["from_state"], r["to_state"]) for r in rows]
    assert ("proposal_submitted", "designing") in pairs
    assert ("designing", "proposal_submitted") in pairs  # the rollback row


async def test_repeated_failures_reach_failed_terminal_with_last_error(
    sqlite_store, fake_compute, fake_artifact_store
) -> None:
    """Re-dispatching until ``design_attempt >= max_design_attempts``
    advances the proposal to the ``failed`` terminal with ``last_error``
    populated via the engine-synthesized retry-exhausted edge."""
    await _seed_proposal(sqlite_store)
    spec = _spec(
        handler=_failing_planner_handler("planner did not finalize"),
        max_design_attempts=2,
    )
    config = EngineConfig()

    final_state: str | None = None
    last_outcome_status: str | None = None
    for _ in range(6):
        rec = await sqlite_store.get_proposal("prop_wedge_test")
        assert rec is not None
        if rec.state in {"failed", "registered", "rejected"}:
            final_state = rec.state
            break
        outcome = await drive_entity_phase3(
            spec=spec,
            metadata_store=sqlite_store,
            artifact_store=fake_artifact_store,
            compute=fake_compute,
            llm=None,
            config=config,
            record=rec,
        )
        last_outcome_status = outcome.status

    assert final_state == "failed", f"got {final_state}"
    # The hop that routed the proposal to the ``failed`` terminal is
    # reported as a *failure* (``dispatch_failed_routed``), not a clean
    # ``advanced`` — the synthesized edge fires inside
    # ``_handle_dispatch_failure``.
    assert last_outcome_status == "dispatch_failed_routed"
    final = await sqlite_store.get_proposal("prop_wedge_test")
    assert final is not None
    assert final.last_error == "planner did not finalize"
    assert final.design_attempt == 2

    rows = await _transition_log_rows(sqlite_store, "prop_wedge_test")
    pairs = [(r["from_state"], r["to_state"]) for r in rows]
    # The final hop into the terminal is recorded with the engine-
    # synthesized edge name.
    assert ("designing", "failed") in pairs
    fail_row = next(r for r in rows if (r["from_state"], r["to_state"]) == ("designing", "failed"))
    assert "synthesized retry-exhausted terminal" in fail_row["edge_name"]


async def test_raising_handler_also_propagates(
    sqlite_store, fake_compute, fake_artifact_store
) -> None:
    """A handler that *raises* (rather than returns an error) takes the
    same path: rollback to ``edge.from_state`` with ``last_error`` set."""

    async def _raises(ctx: DispatchContext) -> DispatchOutcome:
        del ctx
        raise RuntimeError("planner crashed")

    proposal = await _seed_proposal(sqlite_store)
    spec = _spec(handler=_raises, max_design_attempts=2)

    outcome = await drive_entity_phase3(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        llm=None,
        config=EngineConfig(),
        record=proposal,
    )
    assert outcome.status == "dispatch_failed_rolled_back"
    final = await sqlite_store.get_proposal("prop_wedge_test")
    assert final is not None
    assert final.state == "proposal_submitted"
    assert final.last_error is not None
    assert "planner crashed" in final.last_error


async def test_dispatch_outcome_logged_at_info(
    sqlite_store, fake_compute, fake_artifact_store, caplog: pytest.LogCaptureFixture
) -> None:
    """Round-6 item 6: the dispatch + dispatch-outcome lines are emitted at
    INFO so an operator running ``smai dev -v`` sees what the worker is
    doing (and that a dispatch failed)."""
    proposal = await _seed_proposal(sqlite_store)
    spec = _spec(handler=_failing_planner_handler("boom"), max_design_attempts=9)
    with caplog.at_level(logging.INFO, logger="smai_orchestrator.engine.dispatch"):
        await drive_entity_phase3(
            spec=spec,
            metadata_store=sqlite_store,
            artifact_store=fake_artifact_store,
            compute=fake_compute,
            llm=None,
            config=EngineConfig(),
            record=proposal,
        )
    text = "\n".join(r.message for r in caplog.records)
    assert "dispatching proposal.dispatch_planner for proposal/prop_wedge_test" in text
    assert "outcome=failed" in text and "boom" in text
