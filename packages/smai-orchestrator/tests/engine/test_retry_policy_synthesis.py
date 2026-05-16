"""Round-10 declarative retry-bookkeeping primitive — engine-level synthesis.

Pre-round-10 every dispatched state with a retry budget needed two
pieces of manual machinery: a handler-side ``transition_*_state(...,
<counter>=record.<counter>+1)`` call, and a ``*_failed (retry
exhausted)`` ``EdgeDef`` whose gate predicate read the counter and
fired at the cap. Rounds 5 / 6 / 8 each surfaced a state where one or
both pieces were forgotten and the dispatch infinitely retried.

Round 10 made both declarative via :class:`RetryPolicy` on
:class:`DispatchAction`:

* :func:`run_dispatch` bundles the counter bump into step-1's CAS
  ``fields=``, riding the state-transition write — no separate write,
  no version race against handler-internal writes.
* :func:`_handle_dispatch_failure` synthesizes a retry-exhausted
  terminal edge whose gate predicate is
  ``record.<counter> >= max_attempts``, prepended to the dispatch
  state's manually-declared outgoing edges (declaration-first-equivalent
  per the round-8 ordering rule).

These tests pin both halves of the synthesis at the engine level —
independent of any spec. The spec-level wiring tests (proposal /
cg_execution / run / paper) live in :mod:`tests.specs.test_retry_counters`
and the per-spec test modules.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from smai_orchestrator.engine import (
    DispatchAction,
    DispatchContext,
    DispatchOutcome,
    DriveOutcome,
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

# === Helpers ================================================================


async def _seed_proposal(store: Any, state: str = "proposal_submitted") -> ProposalRecord:
    now = datetime(2026, 5, 13, tzinfo=UTC)
    proposal = ProposalRecord(
        id="prop_synth_test",
        submission_kind="novel_technique",
        state=state,  # type: ignore[arg-type]
        created_at=now,
        updated_at=now,
    )
    return await store.create_proposal(proposal)


def _spec_with_retry_policy(
    *,
    handler: Any,
    max_attempts: int,
    on_exhaustion_target_state: str = "failed",
    on_exhaustion_reason: str = "design retry budget exhausted; terminal",
) -> EngineSpec:
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
                    name="dispatch.synth_test",
                    handler=handler,
                    pool="proposal_pipeline",
                    handle_field="planner_job_handle",
                    retry_policy=RetryPolicy(
                        attempt_counter_field="design_attempt",
                        max_attempts=max_attempts,
                        on_exhaustion_target_state=on_exhaustion_target_state,
                        on_exhaustion_reason=on_exhaustion_reason,
                    ),
                ),
            ),
            StateDef(name=on_exhaustion_target_state, is_terminal=True),
        ],
        edges=[
            EdgeDef(
                name="proposal_submitted → designing",
                from_state="proposal_submitted",
                target_state="designing",
                gate_rule=_gate_enter,
                fires_on="dispatch_time",
            ),
        ],
    )


async def _transition_rows(store: Any, entity_id: str) -> list[dict[str, Any]]:
    async with await store.transaction() as tx:
        conn: Any = tx.connection
        result = await conn.execute(
            text(
                "SELECT from_state, to_state, edge_name, gate_outcome_reason "
                "FROM transition_log WHERE entity_kind = :k AND entity_id = :i "
                "ORDER BY id"
            ),
            {"k": "proposal", "i": entity_id},
        )
        return [dict(r) for r in result.mappings().all()]


# === Counter-bump-on-step-1-CAS =============================================


async def test_engine_bumps_counter_on_step_1_cas_on_first_entry(
    sqlite_store, fake_compute, fake_artifact_store
) -> None:
    """Round-10: the engine bumps the counter as part of step-1's CAS
    even when the handler succeeds. A round-trip into ``designing`` →
    handler-succeeds (returns DispatchOutcome with no error and no
    handle for inline) should leave ``design_attempt == 1``."""

    async def _inline_success(ctx: DispatchContext) -> DispatchOutcome:
        del ctx
        return DispatchOutcome()  # success, inline (no handle)

    proposal = await _seed_proposal(sqlite_store)
    # Make ``designing`` an inline state by stripping the handle field
    # via a spec rebuild.
    spec = EngineSpec(
        entity_kind="proposal",
        initial_state="proposal_submitted",
        states=[
            StateDef(name="proposal_submitted"),
            StateDef(
                name="designing",
                on_entry_dispatch=DispatchAction(
                    name="dispatch.inline_success",
                    handler=_inline_success,
                    pool="proposal_pipeline",
                    retry_policy=RetryPolicy(
                        attempt_counter_field="design_attempt",
                        max_attempts=2,
                        on_exhaustion_target_state="failed",
                        on_exhaustion_reason="design retry budget exhausted; terminal",
                    ),
                ),
            ),
            StateDef(name="failed", is_terminal=True),
        ],
        edges=[
            EdgeDef(
                name="proposal_submitted → designing",
                from_state="proposal_submitted",
                target_state="designing",
                gate_rule=_const_advance_gate("enter"),
                fires_on="dispatch_time",
            ),
        ],
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
    assert outcome.status == "advanced"

    after = await sqlite_store.get_proposal("prop_synth_test")
    assert after is not None
    assert after.state == "designing"
    assert after.design_attempt == 1


def _const_advance_gate(reason: str):  # type: ignore[no-untyped-def]
    async def _gate(ctx: GateContext) -> GateOutcome:
        del ctx
        return GateOutcome(advance=True, reason=reason)

    return _gate


async def test_engine_bumps_counter_then_synthesized_terminal_fires(
    sqlite_store, fake_compute, fake_artifact_store
) -> None:
    """A failing handler with ``max_attempts=1`` reaches the
    synthesized terminal on the very first failure — the engine bumps
    the counter to 1 on step-1's CAS, the handler errors, and the
    synthesized edge's gate (``count >= 1``) advances to ``failed``."""

    async def _always_fails(ctx: DispatchContext) -> DispatchOutcome:
        del ctx
        return DispatchOutcome(error="planner crashed on first turn")

    proposal = await _seed_proposal(sqlite_store)
    spec = _spec_with_retry_policy(handler=_always_fails, max_attempts=1)

    outcome = await drive_entity_phase3(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        llm=None,
        config=EngineConfig(),
        record=proposal,
    )
    assert outcome.status == "dispatch_failed_routed"
    assert outcome.error == "planner crashed on first turn"

    final = await sqlite_store.get_proposal("prop_synth_test")
    assert final is not None
    assert final.state == "failed"
    assert final.design_attempt == 1
    assert final.last_error == "planner crashed on first turn"

    rows = await _transition_rows(sqlite_store, "prop_synth_test")
    pairs = [(r["from_state"], r["to_state"]) for r in rows]
    assert ("proposal_submitted", "designing") in pairs
    assert ("designing", "failed") in pairs
    # The terminal hop's edge_name carries the synthesized marker so
    # operators can distinguish engine-synthesized terminals from manual
    # ones in audit queries.
    terminal_row = next(
        r for r in rows if (r["from_state"], r["to_state"]) == ("designing", "failed")
    )
    assert "synthesized retry-exhausted terminal" in terminal_row["edge_name"]
    # gate_outcome_reason matches RetryPolicy.on_exhaustion_reason.
    assert "design retry budget exhausted" in (terminal_row["gate_outcome_reason"] or "")


async def test_no_retry_policy_means_no_engine_bump_or_terminal(
    sqlite_store, fake_compute, fake_artifact_store
) -> None:
    """When the dispatch action has :attr:`retry_policy=None`, the
    engine MUST NOT bump any counter (the bump is opt-in via the
    policy) and MUST NOT synthesize a terminal — a failing handler
    forward-rolls-back to ``edge.from_state`` indefinitely."""

    async def _always_fails(ctx: DispatchContext) -> DispatchOutcome:
        del ctx
        return DispatchOutcome(error="boom")

    proposal = await _seed_proposal(sqlite_store)
    spec = EngineSpec(
        entity_kind="proposal",
        initial_state="proposal_submitted",
        states=[
            StateDef(name="proposal_submitted"),
            StateDef(
                name="designing",
                on_entry_dispatch=DispatchAction(
                    name="dispatch.no_policy",
                    handler=_always_fails,
                    pool="proposal_pipeline",
                    handle_field="planner_job_handle",
                    # No retry_policy — opt-out shape.
                ),
            ),
        ],
        edges=[
            EdgeDef(
                name="proposal_submitted → designing",
                from_state="proposal_submitted",
                target_state="designing",
                gate_rule=_const_advance_gate("enter"),
                fires_on="dispatch_time",
            ),
        ],
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
    final = await sqlite_store.get_proposal("prop_synth_test")
    assert final is not None
    assert final.state == "proposal_submitted"  # rolled back
    assert final.design_attempt == 0  # engine did NOT bump it (no policy)
    assert final.last_error == "boom"


async def test_synthesized_terminal_pre_empts_manual_dispatch_time_edge(
    sqlite_store, fake_compute, fake_artifact_store
) -> None:
    """When the budget is exhausted, the synthesized retry-exhausted
    edge fires BEFORE any manually-declared dispatch_time edge from
    the dispatch state — declaration-first-equivalent, matching the
    round-8 ordering rule. Tests that against a manual edge that would
    otherwise advance the entity somewhere else."""

    async def _always_fails(ctx: DispatchContext) -> DispatchOutcome:
        del ctx
        return DispatchOutcome(error="repeated failure")

    async def _decoy_advance(ctx: GateContext) -> GateOutcome:
        del ctx
        # Always advance — would route to a non-terminal if the
        # synthesized edge weren't pre-empting.
        return GateOutcome(advance=True, reason="decoy")

    proposal = await _seed_proposal(sqlite_store)
    spec = EngineSpec(
        entity_kind="proposal",
        initial_state="proposal_submitted",
        states=[
            StateDef(name="proposal_submitted"),
            StateDef(
                name="designing",
                on_entry_dispatch=DispatchAction(
                    name="dispatch.synth_pre_empts",
                    handler=_always_fails,
                    pool="proposal_pipeline",
                    handle_field="planner_job_handle",
                    retry_policy=RetryPolicy(
                        attempt_counter_field="design_attempt",
                        max_attempts=1,
                        on_exhaustion_target_state="failed",
                        on_exhaustion_reason="design retry budget exhausted; terminal",
                    ),
                ),
            ),
            StateDef(name="designed"),
            StateDef(name="failed", is_terminal=True),
        ],
        edges=[
            EdgeDef(
                name="proposal_submitted → designing",
                from_state="proposal_submitted",
                target_state="designing",
                gate_rule=_const_advance_gate("enter"),
                fires_on="dispatch_time",
            ),
            # Decoy: a manual dispatch_time edge from ``designing`` that
            # would always advance. The synthesized terminal MUST win.
            EdgeDef(
                name="designing → designed (decoy, would always fire)",
                from_state="designing",
                target_state="designed",
                gate_rule=_decoy_advance,
                fires_on="dispatch_time",
            ),
        ],
    )

    outcome: DriveOutcome = await drive_entity_phase3(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        llm=None,
        config=EngineConfig(),
        record=proposal,
    )
    assert outcome.status == "dispatch_failed_routed"
    assert outcome.fired_edge is not None
    assert "synthesized retry-exhausted terminal" in outcome.fired_edge.name
    final = await sqlite_store.get_proposal("prop_synth_test")
    assert final is not None
    assert final.state == "failed"  # NOT ``designed`` — synthesized edge won


async def test_terminal_is_sticky_no_further_transitions(
    sqlite_store, fake_compute, fake_artifact_store
) -> None:
    """After the synthesized terminal fires, subsequent phase-3 drives
    are no-ops (the terminal state has no outgoing edges)."""

    async def _always_fails(ctx: DispatchContext) -> DispatchOutcome:
        del ctx
        return DispatchOutcome(error="boom")

    proposal = await _seed_proposal(sqlite_store)
    spec = _spec_with_retry_policy(handler=_always_fails, max_attempts=1)

    # Drive to terminal.
    await drive_entity_phase3(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        llm=None,
        config=EngineConfig(),
        record=proposal,
    )
    pre = await sqlite_store.get_proposal("prop_synth_test")
    assert pre is not None
    assert pre.state == "failed"
    pre_version = pre.version

    # Re-drive: nothing should change.
    second = await drive_entity_phase3(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        llm=None,
        config=EngineConfig(),
        record=pre,
    )
    assert second.status == "no_match"
    post = await sqlite_store.get_proposal("prop_synth_test")
    assert post is not None
    assert post.state == "failed"
    assert post.version == pre_version
