"""Round-11: the cg_entries ``implementing`` state's retry budget.

Pre-round-11 the entry-spec ``implementing`` :class:`DispatchAction`
carried no :class:`RetryPolicy` and the entry spec declared no
dispatch-failure terminal edge — a dispatch-handler failure rolled the
entry back to ``pending`` and re-dispatched forever. Round 11 added the
:attr:`EntryRecord.entry_dispatch_attempt` counter (distinct from
``implementation_attempt``, which the CG-spec's review-retry gate owns)
plus a :class:`RetryPolicy` on the dispatch action, so the engine
synthesizes a retry-exhausted terminal edge to ``implementation_failed``.

These tests project the real :func:`build_cg_entries_spec` spec to an
:class:`EngineSpec` with the ``implementing`` dispatch handler swapped
for a deterministic failing stub — the real RetryPolicy / states /
edges are preserved, so the synthesis under test is the production
wiring.
"""

from __future__ import annotations

from pathlib import Path

from _helpers import FakeCompute  # type: ignore[import-not-found]
from _specs_fakes import make_cg, make_entry  # type: ignore[import-not-found]
from smai_artifacts_localfs import LocalFsStore
from smai_orchestrator.engine import (
    DispatchAction,
    DispatchContext,
    DispatchOutcome,
    EngineConfig,
    EngineSpec,
    StateDef,
    drive_entity_phase3,
)
from smai_orchestrator.runtime.spec import PipelineSpec
from smai_orchestrator.specs.cg_entries import build_cg_entries_spec
from smai_store_sqlite import SqliteStore
from sqlalchemy import text

# === Spec-structure assertions ==============================================


def test_cg_entries_implementing_declares_retry_policy(tmp_path: Path) -> None:
    """The entry-spec ``implementing`` :class:`DispatchAction` carries the
    round-11 :class:`RetryPolicy` — counter field, cap, terminal, reason."""
    spec = build_cg_entries_spec(workspace_root=tmp_path / "ws", max_entry_dispatch_attempts=2)
    implementing = spec.state_def("implementing")
    assert implementing.on_entry_dispatch is not None
    policy = implementing.on_entry_dispatch.retry_policy
    assert policy is not None
    assert policy.attempt_counter_field == "entry_dispatch_attempt"
    assert policy.max_attempts == 2
    assert policy.on_exhaustion_target_state == "implementation_failed"
    assert "retry budget exhausted" in policy.on_exhaustion_reason


def test_cg_entries_max_attempts_param_threads_into_policy(tmp_path: Path) -> None:
    """``max_entry_dispatch_attempts`` flows into the RetryPolicy cap."""
    spec = build_cg_entries_spec(workspace_root=tmp_path / "ws", max_entry_dispatch_attempts=5)
    policy = spec.state_def("implementing").on_entry_dispatch.retry_policy  # type: ignore[union-attr]
    assert policy is not None
    assert policy.max_attempts == 5


def test_cg_entries_has_no_manual_retry_exhausted_edge(tmp_path: Path) -> None:
    """No manual ``*_failed (retry exhausted)`` :class:`EdgeDef` — the
    engine synthesizes it from the :class:`RetryPolicy` (a manual edge
    alongside would double-evaluate)."""
    spec = build_cg_entries_spec(workspace_root=tmp_path / "ws")
    assert not any("retry exhausted" in e.name for e in spec.edges)


# === Engine-driven synthesized-terminal behavior ============================


def _entry_spec_with_failing_dispatch(spec: PipelineSpec, error: str) -> EngineSpec:
    """Project ``spec`` to an :class:`EngineSpec` with the ``implementing``
    dispatch handler swapped for a failing stub — the real RetryPolicy,
    states, edges, and pools are otherwise preserved."""
    eng = spec.engine_spec()

    async def _failing(ctx: DispatchContext) -> DispatchOutcome:
        del ctx
        return DispatchOutcome(error=error)

    states: list[StateDef] = []
    for s in eng.states:
        if s.name == "implementing" and s.on_entry_dispatch is not None:
            da = s.on_entry_dispatch
            states.append(
                StateDef(
                    name=s.name,
                    is_terminal=s.is_terminal,
                    on_entry_dispatch=DispatchAction(
                        name=da.name,
                        handler=_failing,
                        pool=da.pool,
                        handle_field=da.handle_field,
                        retry_policy=da.retry_policy,
                    ),
                )
            )
        else:
            states.append(s)
    return EngineSpec(
        entity_kind=eng.entity_kind,
        initial_state=eng.initial_state,
        states=states,
        edges=eng.edges,
        pools=eng.pools,
    )


async def test_cg_entries_dispatch_failure_reaches_implementation_failed(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
    tmp_path: Path,
) -> None:
    """A repeated entry implementation-dispatch failure reaches the
    ``implementation_failed`` terminal once ``entry_dispatch_attempt``
    hits ``max_entry_dispatch_attempts`` — via the engine-synthesized
    retry-exhausted edge, with the expected transition_log reason."""
    spec = build_cg_entries_spec(workspace_root=tmp_path / "ws", max_entry_dispatch_attempts=2)
    test_spec = _entry_spec_with_failing_dispatch(
        spec, "implementer agent crashed before submitting a job"
    )

    await sqlite_store.create_cg(make_cg("cg-r11", state="implementing"))
    entry = make_entry("entry-r11", cg_id="cg-r11", state="pending")
    await sqlite_store.create_entry(entry)
    assert entry.entry_dispatch_attempt == 0

    final_state: str | None = None
    for _ in range(6):
        rec = await sqlite_store.get_entry("entry-r11")
        assert rec is not None
        if rec.state in {"implemented", "implementation_failed"}:
            final_state = rec.state
            break
        await drive_entity_phase3(
            spec=test_spec,
            metadata_store=sqlite_store,
            artifact_store=localfs_store,  # type: ignore[arg-type]
            compute=FakeCompute(),  # type: ignore[arg-type]
            llm=None,
            config=EngineConfig(),
            record=rec,
        )
    assert final_state == "implementation_failed", f"got {final_state}"

    final = await sqlite_store.get_entry("entry-r11")
    assert final is not None
    assert final.state == "implementation_failed"
    # The engine bumped the counter on each step-1 CAS into ``implementing``.
    assert final.entry_dispatch_attempt == 2

    async with await sqlite_store.transaction() as tx:
        conn = tx.connection
        rows = (
            (
                await conn.execute(
                    text(
                        "SELECT from_state, to_state, edge_name, gate_outcome_reason "
                        "FROM transition_log WHERE entity_kind='entry' "
                        "AND entity_id='entry-r11' ORDER BY id"
                    )
                )
            )
            .mappings()
            .all()
        )
    terminal = next(
        r
        for r in rows
        if (r["from_state"], r["to_state"]) == ("implementing", "implementation_failed")
    )
    assert "synthesized retry-exhausted terminal" in terminal["edge_name"]
    assert "retry budget exhausted" in (terminal["gate_outcome_reason"] or "")


async def test_cg_entries_dispatch_failure_bumps_counter_each_entry(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
    tmp_path: Path,
) -> None:
    """The engine bumps ``entry_dispatch_attempt`` on each (re-)entry into
    ``implementing`` — one drive from ``pending`` with a budget of 3
    leaves the counter at 1 and rolls the entry back to ``pending``."""
    spec = build_cg_entries_spec(workspace_root=tmp_path / "ws", max_entry_dispatch_attempts=3)
    test_spec = _entry_spec_with_failing_dispatch(spec, "boom")

    await sqlite_store.create_cg(make_cg("cg-r11b", state="implementing"))
    await sqlite_store.create_entry(make_entry("entry-r11b", cg_id="cg-r11b", state="pending"))
    rec = await sqlite_store.get_entry("entry-r11b")
    assert rec is not None
    await drive_entity_phase3(
        spec=test_spec,
        metadata_store=sqlite_store,
        artifact_store=localfs_store,  # type: ignore[arg-type]
        compute=FakeCompute(),  # type: ignore[arg-type]
        llm=None,
        config=EngineConfig(),
        record=rec,
    )
    after = await sqlite_store.get_entry("entry-r11b")
    assert after is not None
    # Budget 3, one failed dispatch → counter 1, rolled back to ``pending``.
    assert after.entry_dispatch_attempt == 1
    assert after.state == "pending"
    assert after.last_error == "boom"
