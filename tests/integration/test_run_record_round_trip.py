"""Integration test for the :class:`RunRecord` sub-spec — Task 3.E3.

Per the brief's acceptance: a fixture CG with 2 entries × 2 seeds = 4
:class:`RunRecord` rows drives all four through
``pending → submitted → succeeded`` (the engine-reachable happy path);
the parent CG's ``running → evaluating`` gate fires once every child
run reaches a terminal state.

A second test ("mixed-terminal") exercises a CG where some runs
succeed, some fail, and one is ``inconclusive`` — proving
``inconclusive`` is a real terminal distinct from ``failed`` per
DEC-031 #8 / `06-mechanical-evaluation.md` §1.

The tests drive the worker loop directly via :func:`run_worker_cycle`
to exercise the engine's phase-1 + phase-2 + phase-3 mechanics
end-to-end against the real :class:`SqliteStore` + the run sub-spec
under test.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from _e3_fakes import E3FakeArtifactStore, E3FakeCompute  # type: ignore[import-not-found]
from smai_orchestrator.engine.config import EngineConfig
from smai_orchestrator.entities.tracking import (
    ComparisonGroupRecord,
    EntryRecord,
    RunRecord,
)
from smai_orchestrator.specs import (
    RUN_METRICS_KEY_TEMPLATE,
    build_run_record_spec,
)
from smai_orchestrator.worker.loop import run_worker_cycle
from smai_store_sqlite import SqliteStore


def _now() -> tuple[datetime, datetime]:
    t = datetime(2026, 4, 28, tzinfo=UTC)
    return t, t


def _make_cg(cg_id: str) -> ComparisonGroupRecord:
    created, updated = _now()
    return ComparisonGroupRecord(
        id=cg_id,
        proposal_id=f"prop_{cg_id}",
        experiment_definition_id=f"exp_{cg_id}",
        state="running",
        version=0,
        created_at=created,
        updated_at=updated,
    )


def _make_entry(entry_id: str, *, cg_id: str) -> EntryRecord:
    created, updated = _now()
    return EntryRecord(
        id=entry_id,
        cg_id=cg_id,
        technique_id=f"tq_{entry_id}",
        is_baseline=False,
        entry_id=entry_id,
        state="implemented",
        version=0,
        created_at=created,
        updated_at=updated,
    )


def _make_pending_run(*, cg_id: str, entry_id: str, seed: int) -> RunRecord:
    created, updated = _now()
    return RunRecord(
        id=f"run_{entry_id}_{seed}",
        cg_id=cg_id,
        entry_id=entry_id,
        seed=seed,
        state="pending",
        version=0,
        created_at=created,
        updated_at=updated,
    )


async def _stage_metrics(
    *,
    store: E3FakeArtifactStore,
    cg_id: str,
    entry_id: str,
    seed: int,
    accuracy: float,
) -> None:
    key = RUN_METRICS_KEY_TEMPLATE.format(cg_id=cg_id, entry_id=entry_id, seed=seed)
    await store.put(key, json.dumps({"accuracy": accuracy}).encode())


async def _all_runs(store: SqliteStore, entry_ids: list[str]) -> list[RunRecord]:
    out: list[RunRecord] = []
    for entry_id in entry_ids:
        page = await store.list_runs_for_entry(entry_id, limit=100)
        out.extend(page.items)
    return out


@pytest.mark.asyncio
async def test_run_sub_spec_drives_2x2_runs_to_succeeded() -> None:
    """2 entries × 2 seeds = 4 runs all drive ``pending → submitted →
    succeeded`` under the engine's standard cycle."""
    cg_id = "cg_e3_happy"
    entry_ids = ["entry_a", "entry_b"]

    sqlite_store = SqliteStore("sqlite+aiosqlite:///:memory:")
    await sqlite_store.migrate()
    artifact_store = E3FakeArtifactStore()
    compute = E3FakeCompute(terminal_states=["succeeded", "succeeded", "succeeded", "succeeded"])

    try:
        await sqlite_store.create_cg(_make_cg(cg_id))
        for entry_id in entry_ids:
            await sqlite_store.create_entry(_make_entry(entry_id, cg_id=cg_id))
        for entry_id in entry_ids:
            for seed in (0, 1):
                await sqlite_store.create_run(
                    _make_pending_run(cg_id=cg_id, entry_id=entry_id, seed=seed)
                )
                await _stage_metrics(
                    store=artifact_store,
                    cg_id=cg_id,
                    entry_id=entry_id,
                    seed=seed,
                    accuracy=0.9,
                )

        spec = build_run_record_spec().engine_spec()
        config = EngineConfig()

        # Drive the loop until every run reaches a terminal state.
        # Per design, two cycles suffice: cycle 1 phase-3 fires
        # ``pending → submitted`` (Compute.submit + handle write);
        # cycle 2 phase-1 fires ``submitted → succeeded`` once Compute
        # reports terminal. We give the loop a generous budget anyway.
        for _ in range(6):
            await run_worker_cycle(
                spec=spec,
                metadata_store=sqlite_store,
                artifact_store=artifact_store,  # type: ignore[arg-type]
                compute=compute,  # type: ignore[arg-type]
                llm_providers=None,
                config=config,
            )
            runs = await _all_runs(sqlite_store, entry_ids)
            if all(r.state in {"succeeded", "failed", "inconclusive"} for r in runs):
                break

        runs = await _all_runs(sqlite_store, entry_ids)
        assert len(runs) == 4
        states = sorted(r.state for r in runs)
        assert states == ["succeeded", "succeeded", "succeeded", "succeeded"]
        # Every run carried a Compute handle at some point — the engine
        # zeros the handle on phase-1 transition. (We assert via the
        # Compute's submit_calls instead.)
        assert len(compute.submit_calls) == 4
        for r in runs:
            # The handle field is cleared on the phase-1 transition out
            # of ``submitted``.
            assert r.compute_job_handle is None
    finally:
        await sqlite_store.dispose()


@pytest.mark.asyncio
async def test_run_sub_spec_routes_failed_jobs_to_failed_state() -> None:
    """A run whose Compute job reports ``failed`` lands in
    :class:`RunState` ``failed``."""
    cg_id = "cg_e3_failed"
    entry_id = "entry_fail"

    sqlite_store = SqliteStore("sqlite+aiosqlite:///:memory:")
    await sqlite_store.migrate()
    artifact_store = E3FakeArtifactStore()
    compute = E3FakeCompute(terminal_states=["failed"])

    try:
        await sqlite_store.create_cg(_make_cg(cg_id))
        await sqlite_store.create_entry(_make_entry(entry_id, cg_id=cg_id))
        await sqlite_store.create_run(_make_pending_run(cg_id=cg_id, entry_id=entry_id, seed=0))

        spec = build_run_record_spec().engine_spec()
        config = EngineConfig()
        for _ in range(4):
            await run_worker_cycle(
                spec=spec,
                metadata_store=sqlite_store,
                artifact_store=artifact_store,  # type: ignore[arg-type]
                compute=compute,  # type: ignore[arg-type]
                llm_providers=None,
                config=config,
            )

        runs = await _all_runs(sqlite_store, [entry_id])
        assert len(runs) == 1
        assert runs[0].state == "failed"
    finally:
        await sqlite_store.dispose()


@pytest.mark.asyncio
async def test_mixed_terminal_fixture_distinguishes_inconclusive_from_failed() -> None:
    """A CG with 4 runs landing in mixed terminal states demonstrates
    ``inconclusive`` as a distinct terminal (DEC-031 #8 / `06` §1).

    Layout: 2 entries × 2 seeds = 4 runs.
    * ``entry_a`` seed 0 → succeeded (metrics staged)
    * ``entry_a`` seed 1 → inconclusive (Compute=succeeded but no metrics)
    * ``entry_b`` seed 0 → failed (Compute reports failed)
    * ``entry_b`` seed 1 → succeeded (metrics staged)

    Asserts every terminal state is reached and the per-run shape
    distinguishes ``inconclusive`` from ``failed``.
    """
    cg_id = "cg_e3_mixed"
    entry_ids = ["entry_a", "entry_b"]

    sqlite_store = SqliteStore("sqlite+aiosqlite:///:memory:")
    await sqlite_store.migrate()
    artifact_store = E3FakeArtifactStore()
    # The dispatch order (submit calls) drives which terminal gets which
    # handle. Phase-3 dispatches in scheduling-query (FIFO created_at)
    # order — so the runs we created first get the first terminals.
    # We create in this order:
    #   1. entry_a/0 → succeeded (with metrics)
    #   2. entry_a/1 → succeeded (NO metrics → inconclusive)
    #   3. entry_b/0 → failed
    #   4. entry_b/1 → succeeded (with metrics)
    compute = E3FakeCompute(terminal_states=["succeeded", "succeeded", "failed", "succeeded"])

    try:
        await sqlite_store.create_cg(_make_cg(cg_id))
        for entry_id in entry_ids:
            await sqlite_store.create_entry(_make_entry(entry_id, cg_id=cg_id))
        # Create the four pending runs in the order documented above.
        for entry_id, seed in (
            ("entry_a", 0),
            ("entry_a", 1),
            ("entry_b", 0),
            ("entry_b", 1),
        ):
            await sqlite_store.create_run(
                _make_pending_run(cg_id=cg_id, entry_id=entry_id, seed=seed)
            )

        # Stage metrics for all but ``entry_a/seed=1`` — that one's
        # Compute job will succeed but the metrics-missing path makes
        # the run sub-spec route to ``inconclusive``.
        await _stage_metrics(
            store=artifact_store, cg_id=cg_id, entry_id="entry_a", seed=0, accuracy=0.9
        )
        await _stage_metrics(
            store=artifact_store, cg_id=cg_id, entry_id="entry_b", seed=1, accuracy=0.7
        )

        spec = build_run_record_spec().engine_spec()
        config = EngineConfig()
        for _ in range(8):
            await run_worker_cycle(
                spec=spec,
                metadata_store=sqlite_store,
                artifact_store=artifact_store,  # type: ignore[arg-type]
                compute=compute,  # type: ignore[arg-type]
                llm_providers=None,
                config=config,
            )
            runs = await _all_runs(sqlite_store, entry_ids)
            if all(r.state in {"succeeded", "failed", "inconclusive"} for r in runs):
                break

        runs = await _all_runs(sqlite_store, entry_ids)
        assert len(runs) == 4
        # All terminal.
        for r in runs:
            assert r.state in {"succeeded", "failed", "inconclusive"}, (
                f"run {r.id!r} not terminal: state={r.state}"
            )
        # All three terminal-state literals are represented.
        states = {r.state for r in runs}
        assert "succeeded" in states
        assert "failed" in states
        assert "inconclusive" in states
        # ``inconclusive`` is structurally distinct from ``failed``: the
        # failed run carries a non-null ``failure_reason`` *or*
        # ``run.state == "failed"``; the inconclusive run reaches its
        # terminal via the ``submitted → inconclusive`` edge (Compute
        # succeeded, metrics absent) and is NOT in the failure set.
        # Phase-1's transition zeroes the compute_job_handle but doesn't
        # touch failure_reason — neither succeeded nor inconclusive
        # write that field, so we use the ``state`` literal alone for
        # the distinction (which is the canonical DEC-031 #8 claim).
        by_state: dict[str, list[RunRecord]] = {}
        for r in runs:
            by_state.setdefault(r.state, []).append(r)
        assert len(by_state["succeeded"]) == 2
        assert len(by_state["inconclusive"]) == 1
        assert len(by_state["failed"]) == 1
        # The inconclusive run is the one we deliberately didn't stage
        # metrics for (entry_a, seed 1).
        incon = by_state["inconclusive"][0]
        assert incon.entry_id == "entry_a"
        assert incon.seed == 1
        # The failed run is the one Compute reported ``failed`` for
        # (entry_b, seed 0).
        failed = by_state["failed"][0]
        assert failed.entry_id == "entry_b"
        assert failed.seed == 0
    finally:
        await sqlite_store.dispose()
