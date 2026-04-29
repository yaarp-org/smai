"""DEC-016 — code-review gate trio at ``implemented → ...``.

Per `03-state-machine.md` §3.3 the implemented state has three
outgoing edges, each with its own gate body. All three gate bodies
share a memoized :class:`CodeReviewResult` written to ArtifactStore at
``comparison-groups/{cg_id}/code-review-attempt-{N}.json``.

These tests construct a minimal CG + entries setup and invoke each
gate body directly, asserting on:

* ``overall_pass=True`` advances on the success edge.
* ``overall_pass=False`` with retry budget remaining advances on the
  retry edge AND increments :attr:`EntryRecord.implementation_attempt`
  + resets entry state to ``pending`` per the brief / DEC-016.
* ``overall_pass=False`` with retry budget exhausted + zero survivors
  advances on the failure-terminal edge.
"""

from __future__ import annotations

from _specs_fakes import (  # type: ignore[import-not-found]
    StubLlmProvider,
    make_cg,
    make_entry,
    make_review_response,
    stage_harness_artifacts,
    stage_per_entry_artifacts,
)
from smai_artifacts_localfs import LocalFsStore
from smai_orchestrator.engine.config import EngineConfig
from smai_orchestrator.engine.types import GateContext
from smai_orchestrator.specs.cg_execution import (
    _make_gate_review_fail_no_survivors,
    _make_gate_review_fail_with_retry,
    _make_gate_review_pass,
)
from smai_store_sqlite import SqliteStore


async def _seed_cg_with_entries(
    *,
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
    cg_id: str,
    entry_specs: list[tuple[str, str | None, str]],
    code_review_attempt: int = 0,
) -> None:
    """Pre-stage CG + entries + harness artifacts.

    ``entry_specs`` is a list of ``(entry_id, technique_id, state)``.
    """
    staged = await stage_harness_artifacts(artifact_store=localfs_store, cg_id=cg_id)
    cg = make_cg(cg_id=cg_id, state="implemented", code_review_attempt=code_review_attempt)
    await sqlite_store.create_cg(cg)
    for entry_id, technique_id, state in entry_specs:
        is_baseline = technique_id is None
        await sqlite_store.create_entry(
            make_entry(
                entry_id,
                cg_id=cg_id,
                state=state,
                technique_id=technique_id,
                is_baseline=is_baseline,
            )
        )
        await stage_per_entry_artifacts(
            artifact_store=localfs_store,
            cg_id=cg_id,
            entry_id=entry_id,
            technique_id=technique_id,
            is_baseline=is_baseline,
            parent_harness_contract_hash=staged.harness_contract.envelope.content_hash,
            code=f"# fake technique code for {entry_id}\n" if technique_id is not None else None,
        )


async def test_review_pass_gate_advances_on_overall_pass(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    cg_id = "cg-pass"
    await _seed_cg_with_entries(
        sqlite_store=sqlite_store,
        localfs_store=localfs_store,
        cg_id=cg_id,
        entry_specs=[
            ("entry-1", "tq-1", "implemented"),
        ],
    )
    review_llm = StubLlmProvider([make_review_response(overall_pass=True)])

    gate = _make_gate_review_pass(
        llm_for_code_reviewer=review_llm,  # type: ignore[arg-type]
        require_human_approval=False,
    )
    cg = await sqlite_store.get_cg(cg_id)
    assert cg is not None
    ctx = GateContext(
        entity_kind="cg",
        entity_id=cg_id,
        entity_state="implemented",
        entity_version=cg.version,
        metadata_store=sqlite_store,
        artifact_store=localfs_store,  # type: ignore[arg-type]
        config=EngineConfig(),
    )
    outcome = await gate(ctx)
    assert outcome.advance is True


async def test_review_pass_gate_holds_when_human_approval_required(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """DEC-003: when ``require_human_approval=True``, the success-gate
    holds the CG in ``implemented`` even on overall_pass."""
    cg_id = "cg-hold"
    await _seed_cg_with_entries(
        sqlite_store=sqlite_store,
        localfs_store=localfs_store,
        cg_id=cg_id,
        entry_specs=[("entry-1", "tq-1", "implemented")],
    )
    review_llm = StubLlmProvider([make_review_response(overall_pass=True)])
    gate = _make_gate_review_pass(
        llm_for_code_reviewer=review_llm,  # type: ignore[arg-type]
        require_human_approval=True,
    )
    cg = await sqlite_store.get_cg(cg_id)
    assert cg is not None
    ctx = GateContext(
        entity_kind="cg",
        entity_id=cg_id,
        entity_state="implemented",
        entity_version=cg.version,
        metadata_store=sqlite_store,
        artifact_store=localfs_store,  # type: ignore[arg-type]
        config=EngineConfig(),
    )
    outcome = await gate(ctx)
    assert outcome.advance is False


async def test_review_fail_with_retry_gate_resets_failing_entries(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """When the review fails on a critical finding for an entry that
    has retry budget remaining, the gate body increments
    :attr:`EntryRecord.implementation_attempt` and resets the entry
    state to ``pending`` per the brief / DEC-016."""
    cg_id = "cg-retry"
    await _seed_cg_with_entries(
        sqlite_store=sqlite_store,
        localfs_store=localfs_store,
        cg_id=cg_id,
        entry_specs=[
            ("entry-bad", "tq-1", "implemented"),
            ("entry-good", "tq-2", "implemented"),
        ],
    )
    review_llm = StubLlmProvider(
        [
            make_review_response(
                overall_pass=False,
                findings=[
                    {
                        "severity": "critical",
                        "target_id": "entry-bad",
                        "target_kind": "entry",
                        "summary": "fake critical finding",
                        "detail": "test",
                        "suggested_fix": "fix it",
                    }
                ],
            )
        ]
    )
    gate = _make_gate_review_fail_with_retry(
        llm_for_code_reviewer=review_llm,  # type: ignore[arg-type]
        max_review_attempts=1,
    )
    cg = await sqlite_store.get_cg(cg_id)
    assert cg is not None
    ctx = GateContext(
        entity_kind="cg",
        entity_id=cg_id,
        entity_state="implemented",
        entity_version=cg.version,
        metadata_store=sqlite_store,
        artifact_store=localfs_store,  # type: ignore[arg-type]
        config=EngineConfig(),
    )
    outcome = await gate(ctx)
    assert outcome.advance is True

    # Entry-bad got reset to pending + attempt bumped.
    bad = await sqlite_store.get_entry("entry-bad")
    assert bad is not None
    assert bad.state == "pending"
    assert bad.implementation_attempt == 1

    # Entry-good was untouched.
    good = await sqlite_store.get_entry("entry-good")
    assert good is not None
    assert good.state == "implemented"
    assert good.implementation_attempt == 0

    # CG's code_review_attempt was bumped (cache key invalidation).
    cg_after = await sqlite_store.get_cg(cg_id)
    assert cg_after is not None
    assert cg_after.code_review_attempt == 1


async def test_review_fail_with_retry_gate_does_not_fire_when_budget_exhausted(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """An entry that already used its retry budget no longer fires the
    retry gate; the next gate (no-survivors terminal) handles routing."""
    cg_id = "cg-exhausted"
    staged = await stage_harness_artifacts(artifact_store=localfs_store, cg_id=cg_id)
    cg = make_cg(cg_id=cg_id, state="implemented", code_review_attempt=0)
    await sqlite_store.create_cg(cg)
    # Entry already at attempt=1 (budget exhausted).
    e = make_entry(
        "entry-burned",
        cg_id=cg_id,
        state="implemented",
        technique_id="tq-1",
        implementation_attempt=1,
    )
    await sqlite_store.create_entry(e)
    await stage_per_entry_artifacts(
        artifact_store=localfs_store,
        cg_id=cg_id,
        entry_id=e.id,
        technique_id="tq-1",
        is_baseline=False,
        parent_harness_contract_hash=staged.harness_contract.envelope.content_hash,
        code="# fake\n",
    )
    review_llm = StubLlmProvider(
        [
            make_review_response(
                overall_pass=False,
                findings=[
                    {
                        "severity": "critical",
                        "target_id": "entry-burned",
                        "target_kind": "entry",
                        "summary": "still bad",
                        "detail": "test",
                        "suggested_fix": "fix it",
                    }
                ],
            )
        ]
    )
    gate = _make_gate_review_fail_with_retry(
        llm_for_code_reviewer=review_llm,  # type: ignore[arg-type]
        max_review_attempts=1,
    )
    ctx = GateContext(
        entity_kind="cg",
        entity_id=cg_id,
        entity_state="implemented",
        entity_version=cg.version,
        metadata_store=sqlite_store,
        artifact_store=localfs_store,  # type: ignore[arg-type]
        config=EngineConfig(),
    )
    outcome = await gate(ctx)
    assert outcome.advance is False
    assert "no entries with retry budget" in (outcome.reason or "")


async def test_review_no_survivors_gate_advances_when_only_critical_remain(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """When every implemented entry has critical findings and no entries
    are clean, the failure-terminal gate fires."""
    cg_id = "cg-no-survivors"
    staged = await stage_harness_artifacts(artifact_store=localfs_store, cg_id=cg_id)
    cg = make_cg(cg_id=cg_id, state="implemented")
    await sqlite_store.create_cg(cg)
    only_entry = make_entry(
        "entry-only",
        cg_id=cg_id,
        state="implemented",
        technique_id="tq-1",
        implementation_attempt=1,
    )
    await sqlite_store.create_entry(only_entry)
    await stage_per_entry_artifacts(
        artifact_store=localfs_store,
        cg_id=cg_id,
        entry_id=only_entry.id,
        technique_id="tq-1",
        is_baseline=False,
        parent_harness_contract_hash=staged.harness_contract.envelope.content_hash,
        code="# fake\n",
    )
    review_llm = StubLlmProvider(
        [
            make_review_response(
                overall_pass=False,
                findings=[
                    {
                        "severity": "critical",
                        "target_id": "entry-only",
                        "target_kind": "entry",
                        "summary": "broken",
                        "detail": "test",
                        "suggested_fix": "rewrite",
                    }
                ],
            )
        ]
    )
    gate = _make_gate_review_fail_no_survivors(
        llm_for_code_reviewer=review_llm,  # type: ignore[arg-type]
    )
    ctx = GateContext(
        entity_kind="cg",
        entity_id=cg_id,
        entity_state="implemented",
        entity_version=cg.version,
        metadata_store=sqlite_store,
        artifact_store=localfs_store,  # type: ignore[arg-type]
        config=EngineConfig(),
    )
    outcome = await gate(ctx)
    assert outcome.advance is True
    assert "no survivors" in (outcome.reason or "")


async def test_code_review_result_cached_in_artifact_store(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """The first gate invocation writes the review result; subsequent
    gates re-read from cache. Verified by counting LLM calls."""
    cg_id = "cg-cached"
    await _seed_cg_with_entries(
        sqlite_store=sqlite_store,
        localfs_store=localfs_store,
        cg_id=cg_id,
        entry_specs=[("entry-1", "tq-1", "implemented")],
    )
    # Provide ONE response — a second LLM call would raise.
    review_llm = StubLlmProvider([make_review_response(overall_pass=True)])
    gate1 = _make_gate_review_pass(
        llm_for_code_reviewer=review_llm,  # type: ignore[arg-type]
        require_human_approval=False,
    )
    gate2 = _make_gate_review_pass(
        llm_for_code_reviewer=review_llm,  # type: ignore[arg-type]
        require_human_approval=False,
    )
    cg = await sqlite_store.get_cg(cg_id)
    assert cg is not None
    ctx = GateContext(
        entity_kind="cg",
        entity_id=cg_id,
        entity_state="implemented",
        entity_version=cg.version,
        metadata_store=sqlite_store,
        artifact_store=localfs_store,  # type: ignore[arg-type]
        config=EngineConfig(),
    )
    out1 = await gate1(ctx)
    out2 = await gate2(ctx)
    assert out1.advance is True
    assert out2.advance is True
    # Only one LLM call was made (second came from the cache).
    assert len(review_llm.calls) == 1
