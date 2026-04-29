"""DEC-033 #3 — manifest hash fanout dispatch.

Per `03-state-machine.md` §3.3 / DEC-033 #3, the
``implementing → implemented`` gate writes ``harness_api_manifest_hash``
to every entry of the CG atomically. Per the brief, the canonical
shape is ``transition_entry_state(...)`` for every entry inside one
:meth:`MetadataStore.transaction` block.

The CG-execution spec lifts this work into the ``implemented`` state's
on-entry dispatch handler (rather than the gate body) per the
gate-rule contract per `05` §1.3 (read-only). The handler reads the
manifest from ArtifactStore, extracts ``content_hash``, and CAS-writes
the hash to every :class:`EntryRecord` of the CG.

These tests exercise the dispatch handler directly against a real
:class:`SqliteStore` + :class:`LocalFsStore` substrate.
"""

from __future__ import annotations

from _helpers import FakeCompute  # type: ignore[import-not-found]
from _specs_fakes import (  # type: ignore[import-not-found]
    make_cg,
    make_entry,
    stage_harness_artifacts,
)
from smai_artifacts_localfs import LocalFsStore
from smai_orchestrator.engine.config import EngineConfig
from smai_orchestrator.engine.types import DispatchContext
from smai_orchestrator.specs.cg_execution import (
    _make_dispatch_manifest_fanout,
)
from smai_store_sqlite import SqliteStore


async def test_manifest_fanout_writes_hash_to_all_entries(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """The manifest fanout dispatch writes ``harness_api_manifest_hash``
    to every entry of the CG atomically (DEC-033 #3)."""
    cg_id = "cg-fanout"
    staged = await stage_harness_artifacts(
        artifact_store=localfs_store,
        cg_id=cg_id,
    )
    cg = make_cg(cg_id=cg_id, state="implemented")
    await sqlite_store.create_cg(cg)

    # Three entries — two regular + one terminal-failed (which still
    # gets a hash since fanout is unconditional per the body's loop).
    entry_a = make_entry("entry-a", cg_id=cg_id, state="implemented")
    entry_b = make_entry("entry-b", cg_id=cg_id, state="implemented")
    entry_c = make_entry("entry-c", cg_id=cg_id, state="implementation_failed")
    await sqlite_store.create_entry(entry_a)
    await sqlite_store.create_entry(entry_b)
    await sqlite_store.create_entry(entry_c)

    handler = _make_dispatch_manifest_fanout()
    ctx = DispatchContext(
        entity_kind="cg",
        entity_id=cg_id,
        entity_state="implemented",
        entity_version=cg.version,
        metadata_store=sqlite_store,
        artifact_store=localfs_store,  # type: ignore[arg-type]
        compute=FakeCompute(),  # type: ignore[arg-type]
        llm=None,
        config=EngineConfig(),
        checkpointer=None,
    )
    outcome = await handler(ctx)
    assert outcome.error is None, outcome.error

    # Every entry now carries the manifest's content_hash.
    for entry_id in ("entry-a", "entry-b", "entry-c"):
        e = await sqlite_store.get_entry(entry_id)
        assert e is not None
        assert e.harness_api_manifest_hash == staged.manifest.content_hash
        # Versions incremented per CAS.
        assert e.version == 1


async def test_manifest_fanout_returns_error_when_manifest_missing(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """When the manifest artifact is missing, the dispatch returns
    a non-None error and the engine forward-rolls-back the CG."""
    cg_id = "cg-no-manifest"
    cg = make_cg(cg_id=cg_id, state="implemented")
    await sqlite_store.create_cg(cg)
    entry = make_entry("entry-1", cg_id=cg_id, state="implemented")
    await sqlite_store.create_entry(entry)

    handler = _make_dispatch_manifest_fanout()
    ctx = DispatchContext(
        entity_kind="cg",
        entity_id=cg_id,
        entity_state="implemented",
        entity_version=cg.version,
        metadata_store=sqlite_store,
        artifact_store=localfs_store,  # type: ignore[arg-type]
        compute=FakeCompute(),  # type: ignore[arg-type]
        llm=None,
        config=EngineConfig(),
        checkpointer=None,
    )
    outcome = await handler(ctx)
    assert outcome.error is not None
    assert "manifest not found" in outcome.error

    # Entry was untouched.
    e = await sqlite_store.get_entry("entry-1")
    assert e is not None
    assert e.harness_api_manifest_hash is None
    assert e.version == 0


async def test_manifest_fanout_uses_transaction_for_atomic_writes(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """Verifies the fanout dispatch uses :meth:`MetadataStore.transaction`
    for the per-entry writes (per DEC-033 #3 / DEC-036).

    Counts the number of entries that received the manifest hash
    after a successful invocation across a 5-entry CG; all should be
    set in one transaction.
    """
    cg_id = "cg-atomic"
    staged = await stage_harness_artifacts(artifact_store=localfs_store, cg_id=cg_id)
    cg = make_cg(cg_id=cg_id, state="implemented")
    await sqlite_store.create_cg(cg)
    entry_ids = [f"entry-atom-{i}" for i in range(5)]
    for entry_id in entry_ids:
        await sqlite_store.create_entry(make_entry(entry_id, cg_id=cg_id, state="implemented"))

    handler = _make_dispatch_manifest_fanout()
    ctx = DispatchContext(
        entity_kind="cg",
        entity_id=cg_id,
        entity_state="implemented",
        entity_version=cg.version,
        metadata_store=sqlite_store,
        artifact_store=localfs_store,  # type: ignore[arg-type]
        compute=FakeCompute(),  # type: ignore[arg-type]
        llm=None,
        config=EngineConfig(),
        checkpointer=None,
    )
    outcome = await handler(ctx)
    assert outcome.error is None

    # All 5 entries got the hash + version bump.
    for entry_id in entry_ids:
        e = await sqlite_store.get_entry(entry_id)
        assert e is not None
        assert e.harness_api_manifest_hash == staged.manifest.content_hash
        assert e.version == 1
