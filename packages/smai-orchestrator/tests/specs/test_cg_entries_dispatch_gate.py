"""Round-15: the cg_entries ``pending → implementing`` dispatch-ready gate.

Pre-round-15 the gate unconditionally advanced — it trusted the SQL
scheduling query :meth:`MetadataStore.get_ready_to_implement_entry` to
have already gated on the harness manifest being committed. The query
never did (a SQL predicate cannot see an :class:`ArtifactStore`), so an
entry was dispatched while the harness builder was still running, the
technique implementer failed on a missing ``harness/manifest.json``, and
the entry burned its round-11 ``entry_dispatch_attempt`` budget on a
transient precondition.

Round 15 moves the "harness manifest committed" check into the gate
body: it reads the entry's ``cg_id``, builds the manifest key, and
``ArtifactStore.exists``-checks it. A missing manifest holds the entry
at ``pending`` (``advance=False``) — and, crucially, consumes no
``entry_dispatch_attempt`` because the entry never transitions into
``implementing`` (the counter is bumped only by the RetryPolicy CAS).
"""

from __future__ import annotations

from pathlib import Path

from _helpers import FakeCompute  # type: ignore[import-not-found]
from _specs_fakes import make_cg, make_entry  # type: ignore[import-not-found]
from smai_artifacts_localfs import LocalFsStore
from smai_orchestrator.engine import (
    EngineConfig,
    GateContext,
    drive_entity_phase3,
)
from smai_orchestrator.specs.cg_entries import (
    _make_entry_dispatch_ready_gate,
    build_cg_entries_spec,
)
from smai_orchestrator.specs.cg_execution import HARNESS_MANIFEST_KEY_TEMPLATE
from smai_store_sqlite import SqliteStore

# === Gate body — manifest-presence check ====================================


def _gate_context(store: SqliteStore, artifact_store: LocalFsStore, entry_id: str) -> GateContext:
    return GateContext(
        entity_kind="entry",
        entity_id=entry_id,
        entity_state="pending",
        entity_version=0,
        metadata_store=store,
        artifact_store=artifact_store,  # type: ignore[arg-type]
        config=EngineConfig(),
    )


async def test_dispatch_ready_gate_holds_entry_when_manifest_absent(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """No harness manifest in the ArtifactStore → gate returns
    ``advance=False`` so the entry stays ``pending``."""
    await sqlite_store.create_cg(make_cg("cg-g1", state="implementing"))
    await sqlite_store.create_entry(make_entry("entry-g1", cg_id="cg-g1", state="pending"))

    gate = _make_entry_dispatch_ready_gate()
    outcome = await gate(_gate_context(sqlite_store, localfs_store, "entry-g1"))

    assert outcome.advance is False
    assert outcome.reason is not None
    assert "manifest" in outcome.reason


async def test_dispatch_ready_gate_advances_when_manifest_present(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """Harness manifest committed to the ArtifactStore → gate advances."""
    await sqlite_store.create_cg(make_cg("cg-g2", state="implementing"))
    await sqlite_store.create_entry(make_entry("entry-g2", cg_id="cg-g2", state="pending"))
    await localfs_store.put(
        HARNESS_MANIFEST_KEY_TEMPLATE.format(cg_id="cg-g2"),
        b"{}",
        content_type="application/json",
    )

    gate = _make_entry_dispatch_ready_gate()
    outcome = await gate(_gate_context(sqlite_store, localfs_store, "entry-g2"))

    assert outcome.advance is True


# === Held-back entry consumes no entry_dispatch_attempt =====================


async def test_held_back_entry_consumes_no_dispatch_attempt(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
    tmp_path: Path,
) -> None:
    """An entry held at ``pending`` by the ``advance=False`` dispatch-ready
    gate (harness manifest absent) does NOT bump ``entry_dispatch_attempt``
    across repeated worker cycles — the counter rides the RetryPolicy CAS
    into ``implementing``, which an ``advance=False`` gate never reaches."""
    spec = build_cg_entries_spec(workspace_root=tmp_path / "ws")
    engine_spec = spec.engine_spec()

    await sqlite_store.create_cg(make_cg("cg-g3", state="implementing"))
    await sqlite_store.create_entry(make_entry("entry-g3", cg_id="cg-g3", state="pending"))

    # No harness manifest staged — the gate holds the entry back every cycle.
    for _ in range(4):
        rec = await sqlite_store.get_entry("entry-g3")
        assert rec is not None
        await drive_entity_phase3(
            spec=engine_spec,
            metadata_store=sqlite_store,
            artifact_store=localfs_store,  # type: ignore[arg-type]
            compute=FakeCompute(),  # type: ignore[arg-type]
            llm=None,
            config=EngineConfig(),
            record=rec,
        )

    final = await sqlite_store.get_entry("entry-g3")
    assert final is not None
    assert final.state == "pending"
    assert final.entry_dispatch_attempt == 0
