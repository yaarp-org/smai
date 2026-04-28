"""Round-trip tests for :class:`MetadataStoreCheckpointer` against
the engine-internal :class:`InMemoryCheckpointBackend`.

Per `05-orchestrator.md` §4.3, the production wiring is the
:class:`MetadataStore` plugin's ``checkpoints`` table CRUD methods —
those are Session-C-pending per Task 2.A2's status note. Until they
land, :class:`InMemoryCheckpointBackend` is the shipped backend; the
tests here prove the wrapper class round-trips correctly against any
:class:`CheckpointBackend` substrate.
"""

from __future__ import annotations

from datetime import UTC, datetime

from smai_orchestrator.checkpointer import (
    Checkpoint,
    CheckpointBackend,
    CheckpointKey,
    InMemoryCheckpointBackend,
    MetadataStoreCheckpointer,
)


async def test_load_returns_none_on_miss() -> None:
    cp = MetadataStoreCheckpointer(InMemoryCheckpointBackend())
    key = CheckpointKey(thread_id="cg_1", step_id="step_v1", input_hash="h0")
    assert await cp.load(key) is None


async def test_save_then_load_round_trip() -> None:
    cp = MetadataStoreCheckpointer(InMemoryCheckpointBackend())
    key = CheckpointKey(thread_id="cg_1", step_id="step_v1", input_hash="h0")
    body = b'{"verdict":"pass"}'
    await cp.save(
        Checkpoint(key=key, result=body, created_at=datetime(2026, 4, 28, tzinfo=UTC))
    )
    loaded = await cp.load(key)
    assert loaded is not None
    assert loaded.result == body
    assert loaded.key == key


async def test_save_overwrites_on_same_key() -> None:
    """Idempotent at the key level per the Protocol contract."""
    cp = MetadataStoreCheckpointer(InMemoryCheckpointBackend())
    key = CheckpointKey(thread_id="cg_1", step_id="step_v1", input_hash="h0")
    await cp.save(
        Checkpoint(key=key, result=b"first", created_at=datetime(2026, 4, 28, tzinfo=UTC))
    )
    await cp.save(
        Checkpoint(key=key, result=b"second", created_at=datetime(2026, 4, 28, tzinfo=UTC))
    )
    loaded = await cp.load(key)
    assert loaded is not None
    assert loaded.result == b"second"


async def test_distinct_keys_isolated() -> None:
    cp = MetadataStoreCheckpointer(InMemoryCheckpointBackend())
    a = CheckpointKey(thread_id="cg_1", step_id="step_v1", input_hash="ha")
    b = CheckpointKey(thread_id="cg_1", step_id="step_v1", input_hash="hb")
    await cp.save(Checkpoint(key=a, result=b"A", created_at=datetime(2026, 4, 28, tzinfo=UTC)))
    await cp.save(Checkpoint(key=b, result=b"B", created_at=datetime(2026, 4, 28, tzinfo=UTC)))
    loaded_a = await cp.load(a)
    loaded_b = await cp.load(b)
    assert loaded_a is not None and loaded_a.result == b"A"
    assert loaded_b is not None and loaded_b.result == b"B"


async def test_step_id_versioning_invalidates() -> None:
    """Per `05` §2: bumping ``step_id`` invalidates prior memos —
    handlers manage version semantics by changing the ``step_id``
    suffix."""
    cp = MetadataStoreCheckpointer(InMemoryCheckpointBackend())
    v1 = CheckpointKey(thread_id="cg_1", step_id="code_review_v1", input_hash="h")
    v2 = CheckpointKey(thread_id="cg_1", step_id="code_review_v2", input_hash="h")
    await cp.save(
        Checkpoint(key=v1, result=b"old", created_at=datetime(2026, 4, 28, tzinfo=UTC))
    )
    assert await cp.load(v2) is None  # different step_id → no memo


async def test_thread_id_isolates_entities() -> None:
    cp = MetadataStoreCheckpointer(InMemoryCheckpointBackend())
    a = CheckpointKey(thread_id="cg_1", step_id="s", input_hash="h")
    b = CheckpointKey(thread_id="cg_2", step_id="s", input_hash="h")
    await cp.save(Checkpoint(key=a, result=b"A", created_at=datetime(2026, 4, 28, tzinfo=UTC)))
    assert await cp.load(b) is None


async def test_in_memory_backend_implements_protocol() -> None:
    backend: CheckpointBackend = InMemoryCheckpointBackend()
    key = CheckpointKey(thread_id="t", step_id="s", input_hash="h")
    assert await backend.get_checkpoint(key) is None
    await backend.put_checkpoint(
        Checkpoint(key=key, result=b"x", created_at=datetime(2026, 4, 28, tzinfo=UTC))
    )
    loaded = await backend.get_checkpoint(key)
    assert loaded is not None and loaded.result == b"x"
