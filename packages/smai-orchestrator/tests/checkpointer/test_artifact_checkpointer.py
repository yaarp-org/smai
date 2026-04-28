"""Round-trip tests for :class:`ArtifactStoreCheckpointer` against the
real :class:`LocalFsStore` plugin (fixture: ``localfs_store``).

Per `05-orchestrator.md` §4.3 / §2 — the artifact-store flavor is the
"large body" path, addressing each checkpoint at
``<prefix>/<thread_id>/<step_id>/<input_hash>.bin`` plus a JSON
sidecar. The tests verify the addressing scheme, the
``ArtifactNotFound`` → ``None`` translation, and that bodies large
enough to embarrass a row-shaped backend round-trip cleanly.
"""

from __future__ import annotations

from datetime import UTC, datetime

from _helpers import FakeClock  # type: ignore[import-not-found]
from smai_artifacts_localfs import LocalFsStore
from smai_orchestrator.checkpointer import (
    DEFAULT_PREFIX,
    ArtifactStoreCheckpointer,
    Checkpoint,
    CheckpointKey,
)


async def test_load_returns_none_on_miss(localfs_store: LocalFsStore) -> None:
    cp = ArtifactStoreCheckpointer(localfs_store)
    key = CheckpointKey(thread_id="cg_1", step_id="trace_v1", input_hash="h0")
    assert await cp.load(key) is None


async def test_save_then_load_round_trip(localfs_store: LocalFsStore) -> None:
    cp = ArtifactStoreCheckpointer(localfs_store)
    key = CheckpointKey(thread_id="cg_1", step_id="trace_v1", input_hash="h0")
    body = b'{"role": "user", "content": "hello"}'
    created_at = datetime(2026, 4, 28, 12, 0, 0, tzinfo=UTC)
    await cp.save(Checkpoint(key=key, result=body, created_at=created_at))
    loaded = await cp.load(key)
    assert loaded is not None
    assert loaded.result == body
    assert loaded.created_at == created_at
    assert loaded.key == key


async def test_default_prefix_addressing(localfs_store: LocalFsStore) -> None:
    """The body path conforms to the documented scheme so external
    tooling (manual debugging, S3 console queries) can find it."""
    cp = ArtifactStoreCheckpointer(localfs_store)
    key = CheckpointKey(thread_id="cg_42", step_id="step", input_hash="abc")
    await cp.save(Checkpoint(key=key, result=b"body", created_at=datetime(2026, 4, 28, tzinfo=UTC)))
    expected_body_key = f"{DEFAULT_PREFIX}/cg_42/step/abc.bin"
    assert await localfs_store.exists(expected_body_key)
    expected_meta_key = f"{DEFAULT_PREFIX}/cg_42/step/abc.meta.json"
    assert await localfs_store.exists(expected_meta_key)


async def test_custom_prefix_round_trip(localfs_store: LocalFsStore) -> None:
    cp = ArtifactStoreCheckpointer(localfs_store, prefix="custom/path")
    key = CheckpointKey(thread_id="t", step_id="s", input_hash="h")
    await cp.save(Checkpoint(key=key, result=b"x", created_at=datetime(2026, 4, 28, tzinfo=UTC)))
    assert await localfs_store.exists("custom/path/t/s/h.bin")


async def test_large_body_round_trip(localfs_store: LocalFsStore) -> None:
    """The artifact-store flavor exists for results too large for a row
    — exercise the path with a 256 KiB body to confirm the binary
    pipeline doesn't silently truncate / re-encode."""
    cp = ArtifactStoreCheckpointer(localfs_store)
    big = b"\x00\x01\x02\x03" * (256 * 1024 // 4)
    assert len(big) == 256 * 1024
    key = CheckpointKey(thread_id="cg_big", step_id="trace", input_hash="big")
    await cp.save(Checkpoint(key=key, result=big, created_at=datetime(2026, 4, 28, tzinfo=UTC)))
    loaded = await cp.load(key)
    assert loaded is not None
    assert loaded.result == big


async def test_save_overwrites_existing(localfs_store: LocalFsStore) -> None:
    cp = ArtifactStoreCheckpointer(localfs_store)
    key = CheckpointKey(thread_id="t", step_id="s", input_hash="h")
    await cp.save(
        Checkpoint(key=key, result=b"first", created_at=datetime(2026, 4, 28, tzinfo=UTC))
    )
    await cp.save(
        Checkpoint(key=key, result=b"second", created_at=datetime(2026, 4, 28, tzinfo=UTC))
    )
    loaded = await cp.load(key)
    assert loaded is not None and loaded.result == b"second"


async def test_wall_clock_seam(localfs_store: LocalFsStore) -> None:
    """The constructor's ``wall_clock`` seam isn't consumed on save
    (the caller passes ``created_at`` directly), but the constructor
    accepts the seam for symmetry with :class:`EngineConfig` — verify
    the seam is honored at construction time without breaking
    round-trip behavior."""
    fake_clock = FakeClock(start=datetime(2030, 1, 1, tzinfo=UTC))
    cp = ArtifactStoreCheckpointer(localfs_store, wall_clock=fake_clock)
    key = CheckpointKey(thread_id="t", step_id="s", input_hash="h")
    await cp.save(Checkpoint(key=key, result=b"x", created_at=datetime(2026, 4, 28, tzinfo=UTC)))
    loaded = await cp.load(key)
    assert loaded is not None
    # The save-time created_at flows through verbatim, not derived from
    # the wall_clock — the seam is an integration point for future
    # cleanup / TTL tasks (not a save-time clock override).
    assert loaded.created_at == datetime(2026, 4, 28, tzinfo=UTC)
