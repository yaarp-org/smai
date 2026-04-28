"""MetadataStore-backed step-memoization checkpointer.

Per ``designs/smai/05-orchestrator.md`` §2 / §4.3. The default flavor
of :class:`Checkpointer` — bodies live in a ``checkpoints`` table
colocated with the rest of the orchestrator's state, keyed on
``(thread_id, step_id, input_hash)``.

**Session-C-pending surface.** Per Task 2.A2's status note,
``transition_log`` and the orchestrator's ``checkpoints`` table CRUD
are plugin-internal pending Session-C of the
:class:`smai_core.plugins.MetadataStore` Protocol settlement. The
Protocol surface today (`07-plugin-interfaces.md` §5) does NOT
enumerate ``get_checkpoint`` / ``put_checkpoint`` methods.

**How this module is shaped today.** The checkpointer takes a
:class:`CheckpointBackend` Protocol — an engine-internal narrow shape
that any persistence substrate can implement. C2 ships:

* :class:`InMemoryCheckpointBackend` — round-trips checkpoints in
  process memory, suitable for tests, the synthetic-spec engine
  fixture, and ``smai dev`` quick-start when the user has not
  configured a persistent backend yet.
* :class:`MetadataStoreCheckpointer` — the public wrapper that takes a
  :class:`CheckpointBackend` and exposes the :class:`Checkpointer`
  Protocol surface dispatch handlers consume.

When Session-C ships ``MetadataStore.get_checkpoint`` /
``MetadataStore.put_checkpoint``, the SQL plugin (``smai-store-sqlite``,
``smai-store-postgres``) will adapt those methods into a
:class:`CheckpointBackend` instance — the same wrapper class continues
to work without change.

The carry-forward is surfaced loudly here and in the Task 2.C2 status
note; supervisor decides whether to land the Protocol surface now (ahead
of full Session-C) or continue with the in-memory backend through Phase
2 wave-3 and 2.C4. C2's tests use :class:`InMemoryCheckpointBackend`
exclusively.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from smai_orchestrator.checkpointer.types import (
    Checkpoint,
    CheckpointKey,
)


@runtime_checkable
class CheckpointBackend(Protocol):
    """Narrow persistence surface for :class:`MetadataStoreCheckpointer`.

    The eventual :class:`MetadataStore` Protocol surface (Session-C —
    pending per Task 2.A2) will expose these methods directly; SQL-
    shaped plugins (SQLite, Postgres) wrap their Protocol method calls
    behind this interface for the checkpointer's convenience. Until
    then, :class:`InMemoryCheckpointBackend` is the only shipped
    implementation.
    """

    async def get_checkpoint(self, key: CheckpointKey) -> Checkpoint | None:
        """Return the stored checkpoint, or ``None`` on miss."""
        ...

    async def put_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Persist (or overwrite) ``checkpoint``. Idempotent."""
        ...


class InMemoryCheckpointBackend:
    """In-process dict-backed :class:`CheckpointBackend` (`05` §2).

    Used for tests and as the default substrate inside ``smai dev``
    when the user has not configured a persistent backend. Fast,
    deterministic, no external dependencies; loses state across
    process restarts (fine for transient memos; the dispatch handler
    re-runs the work on miss).
    """

    def __init__(self) -> None:
        self._data: dict[tuple[str, str, str], Checkpoint] = {}

    async def get_checkpoint(self, key: CheckpointKey) -> Checkpoint | None:
        return self._data.get((key.thread_id, key.step_id, key.input_hash))

    async def put_checkpoint(self, checkpoint: Checkpoint) -> None:
        k = checkpoint.key
        self._data[(k.thread_id, k.step_id, k.input_hash)] = checkpoint


class MetadataStoreCheckpointer:
    """Default :class:`Checkpointer` flavor — bodies in the metadata
    store's ``checkpoints`` table (`05` §2 / §4.3).

    Constructor takes a :class:`CheckpointBackend` rather than a
    :class:`MetadataStore` directly because the Protocol surface for
    checkpoint CRUD is Session-C-pending (per Task 2.A2's status
    note). Once the methods land, plugins implement
    :class:`CheckpointBackend` against their Protocol-typed
    ``get_checkpoint`` / ``put_checkpoint`` calls; this wrapper stays
    unchanged.

    The class is structurally typed against
    :class:`smai_orchestrator.checkpointer.Checkpointer` — its
    ``load`` / ``save`` shape matches the Protocol, so dispatch
    handlers consume it uniformly with the artifact-store flavor.
    """

    def __init__(self, backend: CheckpointBackend) -> None:
        self._backend = backend

    async def load(self, key: CheckpointKey) -> Checkpoint | None:
        return await self._backend.get_checkpoint(key)

    async def save(self, checkpoint: Checkpoint) -> None:
        await self._backend.put_checkpoint(checkpoint)


__all__ = [
    "CheckpointBackend",
    "InMemoryCheckpointBackend",
    "MetadataStoreCheckpointer",
]
