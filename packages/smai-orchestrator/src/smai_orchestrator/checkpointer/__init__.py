"""Step-memoization checkpointer — engine-internal seam.

Per ``designs/smai/05-orchestrator.md`` §2 / DEC-025. Two concrete
flavors ship under this package:

* :class:`MetadataStoreCheckpointer` — bodies in a metadata-store
  ``checkpoints`` table (Session-C-pending Protocol surface; today
  takes a :class:`CheckpointBackend`, with
  :class:`InMemoryCheckpointBackend` as the shipped substrate).
* :class:`ArtifactStoreCheckpointer` — bodies in :class:`ArtifactStore`
  at a content-addressable path; for large results.

Plus the opt-in :func:`memoized` helper that wraps a handler's work in
the canonical load → run → save shape, with the default
:func:`stable_hash` convention per `05` §9 #3.

Per `05` §9 #7 / DEC-028: the checkpointer is **not** a fifth plugin
Protocol. The :class:`Checkpointer` Protocol exported here exists so
dispatch handlers see a uniform shape; the implementations are
concrete classes, internal to :mod:`smai_orchestrator`.

NO CHECKPOINT PERSISTENCE TODAY. The shipped metadata-store checkpointer
backs onto :class:`InMemoryCheckpointBackend`: there is no
``checkpoints`` table, and no checkpointer is instantiated by ``smai dev``
or ``smai start`` (``DispatchContext.checkpointer`` is ``None`` in both).
Consequence: a killed or restarted worker re-runs the in-flight agent
loop from turn 0. Nothing about an in-progress dispatch survives the
process. A SQL-backed checkpointer (the ``checkpoints`` table plus a
metadata-store CRUD surface) is post-M5 backlog; until then the recovery
primitive is orphan-grace reset (`05` §3.1), which re-dispatches the
whole step rather than resuming it.
"""

from smai_orchestrator.checkpointer.artifact import (
    DEFAULT_PREFIX,
    ArtifactStoreCheckpointer,
)
from smai_orchestrator.checkpointer.memoize import memoized, stable_hash
from smai_orchestrator.checkpointer.metadata import (
    CheckpointBackend,
    InMemoryCheckpointBackend,
    MetadataStoreCheckpointer,
)
from smai_orchestrator.checkpointer.types import (
    Checkpoint,
    Checkpointer,
    CheckpointKey,
)

__all__ = [
    "DEFAULT_PREFIX",
    "ArtifactStoreCheckpointer",
    "Checkpoint",
    "CheckpointBackend",
    "CheckpointKey",
    "Checkpointer",
    "InMemoryCheckpointBackend",
    "MetadataStoreCheckpointer",
    "memoized",
    "stable_hash",
]
