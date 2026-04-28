"""Step-memoization checkpointer Protocol and value types.

Per ``designs/smai/05-orchestrator.md`` §2 / DEC-025. The checkpointer is
**not** one of the four plugin Protocols (per `05` §9.7 — the same
"no second implementation yet" reasoning that DEC-028 applied to the
engine itself); it is a seam *internal* to the engine, with two concrete
implementations shipped under :mod:`smai_orchestrator.checkpointer`.

The shape borrows from LangGraph (per DEC-025) but the semantics are
narrower: v2's source of truth for entity state is :class:`MetadataStore`,
not a snapshot inside the engine. The checkpointer memoizes individual
*step results* — typically the output of an expensive LLM call (code
review, planning, contextual evaluation) — keyed on
``(thread_id, step_id, input_hash)``.

Memoization is opt-in at the dispatch-handler level (`05` §2). A handler
that wants memoization wraps its work via the
:func:`smai_orchestrator.checkpointer.memoized` helper (or hand-rolls the
load → run → save shape). The engine never inserts memoization on a
handler's behalf.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict


class CheckpointKey(BaseModel):
    """Identifying tuple for a memoized step result (`05` §2).

    * ``thread_id`` — the entity id (`05` §2). The "thread" terminology
      is borrowed from LangGraph but the shape is specific: one
      checkpoint key per (entity, step) pair, never spanning entities.
    * ``step_id`` — a stable per-dispatch-site label. Spec authors bump
      the suffix (e.g., ``"code_review_v3"`` → ``"code_review_v4"``)
      to invalidate prior memos when the step's semantics change.
    * ``input_hash`` — opaque hash of the inputs that produced the
      result. The dispatch handler chooses the hashing convention; the
      checkpointer never inspects the contents. The
      :func:`smai_orchestrator.checkpointer.stable_hash` helper is the
      default convention (`05` §9 #3).
    """

    model_config = ConfigDict(extra="forbid")

    thread_id: str
    step_id: str
    input_hash: str


class Checkpoint(BaseModel):
    """One memoized step result (`05` §2).

    ``result`` is opaque bytes — the dispatch handler chooses the
    serialization (typically JSON via Pydantic ``model_dump_json()``).
    The checkpointer stores the body as-is and returns it verbatim on
    load; deserialization is the handler's job.

    ``created_at`` is the wall-clock timestamp at save time (UTC); the
    checkpointer populates it via the same wall-clock seam the engine
    uses elsewhere so tests can pin it deterministically.
    """

    model_config = ConfigDict(extra="forbid")

    key: CheckpointKey
    result: bytes
    created_at: datetime


@runtime_checkable
class Checkpointer(Protocol):
    """Engine-internal step-memoization seam (`05` §2).

    Two concrete implementations ship under
    :mod:`smai_orchestrator.checkpointer`:

    * :class:`MetadataStoreCheckpointer` — bodies live in a checkpoints
      table colocated with the rest of the orchestrator's state.
      Default, suited for results small enough for a row.
    * :class:`ArtifactStoreCheckpointer` — bodies live in
      :class:`ArtifactStore` at a stable content-hash key; the row in
      :class:`MetadataStore` (or in-memory) holds only the pointer.
      Suited for large results (full LLM conversation traces).

    The choice between flavors is per-dispatch-handler. Both implement
    this Protocol so handlers see a uniform shape regardless of which
    flavor the spec attached.
    """

    async def load(self, key: CheckpointKey) -> Checkpoint | None:
        """Return the stored checkpoint, or ``None`` if no entry exists.

        Implementations MUST NOT raise on miss — a missing key is the
        normal first-call shape, not an error.
        """
        ...

    async def save(self, checkpoint: Checkpoint) -> None:
        """Persist ``checkpoint`` against its :attr:`Checkpoint.key`.

        Subsequent :meth:`load` calls with the same key MUST return the
        saved checkpoint. Repeated :meth:`save` against the same key
        SHOULD overwrite (idempotent at the key level); this matches
        the LangGraph pattern and keeps the dispatch-handler retry path
        clean.
        """
        ...


__all__ = ["Checkpoint", "CheckpointKey", "Checkpointer"]
