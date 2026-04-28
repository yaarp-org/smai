"""ArtifactStore-backed step-memoization checkpointer.

Per ``designs/smai/05-orchestrator.md`` §2 / §4.3. The "results too
large for a row" flavor of :class:`Checkpointer` — bodies live in
:class:`ArtifactStore` at a stable content-addressable path keyed on
``(thread_id, step_id, input_hash)``.

Suitable for memoizing full LLM conversation traces (10s–100s of KB) or
other handler outputs whose body would bloat the metadata-store row
shape. The artifact-store path encodes the checkpoint key directly —
``checkpoints/<thread_id>/<step_id>/<input_hash>.bin`` — so the key
*is* the addressing primitive; no separate row is needed.

Unlike :class:`MetadataStoreCheckpointer`, this flavor does not depend
on the Session-C-pending checkpoint-table CRUD; today's
:class:`ArtifactStore` Protocol surface (`07` §6) is sufficient.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from smai_core.plugins import ArtifactStore
from smai_core.plugins.artifact_store import ArtifactNotFound

from smai_orchestrator.checkpointer.types import Checkpoint, CheckpointKey
from smai_orchestrator.engine.clock import (
    WallClockProvider,
    default_wall_clock,
)

_UTC = UTC


# Default artifact-store path prefix for memoized step results. The
# leading ``orchestrator/`` namespace matches the convention already in
# use by ``smai-runtime`` / agent traces (`07` §6.2 examples) and keeps
# the checkpoint corpus discoverable by tooling.
DEFAULT_PREFIX = "orchestrator/checkpoints"


class ArtifactStoreCheckpointer:
    """:class:`Checkpointer` flavor that stores bodies in
    :class:`ArtifactStore` (`05` §2 / §4.3).

    The path under :attr:`prefix` is content-addressable —
    ``<prefix>/<thread_id>/<step_id>/<input_hash>.bin`` — so the key
    *is* the addressing primitive; no separate row in the metadata
    store. The ``created_at`` timestamp is recovered on
    :meth:`load` from the artifact's wall-clock header (a small JSON
    sidecar at ``...<input_hash>.meta.json``) — keeps the artifact
    body opaque to the checkpointer while preserving the
    ``Checkpoint.created_at`` round-trip.

    Constructor takes an :class:`ArtifactStore` plus an optional
    ``wall_clock`` callable for save-time ``created_at`` population
    (matching the engine's two-clock convention from
    :mod:`smai_orchestrator.engine.clock`); tests inject a
    :class:`FakeClock` for deterministic timestamps.
    """

    def __init__(
        self,
        artifact_store: ArtifactStore,
        *,
        prefix: str = DEFAULT_PREFIX,
        wall_clock: WallClockProvider = default_wall_clock,
    ) -> None:
        self._store = artifact_store
        self._prefix = prefix.rstrip("/")
        self._wall_clock = wall_clock

    def _body_key(self, key: CheckpointKey) -> str:
        return (
            f"{self._prefix}/{key.thread_id}/{key.step_id}/"
            f"{key.input_hash}.bin"
        )

    def _meta_key(self, key: CheckpointKey) -> str:
        return (
            f"{self._prefix}/{key.thread_id}/{key.step_id}/"
            f"{key.input_hash}.meta.json"
        )

    async def load(self, key: CheckpointKey) -> Checkpoint | None:
        try:
            body = await self._store.get(self._body_key(key))
        except ArtifactNotFound:
            return None
        try:
            meta_bytes = await self._store.get(self._meta_key(key))
        except ArtifactNotFound:
            # Body without a sidecar — the body was stored by an older
            # checkpointer revision or by hand; surface a synthetic
            # ``created_at`` (epoch) rather than fail the load. The
            # dispatch handler treats the result as stale-but-valid.
            created_at = datetime.fromtimestamp(0, tz=_UTC)
        else:
            created_at = _parse_created_at(meta_bytes)
        return Checkpoint(key=key, result=body, created_at=created_at)

    async def save(self, checkpoint: Checkpoint) -> None:
        body_key = self._body_key(checkpoint.key)
        meta_key = self._meta_key(checkpoint.key)
        meta_bytes = _serialize_created_at(checkpoint.created_at)
        await self._store.put(body_key, checkpoint.result)
        await self._store.put(meta_key, meta_bytes, content_type="application/json")


# === Sidecar serialization helpers ==========================================
#
# Tiny JSON shape: ``{"created_at": "<ISO-8601 UTC>"}``. A custom format
# rather than ``Checkpoint.model_dump_json`` would force the loader to
# reconstruct the key — the key is implicit in the artifact path, so
# only the ``created_at`` is sidecar'd.


def _serialize_created_at(dt: datetime) -> bytes:
    return json.dumps({"created_at": dt.isoformat()}).encode("utf-8")


def _parse_created_at(raw: bytes) -> datetime:
    payload = json.loads(raw.decode("utf-8"))
    return datetime.fromisoformat(payload["created_at"])


__all__ = ["ArtifactStoreCheckpointer", "DEFAULT_PREFIX"]
