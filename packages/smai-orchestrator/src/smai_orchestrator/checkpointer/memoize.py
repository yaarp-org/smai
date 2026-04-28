"""Opt-in memoization helper for dispatch handlers.

Per ``designs/smai/05-orchestrator.md`` §2 / §9 #3. Memoization is
opt-in at the dispatch-handler level — handlers that want it call
:func:`memoized` to wrap their work in the canonical
load → run → save shape; handlers that don't simply call their work
directly. The engine never inserts memoization on a handler's behalf.

The default hashing convention — ``stable_hash(model_dump_json(input))``
— is shipped here per `05` §9 #3's settled recommendation: ship the
helper with documentation, mark as opt-in. Handlers that need a
different convention (e.g., hashing only a subset of the inputs to
keep memo hits high across non-load-bearing changes) can pass their
own ``compute_hash`` callable.
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable
from datetime import UTC
from typing import TypeVar

from pydantic import BaseModel

from smai_orchestrator.checkpointer.types import (
    Checkpoint,
    Checkpointer,
    CheckpointKey,
)
from smai_orchestrator.engine.clock import (
    WallClockProvider,
    default_wall_clock,
)

_InputT = TypeVar("_InputT", bound=BaseModel)
_OutputT = TypeVar("_OutputT", bound=BaseModel)


def stable_hash(payload: BaseModel) -> str:
    """Default :class:`CheckpointKey.input_hash` convention (`05` §9 #3).

    ``hashlib.sha256(model_dump_json(payload).encode()).hexdigest()``.
    Stable across runs because Pydantic's ``model_dump_json()`` uses
    deterministic key ordering for ``model_config = ConfigDict()``-
    flavored models.

    Handlers that need a different convention (subset hashing,
    canonicalized JSON via ``smai_core.canonical_json``, etc.) call
    :func:`memoized` with an explicit ``compute_hash`` argument.
    """
    return hashlib.sha256(payload.model_dump_json().encode("utf-8")).hexdigest()


async def memoized(  # noqa: PLR0913
    *,
    checkpointer: Checkpointer,
    thread_id: str,
    step_id: str,
    inputs: _InputT,
    work: Callable[[], Awaitable[_OutputT]],
    serialize: Callable[[_OutputT], bytes],
    deserialize: Callable[[bytes], _OutputT],
    compute_hash: Callable[[_InputT], str] = stable_hash,
    wall_clock: WallClockProvider = default_wall_clock,
) -> _OutputT:
    """Memoize the result of ``work()`` against
    ``(thread_id, step_id, hash(inputs))`` (`05` §2).

    On cache hit: returns the deserialized cached result without
    invoking ``work``. On miss: runs ``work``, serializes the result,
    saves the checkpoint, returns the result.

    ``serialize`` / ``deserialize`` are caller-owned because the
    checkpointer stores opaque bytes — the canonical pattern is
    ``serialize=lambda r: r.model_dump_json().encode()`` and
    ``deserialize=lambda b: ResultModel.model_validate_json(b)``,
    typed against the handler's specific output model.

    ``wall_clock`` matches the engine's two-clock convention from
    :mod:`smai_orchestrator.engine.clock`; tests inject a
    :class:`FakeClock` for deterministic ``Checkpoint.created_at``
    pinning. Production uses ``datetime.now(UTC)``.
    """
    key = CheckpointKey(
        thread_id=thread_id,
        step_id=step_id,
        input_hash=compute_hash(inputs),
    )
    cached = await checkpointer.load(key)
    if cached is not None:
        return deserialize(cached.result)
    result = await work()
    body = serialize(result)
    created_at = wall_clock()
    # Normalize a naive datetime into UTC — the serialized record
    # round-trips an ISO-8601 string, and naive datetimes would lose
    # their offset on ``isoformat()``.
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    await checkpointer.save(
        Checkpoint(key=key, result=body, created_at=created_at)
    )
    return result


__all__ = ["memoized", "stable_hash"]
