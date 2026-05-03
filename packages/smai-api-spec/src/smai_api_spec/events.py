"""SSE event payload shapes per ``designs/smai/11-api.md`` §8.

The SSE channel is opened at ``GET /api/v1/events`` (see
:data:`smai_api_spec.paths.EVENTS`). Two event payload types in v1:
:class:`StateChangeEvent` (one per successful state-machine transition)
and :class:`WorkerHeartbeatEvent` (one per worker cycle).

The wire-format is the SSE standard with ``event:`` lines:

* ``event: state_change`` — body is the JSON form of :class:`StateChangeEvent`.
* ``event: worker_heartbeat`` — body is the JSON form of :class:`WorkerHeartbeatEvent`.
* ``event: refetch_all`` — sentinel event with no body, used per ``11``
  §8.3 when the in-memory ring buffer overflows. Clients respond by
  invalidating their entire cache. No payload type is needed.

Note: ``StateChangeEvent`` serializes ``from_state`` / ``to_state`` as
``"from"`` / ``"to"`` on the wire (Pydantic field aliases). ``from`` is a
reserved Python keyword and cannot be used as an attribute name; the
alias preserves the natural JSON shape per the ``11`` §8.1 example
payload.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import ConfigDict, Field

from smai_api_spec._common import APIBaseModel, EntityKind


class StateChangeEvent(APIBaseModel):
    """Fired on every successful state-machine transition.

    Wire shape per ``11`` §8.1::

        {"kind": "comparison_group", "id": "cg_abc",
         "from": "implementing", "to": "implemented",
         "ts": "2026-04-30T17:23:11Z"}

    Lightweight by design — the SPA's TanStack Query layer treats each
    event as an invalidation signal and refetches the entity body via
    REST (per ``11`` §8.2: SSE-as-invalidator, not SSE-as-data-pusher).

    ``from_state`` / ``to_state`` are the Python attribute names;
    ``from`` / ``to`` are the JSON wire names (Pydantic field aliases —
    ``from`` is a reserved keyword in Python). Use
    ``model_dump(by_alias=True)`` / ``model_dump_json(by_alias=True)``
    when serializing for the wire; alias forms are accepted on
    deserialization regardless of ``by_alias``.
    """

    # `populate_by_name=True` lets callers construct the model using the
    # Python attribute names (`StateChangeEvent(from_state=..., to_state=...)`)
    # while wire JSON uses the aliases. `extra="forbid"` is inherited
    # from APIBaseModel; redeclared here so the extension is explicit.
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    kind: EntityKind
    id: str
    from_state: str = Field(alias="from")
    to_state: str = Field(alias="to")
    ts: datetime


class WorkerHeartbeatEvent(APIBaseModel):
    """Fired once per worker cycle.

    Drives the SPA's "last cycle at" indicator. ``cycle_id`` is a
    monotone per-process integer; ``cycles_processed`` is cumulative
    across the worker's lifetime (resets to 0 on worker restart).
    """

    cycle_id: int
    cycles_processed: int
    ts: datetime


__all__ = ["StateChangeEvent", "WorkerHeartbeatEvent"]
