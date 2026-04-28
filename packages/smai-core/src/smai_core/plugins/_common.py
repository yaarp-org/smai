"""Shared types used across the four plugin Protocols.

Per ``designs/smai/07-plugin-interfaces.md`` §5.6.2 (cursor pagination),
§5.6.7 (lease tokens), and DEC-035 sub-decisions #1 and #2.

The shapes here are not Protocols themselves — they are the value types
exchanged across the Protocol surfaces (cursor pages of records returned
by ``list_*`` and ``get_ready_*`` methods on :class:`MetadataStore`; lease
tokens round-tripped through ``acquire_lease`` / ``release_lease`` /
``extend_lease``).

These types live in ``smai_core.plugins`` rather than ``smai_core.entities``
because they are *plugin-surface* concerns — the methodology layer never
sees a :class:`CursorPage` or a :class:`LeaseToken`. Keeping them under
``smai_core.plugins`` preserves the methodology / pipeline separation
(``entities`` is methodology; ``plugins`` is the seam to pipeline).
"""

from __future__ import annotations

from datetime import datetime
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")

# Entity-kind discriminator used by ``MetadataStore`` lease and counting
# methods. Per ``07-plugin-interfaces.md`` §5.6.6 / §5.6.7. Lowercase
# snake_case canonical form per ``03-state-machine.md`` §2.1 /
# ``01-data-model.md`` §5.2.7.
EntityKind = Literal["cg", "entry", "run", "proposal", "paper"]


class CursorPage(BaseModel, Generic[T]):
    """One page of results from a paginated query (§5.6.2 / DEC-035 #1).

    Per DEC-035 sub-decision #1, every ``list_*`` and ``get_ready_*`` /
    ``get_in_flight_*`` method on :class:`MetadataStore` returns a
    :class:`CursorPage` of the relevant record type. The cursor is
    plugin-internal — typically encoding ``(created_at, id)`` for FIFO
    ordering or ``(tenant_bucket, created_at, id)`` for tenant-fair
    ordering — and consumers (the engine, tooling) treat it as opaque.

    ``total`` is optional: plugins MAY populate it for tooling /
    dashboard use cases that need "showing 100 of N"; the engine's poll
    cycle does not consume it. Computing the total can require a second
    round-trip on Postgres / SQLite, which on large backlogs is cheaper
    to omit.
    """

    model_config = ConfigDict(extra="forbid")

    items: list[T]
    next_cursor: str | None = None
    total: int | None = None


class LeaseToken(BaseModel):
    """Opaque-to-consumers handle for a held lease (§5.6.7 / DEC-035 #2).

    Carries the metadata needed to release / extend without the consumer
    touching plugin internals.

    ``lease_holder_id`` is **informational only** — used for audit /
    debugging / "who's holding this?" surfaces. The canonical CAS gate
    for :meth:`MetadataStore.release_lease` and
    :meth:`MetadataStore.extend_lease` is :attr:`nonce` (per DEC-035 #2),
    which uniquely identifies "this lease, not a successor" and provides
    ABA protection if the lease has been expired-and-reacquired between
    the consumer's acquire and its release / extend call.
    """

    model_config = ConfigDict(extra="forbid")

    entity_kind: EntityKind
    entity_id: str
    acquired_at: datetime
    expires_at: datetime
    lease_holder_id: str
    nonce: str
