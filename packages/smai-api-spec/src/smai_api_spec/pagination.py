"""Cursor-based pagination shapes per ``designs/smai/11-api.md`` §5.1.1.

Every list endpoint accepts ``?cursor=<opaque>`` + ``?limit=<int>`` and
responds with a :class:`CursorPage` body. Cursors are opaque base64-ish
strings — clients round-trip the value without parsing it. Per DEC-035
the underlying ``MetadataStore.CursorPage`` chooses the encoding (Postgres
and SQLite both use ``(updated_at, id)`` tuples today).
"""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import Field

from smai_api_spec._common import APIBaseModel

T = TypeVar("T")


class PaginationParams(APIBaseModel):
    """Shared query-param model for every list endpoint.

    ``cursor`` is the opaque token returned in ``next_cursor`` of the
    prior page; ``None`` (or omitted) requests the first page.

    ``limit`` is the requested max page size. The implementation may
    return fewer items per page; callers must not rely on the exact
    count being honored. Per ``11`` §4.8 / §5.1.1 the per-implementation
    upper bound is plugin-internal — Postgres typically caps at ~500 to
    keep individual queries fast.
    """

    cursor: str | None = None
    limit: int | None = Field(default=None, ge=1, le=500)


class CursorPage(APIBaseModel, Generic[T]):
    """Generic page envelope.

    Per ``11`` §5.1.1 the contract is: ``next_cursor`` MAY be passed back
    as ``?cursor=`` on the same endpoint; ``None`` means "no more pages".
    ``count`` is the size of *this page* — it is NOT a total-result count
    (computing totals against a paginated keyset is expensive on large
    tables and the SPA does not need it).
    """

    items: list[T]
    next_cursor: str | None = None
    count: int


__all__ = ["CursorPage", "PaginationParams"]
