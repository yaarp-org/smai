"""Helpers for slicing an in-memory list against the API's cursor + limit
contract per ``designs/smai/11-api.md`` §5.1.1.

The Runtime services aggregate ``MetadataStore`` cursor pages into
materialized lists (``list_active`` / ``list_active_cgs`` etc.) — they
are bounded by single-user / laptop-deployment scale, not full database
enumerations. The API then re-slices those lists into client-facing
``CursorPage`` responses.

The cursor we emit is a simple base64-encoded integer offset into the
underlying list. Round-tripping a server-emitted cursor lands the
caller on the next page; an opaque-cursor consumer cannot tell this is
an offset and the contract docs warn them not to look. When the
underlying list shape changes (a new entity transitions in mid-page)
the cursor still parses and the client just sees a slightly different
slice — acceptable in v1's single-user pacing.
"""

from __future__ import annotations

import base64
from typing import TypeVar

from smai_api_spec import CursorPage

_DEFAULT_LIMIT = 50
_MAX_LIMIT = 500
_CURSOR_PREFIX = "smai-api-v1:"

_T = TypeVar("_T")


def _encode_cursor(offset: int) -> str:
    """Encode an offset as an opaque base64 cursor."""
    raw = f"{_CURSOR_PREFIX}{offset}".encode()
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_cursor(cursor: str | None) -> int:
    """Decode a previously-emitted cursor back into an offset.

    Unknown / malformed cursors degrade to offset 0 (the start of the
    list). The contract is permissive here — implementations are
    allowed to re-issue cursors across deployments without an explicit
    invalidation event, so a "I don't recognize that cursor" → "give
    me the first page" fallback is safer than a 400.
    """
    if cursor is None or cursor == "":
        return 0
    try:
        # base64 decode tolerates missing padding when given the
        # ``+ "===="`` suffix; pad to a multiple of 4.
        padded = cursor + "=" * (-len(cursor) % 4)
        decoded = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return 0
    if not decoded.startswith(_CURSOR_PREFIX):
        return 0
    suffix = decoded[len(_CURSOR_PREFIX) :]
    try:
        offset = int(suffix)
    except ValueError:
        return 0
    return offset if offset >= 0 else 0


def paginate(
    items: list[_T],
    cursor: str | None,
    limit: int | None,
) -> CursorPage[_T]:
    """Slice ``items`` into a :class:`CursorPage` at the given cursor + limit.

    Returns a one-item-per-row page with a ``next_cursor`` that round-
    trips against the same items list (the caller holds that list
    stable across the request, so this is a thin in-memory pagination
    rather than a real keyset cursor).
    """
    effective_limit = _DEFAULT_LIMIT if limit is None else max(1, min(limit, _MAX_LIMIT))
    offset = _decode_cursor(cursor)
    end = offset + effective_limit
    page_items = items[offset:end]
    next_cursor: str | None = None
    if end < len(items):
        next_cursor = _encode_cursor(end)
    return CursorPage[_T](
        items=page_items,
        next_cursor=next_cursor,
        count=len(page_items),
    )


__all__ = ["paginate"]
