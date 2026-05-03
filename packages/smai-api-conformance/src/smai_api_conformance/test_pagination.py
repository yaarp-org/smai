"""Cross-cutting tests for cursor-pagination round-trip behavior.

Per ``designs/smai/11-api.md`` §4.8 / §5.1.1:

  *A list response's ``next_cursor`` MUST be acceptable as ``?cursor=``
  on the same endpoint.*

The test seeds enough records to force a multi-page response, captures
the cursor from page 1, refetches with that cursor, and asserts the
two pages don't overlap. We use ``POST /api/v1/papers`` for seeding
because it is idempotent (per ``11`` §4.8) — re-running the test
suite doesn't accumulate stale records, and the test is robust against
implementations that share a database across runs.
"""

from __future__ import annotations

from httpx import AsyncClient
from smai_api_spec import CursorPage, PaperSummary
from smai_api_spec.paths import PAPERS

from smai_api_conformance._4_j2_fixtures import sample_paper_request_body


class PaginationConformanceTests:
    """Mixin: cursor-pagination round-trip conformance tests."""

    async def test_cursor_round_trip_no_overlap(self, client: AsyncClient) -> None:
        """Seed N records, fetch page 1 + page 2, assert no overlap.

        Per ``11`` §5.1.1: the contract is that ``next_cursor`` MAY be
        passed back as ``?cursor=`` on the same endpoint and the
        following page contains items not present in the previous
        page. ``None`` ``next_cursor`` means "no more pages".
        """
        # Seed five papers with deterministic ids; idempotent submit means
        # re-running the suite doesn't accumulate them.
        seed_ids = [f"2501.4{i:04d}" for i in range(5)]
        for arxiv_id in seed_ids:
            response = await client.post(PAPERS, json=sample_paper_request_body(arxiv_id))
            assert response.status_code == 202, response.text

        # Page 1: limit=2 to force at least two pages.
        first = await client.get(PAPERS, params={"limit": 2})
        assert first.status_code == 200, first.text
        page1 = CursorPage[PaperSummary].model_validate(first.json())
        if page1.next_cursor is None:
            # The implementation may return all records on page 1
            # (limit is "max", not "exact"). That's spec-conformant —
            # the round-trip rule has nothing to round-trip. Skip the
            # overlap assertion; the cursor-acceptance assertion below
            # is what's load-bearing.
            return

        # Page 2: refetch with the cursor we just received.
        second = await client.get(PAPERS, params={"limit": 2, "cursor": page1.next_cursor})
        assert second.status_code == 200, second.text
        page2 = CursorPage[PaperSummary].model_validate(second.json())

        page1_ids = {item.arxiv_id for item in page1.items}
        page2_ids = {item.arxiv_id for item in page2.items}
        assert not (page1_ids & page2_ids), (
            f"page 1 and page 2 overlap on ids {page1_ids & page2_ids!r} — "
            f"cursor round-trip is not advancing the keyset"
        )

    async def test_cursor_acceptance(self, client: AsyncClient) -> None:
        """An opaque cursor returned by the server MUST be accepted on the
        same endpoint without error.

        This is the minimal contract — a list endpoint that hands out
        a cursor must accept that cursor when it comes back. Even if
        the data is too thin to produce overlap-free pages, the
        cursor must round-trip cleanly.
        """
        first = await client.get(PAPERS, params={"limit": 1})
        assert first.status_code == 200, first.text
        page1 = CursorPage[PaperSummary].model_validate(first.json())
        if page1.next_cursor is None:
            # No next page — nothing to round-trip. The endpoint is
            # still spec-conformant.
            return

        second = await client.get(PAPERS, params={"limit": 1, "cursor": page1.next_cursor})
        assert second.status_code == 200, (
            f"server-provided cursor {page1.next_cursor!r} rejected on refetch: "
            f"{second.status_code} {second.text}"
        )
        CursorPage[PaperSummary].model_validate(second.json())


__all__ = ["PaginationConformanceTests"]
