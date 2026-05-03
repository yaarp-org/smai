"""Conformance tests for the ``/api/v1/papers`` endpoints.

Per ``designs/smai/11-api.md`` §4.2.
"""

from __future__ import annotations

from httpx import AsyncClient
from smai_api_spec import (
    CursorPage,
    PaperDetailResponse,
    PaperSubmissionResponse,
    PaperSummary,
)
from smai_api_spec.paths import PAPER_DETAIL, PAPER_PROMOTE_PARTIAL, PAPERS

from smai_api_conformance._4_j2_fixtures import (
    assert_error_envelope,
    sample_paper_request_body,
)


class PapersConformanceTests:
    """Mixin: ``/api/v1/papers`` endpoint conformance tests."""

    # ---- POST /api/v1/papers ----------------------------------------------

    async def test_submit_paper_returns_202(self, client: AsyncClient) -> None:
        """Submit a paper; 202 + valid PaperSubmissionResponse shape."""
        response = await client.post(PAPERS, json=sample_paper_request_body("2501.10001"))
        assert response.status_code == 202, response.text
        body = PaperSubmissionResponse.model_validate(response.json())
        assert body.arxiv_id == "2501.10001"

    async def test_submit_paper_idempotent(self, client: AsyncClient) -> None:
        """Re-submitting the same arxiv_id is idempotent (202; no error).

        Per ``11-api.md`` §4.2 / §4.8: ``POST /api/v1/papers`` is
        idempotent (creates / promotes / no-ops). Both calls return 202;
        response shapes parse cleanly.
        """
        body = sample_paper_request_body("2501.10002")
        first = await client.post(PAPERS, json=body)
        assert first.status_code == 202, first.text
        PaperSubmissionResponse.model_validate(first.json())

        second = await client.post(PAPERS, json=body)
        assert second.status_code == 202, second.text
        PaperSubmissionResponse.model_validate(second.json())

    async def test_submit_paper_validation_error(self, client: AsyncClient) -> None:
        """Empty body → 400 + VALIDATION_ERROR."""
        response = await client.post(PAPERS, json={})
        assert response.status_code == 400, response.text
        assert_error_envelope(response, expected_code="VALIDATION_ERROR")

    # ---- GET /api/v1/papers -----------------------------------------------

    async def test_list_papers_returns_cursor_page(self, client: AsyncClient) -> None:
        """List endpoint returns a valid ``CursorPage[PaperSummary]``."""
        response = await client.get(PAPERS)
        assert response.status_code == 200, response.text
        page = CursorPage[PaperSummary].model_validate(response.json())
        assert page.count == len(page.items)

    async def test_list_papers_accepts_state_filter(self, client: AsyncClient) -> None:
        """``?state=submitted`` filters by paper state."""
        response = await client.get(PAPERS, params={"state": "submitted"})
        assert response.status_code == 200, response.text
        page = CursorPage[PaperSummary].model_validate(response.json())
        for item in page.items:
            assert item.state == "submitted"

    # ---- GET /api/v1/papers/{arxiv_id} ------------------------------------

    async def test_get_paper_404_for_missing_id(self, client: AsyncClient) -> None:
        """Detail of a nonexistent paper → 404 + PAPER_NOT_FOUND."""
        url = PAPER_DETAIL.format(arxiv_id="9999.99999")
        response = await client.get(url)
        assert response.status_code == 404, response.text
        assert_error_envelope(response, expected_code="PAPER_NOT_FOUND")

    async def test_get_paper_round_trip(self, client: AsyncClient) -> None:
        """Submit a paper then GET its detail; assert spec shape."""
        body = sample_paper_request_body("2501.20001")
        submit = await client.post(PAPERS, json=body)
        assert submit.status_code == 202, submit.text

        url = PAPER_DETAIL.format(arxiv_id="2501.20001")
        response = await client.get(url)
        assert response.status_code == 200, response.text
        detail = PaperDetailResponse.model_validate(response.json())
        assert detail.arxiv_id == "2501.20001"

    # ---- POST /api/v1/papers/{arxiv_id}/promote-partial -------------------

    async def test_promote_partial_invalid_state_returns_409(self, client: AsyncClient) -> None:
        """Promote-partial on a paper not in ``partial`` state → 409.

        A freshly-submitted paper sits in ``submitted`` (lifecycle, not
        contract — but the fact it's NOT ``partial`` is enough to
        trigger 409 + INVALID_STATE).
        """
        body = sample_paper_request_body("2501.30001")
        submit = await client.post(PAPERS, json=body)
        assert submit.status_code == 202, submit.text

        url = PAPER_PROMOTE_PARTIAL.format(arxiv_id="2501.30001")
        response = await client.post(url)
        assert response.status_code == 409, response.text
        assert_error_envelope(response, expected_code="INVALID_STATE")

    async def test_promote_partial_404_for_missing_paper(self, client: AsyncClient) -> None:
        """Promote on nonexistent paper → 404 + PAPER_NOT_FOUND."""
        url = PAPER_PROMOTE_PARTIAL.format(arxiv_id="9999.99999")
        response = await client.post(url)
        assert response.status_code == 404, response.text
        assert_error_envelope(response, expected_code="PAPER_NOT_FOUND")


__all__ = ["PapersConformanceTests"]
