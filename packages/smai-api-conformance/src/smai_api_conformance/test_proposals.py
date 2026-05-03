"""Conformance tests for the ``/api/v1/proposals`` endpoints.

Per ``designs/smai/11-api.md`` §4.1 / §5.2.1. All tests are
shape-only — they do not assert that the underlying engine drives the
proposal through ``proposal_submitted → designing → designed`` (that
is lifecycle, covered by implementation-specific suites).

The mixin defined here is **not** a pytest test class on its own
(``ProposalsConformanceTests`` does not start with ``Test``). It is
composed into :class:`smai_api_conformance.APIConformanceBase` via
multiple inheritance; the inherited ``test_*`` methods run when a
subclass with the ``Test*`` prefix supplies the ``client`` fixture.
"""

from __future__ import annotations

from httpx import AsyncClient
from smai_api_spec import (
    CursorPage,
    ProposalDetailResponse,
    ProposalSubmissionResponse,
    ProposalSummary,
)
from smai_api_spec.paths import PROPOSAL_APPROVE, PROPOSAL_DETAIL, PROPOSAL_REJECT, PROPOSALS

from smai_api_conformance._4_j2_fixtures import (
    assert_error_envelope,
    sample_proposal_request_body,
)


class ProposalsConformanceTests:
    """Mixin: ``/api/v1/proposals`` endpoint conformance tests."""

    # ---- POST /api/v1/proposals -------------------------------------------

    async def test_submit_proposal_returns_202_with_valid_shape(self, client: AsyncClient) -> None:
        """Submit a novel-technique proposal; expect 202 + spec shape."""
        response = await client.post(PROPOSALS, json=sample_proposal_request_body())
        assert response.status_code == 202, response.text
        # Pydantic ``extra="forbid"`` catches drift; field-level coverage
        # lives in smai-api-spec's tests/test_models_basic.py.
        body = ProposalSubmissionResponse.model_validate(response.json())
        assert body.state == "proposal_submitted"
        assert body.submission_kind == "novel_technique"

    async def test_submit_proposal_validation_error_returns_400(self, client: AsyncClient) -> None:
        """Bad payload → 400 + ErrorEnvelope (VALIDATION_ERROR)."""
        # No description fields populated — the model_validator on
        # SubmitProposalRequest rejects this with a 400.
        response = await client.post(PROPOSALS, json={"submission_kind": "novel_technique"})
        assert response.status_code == 400, response.text
        assert_error_envelope(response, expected_code="VALIDATION_ERROR")

    # ---- GET /api/v1/proposals --------------------------------------------

    async def test_list_proposals_returns_cursor_page(self, client: AsyncClient) -> None:
        """List endpoint returns a valid ``CursorPage[ProposalSummary]``."""
        response = await client.get(PROPOSALS)
        assert response.status_code == 200, response.text
        page = CursorPage[ProposalSummary].model_validate(response.json())
        assert page.count == len(page.items)

    async def test_list_proposals_accepts_state_filter(self, client: AsyncClient) -> None:
        """``?state=proposal_submitted`` filters; response is still a CursorPage."""
        response = await client.get(PROPOSALS, params={"state": "proposal_submitted"})
        assert response.status_code == 200, response.text
        page = CursorPage[ProposalSummary].model_validate(response.json())
        # Every returned item must match the filter.
        for item in page.items:
            assert item.state == "proposal_submitted"

    # ---- GET /api/v1/proposals/{id} ---------------------------------------

    async def test_get_proposal_404_for_missing_id(self, client: AsyncClient) -> None:
        """Detail of a nonexistent proposal → 404 + PROPOSAL_NOT_FOUND."""
        url = PROPOSAL_DETAIL.format(proposal_id="prop_does_not_exist_xyz")
        response = await client.get(url)
        assert response.status_code == 404, response.text
        assert_error_envelope(response, expected_code="PROPOSAL_NOT_FOUND")

    async def test_get_proposal_round_trip(self, client: AsyncClient) -> None:
        """Submit a proposal then GET its detail; assert spec shape.

        Per ``11-api.md`` §13 OQ6 RESOLVED 2026-05-03: the
        ``registered_cg_ids`` field is present (not absent) before
        registration, populated as ``[]``.
        """
        submit = await client.post(PROPOSALS, json=sample_proposal_request_body())
        assert submit.status_code == 202, submit.text
        submitted = ProposalSubmissionResponse.model_validate(submit.json())

        url = PROPOSAL_DETAIL.format(proposal_id=submitted.id)
        response = await client.get(url)
        assert response.status_code == 200, response.text
        detail = ProposalDetailResponse.model_validate(response.json())
        assert detail.id == submitted.id
        # Pre-registration: registered_cg_ids is present and empty.
        assert detail.registered_cg_ids == []

    # ---- POST /api/v1/proposals/{id}/approve ------------------------------

    async def test_approve_invalid_state_returns_409(self, client: AsyncClient) -> None:
        """Approving a proposal NOT in ``designed`` returns 409 INVALID_STATE.

        Per ``11-api.md`` §4.1 / §4.8: re-firing an RPC verb on a
        resource not in the right state is the contract's idempotency
        boundary. A freshly-submitted proposal sits in
        ``proposal_submitted`` (lifecycle, not contract — but the fact
        it's NOT ``designed`` is enough to trigger 409).
        """
        submit = await client.post(PROPOSALS, json=sample_proposal_request_body())
        assert submit.status_code == 202, submit.text
        submitted = ProposalSubmissionResponse.model_validate(submit.json())

        url = PROPOSAL_APPROVE.format(proposal_id=submitted.id)
        response = await client.post(url)
        assert response.status_code == 409, response.text
        assert_error_envelope(response, expected_code="INVALID_STATE")

    async def test_approve_404_for_missing_id(self, client: AsyncClient) -> None:
        """Approve on a nonexistent proposal → 404 + PROPOSAL_NOT_FOUND."""
        url = PROPOSAL_APPROVE.format(proposal_id="prop_does_not_exist_xyz")
        response = await client.post(url)
        assert response.status_code == 404, response.text
        assert_error_envelope(response, expected_code="PROPOSAL_NOT_FOUND")

    # ---- POST /api/v1/proposals/{id}/reject -------------------------------

    async def test_reject_invalid_state_returns_409(self, client: AsyncClient) -> None:
        """Rejecting a proposal NOT in ``designed`` returns 409 INVALID_STATE."""
        submit = await client.post(PROPOSALS, json=sample_proposal_request_body())
        assert submit.status_code == 202, submit.text
        submitted = ProposalSubmissionResponse.model_validate(submit.json())

        url = PROPOSAL_REJECT.format(proposal_id=submitted.id)
        response = await client.post(url, json={})
        assert response.status_code == 409, response.text
        assert_error_envelope(response, expected_code="INVALID_STATE")

    async def test_reject_404_for_missing_id(self, client: AsyncClient) -> None:
        """Reject on a nonexistent proposal → 404 + PROPOSAL_NOT_FOUND."""
        url = PROPOSAL_REJECT.format(proposal_id="prop_does_not_exist_xyz")
        response = await client.post(url, json={})
        assert response.status_code == 404, response.text
        assert_error_envelope(response, expected_code="PROPOSAL_NOT_FOUND")

    # ---- Reproduce-paper validation per OQ11 RESOLVED 2026-05-03 ---------

    async def test_reproduce_paper_missing_paper_returns_409(self, client: AsyncClient) -> None:
        """``submission_kind="reproduce_paper"`` referencing a missing
        ``arxiv_id`` returns **409 + ``PAPER_NOT_READY``** per
        ``11-api.md`` §13 OQ11 RESOLVED 2026-05-03.

        The contract is strict: a non-ready paper reference at proposal
        submission MUST surface as 409 + ``PAPER_NOT_READY`` (distinct
        from the generic ``INVALID_STATE`` so clients can render a
        paper-specific error message). Implementations that naturally
        do a paper-detail lookup first and would otherwise return 404
        from that path MUST translate to 409 + ``PAPER_NOT_READY`` at
        the proposal-validation layer.
        """
        body = {
            "submission_kind": "reproduce_paper",
            "reproduce_paper_arxiv_id": "9999.99999",  # never existed
        }
        response = await client.post(PROPOSALS, json=body)
        assert response.status_code == 409, response.text
        err = assert_error_envelope(response, expected_code="PAPER_NOT_READY")
        # `err` is the parsed APIError; the envelope helper has already
        # asserted err.code == "PAPER_NOT_READY" via expected_code.
        del err


__all__ = ["ProposalsConformanceTests"]
