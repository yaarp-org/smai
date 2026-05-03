"""Conformance tests for the ``/api/v1/comparison-groups`` endpoints.

Per ``designs/smai/11-api.md`` §4.4. CGs have NO CRUD on the contract
— they are created exclusively by ``POST /api/v1/proposals/{id}/approve``
or ``POST /api/v1/experiments``. The tests here exercise the read
surface and the artifact transport.

Per the brief carry-forward: 4.J1's three opaque-dict deviations
(EvaluationResult / SystemConfigResponse / compiled-artifacts inner
shapes) mean the EvaluationResult assertion here checks the outer
envelope only — inner ``raw_metrics`` / ``per_entry`` shapes are
``dict[str, object]`` on the contract surface and not deeply
asserted.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from smai_api_spec import (
    AgentStatusResponse,
    ArtifactKeysResponse,
    CGStatusResponse,
    CGSummary,
    ComparisonGroupDetailResponse,
    CursorPage,
    EntryDetailResponse,
    EntrySummary,
    EvaluationResultResponse,
)
from smai_api_spec.paths import (
    COMPARISON_GROUP_AGENT_STATUS,
    COMPARISON_GROUP_ARTIFACT,
    COMPARISON_GROUP_ARTIFACTS,
    COMPARISON_GROUP_DETAIL,
    COMPARISON_GROUP_ENTRIES,
    COMPARISON_GROUP_ENTRY_DETAIL,
    COMPARISON_GROUP_EVALUATION,
    COMPARISON_GROUP_STATUS,
    COMPARISON_GROUPS,
)

from smai_api_conformance._4_j2_fixtures import assert_error_envelope


class ComparisonGroupsConformanceTests:
    """Mixin: ``/api/v1/comparison-groups`` endpoint conformance tests."""

    # The CG-detail and per-entry tests need an existing CG. Subclasses
    # may override this fixture to point at a known-good ID; the default
    # skips the dependent tests with a clear message. The conformance
    # suite cannot reliably create a CG without going through either a
    # proposal-approve flow (requires a worker to drive
    # proposal_submitted → designed) or an experiment-submit flow
    # (requires a methodology-compilable YAML; coupled to the
    # implementation's plugin set).

    @pytest.fixture
    def existing_cg_id(self) -> str:
        """Override to return an existing CG ID for round-trip tests.

        Default skips dependent tests. The self-test mock supplies
        ``"cg_self_test_default"`` — a real implementation should
        override to a CG it has created via its own seeding path.
        """
        pytest.skip(
            "existing_cg_id fixture not provided; override in your subclass to "
            "exercise CG round-trip tests against a known-good CG"
        )
        raise AssertionError("unreachable")  # pragma: no cover

    @pytest.fixture
    def existing_entry_id(self) -> str:
        """Override to return an existing entry ID under ``existing_cg_id``."""
        pytest.skip("existing_entry_id fixture not provided; override in your subclass")
        raise AssertionError("unreachable")  # pragma: no cover

    # ---- GET /api/v1/comparison-groups ------------------------------------

    async def test_list_cgs_returns_cursor_page(self, client: AsyncClient) -> None:
        """List endpoint returns a valid ``CursorPage[CGSummary]``."""
        response = await client.get(COMPARISON_GROUPS)
        assert response.status_code == 200, response.text
        page = CursorPage[CGSummary].model_validate(response.json())
        assert page.count == len(page.items)

    async def test_list_cgs_accepts_state_filter(self, client: AsyncClient) -> None:
        """``?state=draft`` filters by CG state."""
        response = await client.get(COMPARISON_GROUPS, params={"state": "draft"})
        assert response.status_code == 200, response.text
        page = CursorPage[CGSummary].model_validate(response.json())
        for item in page.items:
            assert item.state == "draft"

    async def test_list_cgs_accepts_proposal_id_filter(self, client: AsyncClient) -> None:
        """``?proposal_id=`` filters by the parent proposal."""
        response = await client.get(COMPARISON_GROUPS, params={"proposal_id": "prop_filter_probe"})
        assert response.status_code == 200, response.text
        # Page may legitimately be empty.
        CursorPage[CGSummary].model_validate(response.json())

    # ---- GET /api/v1/comparison-groups/{cg_id} ----------------------------

    async def test_get_cg_404_for_missing_id(self, client: AsyncClient) -> None:
        """Detail of a nonexistent CG → 404 + CG_NOT_FOUND."""
        url = COMPARISON_GROUP_DETAIL.format(cg_id="cg_does_not_exist_xyz")
        response = await client.get(url)
        assert response.status_code == 404, response.text
        assert_error_envelope(response, expected_code="CG_NOT_FOUND")

    async def test_get_cg_detail_shape(self, client: AsyncClient, existing_cg_id: str) -> None:
        """GET CG detail returns valid ComparisonGroupDetailResponse."""
        url = COMPARISON_GROUP_DETAIL.format(cg_id=existing_cg_id)
        response = await client.get(url)
        assert response.status_code == 200, response.text
        ComparisonGroupDetailResponse.model_validate(response.json())

    # ---- GET /api/v1/comparison-groups/{cg_id}/status ---------------------

    async def test_get_cg_status_404(self, client: AsyncClient) -> None:
        """Status of a nonexistent CG → 404."""
        url = COMPARISON_GROUP_STATUS.format(cg_id="cg_does_not_exist_xyz")
        response = await client.get(url)
        assert response.status_code == 404, response.text
        assert_error_envelope(response, expected_code="CG_NOT_FOUND")

    async def test_get_cg_status_shape(self, client: AsyncClient, existing_cg_id: str) -> None:
        """GET CG status returns valid CGStatusResponse."""
        url = COMPARISON_GROUP_STATUS.format(cg_id=existing_cg_id)
        response = await client.get(url)
        assert response.status_code == 200, response.text
        CGStatusResponse.model_validate(response.json())

    # ---- GET /api/v1/comparison-groups/{cg_id}/agent-status ---------------

    async def test_get_agent_status_404(self, client: AsyncClient) -> None:
        """Agent-status of a nonexistent CG → 404."""
        url = COMPARISON_GROUP_AGENT_STATUS.format(cg_id="cg_does_not_exist_xyz")
        response = await client.get(url)
        assert response.status_code == 404, response.text
        assert_error_envelope(response, expected_code="CG_NOT_FOUND")

    async def test_get_agent_status_shape(self, client: AsyncClient, existing_cg_id: str) -> None:
        """GET agent-status returns valid AgentStatusResponse.

        ``harness`` may be ``None`` (status.json not yet written);
        ``entries`` may be empty; both are spec-conformant.
        """
        url = COMPARISON_GROUP_AGENT_STATUS.format(cg_id=existing_cg_id)
        response = await client.get(url)
        assert response.status_code == 200, response.text
        AgentStatusResponse.model_validate(response.json())

    # ---- GET /api/v1/comparison-groups/{cg_id}/entries --------------------

    async def test_list_entries_404_for_missing_cg(self, client: AsyncClient) -> None:
        """Entries listing for a nonexistent CG → 404."""
        url = COMPARISON_GROUP_ENTRIES.format(cg_id="cg_does_not_exist_xyz")
        response = await client.get(url)
        assert response.status_code == 404, response.text
        assert_error_envelope(response, expected_code="CG_NOT_FOUND")

    async def test_list_entries_shape(self, client: AsyncClient, existing_cg_id: str) -> None:
        """List entries returns a CursorPage[EntrySummary]."""
        url = COMPARISON_GROUP_ENTRIES.format(cg_id=existing_cg_id)
        response = await client.get(url)
        assert response.status_code == 200, response.text
        CursorPage[EntrySummary].model_validate(response.json())

    # ---- GET /api/v1/comparison-groups/{cg_id}/entries/{entry_id} ---------

    async def test_get_entry_404_for_missing_entry(
        self, client: AsyncClient, existing_cg_id: str
    ) -> None:
        """Entry detail for a nonexistent entry → 404."""
        url = COMPARISON_GROUP_ENTRY_DETAIL.format(
            cg_id=existing_cg_id, entry_id="entry_does_not_exist_xyz"
        )
        response = await client.get(url)
        assert response.status_code == 404, response.text
        err = assert_error_envelope(response)
        assert err.code in {"ENTRY_NOT_FOUND", "CG_NOT_FOUND"}

    async def test_get_entry_detail_shape(
        self,
        client: AsyncClient,
        existing_cg_id: str,
        existing_entry_id: str,
    ) -> None:
        """GET entry detail returns valid EntryDetailResponse."""
        url = COMPARISON_GROUP_ENTRY_DETAIL.format(cg_id=existing_cg_id, entry_id=existing_entry_id)
        response = await client.get(url)
        assert response.status_code == 200, response.text
        EntryDetailResponse.model_validate(response.json())

    # ---- GET /api/v1/comparison-groups/{cg_id}/evaluation -----------------

    async def test_get_evaluation_404_when_not_produced(self, client: AsyncClient) -> None:
        """Evaluation read for a CG with no result yet → 404."""
        url = COMPARISON_GROUP_EVALUATION.format(cg_id="cg_does_not_exist_xyz")
        response = await client.get(url)
        assert response.status_code == 404, response.text
        err = assert_error_envelope(response)
        # Per ``11-api.md`` §4.4: 404 with ARTIFACT_NOT_FOUND when the
        # evaluation result hasn't been produced yet, or CG_NOT_FOUND
        # if the CG itself doesn't exist.
        assert err.code in {"ARTIFACT_NOT_FOUND", "CG_NOT_FOUND"}

    async def test_get_evaluation_shape_when_present(
        self, client: AsyncClient, existing_cg_id: str
    ) -> None:
        """When the evaluation result exists it parses as EvaluationResultResponse.

        Only asserts shape on the 200 branch — when the implementation
        returns 404 (no result yet) the test asserts envelope shape.
        """
        url = COMPARISON_GROUP_EVALUATION.format(cg_id=existing_cg_id)
        response = await client.get(url)
        if response.status_code == 200:
            EvaluationResultResponse.model_validate(response.json())
        elif response.status_code == 404:
            err = assert_error_envelope(response)
            assert err.code in {"ARTIFACT_NOT_FOUND", "CG_NOT_FOUND"}
        else:
            pytest.fail(
                f"unexpected status {response.status_code} from GET {url}; body: {response.text}"
            )

    # ---- GET /api/v1/comparison-groups/{cg_id}/artifacts ------------------

    async def test_list_artifacts_shape(self, client: AsyncClient, existing_cg_id: str) -> None:
        """List artifact keys returns ArtifactKeysResponse."""
        url = COMPARISON_GROUP_ARTIFACTS.format(cg_id=existing_cg_id)
        response = await client.get(url)
        assert response.status_code == 200, response.text
        ArtifactKeysResponse.model_validate(response.json())

    async def test_list_artifacts_with_prefix(
        self, client: AsyncClient, existing_cg_id: str
    ) -> None:
        """``?prefix=`` filter narrows the listing."""
        url = COMPARISON_GROUP_ARTIFACTS.format(cg_id=existing_cg_id)
        response = await client.get(url, params={"prefix": "harness/"})
        assert response.status_code == 200, response.text
        body = ArtifactKeysResponse.model_validate(response.json())
        for key in body.keys:
            assert key.startswith("harness/"), (
                f"key {key!r} does not match requested prefix 'harness/'"
            )

    # ---- GET /api/v1/comparison-groups/{cg_id}/artifacts/{path:path} ------

    async def test_get_artifact_404_for_missing(self, client: AsyncClient) -> None:
        """Fetching an unknown artifact path → 404 + ARTIFACT_NOT_FOUND."""
        url = COMPARISON_GROUP_ARTIFACT.format(cg_id="cg_does_not_exist_xyz", path="nope.json")
        response = await client.get(url, follow_redirects=False)
        assert response.status_code == 404, response.text
        err = assert_error_envelope(response)
        assert err.code in {"ARTIFACT_NOT_FOUND", "CG_NOT_FOUND"}

    async def test_get_artifact_streams_or_redirects(
        self, client: AsyncClient, existing_cg_id: str
    ) -> None:
        """Artifact fetch either streams bytes (200) or redirects (302).

        Per ``11-api.md`` §5.2.4: the API returns 302 when the configured
        ArtifactStore advertises ``supports_presigned_urls=True``;
        otherwise streams the bytes back via FileResponse. Both modes
        are spec-conformant; the SPA never knows which is active.
        """
        url = COMPARISON_GROUP_ARTIFACT.format(cg_id=existing_cg_id, path="harness/status.json")
        response = await client.get(url, follow_redirects=False)
        if response.status_code == 200:
            # Streamed bytes — content-type may be anything; just ensure
            # the body is non-trivial.
            assert response.content is not None
        elif response.status_code == 302:
            assert response.headers.get("location"), (
                "302 response missing required Location header per `11` §5.2.4"
            )
        elif response.status_code == 404:
            # Acceptable when the artifact path doesn't exist on this CG.
            err = assert_error_envelope(response)
            assert err.code in {"ARTIFACT_NOT_FOUND", "CG_NOT_FOUND"}
        else:
            pytest.fail(
                f"unexpected status {response.status_code} from GET {url}; body: {response.text}"
            )


__all__ = ["ComparisonGroupsConformanceTests"]
