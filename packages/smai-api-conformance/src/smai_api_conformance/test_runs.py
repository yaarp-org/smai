"""Conformance tests for the ``/api/v1/runs`` endpoints.

Per ``designs/smai/11-api.md`` §4.5. Runs are leaf entities — created
by the worker as part of the CG-execution pipeline-spec, never by a
direct API call. The tests here exercise the read surface and the
optional filters (``state``, ``cg_id``, ``entry_id``).
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from smai_api_spec import (
    CursorPage,
    RunDetailResponse,
    RunSummary,
)
from smai_api_spec.paths import RUN_DETAIL, RUNS

from smai_api_conformance._4_j2_fixtures import assert_error_envelope


class RunsConformanceTests:
    """Mixin: ``/api/v1/runs`` endpoint conformance tests."""

    @pytest.fixture
    def existing_run_id(self) -> str:
        """Override to return an existing run ID for round-trip tests.

        Default skips the dependent test. The self-test mock supplies
        ``"run_self_test_default"``; real implementations override
        when they have a known-good run from their seeding path.
        """
        pytest.skip("existing_run_id fixture not provided; override in your subclass")
        raise AssertionError("unreachable")  # pragma: no cover

    # ---- GET /api/v1/runs -------------------------------------------------

    async def test_list_runs_returns_cursor_page(self, client: AsyncClient) -> None:
        """List endpoint returns a valid ``CursorPage[RunSummary]``."""
        response = await client.get(RUNS)
        assert response.status_code == 200, response.text
        page = CursorPage[RunSummary].model_validate(response.json())
        assert page.count == len(page.items)

    async def test_list_runs_accepts_state_filter(self, client: AsyncClient) -> None:
        """``?state=running`` filters by run state."""
        response = await client.get(RUNS, params={"state": "running"})
        assert response.status_code == 200, response.text
        page = CursorPage[RunSummary].model_validate(response.json())
        for item in page.items:
            assert item.state == "running"

    async def test_list_runs_accepts_cg_id_filter(self, client: AsyncClient) -> None:
        """``?cg_id=`` filters by parent CG."""
        response = await client.get(RUNS, params={"cg_id": "cg_filter_probe"})
        assert response.status_code == 200, response.text
        CursorPage[RunSummary].model_validate(response.json())

    async def test_list_runs_accepts_entry_id_filter(self, client: AsyncClient) -> None:
        """``?entry_id=`` filters by parent entry."""
        response = await client.get(RUNS, params={"entry_id": "entry_filter_probe"})
        assert response.status_code == 200, response.text
        CursorPage[RunSummary].model_validate(response.json())

    # ---- GET /api/v1/runs/{run_id} ----------------------------------------

    async def test_get_run_404_for_missing_id(self, client: AsyncClient) -> None:
        """Detail of a nonexistent run → 404 + RUN_NOT_FOUND."""
        url = RUN_DETAIL.format(run_id="run_does_not_exist_xyz")
        response = await client.get(url)
        assert response.status_code == 404, response.text
        assert_error_envelope(response, expected_code="RUN_NOT_FOUND")

    async def test_get_run_detail_shape(self, client: AsyncClient, existing_run_id: str) -> None:
        """GET run detail returns valid RunDetailResponse."""
        url = RUN_DETAIL.format(run_id=existing_run_id)
        response = await client.get(url)
        assert response.status_code == 200, response.text
        RunDetailResponse.model_validate(response.json())


__all__ = ["RunsConformanceTests"]
