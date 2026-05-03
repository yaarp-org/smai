"""Edge-case tests for the papers router."""

from __future__ import annotations

from pathlib import Path

import pytest
from _4_k1_fixtures import (  # type: ignore[import-not-found]
    make_test_runtime,
)
from httpx import ASGITransport, AsyncClient
from smai_api import make_api_app
from smai_api_spec.paths import PAPERS


@pytest.mark.asyncio
async def test_submit_paper_idempotent_returns_existing(tmp_path: Path) -> None:
    """A second submit on the same arxiv_id returns 202 with the existing
    record (per ``11`` §4.2 / §4.8)."""
    runtime = await make_test_runtime(artifact_root=tmp_path / "artifacts")
    app = make_api_app(runtime)  # type: ignore[arg-type]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        first = await client.post(PAPERS, json={"arxiv_id": "2501.99001"})
        assert first.status_code == 202, first.text
        second = await client.post(PAPERS, json={"arxiv_id": "2501.99001"})
        assert second.status_code == 202, second.text
        assert first.json()["arxiv_id"] == second.json()["arxiv_id"]


@pytest.mark.asyncio
async def test_proposal_reproduce_paper_not_ready_returns_409(tmp_path: Path) -> None:
    """Submitting a reproduce-paper proposal against a paper still in
    ``submitted`` (non-terminal) returns 409 + PAPER_NOT_READY (per OQ11
    RESOLVED 2026-05-03)."""
    runtime = await make_test_runtime(artifact_root=tmp_path / "artifacts")
    app = make_api_app(runtime)  # type: ignore[arg-type]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # First create a paper in non-terminal `submitted` state.
        await client.post(PAPERS, json={"arxiv_id": "2501.99002"})
        # Now try to reproduce it — the paper exists but isn't terminal.
        response = await client.post(
            "/api/v1/proposals",
            json={
                "submission_kind": "reproduce_paper",
                "reproduce_paper_arxiv_id": "2501.99002",
            },
        )
        assert response.status_code == 409, response.text
        body = response.json()
        assert body["error"]["code"] == "PAPER_NOT_READY"
