"""Edge-case tests for the proposals router not covered by the
parameterizable conformance suite."""

from __future__ import annotations

from pathlib import Path

import pytest
from _4_k1_fixtures import (  # type: ignore[import-not-found]
    make_test_runtime,
)
from httpx import ASGITransport, AsyncClient
from smai_api import make_api_app
from smai_api_spec.paths import PROPOSALS


@pytest.mark.asyncio
async def test_submit_proposal_with_inline_dict_description(tmp_path: Path) -> None:
    """``technique_description`` (dict) submits cleanly + the artifact
    key is populated on the response (per ``11`` §5.2.1)."""
    runtime = await make_test_runtime(artifact_root=tmp_path / "artifacts")
    app = make_api_app(runtime)  # type: ignore[arg-type]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            PROPOSALS,
            json={
                "submission_kind": "novel_technique",
                "technique_description": {"name": "test", "extension_points": []},
            },
        )
        assert response.status_code == 202, response.text
        body = response.json()
        assert body["technique_description_artifact_key"]


@pytest.mark.asyncio
async def test_submit_proposal_pinned_id_round_trips(tmp_path: Path) -> None:
    """Caller-pinned ``proposal_id`` is honored end-to-end (idempotency
    contract per ``11`` §5.2.1)."""
    runtime = await make_test_runtime(artifact_root=tmp_path / "artifacts")
    app = make_api_app(runtime)  # type: ignore[arg-type]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            PROPOSALS,
            json={
                "submission_kind": "novel_technique",
                "technique_description_text": "edge-case probe",
                "proposal_id": "prop-edge-case-test-001",
            },
        )
        assert response.status_code == 202, response.text
        assert response.json()["id"] == "prop-edge-case-test-001"


@pytest.mark.asyncio
async def test_malformed_json_returns_400_envelope(tmp_path: Path) -> None:
    """A request body that fails Pydantic validation returns 400 +
    structured envelope per ``11`` §6.1."""
    runtime = await make_test_runtime(artifact_root=tmp_path / "artifacts")
    app = make_api_app(runtime)  # type: ignore[arg-type]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Both technique fields populated → spec model_validator rejects.
        response = await client.post(
            PROPOSALS,
            json={
                "submission_kind": "novel_technique",
                "technique_description": {"a": 1},
                "technique_description_text": "also populated",
            },
        )
        assert response.status_code == 400, response.text
        body = response.json()
        assert body["error"]["code"] == "VALIDATION_ERROR"
        assert body["error"]["issues"]
