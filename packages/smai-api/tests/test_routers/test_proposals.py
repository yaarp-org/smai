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

_VALID_PROPOSAL_DESCRIPTION: dict[str, object] = {
    "name": "router_test_probe",
    "summary": (
        "Edge-case-test placeholder TechniqueDescription used by the proposals "
        "router unit tests; not a real technique."
    ),
    "motivation": (
        "Router tests need a minimum-viable body that round-trips through the "
        "typed TechniqueDescription schema introduced at planner-refactor Step 2 "
        "(upstream_requirements §2). This body is the smallest valid proposal."
    ),
    "problem_setting": ("Unit test of the proposals router; no real ML problem setting."),
    "algorithm": {
        "summary": (
            "Placeholder algorithm summary that exists only to satisfy the "
            "schema's minimum length rule on AlgorithmSpec.summary."
        ),
    },
    "context_kind": "proposal",
    "source_proposal_id": "router-test-probe",
    "confidence_flags": [
        {
            "field_path": "/limitations",
            "severity": "unknown",
            "note": "Router test probe; field intentionally unset.",
        },
        {
            "field_path": "/loss_function",
            "severity": "unknown",
            "note": "Router test probe; field intentionally unset.",
        },
        {
            "field_path": "/training_recipe",
            "severity": "unknown",
            "note": "Router test probe; field intentionally unset.",
        },
        {
            "field_path": "/evaluation_protocol",
            "severity": "unknown",
            "note": "Router test probe; field intentionally unset.",
        },
    ],
}


@pytest.mark.asyncio
async def test_submit_proposal_with_inline_dict_description(tmp_path: Path) -> None:
    """``technique_description`` (typed dict per planner_refactor Step 2)
    submits cleanly + the artifact key is populated on the response
    (per ``11`` §5.2.1)."""
    runtime = await make_test_runtime(artifact_root=tmp_path / "artifacts")
    app = make_api_app(runtime)  # type: ignore[arg-type]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            PROPOSALS,
            json={
                "submission_kind": "novel_technique",
                "technique_description": _VALID_PROPOSAL_DESCRIPTION,
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
                "technique_description": _VALID_PROPOSAL_DESCRIPTION,
                "proposal_id": "prop-edge-case-test-001",
            },
        )
        assert response.status_code == 202, response.text
        assert response.json()["id"] == "prop-edge-case-test-001"


@pytest.mark.asyncio
async def test_submit_proposal_with_prose_description(tmp_path: Path) -> None:
    """``description`` (free-text prose) is the PRIMARY novel-technique
    input per DEC-032 — the planner drafts the technique from it.

    Submits cleanly with 202 and the submission artifact key is populated
    on the response (the prose is persisted as the submission artifact).
    """
    runtime = await make_test_runtime(artifact_root=tmp_path / "artifacts")
    app = make_api_app(runtime)  # type: ignore[arg-type]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            PROPOSALS,
            json={
                "submission_kind": "novel_technique",
                "description": (
                    "Add cutout augmentation to the CIFAR-10 training transforms "
                    "and compare top-1 accuracy against an unaugmented baseline."
                ),
            },
        )
        assert response.status_code == 202, response.text
        body = response.json()
        assert body["technique_description_artifact_key"]


@pytest.mark.asyncio
async def test_malformed_json_returns_400_envelope(tmp_path: Path) -> None:
    """A request body that fails Pydantic validation returns 400 +
    structured envelope per ``11`` §6.1."""
    runtime = await make_test_runtime(artifact_root=tmp_path / "artifacts")
    app = make_api_app(runtime)  # type: ignore[arg-type]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # technique_description populated + reproduce_paper_arxiv_id populated
        # → spec model_validator rejects ("exactly one of").
        response = await client.post(
            PROPOSALS,
            json={
                "submission_kind": "novel_technique",
                "technique_description": {"a": 1},
                "reproduce_paper_arxiv_id": "2401.12345",
            },
        )
        assert response.status_code == 400, response.text
        body = response.json()
        assert body["error"]["code"] == "VALIDATION_ERROR"
        assert body["error"]["issues"]
