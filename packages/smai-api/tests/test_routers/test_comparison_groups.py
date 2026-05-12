"""Edge-case tests for the comparison-groups router — artifact transport
modes, agent-status composite read.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _4_k1_fixtures import (  # type: ignore[import-not-found]
    make_test_runtime,
    seed_state,
)
from httpx import ASGITransport, AsyncClient
from smai_api import make_api_app
from smai_api_spec.paths import (
    COMPARISON_GROUP_AGENT_STATUS,
    COMPARISON_GROUP_ARTIFACT,
    COMPARISON_GROUP_ARTIFACTS,
)


@pytest.mark.asyncio
async def test_artifact_endpoint_streams_for_localfs(tmp_path: Path) -> None:
    """LocalFs's ``url_for`` returns ``file://`` — the router should NOT
    redirect; instead it streams the bytes (per the inline note on the
    handler about HTTP-scheme detection)."""
    runtime = await make_test_runtime(artifact_root=tmp_path / "artifacts")
    seeded = await seed_state(runtime)
    app = make_api_app(runtime)  # type: ignore[arg-type]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        url = COMPARISON_GROUP_ARTIFACT.format(cg_id=seeded.cg_id, path="harness/status.json")
        response = await client.get(url, follow_redirects=False)
        assert response.status_code == 200, response.text
        assert b"harness_builder" in response.content


@pytest.mark.asyncio
async def test_artifact_endpoint_redirects_for_http_url(tmp_path: Path) -> None:
    """When the configured store advertises an HTTP-shaped presigned
    URL (the S3-shaped deployment), the router 302-redirects."""
    runtime = await make_test_runtime(
        artifact_root=tmp_path / "artifacts",
        use_inmem_artifact=True,
    )
    seeded = await seed_state(runtime)
    app = make_api_app(runtime)  # type: ignore[arg-type]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        url = COMPARISON_GROUP_ARTIFACT.format(cg_id=seeded.cg_id, path="harness/status.json")
        response = await client.get(url, follow_redirects=False)
        assert response.status_code == 302, response.text
        location = response.headers["location"]
        assert location.startswith("http://test-presigned/")


@pytest.mark.asyncio
async def test_list_artifacts_with_prefix(tmp_path: Path) -> None:
    """``?prefix=harness/`` narrows the listing per ``11`` §5.2.4."""
    runtime = await make_test_runtime(artifact_root=tmp_path / "artifacts")
    seeded = await seed_state(runtime)
    # Seed a non-harness artifact so the filter is meaningful.
    await runtime.plugins.artifact_store.put(
        f"comparison-groups/{seeded.cg_id}/entries/{seeded.entry_id}/code/main.py",
        b"# code\n",
    )
    app = make_api_app(runtime)  # type: ignore[arg-type]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        url = COMPARISON_GROUP_ARTIFACTS.format(cg_id=seeded.cg_id)
        response = await client.get(url, params={"prefix": "harness/"})
        assert response.status_code == 200, response.text
        body = response.json()
        for key in body["keys"]:
            assert key.startswith("harness/")


@pytest.mark.asyncio
async def test_agent_status_composes_harness_and_entries(tmp_path: Path) -> None:
    """Composite agent-status read joins ``harness/status.json`` and per-entry
    ``entries/{id}/status.json`` (per ``11`` §5.2.3).

    Exercises the post-Pass-A payload shape: the projection mirrors the
    real per-turn status keys (``role``, ``turn_count``, ``last_tool_call``,
    ``wall_clock_utc``, ...), and the per-entry path has no ``code/``
    segment (the technique implementer writes ``entries/{id}/status.json``).
    """
    runtime = await make_test_runtime(artifact_root=tmp_path / "artifacts")
    seeded = await seed_state(runtime)
    # Seed a per-entry status.json (no ``code/`` segment — see
    # technique_implementer.DEFAULT_STATUS_KEY_TEMPLATE) so the composite
    # has something to surface beyond the harness blob seed_state writes.
    await runtime.plugins.artifact_store.put(
        f"comparison-groups/{seeded.cg_id}/entries/{seeded.entry_id}/status.json",
        (
            b'{"role": "technique_implementer", "turn_count": 3, '
            b'"last_tool_call": "run_experiment", "last_tool_error": "boom", '
            b'"tool_errors_fired": 1, "attempt_index": 2}'
        ),
    )
    app = make_api_app(runtime)  # type: ignore[arg-type]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        url = COMPARISON_GROUP_AGENT_STATUS.format(cg_id=seeded.cg_id)
        response = await client.get(url)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["harness"] is not None
        assert body["harness"]["role"] == "harness_builder"
        assert body["harness"]["turn_count"] == 1
        assert body["harness"]["last_tool_call"] == "write_file"
        assert body["harness"]["wall_clock_utc"] is not None
        assert seeded.entry_id in body["entries"]
        entry_status = body["entries"][seeded.entry_id]["status"]
        assert entry_status["role"] == "technique_implementer"
        assert entry_status["turn_count"] == 3
        assert entry_status["last_tool_error"] == "boom"
        assert entry_status["attempt_index"] == 2
