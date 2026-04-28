""":func:`smai_agents.retry_context.load_retry_context` round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest
from _agent_fakes import StubArtifactStore  # type: ignore[import-not-found]
from smai_agents import PREV_CONVERSATION_TRACE_FILENAME, load_retry_context


@pytest.mark.asyncio
async def test_load_retry_context_writes_trace_to_workspace(tmp_path: Path) -> None:
    """When the artifact exists, ``load_retry_context`` writes it to
    ``<workspace>/prev-conversation-trace.json`` and returns the path."""
    store = StubArtifactStore()
    artifact_path = "comparison-groups/cg-1/entries/entry-1/conversation-trace.json"
    canned_trace = b'{"turn":1,"role":"technique_implementer"}'
    await store.put(artifact_path, canned_trace)

    workspace = tmp_path / "ws"
    result = await load_retry_context(
        workspace_path=workspace,
        artifact_store=store,
        artifact_path=artifact_path,
    )

    assert result is not None
    assert result == workspace / PREV_CONVERSATION_TRACE_FILENAME
    assert result.read_bytes() == canned_trace


@pytest.mark.asyncio
async def test_load_retry_context_returns_none_on_missing(tmp_path: Path) -> None:
    """First-attempt sessions have no prior trace; the helper must
    return ``None`` rather than raise."""
    store = StubArtifactStore()
    workspace = tmp_path / "ws"
    workspace.mkdir()

    result = await load_retry_context(
        workspace_path=workspace,
        artifact_store=store,
        artifact_path="missing-key.json",
    )
    assert result is None
    assert not (workspace / PREV_CONVERSATION_TRACE_FILENAME).exists()


@pytest.mark.asyncio
async def test_load_retry_context_creates_workspace_if_absent(tmp_path: Path) -> None:
    """The dispatch handler may pre-create the workspace, but it's also
    common for tools to construct paths lazily — accept both."""
    store = StubArtifactStore()
    await store.put("trace-key", b"{}")

    workspace = tmp_path / "fresh-workspace"
    assert not workspace.exists()

    result = await load_retry_context(
        workspace_path=workspace,
        artifact_store=store,
        artifact_path="trace-key",
    )
    assert result is not None
    assert result == workspace / PREV_CONVERSATION_TRACE_FILENAME
    assert result.exists()


@pytest.mark.asyncio
async def test_load_retry_context_filename_is_fixed_per_dec_023(
    tmp_path: Path,
) -> None:
    """§8: the destination filename is hardcoded into the prompt
    variant. Renaming this constant is a coordinated change with B2/B3."""
    assert PREV_CONVERSATION_TRACE_FILENAME == "prev-conversation-trace.json"
