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


@pytest.mark.asyncio
async def test_run_loop_persists_trace_picked_up_by_load_retry_context(
    tmp_path: Path,
) -> None:
    """Round-6 item 5 end-to-end: ``run_loop`` persists ``session.messages``
    to ``conversation_trace_artifact_path`` on exit, and a subsequent
    ``load_retry_context`` against that same key finds it (was always
    ``ArtifactNotFound`` before — the DEC-023 feature was dead)."""
    from _agent_fakes import StubLlmProvider  # type: ignore[import-not-found]
    from _agent_helpers import model_response  # type: ignore[import-not-found]
    from smai_agents.loop import AgentSession, TextContent, run_loop
    from smai_agents.tools import ToolRegistry
    from smai_core.plugins import NormalizedMessage

    store = StubArtifactStore()
    trace_key = "comparison-groups/cg-x/entries/entry-x/conversation-trace.json"
    stub_llm = StubLlmProvider([model_response(text="ok", stop_reason="end_turn")])
    session = AgentSession(
        system_prompt="sys",
        messages=[NormalizedMessage(role="user", content=[TextContent(text="hello")])],
        tools=ToolRegistry(),
        llm_providers={"technique_implementer": stub_llm},  # type: ignore[dict-item]
        current_role="technique_implementer",
        workspace_path=tmp_path / "ws",
        artifact_store=store,  # type: ignore[arg-type]
        conversation_trace_artifact_path=trace_key,
    )
    await run_loop(session)

    assert trace_key in store._data
    # And the retry-context helper now picks it up.
    workspace = tmp_path / "retry-ws"
    result = await load_retry_context(
        workspace_path=workspace,
        artifact_store=store,  # type: ignore[arg-type]
        artifact_path=trace_key,
    )
    assert result is not None
    assert result.read_bytes() == store._data[trace_key]
