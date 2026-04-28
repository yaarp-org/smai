"""Test helpers shared across 2.B2 tool tests."""

from __future__ import annotations

from pathlib import Path

from _agent_fakes import StubArtifactStore, StubLlmProvider  # type: ignore[import-not-found]
from smai_agents.loop import AgentSession
from smai_agents.tools import ToolContext, ToolRegistry
from smai_core.plugins import ArtifactStore, Compute


def make_test_session(
    workspace: Path,
    *,
    artifact_store: ArtifactStore | None = None,
    compute: Compute | None = None,
    time_provider: object | None = None,
) -> AgentSession:
    """Build a minimal :class:`AgentSession` for tool-handler tests.

    The session uses a :class:`StubLlmProvider` with no canned responses
    — tool-handler tests don't drive the loop; they call handlers
    directly with a :class:`ToolContext` derived from the session.
    """
    provider = StubLlmProvider(responses=[])
    fields: dict[str, object] = {
        "system_prompt": "test",
        "tools": ToolRegistry(),
        "llm_providers": {"harness_builder": provider},
        "current_role": "harness_builder",
        "workspace_path": workspace,
    }
    if artifact_store is not None:
        fields["artifact_store"] = artifact_store
    if compute is not None:
        fields["compute"] = compute
    if time_provider is not None:
        fields["time_provider"] = time_provider
    return AgentSession.model_validate(fields)


def make_test_context(session: AgentSession) -> ToolContext:
    """Build a :class:`ToolContext` rooted in ``session``."""
    return ToolContext(
        workspace_path=session.workspace_path,
        artifact_store=session.artifact_store,
        compute=session.compute,
        session=session,
    )


__all__ = [
    "StubArtifactStore",
    "StubLlmProvider",
    "make_test_context",
    "make_test_session",
]
