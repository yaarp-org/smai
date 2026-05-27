""":func:`smai_inline_agents.maybe_run_supervisor_check` — Task 3.G4 hook.

Per ``04-agents.md`` §2.6 / §3.2 / §15 OQ2: the between-turn hook
consumes session state, calls the supervisor, and applies the
decision (continue → no-op; intervene → nudge file write; abort →
session-flag set + loop exit). Integration with :func:`run_loop` is
covered in :file:`test_supervisor_loop_integration.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _agent_fakes import StubArtifactStore, StubLlmProvider  # type: ignore[import-not-found]
from _agent_helpers import model_response  # type: ignore[import-not-found]
from _g4_fakes import canned_supervisor_decision  # type: ignore[import-not-found]
from smai_inline_agents import (
    AgentLoopConfig,
    AgentSession,
    ToolRegistry,
    make_finish_tool,
    maybe_run_supervisor_check,
)


def _new_session(
    *,
    supervised_llm: StubLlmProvider,
    supervisor_llm: StubLlmProvider | None,
    workspace: Path,
    artifact_store: StubArtifactStore | None = None,
    nudge_path: str | None = None,
    status_path: str | None = None,
) -> AgentSession:
    registry = ToolRegistry()
    registry.register(make_finish_tool())
    providers: dict[str, object] = {"harness_builder": supervised_llm}
    if supervisor_llm is not None:
        providers["supervisor"] = supervisor_llm
    return AgentSession(
        system_prompt="sys",
        tools=registry,
        llm_providers=providers,  # type: ignore[arg-type]
        current_role="harness_builder",
        workspace_path=workspace,
        artifact_store=artifact_store,
        nudge_artifact_path=nudge_path,
        status_artifact_path=status_path,
        config=AgentLoopConfig(supervisor_check_every_turns=1),
    )


# === continue → no-op =======================================================


@pytest.mark.asyncio
async def test_hook_continue_is_a_noop(tmp_path: Path) -> None:
    """``action='continue'`` does not write anything and does not abort."""
    store = StubArtifactStore()
    supervised = StubLlmProvider([])
    supervisor = StubLlmProvider(
        [canned_supervisor_decision(action="continue", reason="progressing")]
    )
    session = _new_session(
        supervised_llm=supervised,
        supervisor_llm=supervisor,
        workspace=tmp_path,
        artifact_store=store,
        nudge_path="comparison-groups/cg-1/harness/nudge.txt",
    )
    session.turn_count = 3
    session.recent_tool_call_names = ["read_file"]

    await maybe_run_supervisor_check(session)

    assert session.supervisor_aborted is False
    assert session.supervisor_abort_reason is None
    assert session.supervisor_checks_fired == 1
    # No writes — the supervisor said continue.
    assert store.put_calls == []


# === intervene → nudge file written =========================================


@pytest.mark.asyncio
async def test_hook_intervene_writes_nudge_file(tmp_path: Path) -> None:
    """``action='intervene'`` writes the nudge text to ``nudge_artifact_path``.

    The next-turn ``maybe_check_supervisor_nudge`` (existing surface)
    reads the file and injects it into the conversation. This test
    asserts on the write; ``test_supervisor_loop_integration`` asserts
    on the round-trip via :func:`run_loop`.
    """
    store = StubArtifactStore()
    supervised = StubLlmProvider([])
    supervisor = StubLlmProvider(
        [
            canned_supervisor_decision(
                action="intervene",
                reason="agent stuck on read_file",
                nudge="Move on — you have the file's contents in context.",
            )
        ]
    )
    nudge_path = "comparison-groups/cg-1/harness/nudge.txt"
    session = _new_session(
        supervised_llm=supervised,
        supervisor_llm=supervisor,
        workspace=tmp_path,
        artifact_store=store,
        nudge_path=nudge_path,
    )
    session.turn_count = 3

    await maybe_run_supervisor_check(session)

    assert session.supervisor_aborted is False
    # Nudge file was written.
    assert nudge_path in {key for key, _, _ in store.put_calls}
    body = await store.get(nudge_path)
    assert b"Move on" in body
    # Content type is plain text (per §3.2 the consumer reads it as raw bytes).
    written_ctype = next(ctype for key, _, ctype in store.put_calls if key == nudge_path)
    assert written_ctype == "text/plain"


@pytest.mark.asyncio
async def test_hook_intervene_no_op_when_no_nudge_path(tmp_path: Path) -> None:
    """When the session has no ``nudge_artifact_path``, the hook logs and
    returns rather than crashing — the supervisor's intervene
    decision is undeliverable and that's surfaced via warning, not
    exception."""
    store = StubArtifactStore()
    supervised = StubLlmProvider([])
    supervisor = StubLlmProvider(
        [
            canned_supervisor_decision(
                action="intervene",
                reason="x",
                nudge="hint",
            )
        ]
    )
    session = _new_session(
        supervised_llm=supervised,
        supervisor_llm=supervisor,
        workspace=tmp_path,
        artifact_store=store,
        nudge_path=None,
    )
    session.turn_count = 3

    await maybe_run_supervisor_check(session)

    assert session.supervisor_aborted is False
    assert store.put_calls == []  # No writes possible.


# === abort → session flag ===================================================


@pytest.mark.asyncio
async def test_hook_abort_sets_session_flag(tmp_path: Path) -> None:
    """``action='abort'`` sets :attr:`AgentSession.supervisor_aborted` +
    propagates the reason."""
    store = StubArtifactStore()
    supervised = StubLlmProvider([])
    supervisor = StubLlmProvider(
        [
            canned_supervisor_decision(
                action="abort",
                reason="agent has been stuck for 25 turns; aborting",
            )
        ]
    )
    session = _new_session(
        supervised_llm=supervised,
        supervisor_llm=supervisor,
        workspace=tmp_path,
        artifact_store=store,
        nudge_path="nudge.txt",
    )
    session.turn_count = 3

    await maybe_run_supervisor_check(session)

    assert session.supervisor_aborted is True
    assert session.supervisor_abort_reason == "agent has been stuck for 25 turns; aborting"
    # No nudge file — abort doesn't write nudges.
    assert store.put_calls == []


# === supervisor LLM missing → no-op =========================================


@pytest.mark.asyncio
async def test_hook_no_op_when_no_supervisor_llm(tmp_path: Path) -> None:
    """If no ``'supervisor'`` LlmProvider is wired, the hook logs a
    warning and returns — the supervised agent is not aborted on a
    config error."""
    store = StubArtifactStore()
    supervised = StubLlmProvider([])
    session = _new_session(
        supervised_llm=supervised,
        supervisor_llm=None,
        workspace=tmp_path,
        artifact_store=store,
        nudge_path="nudge.txt",
    )
    session.turn_count = 3

    await maybe_run_supervisor_check(session)

    assert session.supervisor_aborted is False
    assert session.supervisor_checks_fired == 0
    assert store.put_calls == []


# === supervisor StructuredCallFailed → no-op + warning ======================


@pytest.mark.asyncio
async def test_hook_no_op_when_supervisor_protocol_fails(tmp_path: Path) -> None:
    """A supervisor that produces text-not-tool-use twice in a row
    (DEC-018 second-attempt failure) is treated as 'continue' — a
    flaky supervisor must not abort the supervised agent."""
    store = StubArtifactStore()
    supervised = StubLlmProvider([])
    # Both responses are text — no tool_use.
    supervisor = StubLlmProvider(
        [
            model_response(text="thinking out loud", stop_reason="end_turn"),
            model_response(text="still thinking", stop_reason="end_turn"),
        ]
    )
    session = _new_session(
        supervised_llm=supervised,
        supervisor_llm=supervisor,
        workspace=tmp_path,
        artifact_store=store,
        nudge_path="nudge.txt",
    )
    session.turn_count = 3

    await maybe_run_supervisor_check(session)

    assert session.supervisor_aborted is False
    # The supervisor was called (twice — retry-once), but no decision
    # was applied; the counter stays 0 because we return before
    # incrementing.
    assert session.supervisor_checks_fired == 0
    assert store.put_calls == []
