"""End-to-end loop integration for the supervisor (Task 3.G4 acceptance).

Drives :func:`smai_agents.run_loop` against a stuck-agent stub LLM
whose tool calls repeat verbatim turn after turn. The loop's
between-turn supervisor hook fires; the supervisor's stub returns
``intervene`` with a nudge; the nudge file lands in the artifact
store; on the next turn the loop's existing
:func:`maybe_check_supervisor_nudge` consumes the nudge and injects
it as a ``[supervisor nudge] ...`` user message — visible to the
agent via :class:`StubLlmProvider.calls`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _agent_fakes import StubArtifactStore, StubLlmProvider  # type: ignore[import-not-found]
from _agent_helpers import model_response  # type: ignore[import-not-found]
from _g4_fakes import (  # type: ignore[import-not-found]
    canned_supervisor_decision,
    stuck_agent_loop_response,
)
from smai_agents import (
    SUPERVISOR_NUDGE_PREFIX,
    AgentLoopConfig,
    AgentSession,
    ToolRegistry,
    make_finish_tool,
    make_read_file_tool,
    run_loop,
)


def _build_session(
    *,
    supervised: StubLlmProvider,
    supervisor: StubLlmProvider | None,
    workspace: Path,
    artifact_store: StubArtifactStore,
    config: AgentLoopConfig,
    nudge_path: str = "comparison-groups/cg-1/harness/nudge.txt",
    status_path: str = "comparison-groups/cg-1/harness/status.json",
) -> AgentSession:
    registry = ToolRegistry()
    registry.register(make_finish_tool())
    registry.register(make_read_file_tool())
    providers: dict[str, object] = {"harness_builder": supervised}
    if supervisor is not None:
        providers["supervisor"] = supervisor
    return AgentSession(
        system_prompt="sys",
        tools=registry,
        llm_providers=providers,  # type: ignore[arg-type]
        current_role="harness_builder",
        workspace_path=workspace,
        artifact_store=artifact_store,
        nudge_artifact_path=nudge_path,
        status_artifact_path=status_path,
        config=config,
    )


# === Stuck-agent integration fixture (Task 3.G4 acceptance) =================


@pytest.mark.asyncio
async def test_stuck_agent_supervisor_intervene_round_trip(tmp_path: Path) -> None:
    """The acceptance round-trip from §3.4 Task 3.G4:

    A mock LLM that keeps producing the same status write triggers
    the supervisor; the supervisor's `intervene` decision writes a
    nudge file the agent loop consumes on the next turn.

    Flow:
    1. The supervised agent's LLM queue replays
       :func:`stuck_agent_loop_response` three times (read_file
       ``harness/train.py`` over and over) and then a finish call.
    2. Loop config: ``supervisor_check_every_turns=2`` — the hook
       fires after the second turn.
    3. The supervisor's LLM returns ``intervene`` with a nudge.
    4. The nudge lands in :class:`ArtifactStore` at ``nudge_path``.
    5. On the loop's next pre-turn cycle,
       :func:`maybe_check_supervisor_nudge` reads the file and injects
       a ``[supervisor nudge] ...`` user message.
    6. The supervised LLM's *third* call sees the nudge in the
       message history.
    """
    # Materialize a real file the read_file tool can target.
    workspace = tmp_path
    target = workspace / "harness" / "train.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# placeholder\n")

    store = StubArtifactStore()
    supervised = StubLlmProvider(
        [
            stuck_agent_loop_response(),
            stuck_agent_loop_response(),
            stuck_agent_loop_response(),
            # Finally finish — keeps the loop terminating cleanly.
            model_response(
                tool_uses=[
                    (
                        "finish-1",
                        "finish",
                        {"summary": "done", "success": True},
                    )
                ],
                stop_reason="tool_use",
            ),
        ]
    )
    supervisor = StubLlmProvider(
        [
            canned_supervisor_decision(
                action="intervene",
                reason=("agent has read harness/train.py three times in a row without writing"),
                nudge=(
                    "You already have harness/train.py in context. Write the technique module now."
                ),
            ),
        ]
    )

    session = _build_session(
        supervised=supervised,
        supervisor=supervisor,
        workspace=workspace,
        artifact_store=store,
        # Run supervisor every 2 turns; loop's nudge-poll is every 1 turn.
        config=AgentLoopConfig(
            supervisor_check_every_turns=2,
            supervisor_nudge_check_every_turns=1,
            status_write_every_turns=1,
            max_turns=10,
        ),
    )

    outcome = await run_loop(session)

    # The loop terminated cleanly via finish (the supervisor said
    # intervene, not abort).
    assert outcome.kind == "finished"
    # The supervisor was called.
    assert session.supervisor_checks_fired == 1
    # The nudge file was written.
    nudge_path = "comparison-groups/cg-1/harness/nudge.txt"
    # And then consumed (deleted) by ``maybe_check_supervisor_nudge``.
    assert nudge_path in store.delete_calls
    # The supervisor wrote it; the consumer deleted it.
    assert nudge_path in {key for key, _, _ in store.put_calls}
    # ``nudges_consumed`` increments when the loop injects the nudge
    # into the conversation.
    assert session.nudges_consumed == 1
    # And the nudge content was injected with the literal prefix per
    # §3.2's parsing-convention contract.
    nudge_msgs = [
        msg
        for msg in session.messages
        if msg.role == "user"
        and any(getattr(b, "text", "").startswith(SUPERVISOR_NUDGE_PREFIX) for b in msg.content)
    ]
    assert len(nudge_msgs) == 1


# === Stop-decision fixture ==================================================


@pytest.mark.asyncio
async def test_stuck_agent_supervisor_abort_terminates_loop(tmp_path: Path) -> None:
    """``action='abort'`` exits the loop with ``kind='aborted_by_supervisor'``.

    The dispatch handler picks the kind up; routing the parent entity
    to a ``*_failed`` state per its pipeline-spec is the dispatch
    handler's territory, not the loop's. Here we assert on the
    outcome shape only.
    """
    workspace = tmp_path
    target = workspace / "harness" / "train.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# placeholder\n")

    store = StubArtifactStore()
    supervised = StubLlmProvider(
        [
            stuck_agent_loop_response(),
            stuck_agent_loop_response(),
            # Should never be called — the supervisor aborts.
            stuck_agent_loop_response(),
            stuck_agent_loop_response(),
        ]
    )
    supervisor = StubLlmProvider(
        [
            canned_supervisor_decision(
                action="abort",
                reason="agent stuck for two turns without progress",
            ),
        ]
    )

    session = _build_session(
        supervised=supervised,
        supervisor=supervisor,
        workspace=workspace,
        artifact_store=store,
        config=AgentLoopConfig(
            supervisor_check_every_turns=2,
            supervisor_nudge_check_every_turns=1,
            status_write_every_turns=1,
            max_turns=10,
        ),
    )

    outcome = await run_loop(session)

    assert outcome.kind == "aborted_by_supervisor"
    assert outcome.supervisor_reason == "agent stuck for two turns without progress"
    assert session.supervisor_aborted is True
    # The loop terminated before consuming the third stuck response.
    # ``StubLlmProvider`` raises AssertionError if the queue is exhausted,
    # so the queue still has items — verify by counting calls.
    assert len(supervised.calls) == 2  # only two turns ran before abort.


# === Disable-supervisor test ================================================


@pytest.mark.asyncio
async def test_disable_supervisor_skips_hook(tmp_path: Path) -> None:
    """``AgentLoopConfig.supervisor_check_every_turns=0`` skips the hook.

    The orchestrator's ``EngineConfig.supervisor_enabled=False``
    translates to this on the dispatch-handler side; the loop's
    happy-path is unchanged.
    """
    workspace = tmp_path
    target = workspace / "harness" / "train.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# placeholder\n")

    store = StubArtifactStore()
    supervised = StubLlmProvider(
        [
            stuck_agent_loop_response(),
            stuck_agent_loop_response(),
            model_response(
                tool_uses=[
                    (
                        "finish-1",
                        "finish",
                        {"summary": "done", "success": True},
                    )
                ],
                stop_reason="tool_use",
            ),
        ]
    )
    # Supervisor LLM with no responses — the queue would AssertionError
    # if the hook ever invoked it. The disable-supervisor test asserts
    # that the queue is never touched.
    supervisor = StubLlmProvider([])

    session = _build_session(
        supervised=supervised,
        supervisor=supervisor,
        workspace=workspace,
        artifact_store=store,
        config=AgentLoopConfig(
            supervisor_check_every_turns=0,  # disabled
            supervisor_nudge_check_every_turns=1,
            status_write_every_turns=1,
            max_turns=10,
        ),
    )

    outcome = await run_loop(session)

    assert outcome.kind == "finished"
    # Hook never fired.
    assert session.supervisor_checks_fired == 0
    # Supervisor LLM untouched (queue still empty, no calls recorded).
    assert supervisor.calls == []


# === AgentLoopConfig defaults ================================================


def test_agent_loop_config_supervisor_defaults_off() -> None:
    """The loop default is ``0`` (disabled) — ad-hoc test sessions
    that don't wire a supervisor LLM shouldn't activate the hook.
    The orchestrator's ``EngineConfig.supervisor_enabled=True``
    defaults are tested in ``smai-orchestrator``'s own suite."""
    cfg = AgentLoopConfig()
    assert cfg.supervisor_check_every_turns == 0
