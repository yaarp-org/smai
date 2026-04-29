""":func:`smai_agents.run_supervisor_check` — Task 3.G4 acceptance.

Per ``04-agents.md`` §2.6 / §15 OQ2 / §6 / §10 and DEC-018. The wrapper
is exercised with the :class:`StubLlmProvider` from ``_agent_fakes``;
the user-message shape is asserted via the ``calls[]`` snapshot the
stub records. Mirrors the structure of ``test_code_reviewer.py``.
"""

from __future__ import annotations

import pytest
from _agent_fakes import StubLlmProvider  # type: ignore[import-not-found]
from _g4_fakes import canned_supervisor_decision  # type: ignore[import-not-found]
from smai_agents import (
    PromptConfig,
    StructuredCallFailed,
    StructuredOutputTool,
    SupervisorDecision,
    SupervisorInput,
    clear_prompt_config_cache,
    run_supervisor_check,
)
from smai_agents.agents import supervisor as _supervisor_module


@pytest.fixture(autouse=True)
def _clear_prompt_config_cache() -> None:
    """Each test starts with a clean process-local prompt-config cache."""
    clear_prompt_config_cache()


def _input(*, role: str = "harness_builder") -> SupervisorInput:
    return SupervisorInput(
        agent_role=role,  # type: ignore[arg-type]
        entity_id="cg-1",
        current_turn=5,
        turn_budget=50,
        recent_status_snapshots=[
            {"turn_count": 3, "usage_total": {"output_tokens": 100}},
            {"turn_count": 4, "usage_total": {"output_tokens": 100}},
            {"turn_count": 5, "usage_total": {"output_tokens": 100}},
        ],
        recent_tool_calls=["read_file", "read_file", "read_file"],
        signal_kind="periodic",
    )


@pytest.mark.asyncio
async def test_supervisor_returns_continue_decision() -> None:
    """Tool-use response with ``action='continue'`` round-trips."""
    canned = canned_supervisor_decision(
        action="continue", reason="the agent is making steady progress"
    )
    llm = StubLlmProvider([canned])
    result = await run_supervisor_check(llm=llm, input=_input())

    assert isinstance(result, SupervisorDecision)
    assert result.action == "continue"
    assert result.reason == "the agent is making steady progress"
    assert result.nudge is None
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_supervisor_returns_intervene_decision_with_nudge() -> None:
    """``action='intervene'`` requires nudge — round-trip + invariant check."""
    canned = canned_supervisor_decision(
        action="intervene",
        reason="agent has been re-reading the same file for three turns",
        nudge=(
            "Stop re-reading harness/train.py; you already have it cached. "
            "Move on to writing the technique module."
        ),
    )
    llm = StubLlmProvider([canned])
    result = await run_supervisor_check(llm=llm, input=_input())

    assert result.action == "intervene"
    assert "Stop re-reading" in (result.nudge or "")
    assert result.reason.startswith("agent has been re-reading")


@pytest.mark.asyncio
async def test_supervisor_returns_abort_decision() -> None:
    """``action='abort'`` carries reason; nudge stays None."""
    canned = canned_supervisor_decision(
        action="abort",
        reason="agent has been stuck without progress for 25 turns; aborting",
    )
    llm = StubLlmProvider([canned])
    result = await run_supervisor_check(llm=llm, input=_input())

    assert result.action == "abort"
    assert "stuck without progress" in result.reason
    assert result.nudge is None


@pytest.mark.asyncio
async def test_supervisor_tool_name_matches_doc_convention() -> None:
    """§6 / §2.6: the tool name is ``submit_decision``."""
    canned = canned_supervisor_decision(action="continue", reason="ok")
    llm = StubLlmProvider([canned])
    await run_supervisor_check(llm=llm, input=_input())

    tools = llm.calls[0]["tools"]
    assert isinstance(tools, list)
    assert tools[0]["name"] == "submit_decision"
    schema = tools[0]["input_schema"]
    assert isinstance(schema, dict)
    assert "action" in schema["properties"]
    assert "reason" in schema["properties"]
    assert "nudge" in schema["properties"]


@pytest.mark.asyncio
async def test_supervisor_user_message_carries_signal_kind_and_role() -> None:
    """The rendered user message frames the role + recent activity for the LLM."""
    canned = canned_supervisor_decision(action="continue", reason="ok")
    llm = StubLlmProvider([canned])
    await run_supervisor_check(llm=llm, input=_input(role="technique_implementer"))

    user_text = _last_user_text(llm)
    assert "technique_implementer" in user_text
    assert "cg-1" in user_text
    assert "periodic" in user_text
    # Recent tool-call list is rendered.
    assert "read_file" in user_text
    # Recent snapshot detail makes it into the message body.
    assert "turn_count" in user_text


@pytest.mark.asyncio
async def test_supervisor_invariant_intervene_requires_nudge() -> None:
    """SupervisorDecision validator: ``intervene`` without nudge raises."""
    with pytest.raises(ValueError, match="non-empty nudge"):
        SupervisorDecision(action="intervene", reason="x", nudge=None)


@pytest.mark.asyncio
async def test_supervisor_invariant_non_intervene_forbids_nudge() -> None:
    """SupervisorDecision validator: ``continue`` / ``abort`` cannot carry a nudge."""
    with pytest.raises(ValueError, match="must leave nudge=None"):
        SupervisorDecision(action="continue", reason="ok", nudge="hint")
    with pytest.raises(ValueError, match="must leave nudge=None"):
        SupervisorDecision(action="abort", reason="bad", nudge="hint")


@pytest.mark.asyncio
async def test_supervisor_retries_once_on_text_response() -> None:
    """§6 step 2 / DEC-018: text response triggers a single retry."""
    from _agent_helpers import model_response  # type: ignore[import-not-found]  # noqa: PLC0415

    text_only = model_response(text="continue", stop_reason="end_turn")
    valid = canned_supervisor_decision(action="continue", reason="ok")
    llm = StubLlmProvider([text_only, valid])

    result = await run_supervisor_check(llm=llm, input=_input())
    assert result.action == "continue"
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_supervisor_raises_after_two_failed_attempts() -> None:
    """§6 step 3 / DEC-018: no silent fallback."""
    from _agent_helpers import model_response  # type: ignore[import-not-found]  # noqa: PLC0415

    a = model_response(text="not a tool call", stop_reason="end_turn")
    b = model_response(text="still not", stop_reason="end_turn")
    llm = StubLlmProvider([a, b])

    with pytest.raises(StructuredCallFailed) as exc_info:
        await run_supervisor_check(llm=llm, input=_input())
    assert exc_info.value.tool_name == "submit_decision"


@pytest.mark.asyncio
async def test_supervisor_calls_loader_when_prompt_config_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``prompt_config=None`` resolves config via :func:`load_prompt_config`
    keyed on ``role='supervisor'`` (no variant — the doc doesn't
    differentiate supervisor variants per §2.6)."""
    captured: dict[str, object] = {}

    def _fake_loader(*, role: str, variant_name: str | None = None) -> PromptConfig:
        captured["role"] = role
        captured["variant_name"] = variant_name
        return PromptConfig(
            role="supervisor",
            system_prompt="STUB SUPERVISOR SYSTEM PROMPT",
            initial_user_message_template="(unused at single-call site)",
            tools=[],
            structured_output_tool=StructuredOutputTool(
                name="submit_decision",
                description="Submit the supervisor decision.",
                schema_module="smai_agents.schemas.supervisor:SupervisorDecision",
            ),
            layer_chain=["supervisor/base", "supervisor/stub"],
        )

    monkeypatch.setattr(_supervisor_module, "load_prompt_config", _fake_loader)

    canned = canned_supervisor_decision(action="continue", reason="ok")
    llm = StubLlmProvider([canned])
    await run_supervisor_check(llm=llm, input=_input())

    assert captured == {"role": "supervisor", "variant_name": None}
    assert llm.calls[0]["system"] == "STUB SUPERVISOR SYSTEM PROMPT"


@pytest.mark.asyncio
async def test_supervisor_loads_shipped_yaml_base() -> None:
    """The shipped ``supervisor/base.yaml`` round-trips through the loader."""
    canned = canned_supervisor_decision(action="continue", reason="ok")
    llm = StubLlmProvider([canned])
    await run_supervisor_check(llm=llm, input=_input())

    # The shipped base prompt mentions the three actions.
    system = llm.calls[0]["system"]
    assert isinstance(system, str)
    assert "continue" in system
    assert "intervene" in system
    assert "abort" in system


def _last_user_text(llm: StubLlmProvider) -> str:
    msgs = llm.calls[-1]["messages"]
    assert isinstance(msgs, list)
    user = msgs[0]
    assert user["role"] == "user"
    content = user["content"]
    assert isinstance(content, list)
    return content[0]["text"]
