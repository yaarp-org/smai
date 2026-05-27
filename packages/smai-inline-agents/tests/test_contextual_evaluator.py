""":func:`smai_inline_agents.run_contextual_evaluation` — Task 2.B4 acceptance.

Per ``04-agents.md`` §2.5 / §6 / §10 and DEC-031 / DEC-034 #2 / DEC-018.
The wrapper is exercised with the :class:`StubLlmProvider` from
``_agent_fakes`` (queue-driven canned responses); the user-message
shape is asserted via the ``calls[]`` snapshot the stub records.

Per §2.5 the schema is rule-agnostic; only the prompt-config variant
differs between ``compare_to_baseline`` and ``compare_to_target`` CGs.
After the 2.B2/2.B4 reconciliation, the wrapper resolves prompt config
via :func:`smai_inline_agents.prompts.load_prompt_config` (loader-driven YAML
substrate per §10) when ``prompt_config`` is ``None``; we verify
variant selection by inspecting the system prompt the wrapper forwards
to :func:`structured_call`.
"""

from __future__ import annotations

import pytest
from _agent_fakes import StubLlmProvider  # type: ignore[import-not-found]
from _agent_helpers import model_response  # type: ignore[import-not-found]
from _role_fakes import make_evaluation_result  # type: ignore[import-not-found]
from smai_inline_agents import (
    CGMetadata,
    ContextualEvaluatorEntry,
    ContextualEvaluatorInput,
    PromptConfig,
    StructuredCallFailed,
    StructuredOutputTool,
    clear_prompt_config_cache,
    run_contextual_evaluation,
)
from smai_inline_agents.agents import contextual_evaluator as _contextual_evaluator_module


@pytest.fixture(autouse=True)
def _clear_prompt_config_cache() -> None:
    """Each test starts with a clean process-local prompt-config cache."""
    clear_prompt_config_cache()


def _cg_metadata() -> CGMetadata:
    return CGMetadata(
        cg_id="cg-1",
        cg_name="cutout-on-cifar10",
        cg_description="does Cutout improve CIFAR-10 accuracy?",
        factor_dimension="augmentation",
        factor_type="additive",
        controlled_conditions=[
            "dataset: CIFAR-10",
            "optimizer: SGD lr=0.1",
            "epochs: 200",
        ],
        entries=[
            ContextualEvaluatorEntry(
                entry_id="b",
                technique_id=None,
                technique_name="no-augmentation baseline",
                technique_description="harness no-op default",
                is_baseline=True,
            ),
            ContextualEvaluatorEntry(
                entry_id="t1",
                technique_id="t_cutout",
                technique_name="Cutout",
                technique_description="random 16x16 patch occlusion",
                is_baseline=False,
            ),
        ],
    )


def _baseline_input() -> ContextualEvaluatorInput:
    return ContextualEvaluatorInput(
        cg_metadata=_cg_metadata(),
        evaluation_result=make_evaluation_result(
            result="pass", treatment_entry_ids=("t1",), baseline_entry_id="b"
        ),
        comparison_rule="compare_to_baseline",
        comparison_rationale=(
            "no-aug baseline isolates the augmentation effect; expected delta is "
            "+1.5pp from the Cutout paper"
        ),
    )


def _target_input() -> ContextualEvaluatorInput:
    return ContextualEvaluatorInput(
        cg_metadata=_cg_metadata(),
        evaluation_result=make_evaluation_result(
            result="pass", treatment_entry_ids=("t1",), baseline_entry_id="b"
        ),
        comparison_rule="compare_to_target",
        comparison_rationale=None,
    )


def _canned_verdict_response(tu_id: str) -> object:
    return model_response(
        tool_uses=[
            (
                tu_id,
                "submit_evaluation",
                {
                    "overall_verdict": "promising",
                    "summary": "Cutout beat baseline by ~5pp consistently across seeds.",
                    "rankings": [
                        {"entry_id": "t1", "rank": 1, "rationale": "highest accuracy"},
                        {"entry_id": "b", "rank": 2, "rationale": "baseline"},
                    ],
                    "insights": ["all three seeds agreed in direction"],
                    "limitations": ["only three seeds"],
                    "suggested_follow_ups": ["replicate on CIFAR-100"],
                },
            )
        ],
        stop_reason="tool_use",
    )


@pytest.mark.asyncio
async def test_contextual_evaluation_happy_path_returns_typed_verdict() -> None:
    """Tool-use response with valid payload → typed :class:`ContextualVerdict`."""
    llm = StubLlmProvider([_canned_verdict_response("tu-1")])  # type: ignore[list-item]
    result = await run_contextual_evaluation(llm=llm, input=_baseline_input())

    assert result.overall_verdict == "promising"
    assert len(result.rankings) == 2
    assert result.rankings[0].entry_id == "t1"
    assert "Cutout" in result.summary
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_contextual_evaluation_tool_name_matches_v1_convention() -> None:
    """§6 / DEC-018: the tool name is ``submit_evaluation`` per the v1 mapping."""
    llm = StubLlmProvider([_canned_verdict_response("tu-2")])  # type: ignore[list-item]
    await run_contextual_evaluation(llm=llm, input=_baseline_input())

    tools = llm.calls[0]["tools"]
    assert isinstance(tools, list)
    assert tools[0]["name"] == "submit_evaluation"
    schema = tools[0]["input_schema"]
    assert isinstance(schema, dict)
    assert "overall_verdict" in schema["properties"]
    assert "rankings" in schema["properties"]


@pytest.mark.asyncio
async def test_contextual_evaluation_baseline_variant_uses_baseline_framing() -> None:
    """§2.5 / DEC-031 #8 / §10: ``compare_to_baseline`` selects the
    treatment-vs-baseline narrative variant.

    The system prompt is sourced from
    ``prompts/contextual_evaluator/variants/compare_to_baseline.yaml``
    via :func:`load_prompt_config`; we assert against the variant's
    framing line, not the (orthogonal) target-variant framing.
    """
    llm = StubLlmProvider([_canned_verdict_response("tu-3")])  # type: ignore[list-item]
    await run_contextual_evaluation(llm=llm, input=_baseline_input())

    system_prompt = llm.calls[0]["system"]
    assert isinstance(system_prompt, str)
    # Baseline framing line is present; target framing line is not.
    assert "treatment vs no-treatment" in system_prompt
    assert "proposed-technique vs reference-technique" not in system_prompt


@pytest.mark.asyncio
async def test_contextual_evaluation_target_variant_uses_target_framing() -> None:
    """§2.5 / DEC-031 #8 / §10: ``compare_to_target`` selects the
    treatment-vs-target narrative variant."""
    llm = StubLlmProvider([_canned_verdict_response("tu-4")])  # type: ignore[list-item]
    await run_contextual_evaluation(llm=llm, input=_target_input())

    system_prompt = llm.calls[0]["system"]
    assert isinstance(system_prompt, str)
    assert "proposed-technique vs reference-technique" in system_prompt
    assert "treatment vs no-treatment" not in system_prompt


@pytest.mark.asyncio
async def test_contextual_evaluation_user_message_includes_verdict_context() -> None:
    """Per §2.5: the contextual evaluator consumes the deterministic
    ``verdict_context`` (anomalies, statistics, deltas). The user message
    should embed the JSON so the LLM has the grounding."""
    llm = StubLlmProvider([_canned_verdict_response("tu-5")])  # type: ignore[list-item]
    await run_contextual_evaluation(llm=llm, input=_baseline_input())

    user_text = _last_user_text(llm)
    assert "Mechanical Verdict (locked" in user_text
    assert "Verdict Context" in user_text
    assert "treatment_outcomes" in user_text
    assert "delta_summaries" in user_text
    # Comparison rationale (when present) lands in the message body.
    assert "no-aug baseline isolates the augmentation effect" in user_text


@pytest.mark.asyncio
async def test_contextual_evaluation_omits_rationale_when_absent() -> None:
    """``comparison_rationale`` is optional; when ``None`` the message body
    should not carry an empty rationale section."""
    llm = StubLlmProvider([_canned_verdict_response("tu-6")])  # type: ignore[list-item]
    await run_contextual_evaluation(llm=llm, input=_target_input())

    user_text = _last_user_text(llm)
    assert "Comparison Rationale" not in user_text


@pytest.mark.asyncio
async def test_contextual_evaluation_retries_once_on_text_response() -> None:
    """§6 step 2 / DEC-018: text response triggers a single retry."""
    text_only = model_response(
        text="overall it looks promising, here are some thoughts",
        stop_reason="end_turn",
    )
    valid = _canned_verdict_response("tu-7")
    llm = StubLlmProvider([text_only, valid])  # type: ignore[list-item]
    result = await run_contextual_evaluation(llm=llm, input=_baseline_input())

    assert result.overall_verdict == "promising"
    assert len(llm.calls) == 2
    second = llm.calls[1]["messages"]
    assert isinstance(second, list)
    last = second[-1]
    assert last["role"] == "user"
    last_content = last["content"]
    assert isinstance(last_content, list)
    assert "submit_evaluation" in last_content[0]["text"]


@pytest.mark.asyncio
async def test_contextual_evaluation_raises_after_two_failed_attempts() -> None:
    """§6 step 3 / DEC-018: second-attempt failure is loud, not silent."""
    a = model_response(text="not a tool call", stop_reason="end_turn")
    b = model_response(text="still not", stop_reason="end_turn")
    llm = StubLlmProvider([a, b])

    with pytest.raises(StructuredCallFailed) as exc_info:
        await run_contextual_evaluation(llm=llm, input=_baseline_input())
    assert exc_info.value.tool_name == "submit_evaluation"


@pytest.mark.asyncio
async def test_contextual_evaluation_calls_loader_when_prompt_config_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§10 / 2.B2-2.B4 reconciliation: ``prompt_config=None`` resolves
    config via :func:`smai_inline_agents.prompts.load_prompt_config` keyed on
    ``role='contextual_evaluator'`` and ``variant_name=comparison_rule``
    (per §2.5 / DEC-031 #8)."""
    captured: dict[str, object] = {}

    def _fake_loader(*, role: str, variant_name: str | None = None) -> PromptConfig:
        captured["role"] = role
        captured["variant_name"] = variant_name
        return PromptConfig(
            role="contextual_evaluator",
            system_prompt="STUB SYSTEM PROMPT",
            initial_user_message_template="(unused at single-call site)",
            tools=[],
            structured_output_tool=StructuredOutputTool(
                name="submit_evaluation",
                description="Submit the evaluation.",
                schema_module=("smai_inline_agents.schemas.contextual_verdict:ContextualVerdict"),
            ),
            layer_chain=["contextual_evaluator/base", "contextual_evaluator/stub"],
        )

    monkeypatch.setattr(_contextual_evaluator_module, "load_prompt_config", _fake_loader)

    llm = StubLlmProvider([_canned_verdict_response("tu-loader")])  # type: ignore[list-item]
    await run_contextual_evaluation(llm=llm, input=_baseline_input())

    assert captured == {
        "role": "contextual_evaluator",
        "variant_name": "compare_to_baseline",
    }
    assert llm.calls[0]["system"] == "STUB SYSTEM PROMPT"


@pytest.mark.asyncio
async def test_contextual_evaluation_skips_loader_when_prompt_config_supplied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the caller supplies a :class:`PromptConfig`, the loader
    must NOT be called — the caller's config is used verbatim."""

    def _exploding_loader(*, role: str, variant_name: str | None = None) -> PromptConfig:
        raise AssertionError("load_prompt_config must not be called when prompt_config is supplied")

    monkeypatch.setattr(_contextual_evaluator_module, "load_prompt_config", _exploding_loader)

    cfg = PromptConfig(
        role="contextual_evaluator",
        system_prompt="CALLER-PROVIDED SYSTEM PROMPT",
        initial_user_message_template="(unused at single-call site)",
        tools=[],
        structured_output_tool=StructuredOutputTool(
            name="submit_evaluation",
            description="caller's tool description",
            schema_module=("smai_inline_agents.schemas.contextual_verdict:ContextualVerdict"),
        ),
        layer_chain=["test/inline"],
    )

    llm = StubLlmProvider([_canned_verdict_response("tu-supplied")])  # type: ignore[list-item]
    await run_contextual_evaluation(llm=llm, input=_baseline_input(), prompt_config=cfg)

    assert llm.calls[0]["system"] == "CALLER-PROVIDED SYSTEM PROMPT"
    tools = llm.calls[0]["tools"]
    assert isinstance(tools, list)
    assert tools[0]["name"] == "submit_evaluation"
    assert tools[0]["description"] == "caller's tool description"


def _last_user_text(llm: StubLlmProvider) -> str:
    msgs = llm.calls[-1]["messages"]
    assert isinstance(msgs, list)
    user = msgs[0]
    assert user["role"] == "user"
    content = user["content"]
    assert isinstance(content, list)
    return content[0]["text"]
