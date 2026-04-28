""":func:`smai_agents.run_code_review` — Task 2.B4 acceptance.

Per ``04-agents.md`` §2.4 / §11 / §6 / §10 and
DEC-016 / DEC-017 / DEC-018. The wrapper is exercised with the
:class:`StubLlmProvider` from ``_agent_fakes`` (queue-driven canned
responses); the user-message shape is asserted via the ``calls[]``
snapshot the stub records.

After the 2.B2/2.B4 reconciliation, the wrapper resolves prompt config
via :func:`smai_agents.prompts.load_prompt_config` (loader-driven YAML
substrate per §10) when ``prompt_config`` is ``None``.
"""

from __future__ import annotations

import pytest
from _agent_fakes import StubLlmProvider  # type: ignore[import-not-found]
from _agent_helpers import model_response  # type: ignore[import-not-found]
from _role_fakes import (  # type: ignore[import-not-found]
    make_harness_contract,
    make_technique_contract,
)
from smai_agents import (
    CodeReviewerInput,
    EntryUnderReview,
    PromptConfig,
    StructuredCallFailed,
    StructuredOutputTool,
    clear_prompt_config_cache,
    run_code_review,
)
from smai_agents.agents import code_reviewer as _code_reviewer_module


@pytest.fixture(autouse=True)
def _clear_prompt_config_cache() -> None:
    """Each test starts with a clean process-local prompt-config cache."""
    clear_prompt_config_cache()


def _baseline_entry_additive() -> EntryUnderReview:
    return EntryUnderReview(
        entry_id="entry-baseline",
        technique_id=None,
        technique_name="no-augmentation baseline",
        technique_description="harness no-op default; no technique module",
        is_baseline=True,
        technique_contract=make_technique_contract(
            entry_id="entry-baseline",
            technique_id=None,
            is_baseline=True,
        ),
        code=None,
    )


def _baseline_entry_substitutive() -> EntryUnderReview:
    return EntryUnderReview(
        entry_id="entry-baseline",
        technique_id="t_vgg",
        technique_name="VGG (reference)",
        technique_description="reference architecture for the substitutive CG",
        is_baseline=True,
        technique_contract=make_technique_contract(
            entry_id="entry-baseline",
            technique_id="t_vgg",
            is_baseline=True,
        ),
        code="def apply(config):\n    return {'model': vgg.VGG16()}\n",
    )


def _treatment_entry(*, entry_id: str, technique_id: str, name: str, code: str) -> EntryUnderReview:
    return EntryUnderReview(
        entry_id=entry_id,
        technique_id=technique_id,
        technique_name=name,
        technique_description=f"{name} description",
        is_baseline=False,
        technique_contract=make_technique_contract(
            entry_id=entry_id,
            technique_id=technique_id,
            is_baseline=False,
        ),
        code=code,
    )


def _additive_input() -> CodeReviewerInput:
    return CodeReviewerInput(
        cg_id="cg-1",
        factor_type="additive",
        factor_dimension="augmentation",
        harness_contract=make_harness_contract(factor_type="additive"),
        harness_code={"harness/train.py": "# train loop\n"},
        entries=[
            _baseline_entry_additive(),
            _treatment_entry(
                entry_id="entry-cutout",
                technique_id="t_cutout",
                name="Cutout",
                code="def apply(config):\n    return {'image_transform': cutout(16)}\n",
            ),
        ],
    )


def _substitutive_input() -> CodeReviewerInput:
    return CodeReviewerInput(
        cg_id="cg-2",
        factor_type="substitutive",
        factor_dimension="architecture",
        harness_contract=make_harness_contract(
            factor_type="substitutive", factor_name="architecture"
        ),
        harness_code={"harness/train.py": "# train loop\n"},
        entries=[
            _baseline_entry_substitutive(),
            _treatment_entry(
                entry_id="entry-resnet",
                technique_id="t_resnet",
                name="ResNet-50",
                code="def apply(config):\n    return {'model': torchvision.models.resnet50()}\n",
            ),
        ],
    )


@pytest.mark.asyncio
async def test_code_review_happy_path_returns_pass() -> None:
    """Tool-use response with valid payload → typed :class:`CodeReviewResult`."""
    canned = model_response(
        tool_uses=[
            (
                "tu-1",
                "submit_review",
                {
                    "findings": [
                        {
                            "severity": "info",
                            "target_id": "entry-cutout",
                            "target_kind": "entry",
                            "summary": "minor doc nit",
                            "detail": "consider adding a docstring",
                        }
                    ],
                    "overall_pass": True,
                },
            )
        ],
        stop_reason="tool_use",
    )
    llm = StubLlmProvider([canned])
    result = await run_code_review(llm=llm, input=_additive_input())

    assert result.overall_pass is True
    assert len(result.findings) == 1
    assert result.findings[0].severity == "info"
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_code_review_tool_name_matches_v1_convention() -> None:
    """§6 / DEC-018: the tool name is ``submit_review`` per the v1 mapping table."""
    canned = model_response(
        tool_uses=[("tu-2", "submit_review", {"findings": [], "overall_pass": True})],
        stop_reason="tool_use",
    )
    llm = StubLlmProvider([canned])
    await run_code_review(llm=llm, input=_additive_input())

    tools = llm.calls[0]["tools"]
    assert isinstance(tools, list)
    assert tools[0]["name"] == "submit_review"
    # The tool's input schema is the CodeReviewResult Pydantic schema.
    schema = tools[0]["input_schema"]
    assert isinstance(schema, dict)
    assert "findings" in schema["properties"]
    assert "overall_pass" in schema["properties"]


@pytest.mark.asyncio
async def test_code_review_additive_baseline_skips_code_review() -> None:
    """DEC-013 / DEC-017: ``additive`` baselines have no code; the user message
    explicitly flags the omission so the LLM does not hallucinate findings."""
    canned = model_response(
        tool_uses=[("tu-3", "submit_review", {"findings": [], "overall_pass": True})],
        stop_reason="tool_use",
    )
    llm = StubLlmProvider([canned])
    await run_code_review(llm=llm, input=_additive_input())

    user_text = _last_user_text(llm)
    # The baseline section explicitly says "No code to review" rather than
    # being silently omitted (the LLM needs the accounting).
    assert "No code to review for this entry" in user_text
    assert "additive" in user_text
    assert "Cutout" in user_text  # the treatment IS reviewed
    assert "def apply(config):" in user_text  # treatment code is included


@pytest.mark.asyncio
async def test_code_review_substitutive_baseline_includes_code() -> None:
    """DEC-017: ``substitutive`` baselines have real technique modules and are
    reviewed normally — no special skip framing."""
    canned = model_response(
        tool_uses=[("tu-4", "submit_review", {"findings": [], "overall_pass": True})],
        stop_reason="tool_use",
    )
    llm = StubLlmProvider([canned])
    await run_code_review(llm=llm, input=_substitutive_input())

    user_text = _last_user_text(llm)
    # No skip-baseline framing; the baseline's code IS shown.
    assert "vgg.VGG16()" in user_text
    assert "VGG (reference) [BASELINE]" in user_text
    assert "No code to review for this entry" not in user_text
    assert "substitutive" in user_text


@pytest.mark.asyncio
async def test_code_review_retries_once_on_text_response() -> None:
    """§6 step 2 / DEC-018: text response triggers a single retry."""
    text_only = model_response(text="passes lgtm", stop_reason="end_turn")
    valid = model_response(
        tool_uses=[
            (
                "tu-5",
                "submit_review",
                {
                    "findings": [
                        {
                            "severity": "critical",
                            "target_id": "entry-cutout",
                            "target_kind": "entry",
                            "summary": "broken",
                            "detail": "wrong dim",
                        }
                    ],
                    "overall_pass": False,
                },
            )
        ],
        stop_reason="tool_use",
    )
    llm = StubLlmProvider([text_only, valid])

    result = await run_code_review(llm=llm, input=_additive_input())
    assert result.overall_pass is False
    assert len(llm.calls) == 2
    # The retry instruction names the tool the model should invoke.
    second_messages = llm.calls[1]["messages"]
    assert isinstance(second_messages, list)
    last = second_messages[-1]
    assert last["role"] == "user"
    last_content = last["content"]
    assert isinstance(last_content, list)
    assert "submit_review" in last_content[0]["text"]


@pytest.mark.asyncio
async def test_code_review_raises_after_two_failed_attempts() -> None:
    """§6 step 3 / DEC-018: no silent fallback — both-attempts-fail is loud."""
    a = model_response(text="not a tool call", stop_reason="end_turn")
    b = model_response(text="still not", stop_reason="end_turn")
    llm = StubLlmProvider([a, b])

    with pytest.raises(StructuredCallFailed) as exc_info:
        await run_code_review(llm=llm, input=_additive_input())
    assert exc_info.value.tool_name == "submit_review"


@pytest.mark.asyncio
async def test_code_review_uses_single_call_cache_defaults() -> None:
    """§5 final paragraph: rolling=0 for single-call agents (delegated to
    :func:`structured_call`'s default)."""
    canned = model_response(
        tool_uses=[("tu-6", "submit_review", {"findings": [], "overall_pass": True})],
        stop_reason="tool_use",
    )
    llm = StubLlmProvider([canned])
    await run_code_review(llm=llm, input=_additive_input())

    cache_config = llm.calls[0]["cache_config"]
    assert isinstance(cache_config, dict)
    assert cache_config["rolling_cache_count"] == 0


@pytest.mark.asyncio
async def test_code_review_calls_loader_when_prompt_config_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§10 / 2.B2-2.B4 reconciliation: ``prompt_config=None`` resolves
    config via :func:`smai_agents.prompts.load_prompt_config` keyed on
    ``role='code_reviewer'`` (no variant — the code reviewer's
    factor-type framing lives in the user message per DEC-017)."""
    captured: dict[str, object] = {}

    def _fake_loader(*, role: str, variant_name: str | None = None) -> PromptConfig:
        captured["role"] = role
        captured["variant_name"] = variant_name
        return PromptConfig(
            role="code_reviewer",
            system_prompt="STUB SYSTEM PROMPT",
            initial_user_message_template="(unused at single-call site)",
            tools=[],
            structured_output_tool=StructuredOutputTool(
                name="submit_review",
                description="Submit the review.",
                schema_module=("smai_agents.schemas.code_review:CodeReviewResult"),
            ),
            layer_chain=["code_reviewer/base", "code_reviewer/stub"],
        )

    monkeypatch.setattr(_code_reviewer_module, "load_prompt_config", _fake_loader)

    canned = model_response(
        tool_uses=[("tu-loader", "submit_review", {"findings": [], "overall_pass": True})],
        stop_reason="tool_use",
    )
    llm = StubLlmProvider([canned])
    await run_code_review(llm=llm, input=_additive_input())

    assert captured == {"role": "code_reviewer", "variant_name": None}
    assert llm.calls[0]["system"] == "STUB SYSTEM PROMPT"


@pytest.mark.asyncio
async def test_code_review_skips_loader_when_prompt_config_supplied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the caller supplies a :class:`PromptConfig`, the loader
    must NOT be called — the caller's config is used verbatim."""

    def _exploding_loader(*, role: str, variant_name: str | None = None) -> PromptConfig:
        raise AssertionError("load_prompt_config must not be called when prompt_config is supplied")

    monkeypatch.setattr(_code_reviewer_module, "load_prompt_config", _exploding_loader)

    cfg = PromptConfig(
        role="code_reviewer",
        system_prompt="CALLER-PROVIDED SYSTEM PROMPT",
        initial_user_message_template="(unused at single-call site)",
        tools=[],
        structured_output_tool=StructuredOutputTool(
            name="submit_review",
            description="caller's tool description",
            schema_module="smai_agents.schemas.code_review:CodeReviewResult",
        ),
        layer_chain=["test/inline"],
    )

    canned = model_response(
        tool_uses=[("tu-supplied", "submit_review", {"findings": [], "overall_pass": True})],
        stop_reason="tool_use",
    )
    llm = StubLlmProvider([canned])
    await run_code_review(llm=llm, input=_additive_input(), prompt_config=cfg)

    assert llm.calls[0]["system"] == "CALLER-PROVIDED SYSTEM PROMPT"
    tools = llm.calls[0]["tools"]
    assert isinstance(tools, list)
    assert tools[0]["name"] == "submit_review"
    assert tools[0]["description"] == "caller's tool description"


def _last_user_text(llm: StubLlmProvider) -> str:
    """Pull out the user-message text from the most recent call."""
    msgs = llm.calls[-1]["messages"]
    assert isinstance(msgs, list)
    user = msgs[0]
    assert user["role"] == "user"
    content = user["content"]
    assert isinstance(content, list)
    return content[0]["text"]
