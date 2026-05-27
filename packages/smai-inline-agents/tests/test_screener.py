""":func:`smai_inline_agents.run_paper_screening` — Task 3.E2 acceptance.

Per ``04-agents.md`` §6 / ``08-novel-technique-pipeline.md`` §5.2 / DEC-018.
Single-call structured-output paper screener; the wrapper is exercised
with the :class:`StubLlmProvider` from ``_agent_fakes`` (queue-driven
canned responses).

Per DEC-032 / `08` §5.2 the screener decides "is this paper a productive
source of TechniqueRefs?" — NOT "should any CG be created?" — and
returns a structured ``ScreenResult`` whose ``decision`` field drives
the paper-ingestion pipeline-spec's ``screening → planning`` /
``screening → rejected`` gates.
"""

from __future__ import annotations

import pytest
from _agent_fakes import StubLlmProvider  # type: ignore[import-not-found]
from _agent_helpers import model_response  # type: ignore[import-not-found]
from smai_inline_agents import (
    StructuredCallFailed,
    clear_prompt_config_cache,
)
from smai_inline_agents.agents.screener import (
    ScreenerInput,
    run_paper_screening,
)
from smai_inline_agents.schemas.screener import ScreenResult


@pytest.fixture(autouse=True)
def _clear_prompt_config_cache() -> None:
    clear_prompt_config_cache()


def _accept_response() -> object:
    return model_response(
        tool_uses=[
            (
                "tool-1",
                "submit_screening",
                {
                    "decision": "accept",
                    "rejection_reason": None,
                    "summary": (
                        "Paper introduces Cutout, a regularization technique "
                        "for CNN training, and benchmarks it against baselines "
                        "on CIFAR-10 / CIFAR-100 / SVHN."
                    ),
                },
            )
        ],
        stop_reason="tool_use",
    )


def _reject_response(reason: str = "pure theory paper, no empirical comparison") -> object:
    return model_response(
        tool_uses=[
            (
                "tool-1",
                "submit_screening",
                {
                    "decision": "reject",
                    "rejection_reason": reason,
                    "summary": (
                        "Paper is a theoretical analysis with no comparative "
                        "benchmarks, no implementation details."
                    ),
                },
            )
        ],
        stop_reason="tool_use",
    )


@pytest.mark.asyncio
async def test_run_paper_screening_accept() -> None:
    """Accept verdict round-trips through the wrapper."""
    llm = StubLlmProvider([_accept_response()])  # type: ignore[list-item]
    inp = ScreenerInput(
        arxiv_id="1708.04552",
        title="Improved Regularization of Convolutional Neural Networks with Cutout",
        abstract="We introduce Cutout, a simple regularization technique...",
        paper_text="(extracted body text)",
    )
    result = await run_paper_screening(llm=llm, input=inp)
    assert isinstance(result, ScreenResult)
    assert result.decision == "accept"
    assert result.rejection_reason is None
    assert "Cutout" in result.summary
    # The wrapper inspected the prompt config and used the
    # ``submit_screening`` tool name from base.yaml.
    tool_defs = llm.calls[0]["tools"]
    assert tool_defs is not None
    names = {cast_tool["name"] for cast_tool in tool_defs}  # type: ignore[union-attr]
    assert "submit_screening" in names


@pytest.mark.asyncio
async def test_run_paper_screening_reject_carries_reason() -> None:
    """Reject verdicts carry a non-empty rejection_reason; schema enforces."""
    llm = StubLlmProvider([_reject_response()])  # type: ignore[list-item]
    inp = ScreenerInput(
        arxiv_id="9999.99999",
        paper_text="(extracted body text)",
    )
    result = await run_paper_screening(llm=llm, input=inp)
    assert result.decision == "reject"
    assert result.rejection_reason
    assert "theory" in result.rejection_reason.lower()


@pytest.mark.asyncio
async def test_run_paper_screening_reject_without_reason_fails() -> None:
    """A reject verdict missing rejection_reason raises via the schema validator.

    StructuredCallFailed wraps the schema-validation error; the
    structured-call retry-once discipline applies, but with the same
    invalid payload twice the wrapper raises (DEC-018 — no silent
    fallback).
    """
    bad = model_response(
        tool_uses=[
            (
                "tool-1",
                "submit_screening",
                {"decision": "reject", "rejection_reason": None, "summary": "..."},
            )
        ],
        stop_reason="tool_use",
    )
    llm = StubLlmProvider([bad, bad])
    inp = ScreenerInput(arxiv_id="1234.5678", paper_text="...")
    with pytest.raises(StructuredCallFailed):
        await run_paper_screening(llm=llm, input=inp)


@pytest.mark.asyncio
async def test_run_paper_screening_user_message_includes_paper_text() -> None:
    """The user message renders title + abstract + paper_text + decision criteria."""
    llm = StubLlmProvider([_accept_response()])  # type: ignore[list-item]
    inp = ScreenerInput(
        arxiv_id="1708.04552",
        title="Cutout",
        abstract="(abstract here)",
        paper_text="(paper body)",
    )
    await run_paper_screening(llm=llm, input=inp)
    user_msg = llm.calls[0]["messages"][-1]  # type: ignore[index]
    body = "\n".join(
        [block.get("text", "") for block in user_msg["content"]]  # type: ignore[index]
    )
    assert "1708.04552" in body
    assert "(abstract here)" in body
    assert "(paper body)" in body
    assert "Decision Criteria" in body
