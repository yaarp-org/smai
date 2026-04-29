""":func:`smai_agents.run_technique_enrichment` — Task 3.E2 acceptance.

Per ``04-agents.md`` §6 / ``08-novel-technique-pipeline.md`` §5.6 / DEC-018.
Single-call structured-output technique enricher; the wrapper is
exercised with the :class:`StubLlmProvider` from ``_agent_fakes``.

Per DEC-032 / `08` §5.6: paper ingestion's enricher reads a cited
source paper and produces a focused method-extraction for one
specific technique that the planner registered as a skeleton via
``draft_ensure_technique``. Output is an :class:`EnrichmentResult`
carrying ``implementability`` (high / medium / blocked),
``method_extraction`` (free-form Markdown body), optional
``refined_description``, and a ``blocked_reason`` required iff
``implementability == "blocked"``.
"""

from __future__ import annotations

import pytest
from _agent_fakes import StubLlmProvider  # type: ignore[import-not-found]
from _agent_helpers import model_response  # type: ignore[import-not-found]
from smai_agents import (
    StructuredCallFailed,
    clear_prompt_config_cache,
)
from smai_agents.agents.enricher import (
    EnricherInput,
    run_technique_enrichment,
)
from smai_agents.schemas.enricher import EnrichmentResult


@pytest.fixture(autouse=True)
def _clear_prompt_config_cache() -> None:
    clear_prompt_config_cache()


def _enricher_input() -> EnricherInput:
    return EnricherInput(
        technique_id="t_resnet50",
        technique_name="ResNet-50",
        technique_description="50-layer residual network",
        citing_paper_arxiv_id="1708.04552",
        source_paper_arxiv_id="1512.03385",
        source_paper_text=(
            "Deep Residual Learning for Image Recognition. "
            "We introduce a 50-layer ResNet variant..."
        ),
    )


def _high_response() -> object:
    return model_response(
        tool_uses=[
            (
                "tool-1",
                "submit_enrichment",
                {
                    "implementability": "high",
                    "method_extraction": (
                        "## Inputs\n3-channel RGB images (224x224).\n\n"
                        "## Algorithm\nStack 16 bottleneck blocks...\n\n"
                        "## Hyperparameters\n- depth: 50\n- bottleneck: 4\n"
                    ),
                    "refined_description": "50-layer ResNet with bottleneck blocks",
                    "blocked_reason": None,
                },
            )
        ],
        stop_reason="tool_use",
    )


def _blocked_response() -> object:
    return model_response(
        tool_uses=[
            (
                "tool-1",
                "submit_enrichment",
                {
                    "implementability": "blocked",
                    "method_extraction": (
                        "## Notes\nMethod uses proprietary in-house dataset; "
                        "cannot be reproduced from public sources."
                    ),
                    "refined_description": None,
                    "blocked_reason": "uses proprietary dataset not publicly available",
                },
            )
        ],
        stop_reason="tool_use",
    )


@pytest.mark.asyncio
async def test_run_technique_enrichment_high_implementability() -> None:
    """High-implementability path round-trips."""
    llm = StubLlmProvider([_high_response()])  # type: ignore[list-item]
    result = await run_technique_enrichment(llm=llm, input=_enricher_input())
    assert isinstance(result, EnrichmentResult)
    assert result.implementability == "high"
    assert result.blocked_reason is None
    assert "ResNet" in (result.refined_description or "")
    assert "## Algorithm" in result.method_extraction


@pytest.mark.asyncio
async def test_run_technique_enrichment_blocked() -> None:
    """Blocked path requires a blocked_reason; schema enforces."""
    llm = StubLlmProvider([_blocked_response()])  # type: ignore[list-item]
    result = await run_technique_enrichment(llm=llm, input=_enricher_input())
    assert result.implementability == "blocked"
    assert result.blocked_reason is not None
    assert "proprietary" in result.blocked_reason.lower()


@pytest.mark.asyncio
async def test_run_technique_enrichment_blocked_without_reason_fails() -> None:
    """Blocked verdict missing blocked_reason raises (no silent fallback)."""
    bad = model_response(
        tool_uses=[
            (
                "tool-1",
                "submit_enrichment",
                {
                    "implementability": "blocked",
                    "method_extraction": "...",
                    "refined_description": None,
                    "blocked_reason": None,
                },
            )
        ],
        stop_reason="tool_use",
    )
    llm = StubLlmProvider([bad, bad])
    with pytest.raises(StructuredCallFailed):
        await run_technique_enrichment(llm=llm, input=_enricher_input())


@pytest.mark.asyncio
async def test_run_technique_enrichment_user_message_carries_source_text() -> None:
    """The user message includes the source paper text + citing-paper id."""
    llm = StubLlmProvider([_high_response()])  # type: ignore[list-item]
    inp = _enricher_input()
    await run_technique_enrichment(llm=llm, input=inp)
    user_msg = llm.calls[0]["messages"][-1]  # type: ignore[index]
    body = "\n".join(
        [block.get("text", "") for block in user_msg["content"]]  # type: ignore[index]
    )
    assert inp.source_paper_arxiv_id in body
    assert inp.citing_paper_arxiv_id in body
    assert "Deep Residual Learning" in body
