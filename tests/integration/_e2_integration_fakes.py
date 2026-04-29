"""Cross-package fixtures for Task 3.E2 integration tests.

Per the brief's filename-hygiene rule (workspace
``--import-mode=importlib`` puts every package's tests/ on sys.path):
``_e2_*`` is the Task 3.E2 prefix, distinct from ``_e1_*`` /
``_e3_*`` and the smai-agents / smai-orchestrator / smai-cli
``_*_fakes`` modules.

Builders shipped here (workspace-local — the orchestrator-tree
``packages/smai-orchestrator/tests/specs/_e2_fakes.py`` carries the
spec-test versions; this module re-uses the same shapes for
integration tests rooted under ``tests/integration/``):

* :class:`StubLlmProvider` — queue-driven canned-response provider.
* :class:`InProcessFakeFetcher` — :class:`PaperFetcher` returning
  canned :class:`FetchedPaper` content.
* :func:`make_paper_planner_responses` — canned LLM responses for the
  paper-ingestion-variant planner happy path.
* :func:`make_screener_response` — canned :class:`ScreenResult` tool_use.
* :func:`build_smoke_runtime_config_for_papers` — :class:`RuntimeConfig`
  with the Phase-2 plugin defaults and the in-memory SQLite URI; used
  by the round-trip + reproduce-paper integration tests.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Any

from smai_core.plugins import (
    CacheConfig,
    LlmCapabilities,
    ModelResponse,
    NormalizedMessage,
    StopReason,
    TextContent,
    TokenUsage,
    ToolDefinition,
    ToolUseContent,
)
from smai_orchestrator import (
    FetchedPaper,
    PaperFetcher,
    PluginSelection,
    RuntimeConfig,
)
from smai_orchestrator.engine.config import EngineConfig

# === StubLlmProvider ========================================================


class StubLlmProvider:
    """Deterministic :class:`LlmProvider` for integration tests.

    Mirrors the smai-agents and smai-orchestrator local stubs; kept
    workspace-local so the integration tree doesn't pull either
    package's test directory onto ``sys.path``.
    """

    def __init__(
        self,
        responses: Sequence[ModelResponse] | None = None,
        *,
        capabilities: LlmCapabilities | None = None,
        name: str = "stub-e2",
    ) -> None:
        self.name = name
        self.capabilities = capabilities or LlmCapabilities(
            supports_caching=True,
            context_window=200_000,
            max_output_tokens=4_096,
            supports_tool_use=True,
            model_id=f"{name}:test",
        )
        self.model_id = f"{name}:test"
        self._responses: deque[ModelResponse] = deque(responses or [])
        self.calls: list[dict[str, Any]] = []

    async def call(
        self,
        system: str,
        messages: list[NormalizedMessage],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        cache_config: CacheConfig | None = None,
    ) -> ModelResponse:
        del tools, max_tokens, temperature, cache_config
        self.calls.append({"system": system, "messages_n": len(messages)})
        if not self._responses:
            raise AssertionError(f"StubLlmProvider {self.name!r}: ran out of canned responses")
        return self._responses.popleft()


# === Paper fetcher seam =====================================================


class InProcessFakeFetcher(PaperFetcher):
    """:class:`PaperFetcher` returning canned :class:`FetchedPaper` content.

    Used to keep the fetch stage offline. The default fixture is one
    canned paper; tests can pass a per-arxiv-id mapping for richer
    fixtures (e.g., a citing paper plus one or more cited source
    papers for the enricher dedup test).
    """

    def __init__(
        self,
        *,
        default_text: str = (
            "Default fake paper text — empirical comparison of method A vs. method B."
        ),
        default_title: str | None = "Fake Paper Title",
        default_abstract: str | None = "Fake abstract — empirical comparative claim.",
        per_paper: dict[str, FetchedPaper] | None = None,
    ) -> None:
        self._default = FetchedPaper(
            title=default_title,
            abstract=default_abstract,
            authors=["Fake Author"],
            paper_text=default_text,
            figures=[],
            expanded_tex=None,
        )
        self._per_paper = dict(per_paper or {})
        self.fetch_log: list[str] = []

    async def fetch(self, arxiv_id: str) -> FetchedPaper:
        self.fetch_log.append(arxiv_id)
        return self._per_paper.get(arxiv_id, self._default)


# === Canned LLM-response builders ===========================================


def _model_response(
    *,
    tool_uses: list[tuple[str, str, dict[str, Any]]] | None = None,
    text: str | None = None,
    stop_reason: StopReason = "tool_use",
) -> ModelResponse:
    content: list[TextContent | ToolUseContent] = []
    if text is not None:
        content.append(TextContent(text=text))
    if tool_uses is not None:
        for tu_id, tu_name, tu_input in tool_uses:
            content.append(ToolUseContent(id=tu_id, name=tu_name, input=dict(tu_input)))
    return ModelResponse(
        message=NormalizedMessage(role="assistant", content=content),
        stop_reason=stop_reason,
        usage=TokenUsage(input_tokens=1, output_tokens=1),
    )


def make_screener_response(
    *,
    decision: str = "accept",
    rejection_reason: str | None = None,
    summary: str = "Empirical comparison paper; in-scope.",
    tool_name: str = "submit_screening",
) -> ModelResponse:
    payload: dict[str, Any] = {"decision": decision, "summary": summary}
    if rejection_reason is not None:
        payload["rejection_reason"] = rejection_reason
    return _model_response(tool_uses=[("call_1", tool_name, payload)])


def make_paper_planner_responses(
    *,
    arxiv_id: str,
    contribution_techniques: Sequence[dict[str, Any]] | None = None,
) -> list[ModelResponse]:
    """Canned responses driving the paper-ingestion-variant planner
    through a happy-path finalize.

    For each contribution technique: ``draft_create_technique`` with a
    :class:`PaperFidelityAnchor` matching ``arxiv_id``. Then
    ``finalize_paper_techniques`` (no input fields per
    :class:`FinalizePaperTechniquesInput`) and ``finish``
    (``success=True``, summary).
    """
    techniques = (
        list(contribution_techniques)
        if contribution_techniques is not None
        else [
            {
                "symbolic_name": f"{arxiv_id}-tech-contrib",
                "name": "Contribution Technique",
                "description": "Primary contribution technique extracted from the paper.",
                "category": "augmentation",
                "compatible_factor_types": ["additive"],
                "standard": False,
                "fidelity_anchor": {
                    "kind": "paper",
                    "arxiv_id": arxiv_id,
                    "doi": f"arxiv:{arxiv_id}",
                },
            }
        ]
    )
    responses: list[ModelResponse] = []
    for idx, tech in enumerate(techniques):
        responses.append(
            _model_response(
                tool_uses=[
                    (f"tu-create-{idx}", "draft_create_technique", dict(tech)),
                ],
            )
        )
    responses.append(
        _model_response(tool_uses=[("tu-finalize", "finalize_paper_techniques", {})]),
    )
    responses.append(
        _model_response(
            tool_uses=[
                (
                    "tu-finish",
                    "finish",
                    {"success": True, "summary": "paper ingestion complete"},
                )
            ],
        ),
    )
    return responses


# === RuntimeConfig builder ==================================================


def build_smoke_runtime_config_for_papers() -> RuntimeConfig:
    """A :class:`RuntimeConfig` for paper-ingestion integration tests.

    Mirrors :func:`tests.integration.test_smoke_e2e._make_smoke_runtime_config`
    — Phase-2 default plugin selection + in-memory SQLite. The
    pipelines list now includes the paper-ingestion + proposal +
    run-record specs so :func:`Runtime.start_in_band` registers all
    five.
    """
    return RuntimeConfig(
        engine=EngineConfig(poll_interval_seconds=10, supervisor_enabled=False),
        plugins=PluginSelection(
            llm_provider="bedrock",
            metadata_store="sqlite",
            artifact_store="localfs",
            compute="localgpu",
            metadata_store_config={"uri": "sqlite+aiosqlite:///:memory:"},
        ),
        pipelines=[
            "smai_cg_execution",
            "smai_cg_entries",
            "smai_proposal_pipeline",
            "smai_paper_ingestion",
            "smai_run_record",
        ],
    )


__all__ = [
    "InProcessFakeFetcher",
    "StubLlmProvider",
    "build_smoke_runtime_config_for_papers",
    "make_paper_planner_responses",
    "make_screener_response",
]
