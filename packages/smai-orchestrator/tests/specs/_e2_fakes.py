"""Shared fixtures for Task 3.E2 spec tests (paper-ingestion pipeline).

Per the workspace's pytest ``--import-mode=importlib`` filename-hygiene
rule (cross-package fixture filenames must not collide with siblings):
``_e2_*`` keeps these distinct from ``_e1_*`` (proposal — Task 3.E1),
``_e3_*`` (run-record — Task 3.E3), and ``_specs_fakes`` (Task 2.C4).

Builders shipped here:

* :class:`InProcessFakeFetcher` — :class:`PaperFetcher` that returns
  canned :class:`FetchedPaper` content. Used by spec tests + the
  integration round-trip test to keep the fetch stage offline.
* :func:`make_paper_record` — :class:`PaperRecord` builder.
* :func:`build_finalized_paper_buffer_payload` — JSON-shaped finalized
  technique buffer (a list of :class:`TechniqueDescription` dicts) the
  ingestion subagent writes. Used to pre-stage a ``planning →
  registered`` advance / registration without driving the subagent.
* :func:`make_paper_extraction_args` — a paper_extract
  :class:`PaperExtraction` dict for the ingestion output tool.
* :func:`make_ingestion_function_model` — a PydanticAI ``FunctionModel``
  emitting that output (offline Paper Agent run).
* :func:`make_ingestion_corpus_fetcher` — a canned :class:`PaperCorpus`
  fetcher so the subagent's LaTeX-source fetch stays offline.
* :func:`make_screener_response` — canned LLM response emitting a
  :class:`ScreenResult` tool_use.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from smai_core.plugins import (
    ModelResponse,
    NormalizedMessage,
    StopReason,
    TextContent,
    TokenUsage,
    ToolUseContent,
)
from smai_orchestrator.entities.tracking import PaperRecord
from smai_orchestrator.specs.paper_ingestion import FetchedPaper, PaperFetcher


def make_paper_record(
    *,
    arxiv_id: str,
    state: str = "submitted",
    title: str | None = None,
) -> PaperRecord:
    now = datetime.now(UTC)
    return PaperRecord(
        arxiv_id=arxiv_id,
        title=title,
        state=state,  # type: ignore[arg-type]
        version=0,
        created_at=now,
        updated_at=now,
    )


class InProcessFakeFetcher(PaperFetcher):
    """:class:`PaperFetcher` that returns canned content per arxiv id.

    The default constructor ships a single canned paper; tests can pass
    a mapping for multi-paper fixtures (e.g., source-paper enrichment
    tests).
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


def _model_response(
    *,
    tool_uses: list[tuple[str, str, dict[str, Any]]] | None = None,
    text: str | None = None,
    stop_reason: StopReason = "tool_use",
) -> ModelResponse:
    from smai_core.plugins import NormalizedContent  # noqa: PLC0415

    content: list[NormalizedContent] = []
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


def make_paper_extraction_args(
    *,
    name: str = "cutout",
    source_arxiv_id: str,
) -> dict[str, Any]:
    """A JSON-serializable :class:`PaperExtraction` (one paper_extract technique).

    Drives the ingestion Paper Agent's output tool through a
    ``FunctionModel`` (see :func:`make_ingestion_function_model`). The
    single technique is a minimal but valid ``context_kind='paper_extract'``
    :class:`TechniqueDescription` so the planning handler can project it
    to a :class:`TechniqueRef` on registration.
    """
    from smai_inline_agents.ingestion import PaperExtraction  # noqa: PLC0415
    from smai_inline_agents.planner.schemas import (  # noqa: PLC0415
        AlgorithmSpec,
        ConfidenceFlag,
        Hyperparameter,
        SourceExcerpt,
        SourceLocationSection,
        TechniqueDescription,
    )

    technique = TechniqueDescription(
        name=name,
        summary=(
            "Cutout masks a contiguous square region of each input image during "
            "training as a simple, dataset-agnostic regularizer."
        ),
        motivation=(
            "Convolutional networks overfit on small datasets; occluding a region "
            "forces robustness to missing information and improves generalization."
        ),
        problem_setting=(
            "Supervised image classification with convolutional networks on small "
            "labeled datasets such as CIFAR-10 and SVHN."
        ),
        algorithm=AlgorithmSpec(
            summary=(
                "For each training image, sample a random square region and zero "
                "its pixels before the forward pass; the mask location is uniform."
            ),
            source_excerpts=[
                SourceExcerpt(
                    text="we randomly mask out square regions of input during training",
                    location=SourceLocationSection(section_id="3"),
                )
            ],
        ),
        hyperparameters=[
            Hyperparameter(
                name="patch_size",
                summary="side length in pixels of the square cutout region",
                value="16",
                source_excerpt=SourceExcerpt(
                    text="we use a patch length of 16 pixels",
                    location=SourceLocationSection(section_id="4"),
                ),
            )
        ],
        loss_function=None,
        training_recipe=None,
        limitations="Cutout size must be tuned per dataset.",
        context_kind="paper_extract",
        confidence_flags=[
            ConfidenceFlag(
                field_path="/loss_function",
                severity="unknown",
                note="paper introduces no custom loss; standard cross-entropy is used.",
            ),
            ConfidenceFlag(
                field_path="/training_recipe",
                severity="unknown",
                note="paper specifies no special training recipe beyond standard SGD.",
            ),
            ConfidenceFlag(
                field_path="/evaluation_protocol",
                severity="unknown",
                note="paper gives no structured evaluation protocol beyond top-1 accuracy.",
            ),
        ],
        source_arxiv_id=source_arxiv_id,
    )
    return PaperExtraction(
        techniques=[technique],
        paper_summary="The paper introduces Cutout, a regularization technique for CNNs.",
        extraction_caveats=[],
        extraction_audit=[],
    ).model_dump(mode="json")


def make_ingestion_function_model(*, source_arxiv_id: str, name: str = "cutout") -> Any:
    """A PydanticAI ``FunctionModel`` that emits the ingestion output tool.

    Passed as ``ingestion_model`` to :func:`build_paper_ingestion_spec`
    so the planning-state subagent runs offline (no real LLM); the model
    immediately calls ``emit_ingestion_result`` with a paper_extract
    :class:`PaperExtraction`.
    """
    from pydantic_ai.messages import ModelResponse as PydModelResponse  # noqa: PLC0415
    from pydantic_ai.messages import ToolCallPart  # noqa: PLC0415
    from pydantic_ai.models.function import AgentInfo, FunctionModel  # noqa: PLC0415

    def model_fn(messages: list[Any], info: AgentInfo) -> PydModelResponse:
        return PydModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args=make_paper_extraction_args(name=name, source_arxiv_id=source_arxiv_id),
                )
            ]
        )

    return FunctionModel(model_fn)


def make_ingestion_corpus_fetcher(
    *,
    arxiv_id: str,
    title: str = "Fake Ingested Paper",
    abstract: str = "Fake abstract for the ingestion subagent.",
):  # type: ignore[no-untyped-def]
    """An async corpus fetcher returning a canned :class:`PaperCorpus`.

    Passed as ``ingestion_corpus_fetcher`` to
    :func:`build_paper_ingestion_spec` so the subagent's LaTeX-source
    fetch stays offline.
    """
    from smai_inline_agents.ingestion import PaperCorpus  # noqa: PLC0415
    from smai_inline_agents.ingestion.fetch import LatexCiteResolver  # noqa: PLC0415
    from smai_inline_agents.ingestion.schemas import SectionRef  # noqa: PLC0415

    corpus = PaperCorpus(
        arxiv_id=arxiv_id,
        title=title,
        abstract=abstract,
        full_latex="\\section{Method} We randomly mask out square regions of input.",
        sections=[SectionRef(section_id="1", title="Method", depth=1)],
        sections_by_id={"1": "We randomly mask out square regions of input."},
        cite_resolver=LatexCiteResolver({}),
    )

    async def _fetch(requested_arxiv_id: str):  # type: ignore[no-untyped-def]
        del requested_arxiv_id
        return corpus

    return _fetch


def build_finalized_paper_buffer_payload(
    *,
    arxiv_id: str,
    name: str = "cutout",
) -> dict[str, Any]:
    """Build the JSON-shaped finalized technique buffer the ingestion
    subagent writes (Step 3, Sub-PR B).

    The ``techniques`` key is now a list of
    :class:`TechniqueDescription` dicts (the ingestion subagent's output)
    rather than the old planner-buffer dict. Used by spec tests that
    pre-stage the buffer artifact at
    ``papers/{arxiv_id}/draft_techniques.json`` to drive the
    ``planning → registered`` gate / registration without running the
    subagent.
    """
    extraction = make_paper_extraction_args(name=name, source_arxiv_id=arxiv_id)
    return {
        "finalized": True,
        "arxiv_id": arxiv_id,
        "paper_title": "Fake Ingested Paper",
        "paper_level_summary": extraction["paper_summary"],
        "screening": {"decision": "accept", "rejection_reason": None, "summary": "in scope"},
        "extraction_caveats": [],
        "techniques": extraction["techniques"],
    }


__all__ = [
    "InProcessFakeFetcher",
    "build_finalized_paper_buffer_payload",
    "make_ingestion_corpus_fetcher",
    "make_ingestion_function_model",
    "make_paper_extraction_args",
    "make_paper_record",
    "make_screener_response",
]
