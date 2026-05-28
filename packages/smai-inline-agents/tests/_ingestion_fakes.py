"""Fixtures + fakes for the ingestion-subagent tests (Step 3, Sub-PR A).

Per CLAUDE.md filename hygiene: ``_ingestion_*`` prefix so this module
does not collide with the sibling ``_agent_fakes`` / ``_role_fakes`` /
``_search_fakes`` modules under the same ``tests/`` dir on the
importlib sys.path.

Provides:

* :func:`make_paper_extract_technique` -- a valid paper_extract
  :class:`TechniqueDescription`.
* :func:`make_paper_extraction` / :func:`make_paper_extraction_args` --
  the agent's :class:`PaperExtraction` output (object + JSON dict for the
  FunctionModel output tool).
* :class:`FakeCiteResolver` -- a :class:`CiteResolver` fake.
* :func:`make_corpus` / :func:`make_deps` -- a :class:`PaperCorpus` and a
  :class:`PaperAgentDeps` wired with a fake sub-extraction LLM + fetcher.
* :class:`AlwaysSnippetsLlm` + :func:`snippets_response` -- drive the
  in-tool ``structured_call`` sub-extractions.
* :data:`FIXTURE_LATEX` -- a small LaTeX doc for the parser unit test.
"""

from __future__ import annotations

from typing import Any

from _agent_helpers import model_response  # type: ignore[import-not-found]
from smai_core.plugins import (
    CacheConfig,
    LlmCapabilities,
    ModelResponse,
    NormalizedMessage,
    ToolDefinition,
)
from smai_inline_agents.ingestion.corpus import PaperCorpus
from smai_inline_agents.ingestion.schemas import (
    CitedPaperRef,
    PaperAgentDeps,
    PaperExtraction,
    PoolSnapshot,
    SectionRef,
)
from smai_inline_agents.planner.schemas import (
    AlgorithmSpec,
    ConfidenceFlag,
    Hyperparameter,
    SourceExcerpt,
    SourceLocationSection,
    TechniqueDescription,
)

DEFAULT_ARXIV_ID = "1708.04552"


def make_paper_extract_technique(*, name: str = "cutout") -> TechniqueDescription:
    """A minimal but valid ``context_kind='paper_extract'`` technique."""
    return TechniqueDescription(
        name=name,
        summary=(
            "Cutout masks out a contiguous square region of each input image "
            "during training as a simple, dataset-agnostic regularizer."
        ),
        motivation=(
            "Convolutional networks overfit on small datasets; occluding a "
            "contiguous region forces robustness to missing information and "
            "improves generalization without extra parameters."
        ),
        problem_setting=(
            "Supervised image classification with convolutional networks on "
            "small-to-medium labeled datasets such as CIFAR-10 and SVHN."
        ),
        algorithm=AlgorithmSpec(
            summary=(
                "For each training image, sample a random square region and set "
                "its pixels to zero (or the dataset mean) before the forward pass; "
                "the mask location is uniform over the image."
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
        limitations=(
            "Cutout size must be tuned per dataset; an overlarge patch removes "
            "too much signal and can hurt accuracy on low-resolution images."
        ),
        evaluation_protocol=None,
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
        source_arxiv_id=DEFAULT_ARXIV_ID,
    )


def make_paper_extraction(*, name: str = "cutout") -> PaperExtraction:
    """A :class:`PaperExtraction` carrying one paper_extract technique."""
    return PaperExtraction(
        techniques=[make_paper_extract_technique(name=name)],
        paper_summary=(
            "The paper introduces Cutout, a regularization technique that masks "
            "square image regions during training, and benchmarks it on CIFAR/SVHN."
        ),
        extraction_caveats=[],
        extraction_audit=[],
    )


def make_paper_extraction_args(*, name: str = "cutout") -> dict[str, Any]:
    """JSON-serializable args for the FunctionModel output tool call."""
    return make_paper_extraction(name=name).model_dump(mode="json")


class FakeCiteResolver:
    """A :class:`CiteResolver` fake backed by a dict of refs."""

    def __init__(self, refs: dict[str, CitedPaperRef]) -> None:
        self._refs = dict(refs)

    def resolve(self, cite_key: str) -> CitedPaperRef | None:
        return self._refs.get(cite_key.strip())

    def known_keys(self) -> list[str]:
        return sorted(self._refs)


def make_corpus(
    *,
    arxiv_id: str = "2501.00001",
    full_latex: str = "Cited paper LaTeX: the proposed metric is the F-score.",
) -> PaperCorpus:
    """A small cited-paper :class:`PaperCorpus` for search_literature tests."""
    return PaperCorpus(
        arxiv_id=arxiv_id,
        title="A Cited Metric Paper",
        abstract="We propose a metric.",
        full_latex=full_latex,
        sections=[SectionRef(section_id="1", title="Metric", depth=1)],
        sections_by_id={"1": "The proposed metric is the F-score."},
        cite_resolver=FakeCiteResolver({}),
    )


def make_corpus_fetcher(corpus: PaperCorpus | None):
    """An async corpus fetcher that always returns ``corpus``."""

    async def _fetch(arxiv_id: str) -> PaperCorpus | None:
        del arxiv_id
        return corpus

    return _fetch


async def _null_fetcher(arxiv_id: str) -> PaperCorpus | None:
    del arxiv_id
    return None


def make_deps(
    *,
    sub_extraction_llm: Any,
    corpus_fetcher: Any = None,
    sections: list[SectionRef] | None = None,
    sections_by_id: dict[str, str] | None = None,
    cite_resolver: Any = None,
    full_latex: str = "Target paper LaTeX body. We mask square regions.",
    tool_call_count: int = 0,
) -> PaperAgentDeps:
    """Build :class:`PaperAgentDeps` with sensible test defaults."""
    if sections is None:
        sections = [
            SectionRef(section_id="3", title="Method", depth=1),
            SectionRef(section_id="3.1", title="Cutout", depth=2),
        ]
    if sections_by_id is None:
        sections_by_id = {
            "3": "Method body text describing the masking procedure.",
            "3.1": "Cutout details: mask a square region per image.",
        }
    if cite_resolver is None:
        cite_resolver = FakeCiteResolver(
            {"wang2025": CitedPaperRef(cite_key="wang2025", arxiv_id="2501.00001")}
        )
    return PaperAgentDeps(
        arxiv_id=DEFAULT_ARXIV_ID,
        paper_title="Improved Regularization with Cutout",
        paper_abstract="We introduce Cutout.",
        section_list=sections,
        full_latex=full_latex,
        sections_by_id=sections_by_id,
        cite_resolver=cite_resolver,
        pool_snapshot=PoolSnapshot.empty(),
        sub_extraction_llm=sub_extraction_llm,
        corpus_fetcher=corpus_fetcher if corpus_fetcher is not None else _null_fetcher,
        tool_call_count=tool_call_count,
    )


def snippets_response(snippets: list[str], *, found: bool = True) -> ModelResponse:
    """A canned tool_use response for the retrieval ``structured_call``."""
    return model_response(
        tool_uses=[("snip-1", "emit_snippets", {"found": found, "snippets": snippets})],
        stop_reason="tool_use",
    )


def screener_response(
    *,
    decision: str = "accept",
    rejection_reason: str | None = None,
    summary: str = "Empirical comparison paper; in scope for the technique pool.",
) -> ModelResponse:
    """A canned ``submit_screening`` tool_use for the screener single-call.

    Drives ``run_paper_screening`` through the wrapper's internal-screening
    path; the screener prompt's structured-output tool is named
    ``submit_screening``.
    """
    payload: dict[str, Any] = {"decision": decision, "summary": summary}
    if rejection_reason is not None:
        payload["rejection_reason"] = rejection_reason
    return model_response(
        tool_uses=[("screen-1", "submit_screening", payload)],
        stop_reason="tool_use",
    )


class AlwaysSnippetsLlm:
    """An :class:`LlmProvider` that returns the same snippets on every call.

    Unlike the queue-driven ``StubLlmProvider``, it never exhausts -- handy
    for the forced-synthesis test, which drives many tool calls.
    """

    def __init__(self, snippets: list[str], *, found: bool = True) -> None:
        self.name = "always-snippets"
        self.capabilities = LlmCapabilities(
            supports_caching=True,
            context_window=200_000,
            max_output_tokens=4_096,
            supports_tool_use=True,
            model_id="always:test",
        )
        self._response = snippets_response(snippets, found=found)
        self.call_count = 0

    async def call(
        self,
        system: str,
        messages: list[NormalizedMessage],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        cache_config: CacheConfig | None = None,
    ) -> ModelResponse:
        del system, messages, tools, max_tokens, temperature, cache_config
        self.call_count += 1
        return self._response.model_copy(deep=True)

    async def credentials_for_subprocess(self) -> dict[str, str]:
        return {}


# A small but structurally complete LaTeX document for the parser unit
# test: title with nested command, abstract env, nested sections,
# multiple cite forms, and a bibliography with arXiv ids.
FIXTURE_LATEX = r"""
\documentclass{article}
\title{Improved Regularization with \textbf{Cutout}}
\author{T. DeVries}
\begin{document}
\begin{abstract}
We introduce Cutout, a simple regularization technique for CNNs.
\end{abstract}
\section{Introduction}
We build on prior work \cite{devries2017,wang2025}.
\subsection{Background}
The full derivation can be found in Section 2.
\section{Method}
Our metric follows \citep{kingma2014}. We mask square regions of the input.
\begin{thebibliography}{9}
\bibitem{wang2025} J. Wang et al. A useful metric. arXiv:2501.00001.
\bibitem{devries2017} T. DeVries and G. Taylor. Cutout. arXiv:1708.04552.
\end{thebibliography}
\end{document}
"""


__all__ = [
    "DEFAULT_ARXIV_ID",
    "FIXTURE_LATEX",
    "AlwaysSnippetsLlm",
    "FakeCiteResolver",
    "make_corpus",
    "make_corpus_fetcher",
    "make_deps",
    "make_paper_extract_technique",
    "make_paper_extraction",
    "make_paper_extraction_args",
    "screener_response",
    "snippets_response",
]
