"""Paper-ingestion :class:`PipelineSpec` instance — Task 3.E2.

Per ``designs/smai/08-novel-technique-pipeline.md`` §5 and
``designs/smai/03-state-machine.md`` §5. The **supporting utility** per
DEC-032 — paper ingestion produces ``TechniqueRef``s + paper-fidelity-
anchor metadata only; it does NOT produce
:class:`ExperimentDefinition`s, :class:`ComparisonGroupRecord`s, or
:class:`EntryRecord`s. Reproduce-paper-X workflows go through the
proposal pipeline (:mod:`.proposal`) referencing a pre-ingested paper.

States lift from `03` §5.1: ``submitted → fetching → screening →
planning → registered`` plus ``rejected`` / ``failed`` terminals plus
the non-terminal ``partial`` parking spot (created by another paper's
enrichment step per `08` §5.7).

Edge surface lifts from `03` §5.2 — all edges fire on ``dispatch_time``
because every dispatch handler in this pipeline is **inline** (`08`
§7 — paper ingestion uses no :class:`Compute`, just ``LlmProvider`` +
local processing). The §5.2 ``job_succeeded`` / ``job_failed`` edge
triggers in the doc table are v1-vestigial framing that doesn't apply
under the inline-dispatch shape — `phase-1` (per `05` §3.1) only fires
on external-:class:`Compute` :class:`JobStatus` reports, so an inline
fetcher / screener / planner has no phase-1 trigger to consume. Same
rationale as :mod:`.proposal`'s "edges off ``designing`` use
``dispatch_time``" comment. The gate body for a "job succeeded" edge
reads :class:`ArtifactStore` (or :class:`MetadataStore` for the
screener decision artifact) for completion signals; the gate body for
a "job failed" / retry-budget edge reads
:class:`PaperRecord.{screening,planning,registration}_attempt`.

Spec ambiguities resolved
-------------------------

* **The ``partial`` parking spot has no engine-driven edge.** `03`
  §5.2 edge 13 (``partial → submitted``) is declared with
  ``fires_on=dispatch_time`` and the ``paper.partial_user_promotion``
  gate "reads a user-set promotion flag on PaperRecord". Two
  observations:

  1. :class:`PaperRecord` does not carry a ``promotion_requested`` /
     ``promote_partial`` boolean today (see
     :mod:`smai_orchestrator.entities.tracking.paper`). The brief's §1.4
     forbids extending the entity surface unilaterally.
  2. :class:`MetadataStore`'s `07-plugin-interfaces.md` §5.6.5 docstring
     for :meth:`get_partial_pending_promotion` calls promotion "a
     **synchronous user-driven write** through
     :meth:`transition_paper_state` from ``partial`` to ``submitted``"
     and notes that the query exists for tooling/observability — NOT as
     a phase-2 work source. That framing is consistent with `08` §5.7
     line 432-437's "Paper exists with status ``partial``: …
     ``transition_paper_state(...,target_state='submitted')``. Dispatch
     the pipeline."

  The Protocol's framing (synchronous direct write) and the §5.2 edge
  framing (engine-observed dispatch_time gate) are inconsistent —
  exactly the same v1-vestigial split the wave-1.a fix reconciled for
  the proposal pipeline (per the wave-1.a status note on `08` §2.4 row
  3). This spec follows the Protocol framing: ``smai ingest
  --promote-partial <arxiv_id>`` synchronously writes
  ``transition_paper_state(arxiv_id, version, "submitted")`` via
  :class:`smai_cli.runtime.PapersService.promote_partial`. The
  pipeline-spec declares ``partial`` as a non-terminal state with no
  outgoing edges and no scheduling query; the engine never visits a
  partial paper until the synchronous transition writes
  ``state=submitted`` and the next worker cycle's
  ``get_ready_for_paper_fetch`` (or ``...screen`` if content is already
  extracted) picks it up.

  Surfaced for supervisor adjudication. Land the same `08` §5 / `03`
  §5 reconciliation that wave-1.a applied to the proposal pipeline if
  the supervisor agrees.

* **Inline-dispatch edges all use ``dispatch_time``.** Per the module
  docstring above; mirrors :mod:`.proposal`'s pattern. The §5.2 doc
  table's ``job_succeeded`` / ``job_failed`` triggers are v1-vestigial
  framing.

* **``planning`` dispatch handler: the ingestion subagent (planner-
  refactor Step 3, Sub-PR B).** The ``planning`` state's on-entry
  handler (:func:`make_dispatch_paper_ingestion_subagent`) replaces the
  old planner-paper-variant + per-skeleton enricher loop with a single
  call to
  :func:`smai_inline_agents.ingestion.run_ingestion_subagent`. It reuses
  the ``screening`` state's verdict (read from ``screen_result.json``
  and passed as ``screening=``) so the paper is screened ONCE (no
  double-screen — ``ingestion_subagent.md`` §4 vs §7), runs the
  SciReplicate-shaped Paper Agent, and writes the extracted
  :class:`smai_inline_agents.planner.TechniqueDescription` list to the
  finalized technique-buffer artifact. The ``registered`` state's
  registration handler projects each into a paper-anchored
  :class:`TechniqueRef`
  (:func:`_technique_description_to_paper_ref`). The handler is inline;
  it uses no :class:`Compute`. The old enricher-loop's ``partial``-paper
  dedup is superseded by ``search_literature``'s depth-limited deep
  recursion inside the subagent (the v1 ``enricher`` agent survives as a
  removable fallback per ``ingestion_subagent.md`` §4 until
  planner-refactor step 8).

* **arXiv fetching.** Production fetcher uses the `arxiv` PyPI package
  (see :class:`PaperFetcher` for the seam). Tests inject an inline
  fake fetcher via the dispatch-handler factory's ``fetcher`` kwarg to
  keep CI offline.

* **Registration as ``planning → registered`` edge dispatch / on-
  entry of ``registered``.** `03` §5.3 declares the registration
  handler as the ``planning → registered`` edge dispatch handler. The
  current engine drives dispatch from
  :attr:`StateDef.on_entry_dispatch` only (per `05` §1.4 / the
  proposal-spec docstring), so this spec attaches the registration
  handler to the ``registered`` state's on-entry slot — same
  operational shape, the engine fires the handler after the CAS
  transition. ``registered`` is terminal; phase-2 discovery never
  returns terminal states (per `05` §1.5), so the handler fires
  exactly once per paper.

* **Concurrency pool `paper_ingestion`.** Per DEC-034 #4: limit 2,
  priority 10 (lowest of the four pool priorities — paper ingestion is
  CPU/network-only and lower priority than CG-execution + proposal
  pipeline; a burst of paper submissions does not block in-flight
  CGs).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

from smai_core.entities.technique import (
    PaperFidelityAnchor,
    TechniqueRef,
)
from smai_core.plugins import (
    ArtifactNotFound,
    JobHandle,
    LlmProvider,
)

from smai_orchestrator.engine.types import (
    ConcurrencyPool,
    DispatchAction,
    DispatchContext,
    DispatchOutcome,
    EdgeDef,
    GateContext,
    GateOutcome,
    RetryPolicy,
    SchedulingQueryRef,
    StateDef,
)
from smai_orchestrator.runtime.registry import register_pipeline_spec
from smai_orchestrator.runtime.spec import PipelineSpec

# === Constants ==============================================================

PAPER_INGESTION_SPEC_NAME = "smai_paper_ingestion"

# Concurrency pool per DEC-034 #4 — limit 2, priority 10. All paper-
# ingestion stages run inline (no external :class:`Compute`); slot
# accounting bounds the worker process's concurrent paper-ingestion
# work.
POOL_PAPER_INGESTION = "paper_ingestion"
POOL_PAPER_INGESTION_LIMIT = 2
POOL_PAPER_INGESTION_PRIORITY = 10

# Artifact-key conventions — match the v1 layout per
# ``designs/yaarp/paper_ingestion.md`` and the `08` §5 / `01` §5.7
# convention. ``papers/{arxiv_id}/`` rooted.
LATEX_BUNDLE_KEY_TEMPLATE = "papers/{arxiv_id}/source.tar.gz"
EXPANDED_TEX_KEY_TEMPLATE = "papers/{arxiv_id}/expanded.tex"
PAPER_TEXT_KEY_TEMPLATE = "papers/{arxiv_id}/extracted/paper_text.json"
FIGURES_KEY_TEMPLATE = "papers/{arxiv_id}/extracted/figures.json"
SCREEN_RESULT_KEY_TEMPLATE = "papers/{arxiv_id}/screen_result.json"
TECHNIQUE_BUFFER_KEY_TEMPLATE = "papers/{arxiv_id}/draft_techniques.json"
METHOD_EXTRACTION_KEY_TEMPLATE = (
    "papers/{arxiv_id}/techniques/{technique_id}/method_extraction.json"
)

_log = logging.getLogger(__name__)


# === Fetcher seam ===========================================================


@dataclass
class FetchedPaper:
    """In-memory bundle returned by a :class:`PaperFetcher`.

    The fetcher seam abstracts arxiv-fetching so tests can inject
    canned content; production wires :class:`ArxivLatexFetcher`. The
    bundle carries everything the dispatch handler needs to write the
    standard paper-ingestion artifacts to ``ArtifactStore``.
    """

    title: str | None
    abstract: str | None
    authors: list[str]
    paper_text: str
    """Extracted plain-text body of the paper. Production fetcher
    expands the LaTeX and extracts text; test fetchers inject canned
    content."""

    figures: list[dict[str, Any]]
    """Per-figure metadata (caption, asset path, etc.). v1 carry-
    forward shape; the dispatch handler serializes via
    :func:`json.dumps`."""

    expanded_tex: str | None = None
    """Optional expanded LaTeX body. Persisted when present for runbook
    debugging; ``None`` is acceptable when the fetcher cannot expand."""


class PaperFetcher:
    """Protocol-shaped seam for arxiv-fetching.

    Production: :class:`ArxivLatexFetcher` (uses the ``arxiv`` PyPI
    package). Tests: subclass + override :meth:`fetch` with a fake
    that returns canned :class:`FetchedPaper` content.
    """

    async def fetch(self, arxiv_id: str) -> FetchedPaper:
        raise NotImplementedError


class ArxivLatexFetcher(PaperFetcher):
    """Production arxiv fetcher — best-effort LaTeX retrieval.

    Per `08` §5.2 the fetcher downloads the arxiv paper's LaTeX
    source tarball, expands it (best-effort), extracts text, and
    returns a :class:`FetchedPaper`. Failure modes (network errors,
    expansion failures, paper not found) raise
    :class:`PaperFetchError` and are caught by the dispatch handler,
    which routes to the ``failed`` terminal via the retry-budget gate.

    The implementation is intentionally minimal — `08` §5.2 frames
    paper-fetch retries / robust LaTeX parsing as out-of-scope
    operational concerns for v1; deployments that need richer fetch
    (pdf-only papers, mirror-aware retries, etc.) subclass this
    fetcher and inject via the dispatch-handler factory.
    """

    async def fetch(self, arxiv_id: str) -> FetchedPaper:
        try:
            import arxiv  # type: ignore[import-untyped]  # noqa: PLC0415
        except ImportError as exc:
            raise PaperFetchError(
                arxiv_id=arxiv_id,
                reason="the 'arxiv' PyPI package is not installed",
            ) from exc
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            client = arxiv.Client()
            results = list(client.results(search))
        except Exception as exc:  # noqa: BLE001 — any arxiv error is a fetch error
            raise PaperFetchError(
                arxiv_id=arxiv_id,
                reason=f"arxiv search failed: {type(exc).__name__}: {exc}",
            ) from exc
        if not results:
            raise PaperFetchError(
                arxiv_id=arxiv_id,
                reason=f"arxiv id {arxiv_id!r} returned no results",
            )
        result = results[0]
        title = result.title
        abstract = result.summary
        authors = [author.name for author in result.authors]
        # Best-effort source download. ``download_source`` returns the
        # downloaded tarball path; the v1 fetcher expanded the LaTeX
        # and extracted text. v2 ships text-from-abstract as the
        # default fallback (a real production deployment subclasses
        # this fetcher and adds the LaTeX-expansion step).
        paper_text = abstract or ""
        expanded_tex: str | None = None
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                src_path = result.download_source(dirpath=tmpdir)
                # The src_path is a tarball; v1 expanded with latexpand.
                # We surface the raw tarball path's existence as a
                # success signal but do not expand here — operators
                # who need expanded LaTeX subclass + override.
                if src_path and os.path.exists(src_path):
                    paper_text = paper_text or "(LaTeX source downloaded; expansion deferred)"
        except Exception:  # noqa: BLE001 — tarball download is best-effort
            _log.warning("arxiv source download failed for %s; using abstract only", arxiv_id)
        return FetchedPaper(
            title=title,
            abstract=abstract,
            authors=authors,
            paper_text=paper_text,
            figures=[],
            expanded_tex=expanded_tex,
        )


class PaperFetchError(Exception):
    """Raised by :class:`PaperFetcher.fetch` on any fetch failure.

    The dispatch handler wraps this into a :class:`DispatchOutcome`
    error; the next phase-3 cycle's retry-budget gate decides whether
    to re-fire the fetcher or transition to the ``failed`` terminal.
    """

    def __init__(self, *, arxiv_id: str, reason: str) -> None:
        self.arxiv_id = arxiv_id
        self.reason = reason
        super().__init__(f"paper fetch failed for {arxiv_id!r}: {reason}")


# === Gate-rule callable factories ===========================================


def _make_gate_content_already_extracted(
    *,
    paper_text_key_for: Callable[[str], str],
) -> Callable[[GateContext], Awaitable[GateOutcome]]:
    """``submitted → screening`` (dispatch_time, content_already_extracted).

    Reads ``ArtifactStore`` for the paper-text artifact; advances when
    present (the typical case for a ``partial → submitted`` promotion
    whose source content was extracted during a prior paper's
    enrichment per `08` §5.7). Declared **first** in the
    ``submitted``-outgoing edges list per `03` §5.2 so it short-
    circuits the fetch dispatch when content already exists.
    """

    async def _gate(ctx: GateContext) -> GateOutcome:
        key = paper_text_key_for(ctx.entity_id)
        try:
            await ctx.artifact_store.get(key)
        except ArtifactNotFound:
            return GateOutcome(
                advance=False,
                reason=f"paper text not extracted at {key!r}; route to fetching",
            )
        return GateOutcome(
            advance=True,
            reason="paper text already extracted; short-circuit fetch",
        )

    return _gate


def _make_gate_dispatch_fetch_ready() -> Callable[[GateContext], Awaitable[GateOutcome]]:
    """``submitted → fetching`` (dispatch_time, dispatch_fetch_ready).

    Always-fire — the prior edge (content_already_extracted) wins
    when content is present; reaching this gate means content needs
    fetching. Declared after the content-already-extracted edge per
    `03` §5.2's edge-ordering rule.
    """

    async def _gate(ctx: GateContext) -> GateOutcome:
        del ctx
        return GateOutcome(advance=True, reason="paper.dispatch_fetch_ready always-fire")

    return _gate


def _make_gate_fetch_complete(
    *,
    paper_text_key_for: Callable[[str], str],
) -> Callable[[GateContext], Awaitable[GateOutcome]]:
    """``fetching → screening`` (dispatch_time, fetch_complete).

    Reads ``ArtifactStore`` for the paper-text artifact; advances when
    present. The fetcher dispatch handler writes this artifact on
    success.
    """

    async def _gate(ctx: GateContext) -> GateOutcome:
        key = paper_text_key_for(ctx.entity_id)
        try:
            await ctx.artifact_store.get(key)
        except ArtifactNotFound:
            return GateOutcome(advance=False, reason=f"paper text artifact missing at {key!r}")
        return GateOutcome(advance=True, reason="fetch complete; advance to screening")

    return _gate


def _make_gate_screener_decision(
    *,
    expected_decision: str,
    screen_result_key_for: Callable[[str], str],
) -> Callable[[GateContext], Awaitable[GateOutcome]]:
    """``screening → planning`` (screener_pass) or ``screening → rejected``
    (screener_reject).

    Both gates read the screener-result artifact and route based on
    ``decision``. The pass gate (`03` §5.2 edge 5) is declared FIRST
    so accept verdicts route to ``planning``; the reject gate (edge 6)
    is declared after so reject verdicts route to ``rejected``.
    """

    async def _gate(ctx: GateContext) -> GateOutcome:
        key = screen_result_key_for(ctx.entity_id)
        try:
            payload = await ctx.artifact_store.get(key)
        except ArtifactNotFound:
            return GateOutcome(advance=False, reason=f"screen result missing at {key!r}")
        try:
            parsed = json.loads(payload)
        except (ValueError, UnicodeDecodeError):
            return GateOutcome(advance=False, reason=f"screen result at {key!r} is not JSON")
        if not isinstance(parsed, dict):
            return GateOutcome(advance=False, reason=f"screen result at {key!r} is not an object")
        parsed_dict = cast(dict[str, Any], parsed)
        decision = parsed_dict.get("decision")
        if decision == expected_decision:
            return GateOutcome(
                advance=True,
                reason=f"screener decision == {expected_decision!r}",
            )
        return GateOutcome(advance=False, reason=f"screener decision != {expected_decision!r}")

    return _gate


def _make_gate_techniques_finalized(
    *,
    technique_buffer_key_for: Callable[[str], str],
) -> Callable[[GateContext], Awaitable[GateOutcome]]:
    """``planning → registered`` (dispatch_time, techniques_finalized).

    Reads ``ArtifactStore`` for the planner's finalized buffer at
    ``papers/{arxiv_id}/draft_techniques.json``; advances when the
    buffer parses and ``finalized=True``. The structural-soundness
    check per `08` §5.4 is the planner's
    ``finalize_paper_techniques`` tool's responsibility — by the time
    the buffer reaches ``finalized=True``, the check has passed.
    """

    async def _gate(ctx: GateContext) -> GateOutcome:
        key = technique_buffer_key_for(ctx.entity_id)
        try:
            payload = await ctx.artifact_store.get(key)
        except ArtifactNotFound:
            return GateOutcome(advance=False, reason=f"technique buffer missing at {key!r}")
        try:
            parsed = json.loads(payload)
        except (ValueError, UnicodeDecodeError):
            return GateOutcome(advance=False, reason=f"technique buffer at {key!r} is not JSON")
        if not isinstance(parsed, dict):
            return GateOutcome(
                advance=False, reason=f"technique buffer at {key!r} is not an object"
            )
        parsed_dict = cast(dict[str, Any], parsed)
        if not parsed_dict.get("finalized"):
            return GateOutcome(advance=False, reason="technique buffer not finalized")
        return GateOutcome(
            advance=True,
            reason="planner finalized technique buffer; advance to registered",
        )

    return _gate


# === Dispatch handlers ======================================================


def make_dispatch_paper_fetch(
    *,
    fetcher: PaperFetcher | None = None,
    paper_text_key_for: Callable[[str], str] | None = None,
    figures_key_for: Callable[[str], str] | None = None,
    expanded_tex_key_for: Callable[[str], str] | None = None,
) -> Callable[[DispatchContext], Awaitable[DispatchOutcome]]:
    """``fetching`` on-entry dispatch — paper.dispatch_fetch_latex.

    Per `03` §5.3 / `08` §5.2: inline; downloads the paper from arxiv
    (via the injectable :class:`PaperFetcher` seam), persists the
    extracted artifacts to ArtifactStore at the standard
    ``papers/{arxiv_id}/...`` keys, and returns a synthetic
    :class:`JobHandle` so the next cycle's
    ``get_in_flight_paper_fetch`` query picks the entity up.

    Args:
        fetcher: The :class:`PaperFetcher` instance. ``None`` defaults
            to :class:`ArxivLatexFetcher` (production wiring); tests
            inject a fake fetcher.
        paper_text_key_for: Callable resolving ``arxiv_id →
            ArtifactStore key`` for ``paper_text.json``. Defaults to
            :data:`PAPER_TEXT_KEY_TEMPLATE`.
        figures_key_for: Callable resolving ``arxiv_id → ArtifactStore
            key`` for ``figures.json``. Defaults to
            :data:`FIGURES_KEY_TEMPLATE`.
        expanded_tex_key_for: Callable resolving ``arxiv_id →
            ArtifactStore key`` for ``expanded.tex``. Defaults to
            :data:`EXPANDED_TEX_KEY_TEMPLATE`.
    """
    actual_fetcher: PaperFetcher = fetcher if fetcher is not None else ArxivLatexFetcher()
    paper_text_key_fn = paper_text_key_for or _default_paper_text_key
    figures_key_fn = figures_key_for or _default_figures_key
    expanded_tex_key_fn = expanded_tex_key_for or _default_expanded_tex_key

    async def _dispatch(ctx: DispatchContext) -> DispatchOutcome:
        arxiv_id = ctx.entity_id
        paper = await ctx.metadata_store.get_paper(arxiv_id)
        if paper is None:
            return DispatchOutcome(error=f"paper {arxiv_id!r} not found in MetadataStore")
        try:
            fetched = await actual_fetcher.fetch(arxiv_id)
        except PaperFetchError as exc:
            return DispatchOutcome(error=f"paper fetch failed: {exc.reason}")

        # Persist artifacts. Sequential ``put`` calls — the
        # ArtifactStore Protocol does not expose a multi-key
        # transaction surface (per `07` §6).
        paper_text_key = paper_text_key_fn(arxiv_id)
        figures_key = figures_key_fn(arxiv_id)
        await ctx.artifact_store.put(
            paper_text_key,
            json.dumps(
                {
                    "arxiv_id": arxiv_id,
                    "title": fetched.title,
                    "abstract": fetched.abstract,
                    "authors": fetched.authors,
                    "paper_text": fetched.paper_text,
                },
                indent=2,
            ).encode("utf-8"),
        )
        await ctx.artifact_store.put(
            figures_key,
            json.dumps({"figures": fetched.figures}, indent=2).encode("utf-8"),
        )
        if fetched.expanded_tex:
            await ctx.artifact_store.put(
                expanded_tex_key_fn(arxiv_id),
                fetched.expanded_tex.encode("utf-8"),
            )

        # Synthesize a handle so the next cycle's
        # ``get_in_flight_paper_fetch`` query (which filters on
        # ``fetcher_job_handle IS NOT NULL``) returns the entity for
        # phase-3 evaluation. Same shape as the inline-planner pattern.
        handle = JobHandle(plugin="inline", handle=f"inline-paper-fetch-{arxiv_id}")
        return DispatchOutcome(submitted_handles=[handle])

    return _dispatch


def make_dispatch_paper_screener(
    *,
    llm_for_screener: LlmProvider,
    paper_text_key_for: Callable[[str], str] | None = None,
    screen_result_key_for: Callable[[str], str] | None = None,
) -> Callable[[DispatchContext], Awaitable[DispatchOutcome]]:
    """``screening`` on-entry dispatch — paper.dispatch_screener.

    Per `03` §5.3 / `08` §5.2: inline; runs the screener single-call
    agent against the extracted paper text, persists the
    :class:`ScreenResult` to ArtifactStore at
    ``papers/{arxiv_id}/screen_result.json``, and returns a synthetic
    :class:`JobHandle`.
    """
    paper_text_key_fn = paper_text_key_for or _default_paper_text_key
    screen_result_key_fn = screen_result_key_for or _default_screen_result_key

    async def _dispatch(ctx: DispatchContext) -> DispatchOutcome:
        from smai_inline_agents.agents.screener import (  # noqa: PLC0415
            ScreenerInput,
            run_paper_screening,
        )

        arxiv_id = ctx.entity_id
        paper = await ctx.metadata_store.get_paper(arxiv_id)
        if paper is None:
            return DispatchOutcome(error=f"paper {arxiv_id!r} not found in MetadataStore")
        # Read the extracted paper text.
        paper_text_key = paper_text_key_fn(arxiv_id)
        try:
            payload = await ctx.artifact_store.get(paper_text_key)
        except ArtifactNotFound:
            return DispatchOutcome(
                error=f"paper text artifact missing at {paper_text_key!r}",
            )
        try:
            paper_text_blob = cast(dict[str, Any], json.loads(payload))
        except (ValueError, UnicodeDecodeError) as exc:
            return DispatchOutcome(error=f"paper text artifact parse failed: {exc}")
        screener_input = ScreenerInput(
            arxiv_id=arxiv_id,
            title=cast(str | None, paper_text_blob.get("title")) or paper.title,
            abstract=cast(str | None, paper_text_blob.get("abstract")) or paper.abstract,
            paper_text=cast(str, paper_text_blob.get("paper_text", "")),
        )
        result = await run_paper_screening(llm=llm_for_screener, input=screener_input)
        await ctx.artifact_store.put(
            screen_result_key_fn(arxiv_id),
            result.model_dump_json(indent=2).encode("utf-8"),
        )
        handle = JobHandle(plugin="inline", handle=f"inline-paper-screen-{arxiv_id}")
        return DispatchOutcome(submitted_handles=[handle])

    return _dispatch


def make_dispatch_paper_ingestion_subagent(
    *,
    llm_for_ingestion: LlmProvider,
    screen_result_key_for: Callable[[str], str] | None = None,
    technique_buffer_key_for: Callable[[str], str] | None = None,
    corpus_fetcher: Any = None,
    ingestion_model: Any = None,
) -> Callable[[DispatchContext], Awaitable[DispatchOutcome]]:
    """``planning`` on-entry dispatch — paper.dispatch_ingestion_subagent.

    Per ``planner_refactor/design_notes/ingestion_subagent.md`` §7: the
    ``planning`` state cuts over from today's planner-paper-variant +
    enricher loop to the SciReplicate-shaped ingestion subagent. The
    handler:

    1. Reads the screening verdict the ``screening`` state already wrote
       (``papers/{arxiv_id}/screen_result.json``) and passes it into
       :func:`run_ingestion_subagent` as ``screening=`` so the paper is
       screened ONCE (no double-screen — design note §4 vs §7).
    2. Runs the subagent (fetch -> agent run -> assemble). The subagent's
       corpus fetch is LaTeX-source-first via the injectable
       ``corpus_fetcher`` (defaults to :func:`fetch_paper_corpus`; tests
       inject a fake). ``ingestion_model`` overrides the PydanticAI model
       (tests pass a ``FunctionModel``; production leaves ``None`` so the
       agent uses its role-resolved model).
    3. Writes the extracted :class:`TechniqueDescription` list to the
       finalized technique-buffer artifact the ``planning → registered``
       gate reads; the ``registered`` state's registration handler
       projects each into a :class:`TechniqueRef`.

    On an empty / errored result (``fetch_failed``, ``reject``, or no
    techniques) the handler returns a :class:`DispatchOutcome` error so
    the planning :class:`RetryPolicy` routes the paper to ``failed`` once
    the budget is exhausted (the existing failed edge — no new states).
    """
    screen_result_key_fn = screen_result_key_for or _default_screen_result_key
    technique_buffer_key_fn = technique_buffer_key_for or _default_technique_buffer_key

    async def _dispatch(ctx: DispatchContext) -> DispatchOutcome:
        from smai_inline_agents.ingestion import (  # noqa: PLC0415
            PaperAgentDepsFactory,
            ScreeningOutcome,
            fetch_paper_corpus,
            run_ingestion_subagent,
        )
        from smai_inline_agents.schemas.screener import ScreenResult  # noqa: PLC0415

        arxiv_id = ctx.entity_id
        paper = await ctx.metadata_store.get_paper(arxiv_id)
        if paper is None:
            return DispatchOutcome(error=f"paper {arxiv_id!r} not found in MetadataStore")

        # Reuse the screening-state verdict (no double-screen).
        screen_key = screen_result_key_fn(arxiv_id)
        try:
            screen_payload = await ctx.artifact_store.get(screen_key)
        except ArtifactNotFound:
            return DispatchOutcome(error=f"screen result missing at {screen_key!r}")
        try:
            screen = ScreenResult.model_validate_json(screen_payload)
        except Exception as exc:  # noqa: BLE001 — malformed artifact -> dispatch error
            return DispatchOutcome(error=f"screen result parse failed: {exc}")
        screening = ScreeningOutcome.from_screen_result(screen)

        pool = await _render_pool_snapshot(ctx)
        factory = PaperAgentDepsFactory(
            sub_extraction_llm=llm_for_ingestion,
            screener_llm=llm_for_ingestion,
            corpus_fetcher=corpus_fetcher or fetch_paper_corpus,
            pool_snapshot=pool,
            model=ingestion_model,
        )
        try:
            result = await run_ingestion_subagent(arxiv_id, factory, screening=screening)
        except Exception as exc:  # noqa: BLE001 — surface subagent failures via DispatchOutcome
            return DispatchOutcome(error=f"ingestion subagent failed: {type(exc).__name__}: {exc}")

        if result.error_reason or not result.techniques:
            return DispatchOutcome(
                error=(f"ingestion produced no techniques (reason={result.error_reason!r})"),
            )

        buffer = {
            "finalized": True,
            "arxiv_id": arxiv_id,
            "paper_title": result.paper_title,
            "paper_level_summary": result.paper_level_summary,
            "screening": screening.model_dump(mode="json"),
            "extraction_caveats": list(result.extraction_caveats),
            "techniques": [t.model_dump(mode="json") for t in result.techniques],
        }
        await ctx.artifact_store.put(
            technique_buffer_key_fn(arxiv_id),
            json.dumps(buffer, indent=2).encode("utf-8"),
        )
        del paper  # held for runbook context; the registration handler re-reads
        handle = JobHandle(plugin="inline", handle=f"inline-paper-ingest-{arxiv_id}")
        return DispatchOutcome(submitted_handles=[handle])

    return _dispatch


async def _render_pool_snapshot(ctx: DispatchContext) -> Any:
    """Build a :class:`PoolSnapshot` of the technique pool for the subagent.

    Drains :meth:`MetadataStore.list_techniques` (cursor-paginated per
    DEC-035 #1) into the canonical names + a Markdown summary the prompt
    template drops in verbatim (so the agent does not re-emit a technique
    already in the pool).
    """
    from smai_inline_agents.ingestion import PoolSnapshot  # noqa: PLC0415

    names: list[str] = []
    lines: list[str] = []
    cursor: str | None = None
    while True:
        page = await ctx.metadata_store.list_techniques(limit=100, cursor=cursor)
        for technique in page.items:
            names.append(technique.name)
            lines.append(
                f"- {technique.id} | {technique.name} | "
                f"category={technique.category} | standard={technique.standard}"
            )
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    if not lines:
        return PoolSnapshot.empty()
    return PoolSnapshot(technique_names=names[:2000], summary_markdown="\n".join(lines)[:20000])


def make_dispatch_paper_register(
    *,
    technique_buffer_key_for: Callable[[str], str] | None = None,
) -> Callable[[DispatchContext], Awaitable[DispatchOutcome]]:
    """``registered`` on-entry dispatch — paper.dispatch_register.

    Per `03` §5.3 / `08` §5.5: inline; reads the planner's finalized
    technique buffer, projects each draft technique to a
    :class:`TechniqueRef` with a :class:`PaperFidelityAnchor` pointing
    at this paper, and writes them via
    :meth:`MetadataStore.upsert_technique` (idempotent per `07`
    §5.3). The paper's state was already CAS'd to ``registered`` by
    the engine when the ``planning → registered`` edge fired; this
    handler completes the paper-side registration work.

    Per DEC-032: NO CG creation — paper ingestion produces
    ``TechniqueRef``s only. The proposal pipeline owns CG creation.
    """
    technique_buffer_key_fn = technique_buffer_key_for or _default_technique_buffer_key

    async def _dispatch(ctx: DispatchContext) -> DispatchOutcome:
        arxiv_id = ctx.entity_id
        key = technique_buffer_key_fn(arxiv_id)
        try:
            payload = await ctx.artifact_store.get(key)
        except ArtifactNotFound:
            return DispatchOutcome(error=f"technique buffer artifact missing at {key!r}")
        try:
            buffer = cast(dict[str, Any], json.loads(payload))
        except (ValueError, UnicodeDecodeError) as exc:
            return DispatchOutcome(error=f"technique buffer parse failed: {exc}")
        if not buffer.get("finalized"):
            return DispatchOutcome(error="technique buffer not finalized")

        techniques_raw = buffer.get("techniques")
        if not isinstance(techniques_raw, list) or not techniques_raw:
            return DispatchOutcome(
                error=(
                    "technique buffer has no techniques — paper ingestion produces "
                    "TechniqueRefs only; an empty buffer means there is nothing to register"
                ),
            )
        paper_title = cast(str | None, buffer.get("paper_title")) or None
        # Project + upsert each technique. Per DEC-030 the registration
        # could run inside a single ``MetadataStore.transaction``, but
        # ``upsert_technique`` is not on the transaction Protocol (only
        # on the top-level :class:`MetadataStore`); we call it directly.
        # Idempotent per `07` §5.3 — re-running on a registration retry
        # is safe.
        from smai_inline_agents.planner import TechniqueDescription  # noqa: PLC0415

        for raw in cast(list[dict[str, Any]], techniques_raw):
            try:
                desc = TechniqueDescription.model_validate(raw)
            except Exception as exc:  # noqa: BLE001 — malformed buffer entry -> dispatch error
                return DispatchOutcome(
                    error=f"technique buffer entry failed TechniqueDescription validation: {exc}"
                )
            ref = _technique_description_to_paper_ref(
                desc=desc,
                paper_arxiv_id=arxiv_id,
                paper_title=paper_title,
            )
            await ctx.metadata_store.upsert_technique(ref)
        return DispatchOutcome()

    return _dispatch


def _technique_description_to_paper_ref(
    *,
    desc: Any,  # smai_inline_agents.planner.TechniqueDescription
    paper_arxiv_id: str,
    paper_title: str | None = None,
) -> TechniqueRef:
    """Project an ingestion :class:`TechniqueDescription` to a paper-anchored
    :class:`TechniqueRef`.

    The ingestion subagent emits ``context_kind="paper_extract"``
    techniques (``upstream_requirements §1``), so the ref always carries
    a :class:`PaperFidelityAnchor` pointing at this paper and is
    non-standard. The :class:`TechniqueDescription` is the rich
    scientific surface; the methodology-wiring fields the
    :class:`TechniqueRef` needs but the description does not carry
    (``category`` / ``compatible_factor_types`` / extension points /
    parameter schema) default to the v1 additive baseline. A downstream
    proposal-pipeline planner that references the paper refines those at
    CG-construction time.
    """
    anchor = PaperFidelityAnchor(
        arxiv_id=paper_arxiv_id,
        doi=f"arxiv:{paper_arxiv_id}",
        title=paper_title,
    )
    return TechniqueRef(
        id=desc.name,
        name=desc.name,
        description=desc.summary,
        category="uncategorized",
        compatible_factor_types=["additive"],
        standard=False,
        fidelity_anchor=anchor,
        affects_extension_points=[],
        implies_controlled=[],
        parameter_schema=None,
        context_kind="paper_extract",
    )


# === Default key-resolver helpers ===========================================


def _default_paper_text_key(arxiv_id: str) -> str:
    return PAPER_TEXT_KEY_TEMPLATE.format(arxiv_id=arxiv_id)


def _default_figures_key(arxiv_id: str) -> str:
    return FIGURES_KEY_TEMPLATE.format(arxiv_id=arxiv_id)


def _default_expanded_tex_key(arxiv_id: str) -> str:
    return EXPANDED_TEX_KEY_TEMPLATE.format(arxiv_id=arxiv_id)


def _default_screen_result_key(arxiv_id: str) -> str:
    return SCREEN_RESULT_KEY_TEMPLATE.format(arxiv_id=arxiv_id)


def _default_technique_buffer_key(arxiv_id: str) -> str:
    return TECHNIQUE_BUFFER_KEY_TEMPLATE.format(arxiv_id=arxiv_id)


# === Spec factory ===========================================================


def build_paper_ingestion_spec(
    *,
    llm_for_screener: LlmProvider,
    llm_for_ingestion: LlmProvider,
    fetcher: PaperFetcher | None = None,
    max_screening_attempts: int = 1,
    max_planning_attempts: int = 1,
    paper_text_key_for: Callable[[str], str] | None = None,
    figures_key_for: Callable[[str], str] | None = None,
    expanded_tex_key_for: Callable[[str], str] | None = None,
    screen_result_key_for: Callable[[str], str] | None = None,
    technique_buffer_key_for: Callable[[str], str] | None = None,
    pool_limit: int = POOL_PAPER_INGESTION_LIMIT,
    pool_priority: int = POOL_PAPER_INGESTION_PRIORITY,
    ingestion_corpus_fetcher: Any = None,
    ingestion_model: Any = None,
) -> PipelineSpec:
    """Build the SMAI paper-ingestion :class:`PipelineSpec`.

    Args:
        llm_for_screener: :class:`LlmProvider` for the ``screener``
            role (the ``screening`` state's single-call agent).
        llm_for_ingestion: :class:`LlmProvider` for the ``ingestion``
            role — drives the Paper Agent's in-tool ``structured_call``
            sub-extractions (``search_paper`` / ``search_literature``)
            and the internal screener on deep-recursion.
        fetcher: Optional :class:`PaperFetcher`; defaults to
            :class:`ArxivLatexFetcher`. Tests inject a fake.
        max_screening_attempts: Per `08` §5.2 retry budget for the
            screening stage. v1 default 1 (one retry beyond initial).
        max_planning_attempts: Per `08` §5.2 retry budget for the
            planning stage. v1 default 1.
        paper_text_key_for / figures_key_for / expanded_tex_key_for
            / screen_result_key_for / technique_buffer_key_for:
            Optional ArtifactStore key template overrides; defaults are
            the ``papers/{arxiv_id}/...`` shape per the v1 layout.
        pool_limit: ``paper_ingestion`` pool limit. Default 2 per
            DEC-034 #4.
        pool_priority: ``paper_ingestion`` pool priority. Default 10
            per DEC-034 #4 (lowest of the four pools).
        ingestion_corpus_fetcher: Test-only seam — the LaTeX-source-first
            corpus fetcher the ingestion subagent uses. Defaults to
            :func:`smai_inline_agents.ingestion.fetch_paper_corpus`
            (production); tests inject a fake that returns a canned
            :class:`PaperCorpus` to keep the planning stage offline.
        ingestion_model: Test-only seam — the PydanticAI model override
            for the Paper Agent. Production leaves ``None`` (the agent
            uses its role-resolved model); tests pass a ``FunctionModel``.

    Returns:
        A :class:`PipelineSpec` with ``entity_kind="paper"`` ready for
        registration via
        :func:`smai_orchestrator.runtime.register_pipeline_spec` (or
        :func:`register_paper_ingestion_pipeline` for the convenience
        wrapper).
    """
    paper_text_key_fn = paper_text_key_for or _default_paper_text_key
    screen_result_key_fn = screen_result_key_for or _default_screen_result_key
    technique_buffer_key_fn = technique_buffer_key_for or _default_technique_buffer_key

    states: list[StateDef] = [
        StateDef(name="submitted"),
        StateDef(
            name="fetching",
            on_entry_dispatch=DispatchAction(
                name="paper.dispatch_fetch_latex",
                handler=make_dispatch_paper_fetch(
                    fetcher=fetcher,
                    paper_text_key_for=paper_text_key_fn,
                    figures_key_for=figures_key_for or _default_figures_key,
                    expanded_tex_key_for=expanded_tex_key_for or _default_expanded_tex_key,
                ),
                pool=POOL_PAPER_INGESTION,
                handle_field="fetcher_job_handle",
            ),
        ),
        StateDef(
            name="screening",
            on_entry_dispatch=DispatchAction(
                name="paper.dispatch_screener",
                handler=make_dispatch_paper_screener(
                    llm_for_screener=llm_for_screener,
                    paper_text_key_for=paper_text_key_fn,
                    screen_result_key_for=screen_result_key_fn,
                ),
                pool=POOL_PAPER_INGESTION,
                handle_field="screener_job_handle",
                # Round 10: engine bumps ``screening_attempt`` on step-1
                # CAS; synthesized terminal fires on dispatch failure
                # once the budget is exhausted. Pre-round-10 the counter
                # was declared but never incremented (no handler-side
                # bump) so the manual terminal gate
                # ``_make_gate_screening_failed_terminal`` never fired.
                retry_policy=RetryPolicy(
                    attempt_counter_field="screening_attempt",
                    max_attempts=max_screening_attempts,
                    on_exhaustion_target_state="failed",
                    on_exhaustion_reason="screening retry budget exhausted; terminal",
                ),
            ),
        ),
        StateDef(
            name="planning",
            on_entry_dispatch=DispatchAction(
                name="paper.dispatch_ingestion_subagent",
                handler=make_dispatch_paper_ingestion_subagent(
                    llm_for_ingestion=llm_for_ingestion,
                    screen_result_key_for=screen_result_key_fn,
                    technique_buffer_key_for=technique_buffer_key_fn,
                    corpus_fetcher=ingestion_corpus_fetcher,
                    ingestion_model=ingestion_model,
                ),
                pool=POOL_PAPER_INGESTION,
                handle_field="planner_job_handle",
                # Same as screening: pre-round-10 the counter never
                # incremented so the manual terminal never fired.
                retry_policy=RetryPolicy(
                    attempt_counter_field="planning_attempt",
                    max_attempts=max_planning_attempts,
                    on_exhaustion_target_state="failed",
                    on_exhaustion_reason="planning retry budget exhausted; terminal",
                ),
            ),
        ),
        StateDef(
            name="registered",
            is_terminal=True,
            on_entry_dispatch=DispatchAction(
                name="paper.dispatch_register",
                handler=make_dispatch_paper_register(
                    technique_buffer_key_for=technique_buffer_key_fn,
                ),
                pool=POOL_PAPER_INGESTION,
            ),
        ),
        # ``partial`` is non-terminal but has no outgoing edges and no
        # scheduling query (see module docstring's "spec ambiguities
        # resolved" note on the partial parking spot).
        StateDef(name="partial"),
        StateDef(name="rejected", is_terminal=True),
        StateDef(name="failed", is_terminal=True),
    ]

    edges: list[EdgeDef] = [
        # ``submitted`` outgoing — content_already_extracted FIRST so
        # promoted partials short-circuit fetch.
        EdgeDef(
            name="paper.submitted → screening (content already extracted)",
            from_state="submitted",
            target_state="screening",
            gate_rule=_make_gate_content_already_extracted(
                paper_text_key_for=paper_text_key_fn,
            ),
            fires_on="dispatch_time",
        ),
        EdgeDef(
            name="paper.submitted → fetching",
            from_state="submitted",
            target_state="fetching",
            gate_rule=_make_gate_dispatch_fetch_ready(),
            fires_on="dispatch_time",
        ),
        # ``fetching`` outgoing — success path. Round 10: the fetch
        # retry-exhausted edge was a placeholder (gate always returned
        # advance=False); removed since no counter on
        # :class:`PaperRecord` tracks fetch attempts. Deployments
        # needing fetch retries can declare a :class:`RetryPolicy` on
        # the ``fetching`` dispatch action (would require adding a
        # ``fetch_attempt`` field to PaperRecord first).
        EdgeDef(
            name="paper.fetching → screening",
            from_state="fetching",
            target_state="screening",
            gate_rule=_make_gate_fetch_complete(
                paper_text_key_for=paper_text_key_fn,
            ),
            fires_on="dispatch_time",
        ),
        # ``screening`` outgoing — pass first, reject second. The
        # retry-exhausted terminal is engine-synthesized from the
        # screening-state's :class:`RetryPolicy` (round 10).
        EdgeDef(
            name="paper.screening → planning (screener accept)",
            from_state="screening",
            target_state="planning",
            gate_rule=_make_gate_screener_decision(
                expected_decision="accept",
                screen_result_key_for=screen_result_key_fn,
            ),
            fires_on="dispatch_time",
        ),
        EdgeDef(
            name="paper.screening → rejected (screener reject)",
            from_state="screening",
            target_state="rejected",
            gate_rule=_make_gate_screener_decision(
                expected_decision="reject",
                screen_result_key_for=screen_result_key_fn,
            ),
            fires_on="dispatch_time",
        ),
        # ``planning`` outgoing — success path. Retry-exhausted terminal
        # is engine-synthesized from the planning-state's :class:`RetryPolicy`.
        EdgeDef(
            name="paper.planning → registered (techniques finalized)",
            from_state="planning",
            target_state="registered",
            gate_rule=_make_gate_techniques_finalized(
                technique_buffer_key_for=technique_buffer_key_fn,
            ),
            fires_on="dispatch_time",
        ),
    ]

    pools: list[ConcurrencyPool] = [
        ConcurrencyPool(
            name=POOL_PAPER_INGESTION,
            limit=pool_limit,
            priority=pool_priority,
        ),
    ]

    scheduling_queries: dict[str, SchedulingQueryRef] = {
        "submitted": SchedulingQueryRef(
            name="get_ready_for_paper_fetch",
            fn=_drain_to_list("get_ready_for_paper_fetch"),
        ),
        "fetching": SchedulingQueryRef(
            name="get_in_flight_paper_fetch",
            fn=_drain_to_list("get_in_flight_paper_fetch"),
        ),
        "screening": SchedulingQueryRef(
            name="get_in_flight_paper_screen",
            fn=_drain_to_list("get_in_flight_paper_screen"),
        ),
        "planning": SchedulingQueryRef(
            name="get_in_flight_paper_plan",
            fn=_drain_to_list("get_in_flight_paper_plan"),
        ),
        # ``partial`` has no scheduling query — the engine never visits
        # a partial paper. Promotion is synchronous via
        # :meth:`MetadataStore.transition_paper_state` from
        # ``papers_service.promote_partial`` (per the module docstring
        # / `07-plugin-interfaces.md` §5.6.5 docstring).
    }

    return PipelineSpec(
        name=PAPER_INGESTION_SPEC_NAME,
        entity_kind="paper",
        initial_state="submitted",
        states=states,
        edges=edges,
        pools=pools,
        scheduling_queries=scheduling_queries,
    )


def _drain_to_list(method_name: str) -> Callable[[Any], Awaitable[list[Any]]]:
    """Compose a :data:`SchedulingQueryRef.fn` callable that drains pages.

    Mirrors :func:`smai_orchestrator.specs.proposal._drain_to_list` —
    duplicated here to keep the paper-ingestion spec independent of
    the proposal spec's import surface.
    """

    async def _fn(store: Any) -> list[Any]:
        method = getattr(store, method_name)
        out: list[Any] = []
        cursor: str | None = None
        while True:
            page = await method(limit=100, cursor=cursor)
            out.extend(page.items)
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        return out

    _fn.__qualname__ = f"_drain_to_list[{method_name}].fn"
    return _fn


# === Public registration helper =============================================


def register_paper_ingestion_pipeline(
    *,
    llm_for_screener: LlmProvider,
    llm_for_ingestion: LlmProvider,
    fetcher: PaperFetcher | None = None,
    max_screening_attempts: int = 1,
    max_planning_attempts: int = 1,
    paper_text_key_for: Callable[[str], str] | None = None,
    figures_key_for: Callable[[str], str] | None = None,
    expanded_tex_key_for: Callable[[str], str] | None = None,
    screen_result_key_for: Callable[[str], str] | None = None,
    technique_buffer_key_for: Callable[[str], str] | None = None,
    ingestion_corpus_fetcher: Any = None,
    ingestion_model: Any = None,
) -> PipelineSpec:
    """Construct + register the paper-ingestion pipeline-spec.

    Convenience wrapper around :func:`build_paper_ingestion_spec` +
    :func:`smai_orchestrator.runtime.register_pipeline_spec`. Returns
    the constructed spec.
    """
    spec = build_paper_ingestion_spec(
        llm_for_screener=llm_for_screener,
        llm_for_ingestion=llm_for_ingestion,
        fetcher=fetcher,
        max_screening_attempts=max_screening_attempts,
        max_planning_attempts=max_planning_attempts,
        paper_text_key_for=paper_text_key_for,
        figures_key_for=figures_key_for,
        expanded_tex_key_for=expanded_tex_key_for,
        screen_result_key_for=screen_result_key_for,
        technique_buffer_key_for=technique_buffer_key_for,
        ingestion_corpus_fetcher=ingestion_corpus_fetcher,
        ingestion_model=ingestion_model,
    )
    register_pipeline_spec(spec)
    return spec


__all__ = [
    "ArxivLatexFetcher",
    "EXPANDED_TEX_KEY_TEMPLATE",
    "FIGURES_KEY_TEMPLATE",
    "FetchedPaper",
    "LATEX_BUNDLE_KEY_TEMPLATE",
    "METHOD_EXTRACTION_KEY_TEMPLATE",
    "PAPER_INGESTION_SPEC_NAME",
    "PAPER_TEXT_KEY_TEMPLATE",
    "POOL_PAPER_INGESTION",
    "POOL_PAPER_INGESTION_LIMIT",
    "POOL_PAPER_INGESTION_PRIORITY",
    "PaperFetchError",
    "PaperFetcher",
    "SCREEN_RESULT_KEY_TEMPLATE",
    "TECHNIQUE_BUFFER_KEY_TEMPLATE",
    "build_paper_ingestion_spec",
    "make_dispatch_paper_fetch",
    "make_dispatch_paper_ingestion_subagent",
    "make_dispatch_paper_register",
    "make_dispatch_paper_screener",
    "register_paper_ingestion_pipeline",
]
