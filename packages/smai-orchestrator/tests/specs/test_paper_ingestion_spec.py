"""Tests for the paper-ingestion pipeline-spec — Task 3.E2.

Two surfaces:

* Spec-structure tests against :func:`build_paper_ingestion_spec`'s
  return value (no engine, no plugins).
* End-to-end driving the spec through the worker loop against a stub
  :class:`MetadataStore` (in-memory SQLite) + stub :class:`ArtifactStore`
  (LocalFs) + stub :class:`LlmProvider` + fake fetcher. Verifies state
  transitions, gate-rule firing, and ``TechniqueRef`` registration on
  ``registered`` per DEC-032.
"""

from __future__ import annotations

import json  # noqa: F401  # used by test bodies that pre-stage JSON payloads
from pathlib import Path

import pytest
from _e2_fakes import (  # type: ignore[import-not-found]
    InProcessFakeFetcher,
    build_finalized_paper_buffer_payload,
    make_ingestion_corpus_fetcher,
    make_ingestion_function_model,
    make_paper_record,
    make_screener_response,
)
from _specs_fakes import StubLlmProvider  # type: ignore[import-not-found]
from smai_artifacts_localfs import LocalFsStore
from smai_orchestrator.engine.config import EngineConfig
from smai_orchestrator.specs.paper_ingestion import (
    PAPER_INGESTION_SPEC_NAME,
    PAPER_TEXT_KEY_TEMPLATE,
    POOL_PAPER_INGESTION,
    POOL_PAPER_INGESTION_LIMIT,
    POOL_PAPER_INGESTION_PRIORITY,
    SCREEN_RESULT_KEY_TEMPLATE,
    TECHNIQUE_BUFFER_KEY_TEMPLATE,
    ArxivLatexFetcher,
    PaperFetchError,
    build_paper_ingestion_spec,
)
from smai_orchestrator.worker.loop import run_worker_cycle
from smai_store_sqlite import SqliteStore


@pytest.fixture
def paper_spec(tmp_path: Path):  # type: ignore[no-untyped-def]
    del tmp_path
    stub_llm = StubLlmProvider([])
    return build_paper_ingestion_spec(
        llm_for_screener=stub_llm,  # type: ignore[arg-type]
        llm_for_ingestion=stub_llm,  # type: ignore[arg-type]
        fetcher=InProcessFakeFetcher(),
    )


# === Structure tests ========================================================


def test_paper_spec_name_and_entity_kind(paper_spec) -> None:  # type: ignore[no-untyped-def]
    assert paper_spec.name == PAPER_INGESTION_SPEC_NAME
    assert paper_spec.entity_kind == "paper"
    assert paper_spec.initial_state == "submitted"


def test_paper_spec_states_match_03_section_5_1(paper_spec) -> None:  # type: ignore[no-untyped-def]
    """Per `03-state-machine.md` §5.1 — eight states.

    submitted, fetching, screening, planning, registered, partial,
    rejected, failed.
    """
    state_names = {s.name for s in paper_spec.states}
    assert state_names == {
        "submitted",
        "fetching",
        "screening",
        "planning",
        "registered",
        "partial",
        "rejected",
        "failed",
    }


def test_paper_spec_terminal_states(paper_spec) -> None:  # type: ignore[no-untyped-def]
    """`registered`, `rejected`, `failed` are terminals; `partial` is non-terminal."""
    terminals = {s.name for s in paper_spec.states if s.is_terminal}
    assert terminals == {"registered", "rejected", "failed"}
    partial = next(s for s in paper_spec.states if s.name == "partial")
    assert partial.is_terminal is False


def test_paper_spec_pool_per_dec_034_4(paper_spec) -> None:  # type: ignore[no-untyped-def]
    """Single `paper_ingestion` pool per DEC-034 #4 — limit 2, priority 10."""
    pool_names = {p.name for p in paper_spec.pools}
    assert pool_names == {POOL_PAPER_INGESTION}
    pool = next(p for p in paper_spec.pools if p.name == POOL_PAPER_INGESTION)
    assert pool.limit == POOL_PAPER_INGESTION_LIMIT
    assert pool.priority == POOL_PAPER_INGESTION_PRIORITY


def test_paper_spec_edges_per_03_section_5_2(paper_spec) -> None:  # type: ignore[no-untyped-def]
    """Edges per `03` §5.2 (table) — collapsed for the inline-dispatch pattern.

    The canonical `03` §5.2 lists ``job_succeeded`` / ``job_failed``
    triggers for the inline-dispatch edges; per the spec module's
    "spec ambiguities resolved" note (the inline-dispatch retraction
    that mirrors :mod:`smai_orchestrator.specs.proposal`'s pattern) the
    spec uses ``dispatch_time`` for every edge.

    Round 10: the manual ``*_failed (retry exhausted)`` edges off
    ``fetching`` / ``screening`` / ``planning`` are gone — the engine
    synthesizes them from the dispatch actions' :class:`RetryPolicy`
    declarations (where one is declared). The ``fetching → failed``
    edge was a placeholder pre-round-10 (gate always returned
    advance=False; no counter on PaperRecord) and is dropped entirely.
    """
    by_pair = {(e.from_state, e.target_state) for e in paper_spec.edges}
    # ``submitted`` outgoing — content-already-extracted short-circuit + fetch.
    assert ("submitted", "screening") in by_pair
    assert ("submitted", "fetching") in by_pair
    # ``fetching`` outgoing — success path only.
    assert ("fetching", "screening") in by_pair
    # ``screening`` outgoing — accept / reject. Retry-exhausted terminal
    # is engine-synthesized off the dispatch action's RetryPolicy.
    assert ("screening", "planning") in by_pair
    assert ("screening", "rejected") in by_pair
    # ``planning`` outgoing — success. Retry-exhausted terminal is
    # engine-synthesized off the dispatch action's RetryPolicy.
    assert ("planning", "registered") in by_pair

    # Confirm the synthesis hooks: screening + planning dispatch
    # actions declare :class:`RetryPolicy` targeting ``failed``.
    screening = next(s for s in paper_spec.states if s.name == "screening")
    assert screening.on_entry_dispatch is not None
    screening_policy = screening.on_entry_dispatch.retry_policy
    assert screening_policy is not None
    assert screening_policy.on_exhaustion_target_state == "failed"
    planning = next(s for s in paper_spec.states if s.name == "planning")
    assert planning.on_entry_dispatch is not None
    planning_policy = planning.on_entry_dispatch.retry_policy
    assert planning_policy is not None
    assert planning_policy.on_exhaustion_target_state == "failed"


def test_paper_spec_partial_has_no_outgoing_edges(paper_spec) -> None:  # type: ignore[no-untyped-def]
    """`partial → submitted` is NOT a spec-driven edge per the module
    docstring's "spec ambiguities resolved" framing.

    Promotion is a synchronous user-driven write through
    :meth:`MetadataStore.transition_paper_state` from
    :class:`PapersService.promote_partial`. The pipeline-spec declares
    ``partial`` as a non-terminal state with no outgoing edges; the
    next worker cycle's ``submitted`` discovery picks the paper up
    after promotion.
    """
    partial_outgoing = [e for e in paper_spec.edges if e.from_state == "partial"]
    assert partial_outgoing == []


def test_paper_spec_scheduling_queries_declared(paper_spec) -> None:  # type: ignore[no-untyped-def]
    """Scheduling queries for the four non-terminal in-progress states.

    `partial` has no scheduling query — the engine never visits a
    partial paper.
    """
    keys = set(paper_spec.scheduling_queries.keys())
    assert keys == {"submitted", "fetching", "screening", "planning"}


def test_paper_spec_in_progress_states_have_dispatch(paper_spec) -> None:  # type: ignore[no-untyped-def]
    """fetching / screening / planning each have an on-entry dispatch.

    The ``registered`` terminal also carries one (the registration
    handler) per the spec module's docstring on the "registration as
    on-entry of registered" deviation from `03` §5.3's "edge dispatch
    handler" framing.
    """
    fetching = next(s for s in paper_spec.states if s.name == "fetching")
    assert fetching.on_entry_dispatch is not None
    assert fetching.on_entry_dispatch.handle_field == "fetcher_job_handle"

    screening = next(s for s in paper_spec.states if s.name == "screening")
    assert screening.on_entry_dispatch is not None
    assert screening.on_entry_dispatch.handle_field == "screener_job_handle"

    planning = next(s for s in paper_spec.states if s.name == "planning")
    assert planning.on_entry_dispatch is not None
    assert planning.on_entry_dispatch.handle_field == "planner_job_handle"

    registered = next(s for s in paper_spec.states if s.name == "registered")
    assert registered.is_terminal is True
    assert registered.on_entry_dispatch is not None
    # Registration has no handle field (it's the final write).
    assert registered.on_entry_dispatch.handle_field is None


def test_paper_spec_partial_has_no_on_entry_dispatch(paper_spec) -> None:  # type: ignore[no-untyped-def]
    """`partial` is parked indefinitely — no dispatch fires when an
    entity lands here from another paper's enrichment step.
    """
    partial = next(s for s in paper_spec.states if s.name == "partial")
    assert partial.on_entry_dispatch is None


def test_paper_spec_registered_dispatch_persists_techniques(paper_spec) -> None:  # type: ignore[no-untyped-def]
    """Registration handler is named per the spec contract."""
    registered = next(s for s in paper_spec.states if s.name == "registered")
    assert registered.on_entry_dispatch is not None
    assert "register" in registered.on_entry_dispatch.name


# === End-to-end driving =====================================================


@pytest.fixture
async def sqlite_store():  # type: ignore[no-untyped-def]
    """In-memory SqliteStore per test."""
    store = SqliteStore("sqlite+aiosqlite:///:memory:")
    await store.migrate()
    try:
        yield store
    finally:
        await store.dispose()


@pytest.fixture
def localfs(tmp_path: Path) -> LocalFsStore:
    return LocalFsStore(tmp_path / "artifacts")


@pytest.mark.asyncio
async def test_paper_round_trip_subagent_registers_paper_extract_no_double_screen(  # type: ignore[no-untyped-def]
    sqlite_store,
    localfs: LocalFsStore,
) -> None:
    """End-to-end: submitted → fetching → screening → planning → registered
    through the ingestion subagent (Step 3, Sub-PR B).

    The ``screening`` state runs the screener ONCE; the ``planning`` state
    reuses that verdict via ``run_ingestion_subagent(screening=...)`` so
    the paper is NOT re-screened. The ``ingestion`` LLM (the in-tool
    sub-extraction + internal screener provider) must therefore never be
    called: ``ingestion_llm.calls`` stays empty. Asserts a
    ``context_kind='paper_extract'`` :class:`TechniqueRef` is registered
    with a :class:`PaperFidelityAnchor` matching the paper.
    """
    arxiv_id = "2401.99999"
    fetcher = InProcessFakeFetcher()
    screener_llm = StubLlmProvider([make_screener_response(decision="accept")])
    # If the planning handler re-screened or ran a sub-extraction it would
    # call this provider, which raises on an empty queue — proving the
    # no-double-screen contract by construction.
    ingestion_llm = StubLlmProvider([])

    spec = build_paper_ingestion_spec(
        llm_for_screener=screener_llm,  # type: ignore[arg-type]
        llm_for_ingestion=ingestion_llm,  # type: ignore[arg-type]
        fetcher=fetcher,
        ingestion_corpus_fetcher=make_ingestion_corpus_fetcher(arxiv_id=arxiv_id),
        ingestion_model=make_ingestion_function_model(source_arxiv_id=arxiv_id),
    )
    config = EngineConfig(supervisor_enabled=False)

    paper = make_paper_record(arxiv_id=arxiv_id, state="submitted")
    await sqlite_store.create_paper(paper)

    final_state: str | None = None
    for _ in range(10):
        await run_worker_cycle(
            spec=spec.engine_spec(),
            metadata_store=sqlite_store,
            artifact_store=localfs,  # type: ignore[arg-type]
            compute=_NoComputeStub(),  # type: ignore[arg-type]
            llm_providers=None,
            config=config,
        )
        rec = await sqlite_store.get_paper(arxiv_id)
        assert rec is not None
        if rec.state in {"registered", "rejected", "failed"}:
            final_state = rec.state
            break

    assert final_state == "registered", f"got {final_state}"

    # Fetcher was actually called — the ``submitted → fetching`` edge fired.
    assert fetcher.fetch_log == [arxiv_id]
    # Paper text + screen result + technique buffer artifacts landed.
    assert await localfs.exists(PAPER_TEXT_KEY_TEMPLATE.format(arxiv_id=arxiv_id))
    assert await localfs.exists(SCREEN_RESULT_KEY_TEMPLATE.format(arxiv_id=arxiv_id))
    assert await localfs.exists(TECHNIQUE_BUFFER_KEY_TEMPLATE.format(arxiv_id=arxiv_id))

    # No double-screen: the screener fired exactly once (screening state);
    # the ingestion provider was never touched (no re-screen, no
    # sub-extraction in this fixture).
    assert len(screener_llm.calls) == 1
    assert ingestion_llm.calls == []

    # ≥ 1 paper_extract ``TechniqueRef`` registered, paper-anchored.
    techniques = await sqlite_store.list_techniques_for_paper(arxiv_id)
    assert len(techniques.items) >= 1
    technique = techniques.items[0]
    assert technique.context_kind == "paper_extract"
    assert technique.fidelity_anchor is not None
    assert technique.fidelity_anchor.kind == "paper"
    assert technique.fidelity_anchor.arxiv_id == arxiv_id


@pytest.mark.asyncio
async def test_paper_screener_rejection_routes_to_rejected(  # type: ignore[no-untyped-def]
    sqlite_store,
    localfs: LocalFsStore,
) -> None:
    """Screener rejection routes the paper to the ``rejected`` terminal.

    Asserts no ``TechniqueRef`` is registered (the subagent never fires).
    """
    arxiv_id = "2401.88888"
    fetcher = InProcessFakeFetcher()
    screener_llm = StubLlmProvider(
        [
            make_screener_response(
                decision="reject",
                rejection_reason="not an empirical comparison paper",
            )
        ]
    )
    ingestion_llm = StubLlmProvider([])

    spec = build_paper_ingestion_spec(
        llm_for_screener=screener_llm,  # type: ignore[arg-type]
        llm_for_ingestion=ingestion_llm,  # type: ignore[arg-type]
        fetcher=fetcher,
    )
    config = EngineConfig(supervisor_enabled=False)

    paper = make_paper_record(arxiv_id=arxiv_id, state="submitted")
    await sqlite_store.create_paper(paper)

    final_state: str | None = None
    for _ in range(8):
        await run_worker_cycle(
            spec=spec.engine_spec(),
            metadata_store=sqlite_store,
            artifact_store=localfs,  # type: ignore[arg-type]
            compute=_NoComputeStub(),  # type: ignore[arg-type]
            llm_providers=None,
            config=config,
        )
        rec = await sqlite_store.get_paper(arxiv_id)
        assert rec is not None
        if rec.state in {"registered", "rejected", "failed"}:
            final_state = rec.state
            break

    assert final_state == "rejected"

    # No techniques registered.
    techniques = await sqlite_store.list_techniques_for_paper(arxiv_id)
    assert len(techniques.items) == 0


@pytest.mark.asyncio
async def test_paper_content_already_extracted_short_circuits_fetch(  # type: ignore[no-untyped-def]
    sqlite_store,
    localfs: LocalFsStore,
) -> None:
    """When ``papers/{arxiv_id}/extracted/paper_text.json`` already
    exists at the time the paper enters ``submitted``, the
    ``submitted → screening`` edge wins (per `03` §5.2 edge ordering)
    and the fetcher does NOT fire.

    Models the ``partial → submitted`` promotion path: the partial
    paper's content was extracted by another paper's enrichment, so
    when the user promotes, the fetch step short-circuits.
    """
    arxiv_id = "2401.77777"
    # Pre-stage the paper-text artifact — what enrichment would have
    # written when this paper was created as a `partial`.
    await localfs.put(
        PAPER_TEXT_KEY_TEMPLATE.format(arxiv_id=arxiv_id),
        json.dumps(
            {
                "arxiv_id": arxiv_id,
                "title": "Pre-extracted Paper",
                "abstract": "Already extracted.",
                "authors": ["Other Author"],
                "paper_text": "Pre-extracted text from a prior enrichment step.",
            }
        ).encode("utf-8"),
    )

    fetcher = InProcessFakeFetcher()
    screener_llm = StubLlmProvider([make_screener_response(decision="accept")])
    ingestion_llm = StubLlmProvider([])

    spec = build_paper_ingestion_spec(
        llm_for_screener=screener_llm,  # type: ignore[arg-type]
        llm_for_ingestion=ingestion_llm,  # type: ignore[arg-type]
        fetcher=fetcher,
        ingestion_corpus_fetcher=make_ingestion_corpus_fetcher(arxiv_id=arxiv_id),
        ingestion_model=make_ingestion_function_model(source_arxiv_id=arxiv_id),
    )
    config = EngineConfig(supervisor_enabled=False)

    paper = make_paper_record(arxiv_id=arxiv_id, state="submitted")
    await sqlite_store.create_paper(paper)

    final_state: str | None = None
    for _ in range(10):
        await run_worker_cycle(
            spec=spec.engine_spec(),
            metadata_store=sqlite_store,
            artifact_store=localfs,  # type: ignore[arg-type]
            compute=_NoComputeStub(),  # type: ignore[arg-type]
            llm_providers=None,
            config=config,
        )
        rec = await sqlite_store.get_paper(arxiv_id)
        if rec is not None and rec.state in {"registered", "rejected", "failed"}:
            final_state = rec.state
            break

    assert final_state == "registered"
    # Fetcher was NOT invoked — the content-already-extracted gate
    # short-circuited.
    assert fetcher.fetch_log == []


@pytest.mark.asyncio
async def test_paper_techniques_finalized_gate_blocks_on_unfinalized_buffer(  # type: ignore[no-untyped-def]
    sqlite_store,
    localfs: LocalFsStore,
) -> None:
    """The `planning → registered` gate requires the buffer to be finalized.

    Pre-stage a buffer with ``finalized=False`` at the technique-buffer
    artifact key; assert the paper does not advance to ``registered``.
    The subagent's corpus fetch is wired to return ``None`` so that if
    the planning handler re-fires it fails fast (``fetch_failed``)
    without overwriting the pre-staged buffer — keeping the gate the
    thing under test.
    """
    arxiv_id = "2401.66666"
    payload = build_finalized_paper_buffer_payload(arxiv_id=arxiv_id)
    payload["finalized"] = False  # break the finalized invariant

    # Pre-stage paper text + screen result + buffer so the paper sits
    # in ``planning`` with the gate-readable artifacts present but the
    # buffer not finalized.
    await localfs.put(
        PAPER_TEXT_KEY_TEMPLATE.format(arxiv_id=arxiv_id),
        json.dumps(
            {
                "arxiv_id": arxiv_id,
                "title": "Test",
                "abstract": "test",
                "authors": [],
                "paper_text": "test text",
            }
        ).encode("utf-8"),
    )
    await localfs.put(
        SCREEN_RESULT_KEY_TEMPLATE.format(arxiv_id=arxiv_id),
        json.dumps({"decision": "accept", "summary": "test", "rejection_reason": None}).encode(
            "utf-8"
        ),
    )
    await localfs.put(
        TECHNIQUE_BUFFER_KEY_TEMPLATE.format(arxiv_id=arxiv_id),
        json.dumps(payload).encode("utf-8"),
    )

    async def _none_corpus(requested_arxiv_id: str):  # type: ignore[no-untyped-def]
        del requested_arxiv_id
        return None

    fetcher = InProcessFakeFetcher()
    screener_llm = StubLlmProvider([])  # already pre-staged
    ingestion_llm = StubLlmProvider([])

    spec = build_paper_ingestion_spec(
        llm_for_screener=screener_llm,  # type: ignore[arg-type]
        llm_for_ingestion=ingestion_llm,  # type: ignore[arg-type]
        fetcher=fetcher,
        ingestion_corpus_fetcher=_none_corpus,
        max_planning_attempts=10,  # generous so the retry-budget terminal doesn't fire
    )
    config = EngineConfig(supervisor_enabled=False)

    # Pre-stage paper directly in ``planning`` with a synthetic
    # planner_job_handle so the in-flight query returns it.
    from smai_core.plugins import JobHandle  # noqa: PLC0415

    paper = make_paper_record(arxiv_id=arxiv_id, state="planning")
    paper.planner_job_handle = JobHandle(plugin="inline", handle=f"inline-paper-plan-{arxiv_id}")
    await sqlite_store.create_paper(paper)

    # Drive one cycle and assert the paper does not advance to registered.
    await run_worker_cycle(
        spec=spec.engine_spec(),
        metadata_store=sqlite_store,
        artifact_store=localfs,  # type: ignore[arg-type]
        compute=_NoComputeStub(),  # type: ignore[arg-type]
        llm_providers=None,
        config=config,
    )
    rec = await sqlite_store.get_paper(arxiv_id)
    assert rec is not None
    assert rec.state != "registered"


@pytest.mark.asyncio
async def test_paper_registration_disambiguates_same_named_techniques(  # type: ignore[no-untyped-def]
    sqlite_store,
    localfs: LocalFsStore,
) -> None:
    """Two techniques in one paper sharing a name must NOT collide on the
    store PK and silently overwrite each other.

    The registration handler namespaces the ref id by paper
    (``{arxiv_id}:{name}``, cross-paper safety) and suffixes intra-paper
    name collisions (``...-2``). Pre-stage a finalized buffer carrying the
    same technique twice and assert two distinct paper-anchored
    ``TechniqueRef``s register.
    """
    arxiv_id = "2401.55555"
    payload = build_finalized_paper_buffer_payload(arxiv_id=arxiv_id, name="cutout")
    # Two techniques with the same name -> would clobber under id=name.
    payload["techniques"] = [payload["techniques"][0], payload["techniques"][0]]

    await localfs.put(
        PAPER_TEXT_KEY_TEMPLATE.format(arxiv_id=arxiv_id),
        json.dumps(
            {
                "arxiv_id": arxiv_id,
                "title": "Test",
                "abstract": "test",
                "authors": [],
                "paper_text": "test text",
            }
        ).encode("utf-8"),
    )
    await localfs.put(
        SCREEN_RESULT_KEY_TEMPLATE.format(arxiv_id=arxiv_id),
        json.dumps({"decision": "accept", "summary": "test", "rejection_reason": None}).encode(
            "utf-8"
        ),
    )
    await localfs.put(
        TECHNIQUE_BUFFER_KEY_TEMPLATE.format(arxiv_id=arxiv_id),
        json.dumps(payload).encode("utf-8"),
    )

    async def _none_corpus(requested_arxiv_id: str):  # type: ignore[no-untyped-def]
        del requested_arxiv_id
        return None

    spec = build_paper_ingestion_spec(
        llm_for_screener=StubLlmProvider([]),  # type: ignore[arg-type]
        llm_for_ingestion=StubLlmProvider([]),  # type: ignore[arg-type]
        fetcher=InProcessFakeFetcher(),
        ingestion_corpus_fetcher=_none_corpus,
        max_planning_attempts=10,
    )
    config = EngineConfig(supervisor_enabled=False)

    from smai_core.plugins import JobHandle  # noqa: PLC0415

    paper = make_paper_record(arxiv_id=arxiv_id, state="planning")
    paper.planner_job_handle = JobHandle(plugin="inline", handle=f"inline-paper-plan-{arxiv_id}")
    await sqlite_store.create_paper(paper)

    final_state: str | None = None
    for _ in range(10):
        await run_worker_cycle(
            spec=spec.engine_spec(),
            metadata_store=sqlite_store,
            artifact_store=localfs,  # type: ignore[arg-type]
            compute=_NoComputeStub(),  # type: ignore[arg-type]
            llm_providers=None,
            config=config,
        )
        rec = await sqlite_store.get_paper(arxiv_id)
        assert rec is not None
        if rec.state in {"registered", "rejected", "failed"}:
            final_state = rec.state
            break

    assert final_state == "registered", f"got {final_state}"

    techniques = await sqlite_store.list_techniques_for_paper(arxiv_id)
    ids = {t.id for t in techniques.items}
    # Both techniques survived (no silent overwrite) with distinct ids,
    # both paper-namespaced.
    assert ids == {f"{arxiv_id}:cutout", f"{arxiv_id}:cutout-2"}, ids
    assert all(t.context_kind == "paper_extract" for t in techniques.items)


# === Helpers ================================================================


class _NoComputeStub:
    """A :class:`Compute` stub for the inline-only paper-ingestion spec.

    Paper ingestion uses no external :class:`Compute` jobs (`08` §7);
    however the engine's :class:`DispatchContext` is constructed with
    a ``Compute`` value validated against the Protocol shape. We
    return ``"succeeded"`` by default for any phase-1 status query
    (mirroring the proposal-spec test stub) and raise on
    :meth:`submit`.
    """

    name = "no-compute"

    def __init__(self) -> None:
        from smai_core.plugins import ComputeCapabilities  # noqa: PLC0415

        self.capabilities = ComputeCapabilities(
            supports_gpu=False,
            max_timeout_seconds=3600,
        )

    async def submit(  # noqa: PLR0913
        self,
        image: str,
        command: list[str],
        env: dict[str, str],
        gpu: bool = False,
        timeout_seconds: int = 3600,
        **plugin_options: object,
    ) -> object:
        del image, command, env, gpu, timeout_seconds, plugin_options
        raise RuntimeError("_NoComputeStub.submit should not be called by paper ingestion")

    async def status(self, handle):  # type: ignore[no-untyped-def]
        from smai_core.plugins import JobStatus  # noqa: PLC0415

        del handle
        return JobStatus(
            state="succeeded",
            exit_code=0,
            started_at=None,
            finished_at=None,
            failure_reason=None,
        )

    async def logs(self, handle):  # type: ignore[no-untyped-def]
        del handle
        return ""

    async def cancel(self, handle):  # type: ignore[no-untyped-def]
        del handle

    async def stage_workspace(self, local_path):  # type: ignore[no-untyped-def]
        from smai_core.plugins import WorkspaceHandle  # noqa: PLC0415

        return WorkspaceHandle(plugin=self.name, handle=str(local_path))

    async def harvest_workspace(self, handle, local_path):  # type: ignore[no-untyped-def]
        del handle, local_path


# === ArxivLatexFetcher robustness (rate-limit backoff) =======================
#
# These exercise the real :class:`ArxivLatexFetcher` against a fake
# ``arxiv`` module injected into ``sys.modules`` (the fetcher does a local
# ``import arxiv`` inside ``fetch``). They stay offline and patch
# ``asyncio.sleep`` so the exponential backoff doesn't sleep for real.


class _FakeArxivResult:
    """Minimal stand-in for an ``arxiv.Result`` on the success path."""

    title = "Cutout: a fake paper"
    summary = "A canned abstract."

    class _Author:
        def __init__(self, name: str) -> None:
            self.name = name

    authors = [_Author("A. Author")]

    def download_source(self, dirpath: str) -> str:  # noqa: D401
        # No real download — return a path that does not exist so the
        # best-effort source-download branch logs + falls through to the
        # abstract-only ``paper_text`` (the fetcher tolerates this).
        return f"{dirpath}/source.tar.gz"


def _make_fake_arxiv_module(results_fn):  # type: ignore[no-untyped-def]
    """Build a fake ``arxiv`` module with a scripted ``Client.results``.

    ``results_fn`` is a zero-arg callable invoked once per ``client.results``
    call; it returns the result list or raises a fake arxiv exception.
    """
    import types  # noqa: PLC0415

    module = types.ModuleType("arxiv")

    class ArxivError(Exception):
        pass

    class HTTPError(ArxivError):
        def __init__(self, url: str = "u", retry: int = 0, status: int = 500) -> None:
            self.url = url
            self.retry = retry
            self.status = status
            super().__init__(f"HTTP {status}")

    class UnexpectedEmptyPageError(ArxivError):
        def __init__(self, url: str = "u", retry: int = 0, raw_feed: object = None) -> None:
            self.url = url
            self.retry = retry
            self.raw_feed = raw_feed
            super().__init__("empty page")

    class Search:
        def __init__(self, id_list=None) -> None:  # type: ignore[no-untyped-def]
            self.id_list = id_list or []

    class Client:
        # Records the constructor kwargs so a test can assert page_size=1.
        last_kwargs: dict[str, object] = {}

        def __init__(self, **kwargs: object) -> None:
            type(self).last_kwargs = dict(kwargs)

        def results(self, search):  # type: ignore[no-untyped-def]
            del search
            return results_fn()

    module.ArxivError = ArxivError  # type: ignore[attr-defined]
    module.HTTPError = HTTPError  # type: ignore[attr-defined]
    module.UnexpectedEmptyPageError = UnexpectedEmptyPageError  # type: ignore[attr-defined]
    module.Search = Search  # type: ignore[attr-defined]
    module.Client = Client  # type: ignore[attr-defined]
    return module


def _install_fake_arxiv(monkeypatch, results_fn):  # type: ignore[no-untyped-def]
    import sys  # noqa: PLC0415

    module = _make_fake_arxiv_module(results_fn)
    monkeypatch.setitem(sys.modules, "arxiv", module)
    return module


async def test_arxiv_fetcher_retries_transient_429_then_succeeds(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Two 429s then a result -> fetch succeeds after backoff sleeps."""
    import asyncio as _asyncio  # noqa: PLC0415

    calls = {"n": 0}

    def results_fn():  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] <= 2:
            module = sys_module()
            raise module.HTTPError(status=429)
        return [_FakeArxivResult()]

    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    # Install the fake module first so ``sys_module`` resolves it.
    fake_mod = _install_fake_arxiv(monkeypatch, results_fn)

    def sys_module():  # type: ignore[no-untyped-def]
        return fake_mod

    monkeypatch.setattr(_asyncio, "sleep", fake_sleep)

    fetcher = ArxivLatexFetcher(
        max_attempts=5,
        backoff_base_seconds=1.0,
        backoff_max_seconds=8.0,
        backoff_jitter_seconds=0.0,
    )
    fetched = await fetcher.fetch("1708.04552")

    assert fetched.title == "Cutout: a fake paper"
    assert calls["n"] == 3, "expected two failed attempts + one success"
    assert sleeps == [1.0, 2.0], f"expected backoff sleeps after each 429, got {sleeps}"


async def test_arxiv_fetcher_exhausts_budget_on_persistent_429(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A 429 on every attempt -> PaperFetchError after the bounded budget."""
    import asyncio as _asyncio  # noqa: PLC0415

    calls = {"n": 0}
    fake_mod_holder: dict[str, object] = {}

    def results_fn():  # type: ignore[no-untyped-def]
        calls["n"] += 1
        raise fake_mod_holder["mod"].HTTPError(status=429)  # type: ignore[attr-defined]

    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    fake_mod_holder["mod"] = _install_fake_arxiv(monkeypatch, results_fn)
    monkeypatch.setattr(_asyncio, "sleep", fake_sleep)

    fetcher = ArxivLatexFetcher(
        max_attempts=4,
        backoff_base_seconds=1.0,
        backoff_max_seconds=8.0,
        backoff_jitter_seconds=0.0,
    )
    with pytest.raises(PaperFetchError) as excinfo:
        await fetcher.fetch("1708.04552")

    assert calls["n"] == 4, "expected exactly max_attempts calls"
    # One sleep fewer than attempts (no backoff after the final attempt).
    assert sleeps == [1.0, 2.0, 4.0], f"got {sleeps}"
    assert "throttled" in str(excinfo.value)


async def test_arxiv_fetcher_non_transient_404_raises_immediately(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A 404 raises PaperFetchError immediately with NO retry / backoff."""
    import asyncio as _asyncio  # noqa: PLC0415

    calls = {"n": 0}
    fake_mod_holder: dict[str, object] = {}

    def results_fn():  # type: ignore[no-untyped-def]
        calls["n"] += 1
        raise fake_mod_holder["mod"].HTTPError(status=404)  # type: ignore[attr-defined]

    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    fake_mod_holder["mod"] = _install_fake_arxiv(monkeypatch, results_fn)
    monkeypatch.setattr(_asyncio, "sleep", fake_sleep)

    fetcher = ArxivLatexFetcher(max_attempts=5, backoff_base_seconds=1.0)
    with pytest.raises(PaperFetchError) as excinfo:
        await fetcher.fetch("0000.00000")

    assert calls["n"] == 1, "non-transient error must not retry"
    assert sleeps == [], "non-transient error must not back off"
    assert "404" in str(excinfo.value)


async def test_arxiv_fetcher_uses_page_size_one_tuned_client(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The reused client is built with page_size=1 (not the default 100)."""

    def results_fn():  # type: ignore[no-untyped-def]
        return [_FakeArxivResult()]

    fake_mod = _install_fake_arxiv(monkeypatch, results_fn)

    fetcher = ArxivLatexFetcher()
    await fetcher.fetch("1708.04552")

    assert fake_mod.Client.last_kwargs.get("page_size") == 1  # type: ignore[attr-defined]
    assert fake_mod.Client.last_kwargs.get("delay_seconds") == 3.0  # type: ignore[attr-defined]
    assert fake_mod.Client.last_kwargs.get("num_retries") == 2  # type: ignore[attr-defined]


async def test_arxiv_fetcher_reuses_one_client_across_calls(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The tuned client is built once and reused across fetch() calls."""

    def results_fn():  # type: ignore[no-untyped-def]
        return [_FakeArxivResult()]

    fake_mod = _install_fake_arxiv(monkeypatch, results_fn)

    constructed: list[object] = []
    original_client = fake_mod.Client  # type: ignore[attr-defined]

    class _CountingClient(original_client):  # type: ignore[misc, valid-type]
        def __init__(self, **kwargs: object) -> None:
            constructed.append(self)
            super().__init__(**kwargs)

    fake_mod.Client = _CountingClient  # type: ignore[attr-defined]

    fetcher = ArxivLatexFetcher()
    await fetcher.fetch("1708.04552")
    await fetcher.fetch("1708.04552")

    assert len(constructed) == 1, "the tuned arxiv client must be built once and reused"
