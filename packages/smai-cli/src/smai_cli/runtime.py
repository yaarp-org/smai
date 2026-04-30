"""Programmatic Tier-A entry point: :class:`Runtime`.

Per ``designs/smai/09-cli.md`` §9: every CLI verb maps to one
service-surface call, and that surface is independently importable.
The CLI is a thin adapter; programmatic Tier-A integrators (research
teams building automation around SMAI without using the hosted
backend) construct their own :class:`Runtime`, call the sub-services,
and never invoke a CLI subprocess.

Phase-2 deliverables shipped here (per Task 2.D2 brief):

* :meth:`Runtime.start_in_band` — async-context-managed in-band
  Runtime (`09` §5.2). Boots SQLite + LocalFs + Compute + per-role
  Bedrock providers, registers the SMAI CG-execution + entry specs
  (`09` §5.1's default ``pipelines``), and spawns the worker loop in
  the same process. Used by ``smai dev`` and Task 2.D3's smoke test.
* :class:`ExperimentsService` — ``submit_text`` / ``compile_text``
  (the surface ``smai run`` and ``smai compile`` adapt over).
* :class:`StatusService` — ``get`` / ``wait_for_terminal`` (the
  surface ``smai status`` and ``--watch`` adapt over).

Phase 3 lifts ``Runtime.start_worker`` (the out-of-band
``smai start`` shape, `09` §6) and :class:`ProposalsService` /
:class:`PapersService` (`09` §1.2 verb-to-service mapping for the
proposal pipeline + paper ingestion) into peers. Phase-2 explicitly
ships only the in-band Runtime + experiments + status surfaces.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar, cast
from uuid import uuid4

import yaml
from smai_agents.model_selection import TaskRole
from smai_core import (
    ContractArtifactSet,
    DslDocumentAdapter,
    ExperimentDefinition,
    ExperimentDocument,
    FactorModelDocument,
    Registries,
    TechniqueRef,
    compile_experiment,
    load_default_registries,
)
from smai_core.plugins import ArtifactStore, CursorPage, MetadataStore
from smai_orchestrator import (
    CG_EXECUTION_SPEC_NAME,
    DEFAULT_TASK_ROLES,
    ComparisonGroupRecord,
    EntryRecord,
    InstantiatedPlugins,
    PaperFetcher,
    PaperRecord,
    PipelineSpec,
    PluginOverrides,
    ProposalRecord,
    RunRecord,
    RuntimeConfig,
    instantiate_plugins,
    register_paper_ingestion_pipeline,
    register_proposal_pipeline,
    register_run_record_spec,
    register_smai_specs,
    reset_registry,
)
from smai_orchestrator.specs.proposal import (
    TECHNIQUE_DESCRIPTION_KEY_TEMPLATE as PROPOSAL_TECHNIQUE_DESCRIPTION_KEY_TEMPLATE,
)
from smai_orchestrator.worker.loop import (
    WorkerCycleStats,
    run_worker_cycle,
    run_worker_loop,
)

from smai_cli.per_role_llm import build_per_role_llm_providers

# Canonical artifact storage paths per `01-data-model.md` §5 / `02
# §8.6 / matching the existing CG-execution spec's templates.
EXPERIMENT_PLAN_KEY_TEMPLATE = "comparison-groups/{cg_id}/experiment_plan.json"
HARNESS_CONTRACT_KEY_TEMPLATE = "comparison-groups/{cg_id}/harness/contract.json"
TECHNIQUE_CONTRACT_KEY_TEMPLATE = (
    "comparison-groups/{cg_id}/entries/{entry_id}/technique_contract.json"
)
VALIDATION_CONFIG_KEY_TEMPLATE = "comparison-groups/{cg_id}/validation_config.json"

# CG terminal states per `03-state-machine.md` §3.1 + DEC-034 #1
# (three failure terminals + one success terminal).
TERMINAL_CG_STATES: frozenset[str] = frozenset(
    {"complete", "implementation_failed", "running_failed", "evaluation_failed"}
)

# In-flight state catalogs per the :class:`*State` Literal definitions
# in :mod:`smai_orchestrator.entities.tracking`. The dashboard surfaces
# these via :meth:`StatusService.summary_counts` (the index-page count
# pills) and the per-entity ``list_active`` aggregator helpers.
_PROPOSAL_IN_FLIGHT_STATES: frozenset[str] = frozenset(
    {"proposal_submitted", "designing", "designed"}
)
_CG_IN_FLIGHT_STATES: frozenset[str] = frozenset(
    {"draft", "implementing", "implemented", "running", "evaluating"}
)
_RUN_IN_FLIGHT_STATES: frozenset[str] = frozenset({"pending", "submitted", "running"})
# ``partial`` is a non-terminal parking spot per `08` §5.7 — papers in
# ``partial`` need user attention to promote, so they count as in-flight
# from the dashboard's perspective.
_PAPER_IN_FLIGHT_STATES: frozenset[str] = frozenset(
    {"submitted", "fetching", "screening", "planning", "partial"}
)

_log = logging.getLogger(__name__)


class RuntimeNotStartedError(RuntimeError):
    """Raised when sub-service methods are invoked before
    :meth:`Runtime.start_in_band` has entered its context.

    The :meth:`__aenter__`-equivalent is :func:`Runtime.start_in_band`'s
    ``async with`` block; sub-services are only valid inside it.
    """


class CGNotFoundError(LookupError):
    """Raised by :meth:`StatusService.get` when the requested CG ID
    has no record in :class:`MetadataStore`.

    Carries :attr:`cg_id` so the CLI surface can render a clear error.
    """

    def __init__(self, cg_id: str) -> None:
        self.cg_id = cg_id
        super().__init__(f"no CG with id {cg_id!r} found in MetadataStore")


class ProposalNotFoundError(LookupError):
    """Raised by :class:`ProposalsService` when a proposal id is unknown.

    Carries :attr:`proposal_id` for the CLI surface.
    """

    def __init__(self, proposal_id: str) -> None:
        self.proposal_id = proposal_id
        super().__init__(f"no proposal with id {proposal_id!r} found in MetadataStore")


class ProposalStateError(RuntimeError):
    """Raised when a proposal-state operation is illegal.

    E.g. ``approve_proposal`` on a proposal not in the ``designed``
    resting state, or ``reject_proposal`` on an already-terminal
    proposal.
    """

    def __init__(self, proposal_id: str, current_state: str, attempted: str) -> None:
        self.proposal_id = proposal_id
        self.current_state = current_state
        self.attempted = attempted
        super().__init__(
            f"proposal {proposal_id!r} is in state {current_state!r}; "
            f"{attempted} requires state 'designed'"
        )


class PaperNotFoundError(LookupError):
    """Raised by :class:`PapersService` when an arxiv id is unknown.

    Carries :attr:`arxiv_id` for the CLI surface. Mirrors
    :class:`ProposalNotFoundError`.
    """

    def __init__(self, arxiv_id: str) -> None:
        self.arxiv_id = arxiv_id
        super().__init__(f"no paper with arxiv id {arxiv_id!r} found in MetadataStore")


class PaperStateError(RuntimeError):
    """Raised when a paper-state operation is illegal.

    E.g. ``promote_partial`` on a paper not in the ``partial`` resting
    state, or ``submit`` on a paper that already exists in a non-
    terminal state. Mirrors :class:`ProposalStateError`.
    """

    def __init__(self, arxiv_id: str, current_state: str, attempted: str, expected: str) -> None:
        self.arxiv_id = arxiv_id
        self.current_state = current_state
        self.attempted = attempted
        self.expected = expected
        super().__init__(
            f"paper {arxiv_id!r} is in state {current_state!r}; "
            f"{attempted} requires state {expected!r}"
        )


class RunNotFoundError(LookupError):
    """Raised by :meth:`StatusService.get_run_record` when a run id is unknown.

    Carries :attr:`run_id` so the dashboard / CLI surface can render a
    clear error.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(f"no run with id {run_id!r} found in MetadataStore")


class WaitTimeoutError(TimeoutError):
    """Raised by :meth:`StatusService.wait_for_terminal` when the
    timeout elapses without the CG reaching a terminal state.
    """


@dataclass(frozen=True)
class CGStatus:
    """Status snapshot returned by :meth:`StatusService.get`.

    Mirrors the `09` §7 status-detail shape — the CG's current state
    plus its updated_at timestamp. Verdict-detail / contextual-verdict
    artifact pointers are out-of-scope for `09` §7.1's "what status
    does NOT do" framing but the CLI's verb formatter is welcome to
    fetch them separately.
    """

    cg_id: str
    state: str
    updated_at: datetime
    is_terminal: bool


@dataclass(frozen=True)
class SummaryCounts:
    """In-flight entity counts per :meth:`StatusService.summary_counts`.

    Drives the dashboard index page's count pills. ``in_flight``
    membership is defined per the :class:`*State` Literal catalogs in
    :mod:`smai_orchestrator.entities.tracking` — see the ``_*_IN_FLIGHT_STATES``
    frozensets at module top.
    """

    proposals_in_flight: int
    cgs_in_flight: int
    runs_in_flight: int
    papers_in_flight: int


# Cursor-page walker types. The :class:`MetadataStore` ``list_*`` /
# ``get_ready_*`` / ``get_in_flight_*`` Protocol surface returns
# :class:`CursorPage` per DEC-035 #1; the dashboard materializes the
# pages into in-memory lists for rendering.
_T = TypeVar("_T")
_PageFn = Callable[[str | None], Awaitable[CursorPage[_T]]]


async def _walk_cursor_pages(
    fetch: Callable[[str | None], Awaitable[CursorPage[_T]]],
    *,
    page_cap: int = 100,
) -> list[_T]:
    """Walk a paginated source until it yields no more cursors.

    ``page_cap`` is a defensive bound — the dashboard is for laptop /
    single-user deployments at v1, so individual entity counts are
    bounded; the cap exists to avoid runaway loops on a buggy plugin.
    """
    items: list[_T] = []
    cursor: str | None = None
    for _ in range(page_cap):
        page = await fetch(cursor)
        items.extend(page.items)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    return items


async def _walk_many_cursor_pages(
    sources: tuple[Callable[[str | None], Awaitable[CursorPage[_T]]], ...],
) -> list[_T]:
    """Walk multiple paginated sources concurrently and concatenate."""
    page_lists = await asyncio.gather(*(_walk_cursor_pages(s) for s in sources))
    out: list[_T] = []
    for items in page_lists:
        out.extend(items)
    return out


def _dedupe_by_id(items: list[_T], key_fn: Callable[[_T], str]) -> list[_T]:
    """De-duplicate ``items`` by the result of ``key_fn``.

    Multiple per-state scheduling queries can return the same entity
    when state transitions interleave with the dashboard's reads; the
    aggregator helpers de-dupe by primary key so the dashboard renders
    each entity once.
    """
    seen: set[str] = set()
    out: list[_T] = []
    for item in items:
        key = key_fn(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _parse_yaml_text(yaml_text: str) -> dict[str, Any]:
    """Parse YAML text into a mapping.

    The DSL document loader expects a Python dict (loaded out-of-tree
    so :mod:`smai_core` doesn't take pyyaml as a runtime dep — see
    DEC-029 + the comment in :mod:`smai_core.dsl.document`).
    """
    parsed: Any = yaml.safe_load(yaml_text)
    if parsed is None:
        raise ValueError("experiment YAML is empty")
    if not isinstance(parsed, dict):
        raise ValueError(
            f"experiment YAML must be a mapping at the top level; got {type(parsed).__name__}"
        )
    return cast("dict[str, Any]", parsed)


def _resolve_document(yaml_text: str) -> Any:
    """Parse + validate a DSL document from YAML text.

    Convenience wrapper around :data:`DslDocumentAdapter` per
    `02-dsl-and-contracts.md` §2.2. The return type is the
    discriminated union ``ExperimentDocument | FactorModelDocument``;
    callers narrow via ``isinstance``.
    """
    payload = _parse_yaml_text(yaml_text)
    return DslDocumentAdapter.validate_python(payload, context={"smai_mode": "dsl"})


def _experiment_definitions(document: Any) -> list[ExperimentDefinition]:
    """Project a DSL document onto its comprising :class:`ExperimentDefinition`s."""
    if isinstance(document, FactorModelDocument):
        return list(document.experiments)
    return [document.experiment]


# === Sub-services ============================================================


class ExperimentsService:
    """Verb-to-service surface for ``smai run`` / ``smai compile`` (`09`
    §1.2 verb-to-service mapping)."""

    def __init__(
        self,
        *,
        plugins: InstantiatedPlugins,
        registries_factory: Callable[[], Registries] | None = None,
    ) -> None:
        self._plugins = plugins
        self._registries_factory: Callable[[], Registries] = (
            registries_factory if registries_factory is not None else load_default_registries
        )

    async def compile_text(self, yaml_text: str) -> dict[str, ContractArtifactSet]:
        """Compile-without-submit (the ``smai compile`` adapter).

        Always returns a ``dict[experiment_id, ContractArtifactSet]``,
        whether the document is a single-experiment or a factor-model.
        Single-experiment documents return a one-key dict.

        Raises :class:`ValidationError` /
        :class:`smai_core.verification.VerificationError` if the YAML is
        malformed or fails Pass-2 verification.
        """
        document = _resolve_document(yaml_text)
        registries: Registries = self._registries_factory()
        if isinstance(document, FactorModelDocument):
            return compile_experiment(document, registries)
        assert isinstance(document, ExperimentDocument)
        single = compile_experiment(document, registries)
        return {document.experiment.id: single}

    async def submit_text(self, yaml_text: str) -> list[str]:
        """Compile, persist artifacts to :class:`ArtifactStore`, and
        create CGs in :class:`MetadataStore`.

        Returns the list of CG IDs created (one for an
        :class:`ExperimentDocument`; N for a :class:`FactorModelDocument`).
        """
        document = _resolve_document(yaml_text)
        registries: Registries = self._registries_factory()
        if isinstance(document, FactorModelDocument):
            artifact_sets: dict[str, ContractArtifactSet] = compile_experiment(document, registries)
        else:
            assert isinstance(document, ExperimentDocument)
            single = compile_experiment(document, registries)
            artifact_sets = {document.experiment.id: single}

        definitions = {d.id: d for d in _experiment_definitions(document)}

        cg_ids: list[str] = []
        for cg_id, artifact_set in artifact_sets.items():
            definition = definitions[cg_id]
            await _persist_artifacts(self._plugins.artifact_store, cg_id, artifact_set)
            await _create_cg_record(
                self._plugins.metadata_store,
                cg_id=cg_id,
                definition=definition,
                artifact_set=artifact_set,
            )
            cg_ids.append(cg_id)
        return cg_ids


class StatusService:
    """Verb-to-service surface for ``smai status`` (`09` §1.2 / §7)."""

    def __init__(self, *, plugins: InstantiatedPlugins) -> None:
        self._plugins = plugins

    async def get(self, cg_id: str) -> CGStatus:
        """Return the current state of ``cg_id``.

        Reads :class:`ComparisonGroupRecord` from :class:`MetadataStore`.
        Raises :class:`CGNotFoundError` if the row is absent.
        """
        record = await self._plugins.metadata_store.get_cg(cg_id)
        if record is None:
            raise CGNotFoundError(cg_id)
        return CGStatus(
            cg_id=record.id,
            state=record.state,
            updated_at=record.updated_at,
            is_terminal=record.state in TERMINAL_CG_STATES,
        )

    async def wait_for_terminal(
        self,
        cg_id: str,
        *,
        timeout: float | None = None,
        poll_interval_seconds: float = 1.0,
    ) -> CGStatus:
        """Poll until ``cg_id`` reaches a terminal state.

        Returns the :class:`CGStatus` snapshot at the moment of
        terminal observation. If ``timeout`` is supplied and elapses
        before the CG terminates, raises :class:`WaitTimeoutError`.
        """
        deadline = asyncio.get_running_loop().time() + timeout if timeout is not None else None
        while True:
            snapshot = await self.get(cg_id)
            if snapshot.is_terminal:
                return snapshot
            if deadline is not None and asyncio.get_running_loop().time() >= deadline:
                raise WaitTimeoutError(
                    f"CG {cg_id!r} did not reach a terminal state within "
                    f"{timeout:.1f}s; current state: {snapshot.state}"
                )
            await asyncio.sleep(poll_interval_seconds)

    async def get_cg_record(self, cg_id: str) -> ComparisonGroupRecord:
        """Return the full :class:`ComparisonGroupRecord` for ``cg_id``.

        :meth:`get` projects the narrow `09` §7 status surface; the
        dashboard's CG-detail page needs the full record (artifact
        hashes, job handle, code-review state). Raises
        :class:`CGNotFoundError` when missing.
        """
        record = await self._plugins.metadata_store.get_cg(cg_id)
        if record is None:
            raise CGNotFoundError(cg_id)
        return record

    async def get_run_record(self, run_id: str) -> RunRecord:
        """Read a :class:`RunRecord` by id (raises :class:`RunNotFoundError`
        when missing)."""
        record = await self._plugins.metadata_store.get_run(run_id)
        if record is None:
            raise RunNotFoundError(run_id)
        return record

    async def list_entries_for_cg(self, cg_id: str) -> list[EntryRecord]:
        """Return every :class:`EntryRecord` parented by ``cg_id``.

        Walks the cursor pages exposed by
        :meth:`MetadataStore.list_entries_for_cg`. v1 deployments have
        small per-CG entry counts (factor-model fan-out is bounded);
        the dashboard renders the materialized list directly.
        """
        return await _walk_cursor_pages(
            lambda cursor: self._plugins.metadata_store.list_entries_for_cg(cg_id, cursor=cursor),
        )

    async def list_runs_for_entry(self, entry_id: str) -> list[RunRecord]:
        """Return every :class:`RunRecord` parented by ``entry_id``."""
        return await _walk_cursor_pages(
            lambda cursor: self._plugins.metadata_store.list_runs_for_entry(
                entry_id, cursor=cursor
            ),
        )

    async def list_active_cgs(self) -> list[ComparisonGroupRecord]:
        """Aggregate every active CG via per-state scheduling queries.

        The :class:`MetadataStore` Protocol exposes no broad "list all
        CGs" surface (full enumeration is deferred per `07` §5); the
        dashboard composes the ``get_ready_*`` / ``get_in_flight_*``
        queries instead. Covers ``draft``, ``implementing``,
        ``implemented``, ``running``, ``evaluating``. Terminal CGs are
        reachable by direct id-navigation to the detail page.
        """
        store = self._plugins.metadata_store
        sources: tuple[_PageFn[ComparisonGroupRecord], ...] = (
            lambda cursor: store.get_ready_for_harness_build(cursor=cursor),
            lambda cursor: store.get_in_flight_harness_build(cursor=cursor),
            lambda cursor: store.get_ready_for_review_and_run(cursor=cursor),
            lambda cursor: store.get_ready_to_evaluate(cursor=cursor),
            lambda cursor: store.get_in_flight_evaluation(cursor=cursor),
        )
        return _dedupe_by_id(await _walk_many_cursor_pages(sources), lambda r: r.id)

    async def list_active_runs(self) -> list[RunRecord]:
        """Aggregate every active run via per-state scheduling queries.

        Covers ``pending`` (:meth:`get_ready_to_run`) plus the in-flight
        states ``submitted`` / ``running`` (:meth:`get_in_flight_runs`).
        Terminal runs are reachable via the per-entry run list on the
        CG-detail page.
        """
        store = self._plugins.metadata_store
        sources: tuple[_PageFn[RunRecord], ...] = (
            lambda cursor: store.get_ready_to_run(cursor=cursor),
            lambda cursor: store.get_in_flight_runs(cursor=cursor),
        )
        return _dedupe_by_id(await _walk_many_cursor_pages(sources), lambda r: r.id)

    async def summary_counts(self) -> SummaryCounts:
        """In-flight counts per entity kind for the dashboard index.

        Uses :meth:`MetadataStore.count_in_state` (`07` §5.6.6) — one
        round-trip per kind, with the in-flight state lists pinned per
        the :class:`*State` Literal catalogs in
        :mod:`smai_orchestrator.entities.tracking`.
        """
        store = self._plugins.metadata_store
        return SummaryCounts(
            proposals_in_flight=await store.count_in_state(
                "proposal", list(_PROPOSAL_IN_FLIGHT_STATES)
            ),
            cgs_in_flight=await store.count_in_state("cg", list(_CG_IN_FLIGHT_STATES)),
            runs_in_flight=await store.count_in_state("run", list(_RUN_IN_FLIGHT_STATES)),
            papers_in_flight=await store.count_in_state("paper", list(_PAPER_IN_FLIGHT_STATES)),
        )


# === ProposalsService =======================================================


@dataclass(frozen=True)
class ProposalSubmission:
    """Immutable bundle returned by :meth:`ProposalsService.submit`.

    Per `09-cli.md` §1 / DEC-032: ``submit-proposal`` is the v2 primary
    input verb. The synchronous return shape mirrors the v1 API
    convention — the proposal exists in :class:`MetadataStore` and the
    submission artifact is persisted in :class:`ArtifactStore`; the
    pipeline-spec's ``proposal_submitted → designing`` edge fires on
    the next worker cycle.
    """

    proposal_id: str
    state: str
    submission_kind: str
    technique_description_artifact_key: str | None
    reproduce_paper_arxiv_id: str | None


class ProposalsService:
    """Verb-to-service surface for ``smai submit-proposal`` /
    ``smai approve-proposal`` / ``smai reject-proposal`` (`09` §1.2 /
    DEC-032).
    """

    def __init__(self, *, plugins: InstantiatedPlugins) -> None:
        self._plugins = plugins

    async def submit(
        self,
        *,
        proposal_id: str,
        submission_kind: str = "novel_technique",
        technique_description: dict[str, Any] | str | None = None,
        reproduce_paper_arxiv_id: str | None = None,
        submitted_by: str | None = None,
    ) -> ProposalSubmission:
        """Create a :class:`ProposalRecord` in ``proposal_submitted``.

        Per ``08-novel-technique-pipeline.md`` §3.1 / `09` §1: the
        submission step is synchronous in the API surface and runs
        BEFORE the entity enters the pipeline-spec. Persists the
        user's submitted technique description (or reproduce-paper
        reference) to ArtifactStore at
        :data:`PROPOSAL_TECHNIQUE_DESCRIPTION_KEY_TEMPLATE`, then
        creates the :class:`ProposalRecord` in ``proposal_submitted``.
        """
        if submission_kind not in {"novel_technique", "reproduce_paper"}:
            raise ValueError(
                f"submission_kind {submission_kind!r} must be "
                "'novel_technique' or 'reproduce_paper'"
            )
        artifact_key: str | None = None
        if technique_description is not None:
            artifact_key = PROPOSAL_TECHNIQUE_DESCRIPTION_KEY_TEMPLATE.format(
                proposal_id=proposal_id,
            )
            payload: bytes
            if isinstance(technique_description, str):
                payload = technique_description.encode("utf-8")
            else:
                payload = json.dumps(technique_description, indent=2, sort_keys=True).encode(
                    "utf-8"
                )
            await self._plugins.artifact_store.put(artifact_key, payload)
        now = datetime.now(UTC)
        record = ProposalRecord(
            id=proposal_id,
            submitted_by=submitted_by,
            submission_kind=submission_kind,  # type: ignore[arg-type]
            technique_description_artifact_key=artifact_key,
            reproduce_paper_arxiv_id=reproduce_paper_arxiv_id,
            state="proposal_submitted",
            created_at=now,
            updated_at=now,
        )
        await self._plugins.metadata_store.create_proposal(record)
        return ProposalSubmission(
            proposal_id=proposal_id,
            state=record.state,
            submission_kind=record.submission_kind,
            technique_description_artifact_key=artifact_key,
            reproduce_paper_arxiv_id=reproduce_paper_arxiv_id,
        )

    async def get(self, proposal_id: str) -> ProposalRecord:
        """Read the :class:`ProposalRecord` (raises if missing)."""
        record = await self._plugins.metadata_store.get_proposal(proposal_id)
        if record is None:
            raise ProposalNotFoundError(proposal_id)
        return record

    async def approve(self, proposal_id: str) -> ProposalRecord:
        """Approve a proposal in the ``designed`` resting state.

        Per ``03-state-machine.md`` §4.2 (edge 5,
        ``proposal.user_approved``) / `09` §1: writes
        ``user_decision = "approved"`` to the proposal record. The
        worker's next cycle picks up the ``designed → registered`` gate
        and fires the registration handler.
        """
        record = await self.get(proposal_id)
        if record.state != "designed":
            raise ProposalStateError(proposal_id, record.state, "approve_proposal")
        return await self._plugins.metadata_store.transition_proposal_state(
            proposal_id,
            record.version,
            "designed",
            user_decision="approved",
            user_decided_at=datetime.now(UTC),
        )

    async def reject(self, proposal_id: str, *, reason: str | None = None) -> ProposalRecord:
        """Reject a proposal in the ``designed`` resting state.

        Per ``03`` §4.2 (edge 6, ``proposal.user_rejected``). Writes
        ``user_decision = "rejected"``; the next worker cycle fires
        the ``designed → rejected`` edge.
        """
        record = await self.get(proposal_id)
        if record.state != "designed":
            raise ProposalStateError(proposal_id, record.state, "reject_proposal")
        # ``reason`` is currently informational — surfaced as
        # last_error per the LeaseableRecord audit shape; the
        # design-time gate body doesn't read it.
        last_error = f"rejected by user: {reason}" if reason is not None else "rejected by user"
        return await self._plugins.metadata_store.transition_proposal_state(
            proposal_id,
            record.version,
            "designed",
            user_decision="rejected",
            user_decided_at=datetime.now(UTC),
            last_error=last_error,
        )

    async def list_active(self) -> list[ProposalRecord]:
        """Aggregate every active proposal via per-state scheduling queries.

        Covers ``proposal_submitted`` (:meth:`get_ready_for_proposal_design`),
        ``designing`` (:meth:`get_in_flight_proposal_design`), and
        ``designed`` (:meth:`get_proposals_at_human_gate` — returns ALL
        proposals at the human gate per the Protocol docstring's
        reconciliation note). Terminal proposals (``registered`` /
        ``rejected`` / ``failed``) are reachable by direct id-navigation
        to the detail page.
        """
        store = self._plugins.metadata_store
        sources: tuple[_PageFn[ProposalRecord], ...] = (
            lambda cursor: store.get_ready_for_proposal_design(cursor=cursor),
            lambda cursor: store.get_in_flight_proposal_design(cursor=cursor),
            lambda cursor: store.get_proposals_at_human_gate(cursor=cursor),
        )
        return _dedupe_by_id(await _walk_many_cursor_pages(sources), lambda r: r.id)

    async def list_cgs(self, proposal_id: str) -> list[ComparisonGroupRecord]:
        """Return every CG parented by ``proposal_id``.

        Walks :meth:`MetadataStore.list_cgs_for_proposal`'s cursor pages.
        Used by the proposal-detail page to surface the registered
        children of an approved proposal.
        """
        return await _walk_cursor_pages(
            lambda cursor: self._plugins.metadata_store.list_cgs_for_proposal(
                proposal_id, cursor=cursor
            ),
        )


# === PapersService ==========================================================


@dataclass(frozen=True)
class PaperSubmission:
    """Immutable bundle returned by :meth:`PapersService.submit`.

    Per `09-cli.md` §1 / DEC-032: ``ingest`` is the v2 paper-ingestion
    verb. Submission is synchronous in the API surface — the
    :class:`PaperRecord` exists in :class:`MetadataStore` after the
    call returns; the worker's next cycle picks the paper up via
    ``get_ready_for_paper_fetch`` and drives the ingestion pipeline.

    For partial-promotion submissions the returned ``state`` reflects
    the post-promotion state (``submitted``); the ``promoted`` flag
    indicates whether the call promoted an existing partial vs.
    creating a new submission.
    """

    arxiv_id: str
    state: str
    promoted: bool


class PapersService:
    """Verb-to-service surface for ``smai ingest`` (`09` §1.2 / DEC-032).

    Mirrors :class:`ProposalsService`'s shape. Three operations:

    * :meth:`submit` — create a :class:`PaperRecord` in ``submitted``
      (the default ``smai ingest <arxiv-id>`` flow).
    * :meth:`get` — read a :class:`PaperRecord` (raises if missing).
    * :meth:`promote_partial` — synchronous transition of a paper from
      ``partial`` to ``submitted`` (the ``smai ingest --promote-partial
      <arxiv-id>`` flow).

    Per `08-novel-technique-pipeline.md` §5.7 / `07-plugin-interfaces.md`
    §5.6.5 docstring: promotion is a synchronous user-driven write
    through :meth:`MetadataStore.transition_paper_state`. The pipeline-
    spec does not declare an engine-driven ``partial → submitted``
    edge; the synchronous transition + the next cycle's ``submitted``
    discovery is the canonical promotion shape (see the paper-
    ingestion spec module's "spec ambiguities resolved" note).
    """

    def __init__(self, *, plugins: InstantiatedPlugins) -> None:
        self._plugins = plugins

    async def submit(
        self,
        *,
        arxiv_id: str,
        title: str | None = None,
    ) -> PaperSubmission:
        """Create a :class:`PaperRecord` in ``submitted``, OR promote an
        existing ``partial`` to ``submitted``, OR no-op on an already-
        in-flight or terminal paper.

        Per `08` §5.7 line 432-437:

        * Paper doesn't exist: create entity in ``submitted``, dispatch
          the pipeline.
        * Paper exists with status ``partial``: transition to
          ``submitted`` synchronously; pipeline runs (fetch step
          short-circuits via ``content_already_extracted`` per `08`
          §5.2 edge 1).
        * Paper exists in a terminal state (``registered`` / ``rejected``
          / ``failed``): return existing paper.
        * Paper exists and is in progress: return existing paper with
          current status.

        Returns a :class:`PaperSubmission` recording the post-call
        state and whether the call promoted an existing partial.
        """
        existing = await self._plugins.metadata_store.get_paper(arxiv_id)
        if existing is None:
            now = datetime.now(UTC)
            record = PaperRecord(
                arxiv_id=arxiv_id,
                title=title,
                state="submitted",
                created_at=now,
                updated_at=now,
            )
            await self._plugins.metadata_store.create_paper(record)
            return PaperSubmission(
                arxiv_id=arxiv_id,
                state=record.state,
                promoted=False,
            )
        if existing.state == "partial":
            promoted = await self._plugins.metadata_store.transition_paper_state(
                arxiv_id,
                existing.version,
                "submitted",
            )
            return PaperSubmission(
                arxiv_id=arxiv_id,
                state=promoted.state,
                promoted=True,
            )
        # Terminal or in-progress — return existing snapshot. The CLI
        # surface formats the response so the user can see "already
        # ingested at registered" or "in flight at planning".
        return PaperSubmission(
            arxiv_id=arxiv_id,
            state=existing.state,
            promoted=False,
        )

    async def get(self, arxiv_id: str) -> PaperRecord:
        """Read the :class:`PaperRecord` (raises :class:`PaperNotFoundError`
        if missing)."""
        record = await self._plugins.metadata_store.get_paper(arxiv_id)
        if record is None:
            raise PaperNotFoundError(arxiv_id)
        return record

    async def promote_partial(self, arxiv_id: str) -> PaperRecord:
        """Promote a partial paper to ``submitted``.

        Per `08` §5.7 / `07` §5.6.5: synchronous user-driven write
        through :meth:`MetadataStore.transition_paper_state`. Raises
        :class:`PaperStateError` if the paper is not in ``partial``.
        """
        record = await self.get(arxiv_id)
        if record.state != "partial":
            raise PaperStateError(arxiv_id, record.state, "promote_partial", "partial")
        return await self._plugins.metadata_store.transition_paper_state(
            arxiv_id,
            record.version,
            "submitted",
        )

    async def list_active(self) -> list[PaperRecord]:
        """Aggregate every active paper via per-state scheduling queries.

        Composes the paper-pipeline ``get_ready_*`` /
        ``get_in_flight_*`` queries plus
        :meth:`get_partial_pending_promotion` (papers parked in
        ``partial`` per `08` §5.7 require user attention so they belong
        on the dashboard). Terminal papers (``registered`` / ``rejected``
        / ``failed``) are reachable by direct arxiv-id navigation.
        """
        store = self._plugins.metadata_store
        sources: tuple[_PageFn[PaperRecord], ...] = (
            lambda cursor: store.get_ready_for_paper_fetch(cursor=cursor),
            lambda cursor: store.get_in_flight_paper_fetch(cursor=cursor),
            lambda cursor: store.get_ready_for_paper_screen(cursor=cursor),
            lambda cursor: store.get_in_flight_paper_screen(cursor=cursor),
            lambda cursor: store.get_ready_for_paper_plan(cursor=cursor),
            lambda cursor: store.get_in_flight_paper_plan(cursor=cursor),
            lambda cursor: store.get_partial_pending_promotion(cursor=cursor),
        )
        return _dedupe_by_id(
            await _walk_many_cursor_pages(sources),
            lambda r: r.arxiv_id,
        )

    async def list_techniques(self, arxiv_id: str) -> list[TechniqueRef]:
        """Techniques registered with ``fidelity_anchor`` pointing at this paper.

        Walks :meth:`MetadataStore.list_techniques_for_paper`'s cursor
        pages. The paper-detail page surfaces these as the visible
        outcome of a successful ingestion.
        """
        return await _walk_cursor_pages(
            lambda cursor: self._plugins.metadata_store.list_techniques_for_paper(
                arxiv_id, cursor=cursor
            ),
        )


# === Persistence helpers (used by ExperimentsService.submit_text) ===========


async def _persist_artifacts(
    artifact_store: ArtifactStore,
    cg_id: str,
    artifact_set: ContractArtifactSet,
) -> None:
    """Write the four contract artifacts to :class:`ArtifactStore`.

    Per `09` §5.2 step 2: registration writes the four artifacts
    atomically. The current ArtifactStore Protocol does not expose a
    multi-key transaction (per `07-plugin-interfaces.md` §6); we issue
    sequential ``put`` calls. The CG-execution spec's first phase-2
    cycle reads the artifacts; if a partial write happens the CG
    parks in ``draft`` and a future ``smai run`` re-submission overwrites.
    """
    await artifact_store.put(
        EXPERIMENT_PLAN_KEY_TEMPLATE.format(cg_id=cg_id),
        artifact_set.experiment_plan.model_dump_json().encode("utf-8"),
    )
    await artifact_store.put(
        HARNESS_CONTRACT_KEY_TEMPLATE.format(cg_id=cg_id),
        artifact_set.harness_contract.model_dump_json().encode("utf-8"),
    )
    for technique_contract in artifact_set.technique_contracts:
        entry_id = technique_contract.body.entry_id
        await artifact_store.put(
            TECHNIQUE_CONTRACT_KEY_TEMPLATE.format(cg_id=cg_id, entry_id=entry_id),
            technique_contract.model_dump_json().encode("utf-8"),
        )
    await artifact_store.put(
        VALIDATION_CONFIG_KEY_TEMPLATE.format(cg_id=cg_id),
        artifact_set.validation_config.model_dump_json().encode("utf-8"),
    )


async def _create_cg_record(
    metadata_store: MetadataStore,
    *,
    cg_id: str,
    definition: ExperimentDefinition,
    artifact_set: ContractArtifactSet,
) -> None:
    """Create the :class:`ComparisonGroupRecord` + per-entry rows.

    Phase-2 simplification: ``smai run`` is the sole CG-creation
    pathway in Phase 2 (the proposal pipeline is Phase 3). The CG
    starts in ``draft`` (per `03-state-machine.md` §3.1) with the
    artifact hashes pre-populated so the ``draft → implementing``
    edge has the contract surfaces it needs immediately. The proposal
    side (Phase 3) populates the same hashes in its registration
    transaction (per `01-data-model.md` §5.3).

    Additive baselines (``is_baseline=True`` and ``technique_id is None``)
    are written directly in ``implemented`` per the :class:`EntryRecord`
    docstring / DEC-013 / DEC-017 — they have no technique to implement
    so the orchestrator's per-entry implementer dispatch never fires
    for them. The CG-creation transaction is the canonical lift point
    in both Phase 2 (this function — ``smai run`` direct creation) and
    Phase 3 (the proposal pipeline's registration transaction).
    Substitutive baselines (``is_baseline=True`` and
    ``technique_id is not None``) and treatments (``is_baseline=False``)
    stay in ``pending`` — both have technique code the agent must
    implement.
    """
    now = datetime.now(UTC)
    cg = ComparisonGroupRecord(
        id=cg_id,
        proposal_id=f"smai-cli-direct-{cg_id}",  # synthetic — Phase 2 has no proposals
        factor_model_id=definition.factor_model_id,
        experiment_definition_id=cg_id,
        experiment_plan_hash=artifact_set.experiment_plan.envelope.content_hash,
        harness_contract_hash=artifact_set.harness_contract.envelope.content_hash,
        validation_config_hash=artifact_set.validation_config.envelope.content_hash,
        state="draft",
        created_at=now,
        updated_at=now,
    )
    contract_by_entry = {
        contract.body.entry_id: contract for contract in artifact_set.technique_contracts
    }
    async with await metadata_store.transaction() as txn:
        await txn.create_cg(cg)
        for entry in definition.entries:
            contract = contract_by_entry.get(entry.id)
            is_additive_baseline = entry.is_baseline and entry.level.technique_id is None
            entry_record = EntryRecord(
                id=entry.id,
                cg_id=cg_id,
                technique_id=entry.level.technique_id,
                is_baseline=entry.is_baseline,
                entry_id=entry.id,
                technique_contract_hash=(
                    contract.envelope.content_hash if contract is not None else None
                ),
                state="implemented" if is_additive_baseline else "pending",
                created_at=now,
                updated_at=now,
            )
            await txn.create_entry(entry_record)


# === Runtime =================================================================


@dataclass
class _RuntimeState:
    """Internal handle on the worker loop's lifecycle."""

    plugins: InstantiatedPlugins
    specs: list[PipelineSpec]
    config: RuntimeConfig
    workspace_root: Path
    worker_id: str
    shutdown_event: asyncio.Event = field(default_factory=asyncio.Event)
    worker_task: asyncio.Task[None] | None = None
    cycle_stats: list[WorkerCycleStats] = field(default_factory=list[WorkerCycleStats])


class Runtime:
    """In-band Runtime — boots SQLite + LocalFs + Compute + per-role
    LlmProviders, registers the SMAI specs, and runs the worker loop in
    the same process.

    The user-facing surface is :meth:`start_in_band` (an async context
    manager that yields a constructed instance). Sub-services live on
    :attr:`experiments` / :attr:`status`.

    Per `09` §9: ``Runtime.start_in_band`` is the canonical Tier-A
    entry path. Task 2.D3's smoke test wraps this same surface to
    drive the end-to-end test.
    """

    @classmethod
    @asynccontextmanager
    async def start_worker(  # noqa: PLR0913
        cls,
        config: RuntimeConfig,
        *,
        worker_id: str,
        workspace_root: Path | None = None,
        plugin_overrides: PluginOverrides | None = None,
        per_role_overrides: dict[str, str] | None = None,
        env: Mapping[str, str] | None = None,
        runtime_image: str = "smai-runtime:dev",
        paper_fetcher: PaperFetcher | None = None,
    ) -> AsyncGenerator[Runtime, None]:
        """Out-of-band production worker (`09` §6 / `05` §7.2 / DEC-024).

        The :meth:`start_in_band` counterpart for ``smai start``: same
        spec registration + plugin instantiation + worker-loop machinery,
        but production-shaped — the :class:`Runtime` yielded by this
        context manager is meant to be observed (e.g., for a graceful
        SIGTERM signal-handler that sets ``state.shutdown_event``), not
        used as a one-stop shell. Submit verbs / dashboards run as
        separate processes against the same :class:`MetadataStore`.

        Implementation choice: the worker runs as a separate
        :class:`asyncio.Task` within this CLI process — same shape as
        :meth:`start_in_band`. The "out-of-band" framing of `09` §6.1
        is operational, not implementation-level: the *deployment*
        boundary (this CLI process == the worker process) is enforced
        by systemd / supervisor / launchd running ``smai start`` as
        its own long-lived service, with submit verbs / dashboards
        invoked separately. A separate-OS-process design was
        considered but rejected: it would require ``multiprocessing``-
        style fork/spawn semantics which complicate signal handling
        and plugin-state pickling, and adds no operational benefit
        over "run ``smai start`` under systemd, kill it with SIGTERM"
        — the same code, different config principle (`00` §4 #7) is
        preserved with the asyncio-task design.

        ``worker_id`` is **required** here (vs auto-generated in
        :meth:`start_in_band`): production deployments pin a stable
        identity per `01` §5.6 / DEC-035 #2 so the lease-holder field
        on each entity row is human-readable in operator queries.
        :func:`smai_cli.main.smai_start` resolves this from the
        ``--worker-id`` flag, the ``SMAI_WORKER_ID`` env var, or a
        ``f"{hostname}-{pid}-{uuid8}"`` fallback.

        Other parameters mirror :meth:`start_in_band` exactly. The
        body delegates to it with ``run_worker=True``; the only
        difference is the docstring + the required ``worker_id``.
        """
        async with cls.start_in_band(
            config,
            workspace_root=workspace_root,
            plugin_overrides=plugin_overrides,
            per_role_overrides=per_role_overrides,
            env=env,
            runtime_image=runtime_image,
            run_worker=True,
            paper_fetcher=paper_fetcher,
            worker_id=worker_id,
        ) as runtime:
            yield runtime

    @classmethod
    @asynccontextmanager
    async def start_in_band(  # noqa: PLR0913
        cls,
        config: RuntimeConfig,
        *,
        workspace_root: Path | None = None,
        plugin_overrides: PluginOverrides | None = None,
        per_role_overrides: dict[str, str] | None = None,
        env: Mapping[str, str] | None = None,
        runtime_image: str = "smai-runtime:dev",
        run_worker: bool = True,
        paper_fetcher: PaperFetcher | None = None,
        worker_id: str | None = None,
    ) -> AsyncGenerator[Runtime, None]:
        """Boot the in-band Runtime; yield a configured instance.

        ``workspace_root`` defaults to ``~/.smai/workspaces``. The
        agent dispatch handlers materialize per-CG workspaces under
        this root.

        ``plugin_overrides`` accepts pre-instantiated plugins that
        bypass entry-point discovery (test injection per
        :class:`PluginOverrides`). When ``plugin_overrides.llm_providers``
        is provided, the per-role construction step is skipped — the
        caller is responsible for supplying the role map.

        ``per_role_overrides`` is the explicit per-role
        ``"<provider>:<model_id>"`` map (alternative to the
        ``SMAI_MODEL_<ROLE>`` env-var path); the typed dict is
        per-role-name string keyed.

        ``run_worker=False`` is for tests that drive
        :func:`run_worker_cycle` manually; production callers
        (``smai dev``) leave it ``True``.

        ``worker_id`` is the deployment-stable identity threaded into
        the phase-3 lease wrapper (`05` §3.5 / DEC-035 #2 / Task 3.G1).
        ``None`` (the default) auto-generates ``f"in-band-{uuid4()}"``.
        Multi-worker deployments (Task 3.G3 ``smai start``) pin a
        process-stable id (typically host+pid+uuid).
        """
        resolved_worker_id = worker_id if worker_id is not None else f"in-band-{uuid4().hex[:8]}"
        workspace_root = workspace_root or (Path.home() / ".smai" / "workspaces")
        workspace_root.mkdir(parents=True, exist_ok=True)

        # Build per-role LLM providers for the worker / specs unless
        # overridden by the caller.
        overrides = plugin_overrides or PluginOverrides()
        if overrides.llm_providers is None:
            roles_overrides = (
                _parse_per_role_overrides(per_role_overrides)
                if per_role_overrides is not None
                else None
            )
            try:
                llm_providers = build_per_role_llm_providers(
                    config.plugins,
                    env=env,
                    overrides=roles_overrides,
                )
            except Exception:
                # If we can't build providers (e.g., live AWS unreachable),
                # the caller probably wants to know and re-run with
                # ``plugin_overrides`` for tests.
                raise
            overrides = PluginOverrides(
                llm_providers=llm_providers,
                metadata_store=overrides.metadata_store,
                artifact_store=overrides.artifact_store,
                compute=overrides.compute,
                extra_teardown=overrides.extra_teardown,
            )

        # Reset the spec registry on entry — `smai dev` is the sole
        # consumer in-band, and idempotent registration matches the
        # zero-config laptop ergonomic per `09` §5.
        reset_registry()

        async with instantiate_plugins(config.plugins, overrides=overrides) as plugins:
            # Register the SMAI Phase-2 specs (CG-execution + entry).
            llm_for_code_reviewer = plugins.llm_providers["code_reviewer"]
            llm_for_contextual_evaluator = plugins.llm_providers["contextual_evaluator"]
            cg_spec, entry_spec = register_smai_specs(
                workspace_root=workspace_root,
                llm_for_code_reviewer=llm_for_code_reviewer,
                llm_for_contextual_evaluator=llm_for_contextual_evaluator,
                runtime_image=runtime_image,
            )
            # Register the :class:`RunRecord` sub-state-machine spec
            # alongside the Phase-2 specs (per Task 3.E3 / DEC-034 #3).
            # The supervisor merges this into :func:`register_smai_specs`
            # at commit time once the parallel Task 3.E1 (proposal
            # pipeline-spec) lands too — keeping the registration
            # explicit here for now avoids a multi-task signature change
            # collision.
            # Register the :class:`RunRecord` sub-state-machine spec
            # alongside the Phase-2 specs (per Task 3.E3 / DEC-034 #3).
            # The supervisor merges this into :func:`register_smai_specs`
            # at commit time once the parallel Task 3.E1 (proposal
            # pipeline-spec) lands too — keeping the registration
            # explicit here for now avoids a multi-task signature change
            # collision.
            run_spec = register_run_record_spec(runtime_image=runtime_image)
            # Register the proposal pipeline-spec (per Task 3.E1 /
            # DEC-032 — the v2 primary input path). Same supervisor-
            # merge note as the run-record registration above.
            llm_for_planner = plugins.llm_providers["planner"]
            proposal_spec = register_proposal_pipeline(
                workspace_root=workspace_root,
                llm_for_planner=llm_for_planner,
                # `smai dev` defaults to require_human_approval=True
                # (the laptop deployment shape); production overrides
                # via the same flag.
                require_human_approval=True,
            )
            # Register the paper-ingestion pipeline-spec (per Task 3.E2
            # / DEC-032 — the supporting utility per the
            # ``08-novel-technique-pipeline.md`` §5 framing). Reuses the
            # same per-role :class:`LlmProvider` map; the screener,
            # planner (paper-ingestion variant), and enricher each
            # receive their own role-bound provider. ``paper_fetcher``
            # defaults to :class:`ArxivLatexFetcher` (production); test
            # callers inject a fake fetcher that returns canned content
            # to keep the fetch-stage offline.
            llm_for_screener = plugins.llm_providers.get("screener", llm_for_planner)
            llm_for_enricher = plugins.llm_providers.get("enricher", llm_for_planner)
            paper_spec = register_paper_ingestion_pipeline(
                workspace_root=workspace_root,
                llm_for_screener=llm_for_screener,
                llm_for_planner=llm_for_planner,
                llm_for_enricher=llm_for_enricher,
                fetcher=paper_fetcher,
            )
            # Drive specs in this order each cycle:
            #   1. proposal spec — advance proposals through designing
            #      / designed / registered (creates new CGs in draft)
            #   2. paper spec — advance papers through fetching /
            #      screening / planning / registered (writes
            #      ``TechniqueRef``s to MetadataStore; CGs are NOT
            #      created from this spec per DEC-032)
            #   3. entry spec — advance per-entry implementation phase-1
            #   4. cg spec — phase-1 (harness) + phase-2/3 (gate trio,
            #      runs creation, evaluation dispatch)
            #   5. run spec — drive newly-pending runs to ``submitted``
            #      and phase-1-poll already-in-flight runs to terminal
            # so within one cycle a proposal that lands in ``registered``
            # has its CGs created before the next cycle reads them via
            # the CG-execution spec's ``draft → implementing`` gate.
            # Paper ingestion is independent of the CG-spec lifecycle —
            # it produces ``TechniqueRef``s consumed by future proposal-
            # pipeline planner sessions, not CGs.
            state = _RuntimeState(
                plugins=plugins,
                specs=[proposal_spec, paper_spec, entry_spec, cg_spec, run_spec],
                config=config,
                workspace_root=workspace_root,
                worker_id=resolved_worker_id,
            )
            runtime = cls(state)

            if run_worker:
                state.worker_task = asyncio.create_task(_run_worker_until_shutdown(state))
            try:
                yield runtime
            finally:
                state.shutdown_event.set()
                if state.worker_task is not None:
                    try:
                        await asyncio.wait_for(state.worker_task, timeout=5.0)
                    except TimeoutError:
                        state.worker_task.cancel()
                        try:
                            await state.worker_task
                        except (asyncio.CancelledError, Exception):  # noqa: BLE001
                            pass
                # Spec registry reset on exit so a subsequent
                # ``Runtime.start_in_band`` call doesn't trip
                # DuplicateSpecError.
                reset_registry()

    def __init__(self, state: _RuntimeState) -> None:
        self._state = state
        self._experiments = ExperimentsService(plugins=state.plugins)
        self._status = StatusService(plugins=state.plugins)
        self._proposals = ProposalsService(plugins=state.plugins)
        self._papers = PapersService(plugins=state.plugins)

    @property
    def experiments(self) -> ExperimentsService:
        return self._experiments

    @property
    def status(self) -> StatusService:
        return self._status

    @property
    def proposals(self) -> ProposalsService:
        return self._proposals

    @property
    def papers(self) -> PapersService:
        return self._papers

    @property
    def workspace_root(self) -> Path:
        return self._state.workspace_root

    @property
    def config(self) -> RuntimeConfig:
        return self._state.config

    @property
    def plugins(self) -> InstantiatedPlugins:
        return self._state.plugins

    @property
    def worker_id(self) -> str:
        """The deployment-stable worker identity threaded into the
        phase-3 lease wrapper (`05` §3.5 / DEC-035 #2 / Task 3.G1).

        Set at :meth:`start_in_band` entry — auto-generated as
        ``f"in-band-{uuid4()}"`` unless the caller passed a
        ``worker_id=`` override.
        """
        return self._state.worker_id

    async def run_one_cycle(self) -> list[WorkerCycleStats]:
        """Run a single worker cycle synchronously across every
        registered :class:`PipelineSpec`.

        Returns one :class:`WorkerCycleStats` per spec, in the same
        order the specs are stored on :class:`_RuntimeState` (per-entry
        spec first, then CG spec — see :meth:`start_in_band`). Tests
        that drive cycles deterministically (``run_worker=False``) get
        per-spec stats so they can assert per-spec phase counters
        independently.
        """
        stats_per_spec: list[WorkerCycleStats] = []
        for spec in self._state.specs:
            stats = await run_worker_cycle(
                spec=spec.engine_spec(),
                metadata_store=self._state.plugins.metadata_store,
                artifact_store=self._state.plugins.artifact_store,
                compute=self._state.plugins.compute,
                llm_providers=dict(self._state.plugins.llm_providers),
                config=self._state.config.engine,
                worker_id=self._state.worker_id,
            )
            stats_per_spec.append(stats)
        return stats_per_spec


def _parse_per_role_overrides(
    raw: Mapping[str, str],
) -> dict[TaskRole, tuple[str, str]]:
    """Convert ``{"planner": "bedrock:opus"}`` into the
    :func:`get_model_for_task` overrides shape."""
    from typing import cast as _cast  # noqa: PLC0415

    out: dict[TaskRole, tuple[str, str]] = {}
    for role, raw_value in raw.items():
        if ":" not in raw_value:
            raise ValueError(
                f"per-role override for {role!r} must be 'provider:model_id'; got {raw_value!r}"
            )
        provider, model_id = raw_value.split(":", 1)
        provider = provider.strip()
        model_id = model_id.strip()
        if not provider or not model_id:
            raise ValueError(
                f"per-role override for {role!r} must be 'provider:model_id'; got {raw_value!r}"
            )
        out[_cast(TaskRole, role)] = (provider, model_id)
    return out


async def _run_worker_until_shutdown(state: _RuntimeState) -> None:
    """Background worker task body — one :func:`run_worker_loop` per
    registered :class:`PipelineSpec`.

    All loops share the same :attr:`_RuntimeState.shutdown_event`, so
    the context-manager exit cleanly stops every spec at once. Per-spec
    cycle stats are recorded into the shared ``state.cycle_stats`` list
    (interleaved across specs) — tests that need per-spec slicing call
    :meth:`Runtime.run_one_cycle` instead of inspecting the background
    cycle log.
    """
    on_cycle = _record_cycle(state)
    tasks = [
        asyncio.create_task(
            run_worker_loop(
                spec=spec.engine_spec(),
                metadata_store=state.plugins.metadata_store,
                artifact_store=state.plugins.artifact_store,
                compute=state.plugins.compute,
                llm_providers=dict(state.plugins.llm_providers),
                config=state.config.engine,
                shutdown_event=state.shutdown_event,
                on_cycle_complete=on_cycle,
                worker_id=state.worker_id,
            )
        )
        for spec in state.specs
    ]
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        raise
    except Exception:  # noqa: BLE001 — log so background failures aren't silent
        _log.exception("smai dev worker loop crashed")
        for task in tasks:
            task.cancel()
        raise


def _record_cycle(state: _RuntimeState) -> Callable[[WorkerCycleStats], Awaitable[None]]:
    """Closure: append cycle stats to ``state.cycle_stats``."""

    async def _on_cycle(stats: WorkerCycleStats) -> None:
        state.cycle_stats.append(stats)

    return _on_cycle


__all__ = [
    "CGNotFoundError",
    "CG_EXECUTION_SPEC_NAME",
    "CGStatus",
    "DEFAULT_TASK_ROLES",
    "EXPERIMENT_PLAN_KEY_TEMPLATE",
    "ExperimentsService",
    "HARNESS_CONTRACT_KEY_TEMPLATE",
    "PROPOSAL_TECHNIQUE_DESCRIPTION_KEY_TEMPLATE",
    "PaperNotFoundError",
    "PaperStateError",
    "PaperSubmission",
    "PapersService",
    "ProposalNotFoundError",
    "ProposalStateError",
    "ProposalSubmission",
    "ProposalsService",
    "RunNotFoundError",
    "Runtime",
    "RuntimeNotStartedError",
    "StatusService",
    "SummaryCounts",
    "TECHNIQUE_CONTRACT_KEY_TEMPLATE",
    "TERMINAL_CG_STATES",
    "VALIDATION_CONFIG_KEY_TEMPLATE",
    "WaitTimeoutError",
]
