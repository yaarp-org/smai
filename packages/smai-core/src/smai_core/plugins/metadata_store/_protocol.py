""":class:`MetadataStore` Protocol — SQL-shaped persistence for pipeline-
tracking entities and a small set of methodology-referenced lookups.

Per ``designs/smai/07-plugin-interfaces.md`` §5; DEC-028, DEC-029,
DEC-030, DEC-035.

The Protocol's surface decomposes into four groups:

* **Entity CRUD** for pipeline-tracking entities (§5.3) — uniform
  ``get_*`` / ``list_*`` / ``create_*`` shape; pagination via
  :class:`CursorPage` per DEC-035 #1.
* **Methodology-touching lookups** — the ``TechniqueRef`` registry mirror
  for long-running deployments where the in-memory registry is too big
  (§5.3 second block).
* **Conditional state transitions** (§5.4) — compare-and-swap on
  ``version``; raises :class:`ConflictError` on mismatch.
* **Scheduling queries / leasing / in-flight counts** (§5.6) — what the
  orchestrator's poll cycle consumes; lease ops decoupled from
  entity-state CAS per DEC-035 #2; ``count_with_in_flight_jobs`` takes
  caller-resolved state list per DEC-035 #3.
* **Transactional grouping** (§5.3 last block) — cross-entity atomicity
  via an async context manager; enabled by SQL-only commitment per
  DEC-030.

Concurrent transitions on the same row: only one wins; the loser raises
:class:`ConflictError`. Lease-acquisition contention: failure is
*normal* in the poll loop — :meth:`MetadataStore.acquire_lease` returns
``None`` rather than raising, since it's an opportunistic try.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from smai_core.plugins._common import CursorPage, EntityKind, LeaseToken
from smai_core.plugins.metadata_store._capabilities import MetadataStoreCapabilities

if TYPE_CHECKING:
    from datetime import datetime

    # ``TechniqueRef`` is a methodology entity (smai-core internal); imported
    # under TYPE_CHECKING to avoid a runtime import cycle through
    # ``smai_core.entities``. Used by the technique-registry lookups in §5.3.
    # Pipeline-tracking record types live in ``smai-orchestrator`` per
    # ``07-plugin-interfaces.md`` §3.1 + DEC-029 (one-way dependency:
    # pipeline → methodology, never reverse). The Protocol references them
    # by name only; they are never imported at smai-core runtime.
    # ``tools/check_deps.py`` exempts these TYPE_CHECKING imports from
    # rule 2 — see the lint module for details.
    from smai_orchestrator.entities.tracking import (
        AgentSessionRecord,
        CGState,
        ComparisonGroupRecord,
        EntryRecord,
        EntryState,
        PaperRecord,
        PaperState,
        ProposalRecord,
        ProposalState,
        RunRecord,
        RunState,
    )

    from smai_core.entities import TechniqueRef


@runtime_checkable
class Transaction(Protocol):
    """Scoped to one transaction (§5.3).

    All write methods on the parent :class:`MetadataStore` are mirrored
    here; the :class:`Transaction`'s writes commit together when the
    context exits without exception, rollback on exception.

    Reads outside the transaction see committed state only. Whether
    in-transaction reads expose uncommitted state is unspecified at v1
    (per §12 OQ10 — Session C).
    """

    @property
    def connection(self) -> object:
        """Plugin-internal handle for the underlying transactional
        connection — used by Task 4.K3's :class:`PgNotifyEventChannel`
        so it can issue ``pg_notify('smai_events', ...)`` against the
        same asyncpg transaction the CAS ``UPDATE`` ran in (per
        ``12-ui-process.md`` §6.5).

        Generic engine code MUST treat this attribute as opaque. Only
        plugin-specific :class:`smai_events.EventChannel`
        implementations that know which plugin they pair with should
        narrow the type and access driver-specific operations on it.
        For the Postgres plugin this is a SQLAlchemy ``AsyncConnection``;
        for the SQLite plugin the same; for hypothetical non-SQL
        plugins it could be anything (or ``None``).
        """
        ...

    async def create_cg(self, cg: ComparisonGroupRecord) -> ComparisonGroupRecord: ...
    async def create_entry(self, entry: EntryRecord) -> EntryRecord: ...
    async def create_run(self, run: RunRecord) -> RunRecord: ...
    async def create_proposal(self, proposal: ProposalRecord) -> ProposalRecord: ...
    async def create_paper(self, paper: PaperRecord) -> PaperRecord: ...

    async def transition_cg_state(
        self,
        cg_id: str,
        expected_version: int,
        target_state: CGState,
        **fields: object,
    ) -> ComparisonGroupRecord: ...

    async def transition_entry_state(
        self,
        entry_id: str,
        expected_version: int,
        target_state: EntryState,
        **fields: object,
    ) -> EntryRecord: ...

    async def transition_run_state(
        self,
        run_id: str,
        expected_version: int,
        target_state: RunState,
        **fields: object,
    ) -> RunRecord: ...

    async def transition_proposal_state(
        self,
        proposal_id: str,
        expected_version: int,
        target_state: ProposalState,
        **fields: object,
    ) -> ProposalRecord: ...

    async def transition_paper_state(
        self,
        arxiv_id: str,
        expected_version: int,
        target_state: PaperState,
        **fields: object,
    ) -> PaperRecord: ...

    async def append_transition_log(
        self,
        *,
        entity_kind: str,
        entity_id: str,
        from_state: str,
        to_state: str,
        edge_name: str,
        worker_id: str | None,
        gate_outcome_reason: str | None = None,
    ) -> None:
        """Append one row to the ``transition_log`` audit table (DEC-033 #1).

        Mirrored on the parent :class:`MetadataStore`; the transactional
        variant writes the audit row in the *same* transaction as the
        CAS ``UPDATE`` (so the Pg-``NOTIFY`` / K3 path stays atomic).
        ``occurred_at`` is set by the implementation to "now".
        """
        ...


@runtime_checkable
class MetadataStore(Protocol):
    """SQL-shaped persistence for pipeline tracking entities and a small
    set of methodology-referenced lookups (§5.1).

    Plugins register via::

        [project.entry-points."smai.metadata_stores"]
        <name> = "<module>:<class>"

    Per DEC-029 / DEC-030. Methodology entities never touch
    :class:`MetadataStore` (per DEC-029) — Tier B integrators construct
    methodology entities in memory, compile them, evaluate metrics
    against raw data, return a verdict; the Protocol is irrelevant to
    that path.
    """

    name: str
    capabilities: MetadataStoreCapabilities

    # === Entity CRUD — pipeline tracking entities (§5.3) ===

    async def get_cg(self, cg_id: str) -> ComparisonGroupRecord | None: ...

    async def list_entries_for_cg(
        self,
        cg_id: str,
        limit: int = 100,
        cursor: str | None = None,
    ) -> CursorPage[EntryRecord]: ...

    async def get_entry(self, entry_id: str) -> EntryRecord | None: ...

    async def list_runs_for_entry(
        self,
        entry_id: str,
        limit: int = 100,
        cursor: str | None = None,
    ) -> CursorPage[RunRecord]: ...

    async def get_run(self, run_id: str) -> RunRecord | None: ...

    async def get_proposal(self, proposal_id: str) -> ProposalRecord | None: ...

    async def list_cgs_for_proposal(
        self,
        proposal_id: str,
        limit: int = 100,
        cursor: str | None = None,
    ) -> CursorPage[ComparisonGroupRecord]:
        """List CGs parented by ``proposal_id`` (§5.3).

        Per DEC-032 OQ2 a proposal can parent 1..N CGs; cursor-based
        per ``01`` §5.10.
        """
        ...

    async def get_paper(self, arxiv_id: str) -> PaperRecord | None: ...

    async def list_techniques_for_paper(
        self,
        arxiv_id: str,
        limit: int = 100,
        cursor: str | None = None,
    ) -> CursorPage[TechniqueRef]:
        """List techniques whose ``fidelity_anchor`` points at this paper
        (§5.3)::

            SELECT ... FROM techniques
            WHERE fidelity_anchor->>'kind' = 'paper'
              AND fidelity_anchor->>'arxiv_id' = :a

        per ``01`` §5.7; cursor-based.
        """
        ...

    async def create_cg(self, cg: ComparisonGroupRecord) -> ComparisonGroupRecord: ...
    async def create_entry(self, entry: EntryRecord) -> EntryRecord: ...
    async def create_run(self, run: RunRecord) -> RunRecord: ...
    async def create_proposal(self, proposal: ProposalRecord) -> ProposalRecord: ...
    async def create_paper(self, paper: PaperRecord) -> PaperRecord: ...

    # Updates, deletes, and full enumeration deferred to Session C
    # (``01-data-model.md`` §5).

    # === Observability tables — agent sessions + transition log (DEC-033) ===

    async def create_agent_session(
        self,
        *,
        parent_kind: str,
        parent_id: str,
        agent_role: str,
        llm_provider: str,
        model_id: str,
    ) -> str:
        """Insert an ``agent_sessions`` row at dispatch start; return its id.

        The new row carries ``started_at = now()``, ``ended_at = None``,
        the per-token USD rate columns ``NULL`` (cost-rate work is
        backlogged), and the token / turn counters at ``0``. The
        returned id is a ULID-shaped string matching the id convention
        used elsewhere.
        """
        ...

    async def update_agent_session(
        self,
        session_id: str,
        *,
        turn_count: int,
        input_tokens: int,
        cached_input_tokens: int,
        output_tokens: int,
        ended_at: datetime | None = None,
    ) -> None:
        """UPDATE the running token / turn counts on an agent-session row.

        The agent loop's per-turn ``progress_sink`` calls this with the
        latest running totals; the dispatch handler calls it once more on
        loop exit with ``ended_at`` set. A no-op (silently) when no row
        with ``session_id`` exists is acceptable — callers don't depend
        on a "not found" signal here.
        """
        ...

    async def get_agent_session(self, session_id: str) -> AgentSessionRecord | None:
        """Read back one agent-session row by id (``None`` if absent)."""
        ...

    async def append_transition_log(
        self,
        *,
        entity_kind: str,
        entity_id: str,
        from_state: str,
        to_state: str,
        edge_name: str,
        worker_id: str | None,
        gate_outcome_reason: str | None = None,
    ) -> None:
        """Append one row to the ``transition_log`` audit table (DEC-033 #1).

        Called by the engine's transition driver after a successful CAS
        that actually changed state. ``occurred_at`` is set by the
        implementation. Same-state handle-only writes do NOT call this.
        The transactional variant (on :class:`Transaction`) writes the
        row in the CAS transaction; this non-transactional variant is
        used on the post-commit / best-effort path.
        """
        ...

    # === Methodology-touching lookups (§5.3) ===

    async def get_technique(self, technique_id: str) -> TechniqueRef | None:
        """Look up a technique by ID (§5.3).

        The technique registry is methodology-layer state but is too
        large to fit in-memory in long-running deployments — it lives
        behind the :class:`MetadataStore`. Methodology code never calls
        these; the pipeline layer assembles a :class:`Registries` from
        these lookups before invoking the compiler.
        """
        ...

    async def list_techniques(
        self,
        limit: int = 100,
        cursor: str | None = None,
    ) -> CursorPage[TechniqueRef]:
        """Cursor-based per ``01`` §5.10 (§5.3).

        The Protocol does NOT carry a ``kind``-filter parameter for
        ``fidelity_anchor.kind`` — clients filter client-side (the
        registry is small at v1 scale, ≪10K rows).
        """
        ...

    async def upsert_technique(self, technique: TechniqueRef) -> None: ...

    # === Conditional state transitions (§5.4) ===

    async def transition_cg_state(
        self,
        cg_id: str,
        expected_version: int,
        target_state: CGState,
        **fields: object,
    ) -> ComparisonGroupRecord:
        """Atomic compare-and-swap state transition (§5.4).

        Translates to (illustrative SQL)::

            UPDATE comparison_groups
               SET state = :target_state,
                   version = version + 1,
                   <fields>
             WHERE id = :cg_id
               AND version = :expected_version
            RETURNING *

        On no-row-affected (the version did not match), MUST raise
        :class:`ConflictError` with ``entity_type='cg'``, ``entity_id=cg_id``.

        On success, returns the updated record (with version
        incremented).

        ``**fields`` carries optional fields the transition is allowed
        to set atomically with the state change (e.g., setting
        ``harness_job_handle`` when transitioning to ``implementing``).
        Per ``01-data-model.md`` §5.3.
        """
        ...

    async def transition_entry_state(
        self,
        entry_id: str,
        expected_version: int,
        target_state: EntryState,
        **fields: object,
    ) -> EntryRecord: ...

    async def transition_run_state(
        self,
        run_id: str,
        expected_version: int,
        target_state: RunState,
        **fields: object,
    ) -> RunRecord: ...

    async def transition_proposal_state(
        self,
        proposal_id: str,
        expected_version: int,
        target_state: ProposalState,
        **fields: object,
    ) -> ProposalRecord: ...

    async def transition_paper_state(
        self,
        arxiv_id: str,
        expected_version: int,
        target_state: PaperState,
        **fields: object,
    ) -> PaperRecord: ...

    # === Scheduling queries — CG-execution pipeline (§5.6.3) ===

    async def get_ready_for_harness_build(
        self,
        limit: int = 100,
        cursor: str | None = None,
    ) -> CursorPage[ComparisonGroupRecord]:
        """CGs ready for harness-builder dispatch (§5.6.3).

        CGs in state ``draft`` (or ``implementing`` per ``03``'s
        settling, whichever state harness build is the on-entry
        dispatch for) with no in-flight harness-builder job —
        ``harness_job_handle is null``. Ordered FIFO by ``created_at``
        (single-tenant); tenant-fair when
        ``capabilities.is_tenant_aware``.
        """
        ...

    async def get_in_flight_harness_build(
        self,
        limit: int = 100,
        cursor: str | None = None,
    ) -> CursorPage[ComparisonGroupRecord]:
        """CGs whose ``harness_job_handle is not null`` and whose state
        is the in-progress harness-build state (§5.6.3).

        Used by phase 1 of the poll cycle (per ``05`` §3.1) to drive
        :meth:`Compute.status` polling.
        """
        ...

    async def get_ready_to_implement_entry(
        self,
        limit: int = 100,
        cursor: str | None = None,
    ) -> CursorPage[EntryRecord]:
        """Entries ready for implementation dispatch (§5.6.3).

        Non-baseline entries with ``state='pending'``, in CGs whose
        state allows implementation dispatch (e.g., ``implementing``
        with a completed harness). Baseline entries with
        ``technique_id is null`` per DEC-013 are NOT returned — they
        are runnable as-is and never need implementation.

        Ordering: FIFO by entry ``created_at``. Returns entries with
        their parent CG and referenced technique sufficient for the
        orchestrator to assemble a dispatch payload (the exact joined
        shape — eager-load vs. lazy lookup — is a plugin-implementation
        choice; the Protocol commits only to the entity returned).
        """
        ...

    async def get_in_flight_entry_implementation(
        self,
        limit: int = 100,
        cursor: str | None = None,
    ) -> CursorPage[EntryRecord]:
        """Entries with a non-null ``implementation_job_handle``.
        Phase-1 polling (§5.6.3)."""
        ...

    async def get_ready_for_review_and_run(
        self,
        limit: int = 100,
        cursor: str | None = None,
    ) -> CursorPage[ComparisonGroupRecord]:
        """CGs ready for the inline-review gate at
        ``implemented → running`` (§5.6.3).

        CGs in state ``implemented`` whose entries have all reached a
        terminal implementation state (``implemented`` or
        ``implementation_failed``), with ≥ 1 entry ``implemented``.
        Per DEC-016 / ``03-state-machine.md`` §3.3, code review is an
        inline gate at the ``implemented → running`` edge (NOT a
        separate state); this single CG-level query feeds that gate's
        evaluation, which decides advance-to-running vs.
        retry-into-implementing based on the inline review's
        ``CodeReviewResult``. v1's separate ``getReadyToReview`` +
        ``getReadyToRun`` are collapsed here.

        The "all entries terminal" predicate is a join + aggregation;
        plugin implementations push it into SQL (the SQL-shape
        commitment per DEC-030 makes this natural).
        """
        ...

    async def get_ready_to_run(
        self,
        limit: int = 100,
        cursor: str | None = None,
    ) -> CursorPage[RunRecord]:
        """:class:`RunRecord` rows in state ``pending`` (§5.6.3).

        Per ``01-data-model.md`` §5.5 / ``03-state-machine.md`` §3.9 —
        the run sub-state-machine pipeline-spec. Equivalent to
        "implemented entries × seeds whose run record exists with
        state pending" — the run records are pre-created at CG-entry
        into ``running`` state per the v1 pattern (``dao_design.md``
        §3.3), so this query is a simple state filter on
        :class:`RunRecord`. Ordered FIFO by ``created_at``. This query
        feeds the run sub-spec's on-entry dispatch handler for
        ``pending`` (which calls :meth:`Compute.submit`).

        Note that this method returns ``CursorPage[RunRecord]``
        (per-run granularity), distinct from
        :meth:`get_ready_for_review_and_run` which returns
        ``CursorPage[ComparisonGroupRecord]`` (per-CG granularity for
        the inline review gate).
        """
        ...

    async def get_in_flight_runs(
        self,
        limit: int = 100,
        cursor: str | None = None,
    ) -> CursorPage[RunRecord]:
        """Runs with a non-null ``compute_job_handle`` whose state is
        in-progress (e.g., ``submitted``, ``running``). Phase-1
        polling (§5.6.3)."""
        ...

    async def get_ready_to_evaluate(
        self,
        limit: int = 100,
        cursor: str | None = None,
    ) -> CursorPage[ComparisonGroupRecord]:
        """CGs ready for evaluation dispatch (§5.6.3).

        CGs in state ``running`` whose runs have all reached a
        terminal run state. Same all-children-terminal join shape as
        :meth:`get_ready_for_review_and_run`. Per ``03`` §3.7's
        settlement of the single-vs-two-state evaluation question —
        single ``evaluating`` state with sequential dispatches — this
        one query suffices; no split into
        ``get_ready_for_mechanical_evaluation`` /
        ``get_ready_for_contextual_evaluation``.
        """
        ...

    async def get_in_flight_evaluation(
        self,
        limit: int = 100,
        cursor: str | None = None,
    ) -> CursorPage[ComparisonGroupRecord]:
        """CGs in state ``evaluating`` where the on-entry inline-dispatch
        handler is mid-flight (§5.6.3).

        Phase-1 polling for the inline-dispatch lifecycle (per ``05``
        §1.4 — inline failures surface via the same ``job_failed``
        channel as external compute-job failures).
        """
        ...

    # === Scheduling queries — proposal pipeline (§5.6.4) ===

    async def get_ready_for_proposal_design(
        self,
        limit: int = 100,
        cursor: str | None = None,
    ) -> CursorPage[ProposalRecord]:
        """Proposals in state ``proposal_submitted`` with no in-flight
        planner job (§5.6.4). Per ``03`` §4.5 / ``08`` §2.4."""
        ...

    async def get_in_flight_proposal_design(
        self,
        limit: int = 100,
        cursor: str | None = None,
    ) -> CursorPage[ProposalRecord]:
        """Proposals in state ``designing`` with a non-null planner-job
        handle (§5.6.4). Per ``03`` §4.5 / ``08`` §2.4. Phase-1
        polling."""
        ...

    async def get_proposals_at_human_gate(
        self,
        limit: int = 100,
        cursor: str | None = None,
    ) -> CursorPage[ProposalRecord]:
        """Proposals parked in state ``designed`` (§5.6.4).

        Returns ALL proposals in ``designed`` regardless of
        :attr:`ProposalRecord.user_decision`. The worker drives
        gate-rule evaluation each cycle: the three ``dispatch_time``
        edges out of ``designed`` (``proposal.user_approved``,
        ``proposal.user_rejected``, ``proposal.registration_exhausted``
        per ``03`` §4.2 / ``08`` §2.2) read ``user_decision`` from the
        record and determine fire/wait. Registration runs as the
        on-entry dispatch of ``registered``.

        Note: ``08`` §2.4 line 113's "with no user-decision flag set"
        phrasing predates the gate-driven dispatch model in §2.2 line 86
        and is slated for Yaarp-side reconciliation; this Protocol
        surface follows §2.2.
        """
        ...

    # === Scheduling queries — paper-ingestion supporting utility (§5.6.5) ===

    async def get_ready_for_paper_fetch(
        self,
        limit: int = 100,
        cursor: str | None = None,
    ) -> CursorPage[PaperRecord]:
        """Papers in state ``submitted`` with no in-flight fetch job
        (§5.6.5)."""
        ...

    async def get_in_flight_paper_fetch(
        self,
        limit: int = 100,
        cursor: str | None = None,
    ) -> CursorPage[PaperRecord]: ...

    async def get_ready_for_paper_screen(
        self,
        limit: int = 100,
        cursor: str | None = None,
    ) -> CursorPage[PaperRecord]:
        """Papers in state ``screening`` with no in-flight screener job
        (``screener_job_handle is null``) (§5.6.5).

        The screener dispatch fires from the ``screening`` state's
        on-entry handler.
        """
        ...

    async def get_in_flight_paper_screen(
        self,
        limit: int = 100,
        cursor: str | None = None,
    ) -> CursorPage[PaperRecord]: ...

    async def get_ready_for_paper_plan(
        self,
        limit: int = 100,
        cursor: str | None = None,
    ) -> CursorPage[PaperRecord]:
        """Papers in state ``planning`` with no in-flight planner job
        (§5.6.5).

        The paper-ingestion-variant planner per ``08`` §5.3.
        """
        ...

    async def get_in_flight_paper_plan(
        self,
        limit: int = 100,
        cursor: str | None = None,
    ) -> CursorPage[PaperRecord]: ...

    async def get_partial_pending_promotion(
        self,
        limit: int = 100,
        cursor: str | None = None,
    ) -> CursorPage[PaperRecord]:
        """Papers in state ``partial`` with a user-set promotion flag
        (§5.6.5).

        The engine just observes; phase-3 dispatch fires when the flag
        flips. Per ``03`` §5.4. Promotion is a synchronous user-driven
        write through :meth:`transition_paper_state` from ``partial``
        to ``submitted``; this query exists for tooling/observability
        ("what's awaiting promotion?"), not as a phase-2 work source —
        ``partial`` papers wait there indefinitely until the user
        explicitly promotes.
        """
        ...

    # === In-flight count aggregations (§5.6.6) ===

    async def count_in_state(
        self,
        entity_kind: Literal["cg", "entry", "run", "proposal", "paper"],
        states: list[str],
    ) -> int:
        """Count of entities of the given kind whose state is in
        ``states`` (§5.6.6).

        Multi-state input lets the engine aggregate "all in-progress
        states in pool P" in one call rather than N (e.g., the run
        pool's in-progress states are ``{submitted, running}``).

        SQL implementation: ``SELECT COUNT(*) FROM <table> WHERE state
        IN (...)``. Single round-trip; index on ``state`` per
        ``01-data-model.md`` §5 (Session-C-settled).
        """
        ...

    async def count_with_in_flight_jobs(
        self,
        entity_kind: Literal["cg", "entry", "run", "proposal", "paper"],
        in_flight_states: list[str],
    ) -> int:
        """Count of entities of the given kind currently holding an
        external :class:`Compute` job (§5.6.6 / DEC-035 #3).

        The "is in-flight" predicate is ``state IN :in_flight_states
        AND job_handle IS NOT NULL``. The caller (the orchestrator
        engine) resolves the pool-to-states mapping from the
        pipeline-spec it owns and passes the resolved state list here
        — the plugin stays spec-agnostic per DEC-029 / DEC-030.

        SQL implementation: ``SELECT COUNT(*) FROM <table> WHERE state
        IN (...) AND job_handle IS NOT NULL``. Single round-trip;
        partial index on ``(state, job_handle IS NOT NULL)`` scoped to
        the in-progress states keeps this O(log n) on the hot path.
        """
        ...

    # === Lease acquisition / release / extension (§5.6.7) ===

    async def acquire_lease(
        self,
        entity_kind: EntityKind,
        entity_id: str,
        lease_seconds: int,
        lease_holder_id: str,
    ) -> LeaseToken | None:
        """Atomically acquire a lease on the entity (§5.6.7 / DEC-035 #2).

        Returns a :class:`LeaseToken` on success; returns ``None`` on
        failure (the lease is held by another worker AND not yet
        expired).

        Lease acquisition is **independent** of entity-state CAS —
        there is NO ``expected_version`` parameter. The entity's
        ``version`` is managed by ``transition_*_state`` and is
        unrelated to lease ops; coupling would over-determine the
        failure cases (a busy lease AND a stale version would both
        surface as ``None``, hiding which condition fired).

        Implementation (illustrative SQL)::

            UPDATE <entity_table>
               SET leased_by = :holder,
                   lease_expires_at = now() + :seconds * interval '1 second',
                   lease_nonce = gen_random_uuid()
             WHERE id = :entity_id
               AND (lease_expires_at IS NULL OR lease_expires_at < now())
            RETURNING leased_by, lease_expires_at, lease_nonce

        No version condition in the predicate. The SET clause does
        not bump ``version`` — lease ops are not state-affecting
        writes.

        On no-row-affected: return ``None`` (caller treats as "another
        worker holds it; skip this entity this cycle"). Note that
        this method does NOT raise :class:`ConflictError`; callers
        use lease acquisition as an opportunistic try; failure is
        normal.

        On success: return :class:`LeaseToken`.
        """
        ...

    async def release_lease(self, lease_token: LeaseToken) -> None:
        """Voluntary release (§5.6.7).

        Clears ``leased_by`` / ``lease_expires_at`` / ``lease_nonce``
        on the entity, gated **on ``nonce`` matching only** (so a
        stale token from an expired-and-reacquired lease is a no-op,
        not a corruption).

        Idempotent: releasing an already-released or already-expired
        lease is a no-op (no raise). A token whose ``entity_id``
        doesn't exist in :class:`MetadataStore` is also silently a
        no-op (the UPDATE matches 0 rows); same idempotency contract
        as already-stale tokens.

        Note: voluntary release is rare in normal operation —
        workers complete the dispatch handler, then transition the
        entity (which also clears the lease as part of the same
        write per the spec's transition handler). Voluntary release
        exists for the dispatch-handler-failed path (the engine
        releases the lease so the next poll cycle can retry without
        waiting for ``lease_seconds``).
        """
        ...

    async def extend_lease(
        self,
        lease_token: LeaseToken,
        additional_seconds: int,
    ) -> LeaseToken:
        """Extend the lease by ``additional_seconds`` past the current
        ``expires_at`` (§5.6.7).

        Returns an updated :class:`LeaseToken` with the new
        ``expires_at`` and a fresh ``nonce``. Used by long-running
        inline dispatch handlers (e.g., a multi-turn planner agent
        loop running > ``lease_seconds``) to keep the lease alive
        without releasing.

        Gated **on the current nonce matching** — if the lease has
        already expired and been reacquired by another worker,
        extension fails and MUST raise :class:`LeaseLostError` (NOT
        silently succeed). The original worker has lost its lease
        and must abort its dispatch.
        """
        ...

    # === Transactional grouping (§5.3) ===

    async def transaction(self) -> AbstractAsyncContextManager[Transaction]:
        """Provide cross-entity atomicity (§5.3).

        Inside the context, all write operations succeed or fail
        together. Enabled by SQL-only commitment per DEC-030 — no
        per-backend bound on operation count.
        """
        ...
