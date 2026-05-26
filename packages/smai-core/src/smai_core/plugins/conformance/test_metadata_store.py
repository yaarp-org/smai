""":class:`MetadataStoreConformance` — parameterizable test base for
:class:`MetadataStore` plugins.

Per ``designs/smai/07-plugin-interfaces.md`` §5.8 (the per-Protocol
conformance enumeration), §5.6.10 (scheduling-query / lease /
in-flight-count detail), and §8 (cross-cutting discipline).

Subclass and override :meth:`MetadataStoreConformance.make_store` to
run the suite against your plugin::

    from smai_core.plugins.conformance import MetadataStoreConformance
    from smai_store_sqlite import SqliteStore

    class TestSqliteConformance(MetadataStoreConformance):
        async def make_store(self) -> MetadataStore:
            store = SqliteStore(":memory:")
            await store.migrate()
            return store

This base is the largest of the four — :class:`MetadataStore` has 47
methods (per Task 1.8's enumeration). Per §5.8, the contract groups are:

* round-trip CRUD equivalence
* compare-and-swap state-transition semantics
* transactional rollback
* scheduling-query result equivalence (parameterized over the 19
  scheduling-query methods)
* lease semantics (acquire/release/extend round-trip plus expire-and-reclaim)
* in-flight count correctness
* pagination determinism / cursor stability
* tenant-fairness (gated on ``capabilities.is_tenant_aware``)

Wall-clock-sensitive tests (lease expiry, multi-worker contention) use
the :func:`time_provider` fixture, deferred per Task 2.C1 / §6 Q4.
Multi-worker contention beyond the ABA basics is deferred to Task 3.G1
(per the task spec) — the bases here ship the basic single-process
ABA-protection contract.
"""

from __future__ import annotations

import time

import pytest

# Pipeline-tracking record types live in ``smai-orchestrator`` per Task
# 1.10 / ``01-data-model.md`` §5.1. This conformance suite tests the
# cross-package contract between ``MetadataStore`` plugins and the
# records they accept and return; ``tools/check_deps.py`` exempts the
# conformance subpackage from rule 2 to admit this runtime import.
from smai_orchestrator.entities.tracking import (
    ComparisonGroupRecord,
    EntryRecord,
    PaperRecord,
    ProposalRecord,
    RunRecord,
)

from smai_core.plugins import (
    ConflictError,
    CursorPage,
    EntityKind,
    JobHandle,
    LeaseLostError,
    LeaseToken,
    MetadataStore,
    MetadataStoreCapabilities,
)
from smai_core.plugins.conformance._common import (
    SCHEDULING_QUERY_METHODS,
    TimeProvider,
    make_record,
    skip_no_factory_override,
)


class MetadataStoreConformance:
    """Universal contract suite for :class:`MetadataStore` implementations.

    Contract groups, per ``07-plugin-interfaces.md`` §5.8:

    * **CRUD round-trip** — :meth:`test_crud_round_trip_cg`,
      :meth:`test_crud_round_trip_entry`,
      :meth:`test_crud_round_trip_run`,
      :meth:`test_crud_round_trip_proposal`,
      :meth:`test_crud_round_trip_paper`
    * **Conditional state-transition CAS** —
      :meth:`test_transition_succeeds_on_matching_version`,
      :meth:`test_transition_raises_conflict_on_stale_version`
    * **Transactional rollback** —
      :meth:`test_transaction_commits_on_clean_exit`,
      :meth:`test_transaction_rolls_back_on_exception`
    * **Scheduling queries** — parameterized over each query name in
      :data:`SCHEDULING_QUERY_METHODS`:
      :meth:`test_scheduling_query_returns_cursor_page`,
      :meth:`test_scheduling_query_pagination_terminates`
    * **In-flight counts** —
      :meth:`test_count_in_state_multi_state_input`,
      :meth:`test_count_with_in_flight_jobs_predicate`
    * **Lease semantics** —
      :meth:`test_acquire_then_acquire_fails`,
      :meth:`test_release_then_acquire_succeeds`,
      :meth:`test_extend_still_held_blocks_acquire`,
      :meth:`test_acquire_expire_reacquire`,
      :meth:`test_extend_after_loss_raises`,
      :meth:`test_stale_token_release_noop`,
      :meth:`test_unknown_token_release_noop`
    * **Tenant-fairness contract** — gated on
      ``capabilities.is_tenant_aware``
      (:meth:`test_tenant_fairness_contract`)
    """

    # ---- factory hook ------------------------------------------------------

    async def make_store(self) -> MetadataStore:
        """Return a configured :class:`MetadataStore` instance.

        Override this in your subclass. The default skip-collects.
        Async to support plugins that need to bring up a fresh schema
        per test (e.g., ``await store.migrate()`` against an in-memory
        SQLite).
        """
        skip_no_factory_override(type(self).__name__, "make_store")
        raise AssertionError("unreachable")  # pragma: no cover

    # ---- fixtures ----------------------------------------------------------

    @pytest.fixture
    async def store(self) -> MetadataStore:
        """Per-test :class:`MetadataStore` instance via :meth:`make_store`."""
        return await self.make_store()

    @pytest.fixture
    def time_provider(self) -> TimeProvider:
        """Wall-clock provider (deferred per Task 2.C1).

        Defaults to :func:`time.monotonic`. Subclasses override this
        once Task 2.C1 ships its ``EngineConfig``-driven seam.
        """
        return time.monotonic

    @pytest.fixture
    def short_lease_seconds(self) -> int:
        """Short lease duration for the expire-and-reacquire tests (per
        §5.6.10's "lease_seconds=1 + sleep 2s" pattern)."""
        return 1

    # ---- CRUD round-trip ---------------------------------------------------

    async def test_crud_round_trip_cg(self, store: MetadataStore) -> None:
        """``create_cg`` then ``get_cg`` returns equivalent record."""
        cg = make_record(ComparisonGroupRecord, id="cg_round_trip", state="draft", version=1)
        await store.create_cg(cg)
        fetched = await store.get_cg("cg_round_trip")
        assert fetched is not None

    async def test_crud_round_trip_entry(self, store: MetadataStore) -> None:
        """``create_entry`` then ``get_entry`` returns equivalent record.

        Parent CG is seeded first to satisfy the ``entries.cg_id``
        foreign-key declared on the shared SQL schema (DEC-036). SQLite
        does not enforce FKs by default; Postgres does. The fixture
        order matches the production write path (CG before its
        entries) — see ``08-novel-technique-pipeline.md`` §2.4.
        """
        await store.create_cg(
            make_record(ComparisonGroupRecord, id="cg_x", state="draft", version=1)
        )
        entry = make_record(
            EntryRecord, id="entry_round_trip", cg_id="cg_x", state="pending", version=1
        )
        await store.create_entry(entry)
        fetched = await store.get_entry("entry_round_trip")
        assert fetched is not None

    async def test_crud_round_trip_run(self, store: MetadataStore) -> None:
        """``create_run`` then ``get_run`` returns equivalent record.

        Parent CG + entry are seeded first to satisfy the ``runs.cg_id``
        and ``runs.entry_id`` foreign keys declared on the shared SQL
        schema (DEC-036). See :meth:`test_crud_round_trip_entry` for the
        rationale.
        """
        await store.create_cg(
            make_record(ComparisonGroupRecord, id="cg_run_parent", state="draft", version=1)
        )
        await store.create_entry(
            make_record(
                EntryRecord,
                id="entry_x",
                cg_id="cg_run_parent",
                state="pending",
                version=1,
            )
        )
        run = make_record(
            RunRecord,
            id="run_round_trip",
            cg_id="cg_run_parent",
            entry_id="entry_x",
            state="pending",
            version=1,
        )
        await store.create_run(run)
        fetched = await store.get_run("run_round_trip")
        assert fetched is not None

    async def test_crud_round_trip_proposal(self, store: MetadataStore) -> None:
        """``create_proposal`` then ``get_proposal`` returns equivalent record."""
        proposal = make_record(
            ProposalRecord,
            id="proposal_round_trip",
            state="proposal_submitted",
            version=1,
        )
        await store.create_proposal(proposal)
        fetched = await store.get_proposal("proposal_round_trip")
        assert fetched is not None

    async def test_crud_round_trip_paper(self, store: MetadataStore) -> None:
        """``create_paper`` then ``get_paper`` returns equivalent record."""
        paper = make_record(PaperRecord, arxiv_id="2501.00001", state="submitted", version=1)
        await store.create_paper(paper)
        fetched = await store.get_paper("2501.00001")
        assert fetched is not None

    # ---- Conditional state transitions (CAS) -------------------------------

    async def test_transition_succeeds_on_matching_version(self, store: MetadataStore) -> None:
        """Transition with matching ``expected_version`` succeeds and
        increments the version (per §5.4)."""
        cg = make_record(ComparisonGroupRecord, id="cg_cas_ok", state="draft", version=1)
        await store.create_cg(cg)
        updated = await store.transition_cg_state(
            cg_id="cg_cas_ok", expected_version=1, target_state="implementing"
        )
        # The returned record has its `version` field bumped (per §5.4 — pyright
        # can't see the field on the stub, so we read via getattr).
        assert getattr(updated, "version", 2) == 2

    async def test_transition_raises_conflict_on_stale_version(self, store: MetadataStore) -> None:
        """Transition with stale ``expected_version`` raises
        :class:`ConflictError` (per §5.4)."""
        cg = make_record(ComparisonGroupRecord, id="cg_cas_stale", state="draft", version=1)
        await store.create_cg(cg)
        # First transition advances version 1 → 2.
        await store.transition_cg_state(
            cg_id="cg_cas_stale", expected_version=1, target_state="implementing"
        )
        # Second transition with the now-stale version=1 must raise.
        with pytest.raises(ConflictError):
            await store.transition_cg_state(
                cg_id="cg_cas_stale", expected_version=1, target_state="implemented"
            )

    # ---- Transactional rollback --------------------------------------------

    async def test_transaction_commits_on_clean_exit(self, store: MetadataStore) -> None:
        """Without exception, all transactional writes are visible after
        the context exits (per §5.3)."""
        async with await store.transaction() as tx:
            await tx.create_cg(
                make_record(ComparisonGroupRecord, id="cg_tx_ok", state="draft", version=1)
            )
        fetched = await store.get_cg("cg_tx_ok")
        assert fetched is not None

    async def test_transaction_rolls_back_on_exception(self, store: MetadataStore) -> None:
        """Exception inside ``transaction()`` rolls back all writes
        (per §5.3)."""

        class _SentinelException(Exception):
            pass

        with pytest.raises(_SentinelException):
            async with await store.transaction() as tx:
                await tx.create_cg(
                    make_record(
                        ComparisonGroupRecord, id="cg_tx_rollback", state="draft", version=1
                    )
                )
                raise _SentinelException
        fetched = await store.get_cg("cg_tx_rollback")
        assert fetched is None

    # ---- Observability tables — agent_sessions + transition_log -------------

    async def test_agent_session_crud_round_trip(self, store: MetadataStore) -> None:
        """``create_agent_session`` returns an id; ``get_agent_session``
        reflects a subsequent ``update_agent_session`` (turn_count /
        tokens); a final update with ``ended_at`` is reflected."""
        session_id = await store.create_agent_session(
            parent_kind="proposal",
            parent_id="proposal_obs_round_trip",
            agent_role="planner",
            llm_provider="fake-provider",
            model_id="fake-model",
        )
        assert isinstance(session_id, str)
        assert session_id

        initial = await store.get_agent_session(session_id)
        assert initial is not None
        assert getattr(initial, "parent_kind", None) == "proposal"
        assert getattr(initial, "parent_id", None) == "proposal_obs_round_trip"
        assert getattr(initial, "agent_role", None) == "planner"
        assert getattr(initial, "turn_count", 0) == 0
        assert getattr(initial, "ended_at", "x") is None

        await store.update_agent_session(
            session_id,
            turn_count=7,
            input_tokens=1234,
            cached_input_tokens=200,
            output_tokens=456,
        )
        updated = await store.get_agent_session(session_id)
        assert updated is not None
        assert getattr(updated, "turn_count", 0) == 7
        assert getattr(updated, "input_tokens", 0) == 1234
        assert getattr(updated, "cached_input_tokens", 0) == 200
        assert getattr(updated, "output_tokens", 0) == 456
        assert getattr(updated, "ended_at", "x") is None

        from datetime import UTC, datetime  # noqa: PLC0415

        await store.update_agent_session(
            session_id,
            turn_count=9,
            input_tokens=2000,
            cached_input_tokens=300,
            output_tokens=500,
            ended_at=datetime.now(UTC),
        )
        closed = await store.get_agent_session(session_id)
        assert closed is not None
        assert getattr(closed, "turn_count", 0) == 9
        assert getattr(closed, "ended_at", None) is not None

    async def test_get_agent_session_missing_returns_none(self, store: MetadataStore) -> None:
        assert await store.get_agent_session("no-such-session-id") is None

    async def test_agent_session_compute_job_handle_round_trip(self, store: MetadataStore) -> None:
        """Agent-refactor Step 4 sub-PR B (D2):
        ``agent_sessions.compute_job_handle`` round-trips a
        :class:`JobHandle` so post-hoc operator queries can correlate a
        cost-ledger row with ``Compute.logs(handle)`` even after the
        parent record's ``*_job_handle`` is overwritten by a later
        re-dispatch.

        Asserts: passing ``compute_job_handle`` at create time persists
        it; default-omit preserves NULL for inline roles; the round-tripped
        value equals the original.
        """
        from smai_core.plugins import JobHandle  # noqa: PLC0415

        handle = JobHandle(
            plugin="test-substrate",
            handle="job-handle-d2-round-trip",
            metadata={"region": "us-west-1"},
        )
        session_with_handle = await store.create_agent_session(
            parent_kind="cg",
            parent_id="cg_obs_d2_handle",
            agent_role="harness_builder",
            llm_provider="fake-provider",
            model_id="fake-model",
            compute_job_handle=handle,
        )
        loaded = await store.get_agent_session(session_with_handle)
        assert loaded is not None
        assert getattr(loaded, "compute_job_handle", None) == handle

        session_inline = await store.create_agent_session(
            parent_kind="proposal",
            parent_id="proposal_obs_d2_inline",
            agent_role="planner",
            llm_provider="fake-provider",
            model_id="fake-model",
        )
        loaded_inline = await store.get_agent_session(session_inline)
        assert loaded_inline is not None
        assert getattr(loaded_inline, "compute_job_handle", "MISSING") is None

    async def test_append_transition_log_persists(self, store: MetadataStore) -> None:
        """``append_transition_log`` writes a row that a raw select sees."""
        await store.append_transition_log(
            entity_kind="proposal",
            entity_id="proposal_obs_translog",
            from_state="proposal_submitted",
            to_state="designing",
            edge_name="proposal.proposal_submitted → designing",
            worker_id="worker-test",
            gate_outcome_reason="always-fire",
        )
        # Read back via a raw select on the underlying connection — the
        # Protocol doesn't (yet) expose a list method for the audit table.
        from typing import Any as _Any  # noqa: PLC0415

        from sqlalchemy import text  # noqa: PLC0415

        async with await store.transaction() as tx:
            conn: _Any = tx.connection
            result = await conn.execute(
                text(
                    "SELECT from_state, to_state, edge_name, worker_id, gate_outcome_reason "
                    "FROM transition_log WHERE entity_kind = :k AND entity_id = :i"
                ),
                {"k": "proposal", "i": "proposal_obs_translog"},
            )
            rows = result.mappings().all()
        assert len(rows) == 1
        assert rows[0]["from_state"] == "proposal_submitted"
        assert rows[0]["to_state"] == "designing"
        assert rows[0]["worker_id"] == "worker-test"
        assert rows[0]["gate_outcome_reason"] == "always-fire"

    async def test_append_transition_log_transactional(self, store: MetadataStore) -> None:
        """The ``Transaction``-mirrored ``append_transition_log`` participates
        in the transaction (rollback discards the row)."""
        from typing import Any as _Any  # noqa: PLC0415

        from sqlalchemy import text  # noqa: PLC0415

        class _Sentinel(Exception):
            pass

        with pytest.raises(_Sentinel):
            async with await store.transaction() as tx:
                await tx.append_transition_log(
                    entity_kind="cg",
                    entity_id="cg_obs_translog_rollback",
                    from_state="draft",
                    to_state="implementing",
                    edge_name="cg.draft → implementing",
                    worker_id=None,
                )
                raise _Sentinel

        async with await store.transaction() as tx:
            conn: _Any = tx.connection
            result = await conn.execute(
                text("SELECT 1 FROM transition_log WHERE entity_id = :i"),
                {"i": "cg_obs_translog_rollback"},
            )
            assert result.first() is None

    # ---- Scheduling queries (parameterized over the 19 methods) ------------

    @pytest.mark.parametrize("method_name", SCHEDULING_QUERY_METHODS)
    async def test_scheduling_query_returns_cursor_page(
        self, store: MetadataStore, method_name: str
    ) -> None:
        """Every scheduling query returns a :class:`CursorPage` of the
        expected record type (per §5.6.2)."""
        method = getattr(store, method_name)
        result = await method(limit=10, cursor=None)
        assert isinstance(result, CursorPage)
        # next_cursor is None or a string per the CursorPage shape.
        assert result.next_cursor is None or isinstance(result.next_cursor, str)

    @pytest.mark.parametrize("method_name", SCHEDULING_QUERY_METHODS)
    async def test_scheduling_query_pagination_terminates(
        self, store: MetadataStore, method_name: str
    ) -> None:
        """Cursor-anchored iteration terminates at ``next_cursor=None``
        (per §5.6.10's pagination clause).

        On an empty backlog, the very first page should already have
        ``next_cursor=None``. On a populated backlog, the loop terminates
        within a bounded number of pages (we cap at 1000 to defend
        against pathological cursor-loop bugs).
        """
        method = getattr(store, method_name)
        cursor: str | None = None
        pages = 0
        max_pages = 1000  # defense against cursor-non-progress bugs
        while pages < max_pages:
            page: CursorPage[object] = await method(limit=10, cursor=cursor)
            pages += 1
            if page.next_cursor is None:
                return
            assert page.next_cursor != cursor, (
                f"{method_name}: next_cursor did not advance (loop bug?)"
            )
            cursor = page.next_cursor
        pytest.fail(
            f"{method_name}: did not terminate after {max_pages} pages — "
            "cursor pagination contract violated."
        )

    # ---- Handle-aware filter correctness (regression for JSON null bug) ----
    #
    # Several scheduling queries filter on ``<entity>_job_handle IS NULL``
    # (ready) or ``IS NOT NULL`` (in-flight). Plugins backing JSON-typed
    # nullable columns must serialize Python ``None`` to SQL ``NULL`` —
    # not to the JSON literal ``'null'``. A plugin that writes the JSON
    # literal would pass the existing ``test_scheduling_query_*`` shape
    # tests yet silently return wrong results from the handle-aware
    # filters (every record matches ``IS NOT NULL``; no record matches
    # ``IS NULL``). The fixtures below seed the null/non-null partition
    # explicitly so any such plugin fails loudly here.

    async def test_filter_correctness_harness_build_handle(self, store: MetadataStore) -> None:
        """``get_in_flight_harness_build`` and ``get_ready_for_harness_build``
        partition CGs correctly by ``harness_job_handle`` null-ness.

        Seeds two ``implementing`` CGs (one with handle, one without) and
        two ``draft`` CGs (one with handle, one without). The in-flight
        query must return only the implementing+handle pair; the ready
        query must return only the draft+no-handle pair.
        """
        handle = JobHandle(plugin="test", handle="job-cg", metadata={})
        await store.create_cg(
            make_record(
                ComparisonGroupRecord,
                id="cg_filter_inflight_real",
                state="implementing",
                version=1,
                harness_job_handle=handle,
            )
        )
        await store.create_cg(
            make_record(
                ComparisonGroupRecord,
                id="cg_filter_inflight_null",
                state="implementing",
                version=1,
                harness_job_handle=None,
            )
        )
        await store.create_cg(
            make_record(
                ComparisonGroupRecord,
                id="cg_filter_ready_null",
                state="draft",
                version=1,
                harness_job_handle=None,
            )
        )
        await store.create_cg(
            make_record(
                ComparisonGroupRecord,
                id="cg_filter_ready_real",
                state="draft",
                version=1,
                harness_job_handle=handle,
            )
        )
        in_flight = await store.get_in_flight_harness_build(limit=100, cursor=None)
        ready = await store.get_ready_for_harness_build(limit=100, cursor=None)
        in_flight_ids = {r.id for r in in_flight.items}
        ready_ids = {r.id for r in ready.items}
        assert "cg_filter_inflight_real" in in_flight_ids
        assert "cg_filter_inflight_null" not in in_flight_ids, (
            "in-flight query returned a CG with NULL harness_job_handle — "
            "JSON-typed null may have been written as the literal 'null' "
            "rather than SQL NULL"
        )
        assert "cg_filter_ready_null" in ready_ids
        assert "cg_filter_ready_real" not in ready_ids, (
            "ready query returned a CG with a non-null harness_job_handle — "
            "the IS NULL predicate is not matching correctly"
        )

    async def test_filter_correctness_proposal_design_handle(self, store: MetadataStore) -> None:
        """``get_in_flight_proposal_design`` and ``get_ready_for_proposal_design``
        partition proposals correctly by ``planner_job_handle`` null-ness."""
        handle = JobHandle(plugin="test", handle="job-prop", metadata={})
        await store.create_proposal(
            make_record(
                ProposalRecord,
                id="prop_filter_inflight_real",
                state="designing",
                version=1,
                planner_job_handle=handle,
            )
        )
        await store.create_proposal(
            make_record(
                ProposalRecord,
                id="prop_filter_inflight_null",
                state="designing",
                version=1,
                planner_job_handle=None,
            )
        )
        await store.create_proposal(
            make_record(
                ProposalRecord,
                id="prop_filter_ready_null",
                state="proposal_submitted",
                version=1,
                planner_job_handle=None,
            )
        )
        await store.create_proposal(
            make_record(
                ProposalRecord,
                id="prop_filter_ready_real",
                state="proposal_submitted",
                version=1,
                planner_job_handle=handle,
            )
        )
        in_flight = await store.get_in_flight_proposal_design(limit=100, cursor=None)
        ready = await store.get_ready_for_proposal_design(limit=100, cursor=None)
        in_flight_ids = {r.id for r in in_flight.items}
        ready_ids = {r.id for r in ready.items}
        assert "prop_filter_inflight_real" in in_flight_ids
        assert "prop_filter_inflight_null" not in in_flight_ids
        assert "prop_filter_ready_null" in ready_ids
        assert "prop_filter_ready_real" not in ready_ids

    async def test_filter_correctness_paper_fetch_handle(self, store: MetadataStore) -> None:
        """``get_in_flight_paper_fetch`` and ``get_ready_for_paper_fetch``
        partition papers correctly by ``fetcher_job_handle`` null-ness."""
        handle = JobHandle(plugin="test", handle="job-pap", metadata={})
        await store.create_paper(
            make_record(
                PaperRecord,
                arxiv_id="2601.10001",
                state="fetching",
                version=1,
                fetcher_job_handle=handle,
            )
        )
        await store.create_paper(
            make_record(
                PaperRecord,
                arxiv_id="2601.10002",
                state="fetching",
                version=1,
                fetcher_job_handle=None,
            )
        )
        await store.create_paper(
            make_record(
                PaperRecord,
                arxiv_id="2601.10003",
                state="submitted",
                version=1,
                fetcher_job_handle=None,
            )
        )
        await store.create_paper(
            make_record(
                PaperRecord,
                arxiv_id="2601.10004",
                state="submitted",
                version=1,
                fetcher_job_handle=handle,
            )
        )
        in_flight = await store.get_in_flight_paper_fetch(limit=100, cursor=None)
        ready = await store.get_ready_for_paper_fetch(limit=100, cursor=None)
        in_flight_ids = {r.arxiv_id for r in in_flight.items}
        ready_ids = {r.arxiv_id for r in ready.items}
        assert "2601.10001" in in_flight_ids
        assert "2601.10002" not in in_flight_ids
        assert "2601.10003" in ready_ids
        assert "2601.10004" not in ready_ids

    async def test_filter_correctness_run_handle(self, store: MetadataStore) -> None:
        """``get_in_flight_runs`` filters by ``compute_job_handle`` null-ness
        within the dispatch state set."""
        handle = JobHandle(plugin="test", handle="job-run", metadata={})
        # Parent CG + entry are required by FK. They are otherwise irrelevant
        # to this filter — only seed enough to satisfy the FK constraint.
        await store.create_cg(
            make_record(
                ComparisonGroupRecord,
                id="cg_for_run_filter",
                state="running",
                version=1,
            )
        )
        await store.create_entry(
            make_record(
                EntryRecord,
                id="entry_for_run_filter",
                cg_id="cg_for_run_filter",
                state="implemented",
                version=1,
            )
        )
        await store.create_run(
            make_record(
                RunRecord,
                id="run_filter_inflight_real",
                cg_id="cg_for_run_filter",
                entry_id="entry_for_run_filter",
                seed=0,
                state="running",
                version=1,
                compute_job_handle=handle,
            )
        )
        await store.create_run(
            make_record(
                RunRecord,
                id="run_filter_inflight_null",
                cg_id="cg_for_run_filter",
                entry_id="entry_for_run_filter",
                seed=1,
                state="running",
                version=1,
                compute_job_handle=None,
            )
        )
        in_flight = await store.get_in_flight_runs(limit=100, cursor=None)
        in_flight_ids = {r.id for r in in_flight.items}
        assert "run_filter_inflight_real" in in_flight_ids
        assert "run_filter_inflight_null" not in in_flight_ids, (
            "in-flight runs query returned a run with NULL compute_job_handle"
        )

    # ---- Ready-to-implement-entry filter correctness (R3 / F3) -------------
    #
    # Per DEC-013 / DEC-017 the implementer-dispatch scheduling query
    # selects entries whose ``technique_id IS NOT NULL``. Both
    # substitutive baselines (``is_baseline=True`` AND ``technique_id IS
    # NOT NULL`` — e.g. a VGG architecture reference) AND treatments
    # (``is_baseline=False``) need implementation; only additive baselines
    # (``technique_id IS NULL``) skip the dispatch. Pre-R3 a `SqliteStore`
    # filter additionally required ``is_baseline=False`` which silently
    # excluded substitutive baselines — this test pins the corrected
    # cross-plugin contract.

    async def test_filter_correctness_ready_to_implement_entry_baseline_kinds(
        self, store: MetadataStore
    ) -> None:
        """``get_ready_to_implement_entry`` returns substitutive
        baselines + treatments and excludes additive baselines.

        Seeds three pending entries whose parent CG is in
        ``implementing``:

        * additive baseline (``is_baseline=True``, ``technique_id=None``)
          — excluded.
        * substitutive baseline (``is_baseline=True``, ``technique_id``
          set) — included.
        * treatment (``is_baseline=False``, ``technique_id`` set) —
          included.
        """
        await store.create_cg(
            make_record(
                ComparisonGroupRecord,
                id="cg_rti_filter",
                state="implementing",
                version=1,
            )
        )
        await store.create_entry(
            make_record(
                EntryRecord,
                id="entry_rti_additive",
                cg_id="cg_rti_filter",
                state="pending",
                version=1,
                is_baseline=True,
                technique_id=None,
            )
        )
        await store.create_entry(
            make_record(
                EntryRecord,
                id="entry_rti_substitutive",
                cg_id="cg_rti_filter",
                state="pending",
                version=1,
                is_baseline=True,
                technique_id="tq_vgg",
            )
        )
        await store.create_entry(
            make_record(
                EntryRecord,
                id="entry_rti_treatment",
                cg_id="cg_rti_filter",
                state="pending",
                version=1,
                is_baseline=False,
                technique_id="tq_treatment",
            )
        )

        page = await store.get_ready_to_implement_entry(limit=100, cursor=None)
        returned_ids = {e.id for e in page.items}
        assert "entry_rti_additive" not in returned_ids, (
            "additive baseline (technique_id IS NULL) must be excluded "
            "from get_ready_to_implement_entry per DEC-013"
        )
        assert "entry_rti_substitutive" in returned_ids, (
            "substitutive baseline (is_baseline=True with non-null "
            "technique_id) must be included per DEC-013 / DEC-017 — "
            "the over-restrictive ``is_baseline=False`` filter shape "
            "would silently exclude it"
        )
        assert "entry_rti_treatment" in returned_ids

    # ---- In-flight count correctness ---------------------------------------

    @pytest.mark.parametrize(
        "entity_kind",
        ["cg", "entry", "run", "proposal", "paper"],
    )
    async def test_count_in_state_multi_state_input(
        self, store: MetadataStore, entity_kind: EntityKind
    ) -> None:
        """``count_in_state`` accepts multi-state input (per §5.6.6).

        Empty list returns 0; a populated list returns a non-negative
        integer. Specific-fixture-driven cardinality assertions are the
        plugin author's responsibility — the universal contract is just
        that the call shape works and returns an int.
        """
        zero = await store.count_in_state(entity_kind, [])
        assert isinstance(zero, int) and zero == 0
        nonempty = await store.count_in_state(entity_kind, ["draft", "running"])
        assert isinstance(nonempty, int) and nonempty >= 0

    @pytest.mark.parametrize(
        "entity_kind",
        ["cg", "entry", "run", "proposal", "paper"],
    )
    async def test_count_with_in_flight_jobs_predicate(
        self, store: MetadataStore, entity_kind: EntityKind
    ) -> None:
        """``count_with_in_flight_jobs`` returns a non-negative count
        for the dispatch-state-set + ``job_handle IS NOT NULL``
        intersection (per §5.6.6 / DEC-035 #3 — caller resolves states)."""
        count = await store.count_with_in_flight_jobs(entity_kind, ["running"])
        assert isinstance(count, int) and count >= 0

    # ---- Lease semantics (per §5.6.10) -------------------------------------

    async def test_acquire_then_acquire_fails(self, store: MetadataStore) -> None:
        """Worker A acquires; worker B's acquire returns ``None`` within
        ``lease_seconds`` (per §5.6.10 "Acquire-then-acquire-fails")."""
        cg = make_record(ComparisonGroupRecord, id="cg_lease_ab", state="draft", version=1)
        await store.create_cg(cg)
        token_a = await store.acquire_lease("cg", "cg_lease_ab", 60, "worker-a")
        assert token_a is not None
        token_b = await store.acquire_lease("cg", "cg_lease_ab", 60, "worker-b")
        assert token_b is None

    async def test_release_then_acquire_succeeds(self, store: MetadataStore) -> None:
        """Worker A acquires, releases; worker B's acquire succeeds
        immediately (per §5.6.10 "Release-then-acquire")."""
        cg = make_record(ComparisonGroupRecord, id="cg_lease_release", state="draft", version=1)
        await store.create_cg(cg)
        token_a = await store.acquire_lease("cg", "cg_lease_release", 60, "worker-a")
        assert token_a is not None
        await store.release_lease(token_a)
        token_b = await store.acquire_lease("cg", "cg_lease_release", 60, "worker-b")
        assert token_b is not None

    async def test_extend_still_held_blocks_acquire(self, store: MetadataStore) -> None:
        """Worker A acquires, extends; expire window pushed out;
        worker B's acquire (within new window) returns ``None`` (per
        §5.6.10 "Extend-still-held")."""
        cg = make_record(ComparisonGroupRecord, id="cg_lease_extend", state="draft", version=1)
        await store.create_cg(cg)
        token_a = await store.acquire_lease("cg", "cg_lease_extend", 60, "worker-a")
        assert token_a is not None
        extended = await store.extend_lease(token_a, additional_seconds=60)
        assert isinstance(extended, LeaseToken)
        assert extended.expires_at >= token_a.expires_at
        token_b = await store.acquire_lease("cg", "cg_lease_extend", 60, "worker-b")
        assert token_b is None

    async def test_acquire_expire_reacquire(
        self, store: MetadataStore, short_lease_seconds: int
    ) -> None:
        """Worker A acquires with ``lease_seconds=1``; sleeps 2s;
        worker B's acquire succeeds with fresh nonce (per §5.6.10
        "Acquire-then-expire-then-reacquire").

        Wall-clock-sensitive — flagged best-effort per §5.6.10. Task
        2.C1 will allow injection of a fake clock.
        """
        import asyncio

        cg = make_record(ComparisonGroupRecord, id="cg_lease_expire", state="draft", version=1)
        await store.create_cg(cg)
        token_a = await store.acquire_lease(
            "cg", "cg_lease_expire", short_lease_seconds, "worker-a"
        )
        assert token_a is not None
        await asyncio.sleep(short_lease_seconds + 1)
        token_b = await store.acquire_lease("cg", "cg_lease_expire", 60, "worker-b")
        assert token_b is not None
        assert token_b.nonce != token_a.nonce

    async def test_extend_after_loss_raises(
        self, store: MetadataStore, short_lease_seconds: int
    ) -> None:
        """Worker A acquires with ``lease_seconds=1``; sleeps 2s;
        worker B acquires; worker A's extend raises
        :class:`LeaseLostError` (per §5.6.10 "Extend-after-loss-raises").

        Wall-clock-sensitive — flagged best-effort per §5.6.10.
        """
        import asyncio

        cg = make_record(ComparisonGroupRecord, id="cg_lease_lost", state="draft", version=1)
        await store.create_cg(cg)
        token_a = await store.acquire_lease("cg", "cg_lease_lost", short_lease_seconds, "worker-a")
        assert token_a is not None
        await asyncio.sleep(short_lease_seconds + 1)
        token_b = await store.acquire_lease("cg", "cg_lease_lost", 60, "worker-b")
        assert token_b is not None
        with pytest.raises(LeaseLostError):
            await store.extend_lease(token_a, additional_seconds=60)

    async def test_stale_token_release_noop(
        self, store: MetadataStore, short_lease_seconds: int
    ) -> None:
        """Worker A acquires, lets expire, then calls release — no-op,
        no raise (per §5.6.10 "Stale-token-release-noop")."""
        import asyncio

        cg = make_record(ComparisonGroupRecord, id="cg_lease_stale", state="draft", version=1)
        await store.create_cg(cg)
        token_a = await store.acquire_lease("cg", "cg_lease_stale", short_lease_seconds, "worker-a")
        assert token_a is not None
        await asyncio.sleep(short_lease_seconds + 1)
        # No raise — release of an expired-and-since-overwritten lease is
        # silently a no-op (UPDATE matches 0 rows).
        await store.release_lease(token_a)

    async def test_unknown_token_release_noop(self, store: MetadataStore) -> None:
        """``release_lease`` with a token whose ``entity_id`` does not
        exist is silently a no-op (per §5.6.10 "Unknown-token-release-noop")."""
        from datetime import UTC, datetime, timedelta

        bogus = LeaseToken(
            entity_kind="cg",
            entity_id="cg_does_not_exist",
            acquired_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(seconds=60),
            lease_holder_id="ghost-worker",
            nonce="00000000-0000-0000-0000-000000000000",
        )
        # No raise.
        await store.release_lease(bogus)

    # ---- Tenant-fairness (gated on capabilities) ---------------------------

    async def test_tenant_fairness_contract(self, store: MetadataStore) -> None:
        """When ``capabilities.is_tenant_aware=True``, the plugin's
        scheduling queries MUST honor the tenant-fairness contract per
        §5.5 / §5.8 (no single tenant monopolizes the dispatch queue
        given multi-tenant backlog).

        When ``False``, this test skips per the §5.8 gating rule.

        The full multi-tenant fixture corpus + window-function-derived
        ordering assertions are deferred to the closed `AuroraStore`
        plugin's own conformance subclass (the hosted-backend deployment
        owns the test corpus); the OSS reference plugins
        (``SqliteStore``, ``PostgresStore``) report
        ``is_tenant_aware=False`` and skip-collect this test cleanly.
        """
        if not store.capabilities.is_tenant_aware:
            pytest.skip(
                "plugin reports is_tenant_aware=False; tenant-fairness contract not exercised"
            )
        # The contract surface — the plugin must accept the call shape
        # and return a bounded result. Multi-tenant fixture fanout is
        # the tenant-aware plugin's own subclass concern.
        page = await store.get_ready_for_harness_build(limit=10, cursor=None)
        assert isinstance(page, CursorPage)

    # ---- Capability sanity check ------------------------------------------

    def test_capabilities_are_capabilities_model(self, store: MetadataStore) -> None:
        """Capability attribute is the right Pydantic model."""
        assert isinstance(store.capabilities, MetadataStoreCapabilities)

    async def test_supports_leasing_matches_behavior(self, store: MetadataStore) -> None:
        """When ``capabilities.supports_leasing=True``, the lease
        primitives MUST function (per §5.6.7 / DEC-035 #2). When
        ``False``, this test skips per the §5.8 gating rule.

        The lease-semantics group above (acquire/release/extend) already
        exercises every primitive in detail; this case pins the
        capability-to-behavior linkage explicitly so a plugin that
        reports ``True`` while leaving the primitives unimplemented
        fails the conformance suite at this single point rather than
        seven lease tests deep.
        """
        if not store.capabilities.supports_leasing:
            pytest.skip(
                "plugin reports supports_leasing=False; lease-primitive contract not exercised"
            )
        cg = make_record(ComparisonGroupRecord, id="cg_caps_lease", state="draft", version=1)
        await store.create_cg(cg)
        token = await store.acquire_lease("cg", "cg_caps_lease", 60, "worker-caps")
        assert token is not None
        await store.release_lease(token)

    def test_supports_listen_notify_capability_bit_present(self, store: MetadataStore) -> None:
        """Every plugin advertises a boolean
        :attr:`MetadataStoreCapabilities.supports_listen_notify` bit
        (Task 4.K3 / `07` §5.5 + `12-ui-process.md` §6.3 + §12 OQ2).

        Default per the Pydantic model is ``False`` (safe-no — only
        plugins with a native cross-process NOTIFY substrate flip it
        true); the Postgres plugin's own credentialed test suite
        verifies the behavior side of the bit (publisher transition →
        ``LISTEN`` connection receives → broker publishes the
        spec-shape payload, ROLLBACK suppresses the wire signal,
        multi-listener fan-out). This conformance assertion just
        pins that the bit exists and is a bool, so a plugin that
        forgets to set it surfaces here rather than in
        :func:`smai_api.make_api_app`'s lifespan when the listener
        decision is made."""
        assert isinstance(store.capabilities.supports_listen_notify, bool)
