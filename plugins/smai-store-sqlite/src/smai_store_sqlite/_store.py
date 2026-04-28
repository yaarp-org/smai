""":class:`SqliteStore` — the reference :class:`MetadataStore` plugin.

Per ``designs/smai/07-plugin-interfaces.md`` §5, DEC-029 (Protocol defs
in smai-core), DEC-030 (SQL-shape only), DEC-033 (cost tables exist;
no Protocol CRUD yet), DEC-035 (cursor pagination, lease nonce,
caller-resolved pool→states), DEC-036 (SQLAlchemy 2.0 async Core,
Alembic deferred to Task 3.H2).

The 47-method Protocol surface decomposes:

* CRUD round-trip (14 methods)
* Methodology lookups for ``TechniqueRef`` (3 methods)
* Conditional state-transition CAS (5 methods)
* Scheduling queries — CG-execution (9), proposal (3), paper (7) (19 total)
* In-flight count aggregations (2 methods)
* Lease acquire / release / extend (3 methods)
* Transactional grouping (1 method, returns context manager)

Plus the :class:`Transaction` Protocol's 10 mirrored methods (5 ``create_*``
+ 5 ``transition_*_state``).

The implementation files in this package:

* ``_schema.py`` — declarative ``MetaData`` + ``Table`` set (9 tables).
* ``_records.py`` — Pydantic types for the DEC-033 cost tables; plugin-
  internal until the Protocol surface gains CRUD for them (Session C).
* ``_serde.py`` — Pydantic ↔ SQL row conversion.
* this file — :class:`SqliteStore` and the inner :class:`_Transaction`
  + :class:`_TransactionContextManager`.
"""

from __future__ import annotations

import base64
import json
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import uuid4

from smai_core.entities import TechniqueRef
from smai_core.plugins import (
    ConflictError,
    CursorPage,
    EntityKind,
    LeaseLostError,
    LeaseToken,
    MetadataStoreCapabilities,
)
from smai_orchestrator.entities.tracking import (
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
from sqlalchemy import (
    Column,
    ColumnElement,
    Select,
    Table,
    and_,
    delete,
    exists,
    func,
    insert,
    not_,
    or_,
    select,
    update,
)
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    create_async_engine,
)

from smai_store_sqlite._schema import (
    ENTITY_PK_COLUMN,
    ENTITY_TABLE,
    agent_sessions_table,
    cgs_table,
    entries_table,
    factor_models_table,
    metadata,
    papers_table,
    proposals_table,
    run_costs_table,
    runs_table,
    techniques_table,
    transition_log_table,
)
from smai_store_sqlite._serde import row_to_record

if TYPE_CHECKING:
    from collections.abc import Sequence

# Default for ``smai dev`` — in-process SQLite. Production deployments
# pass an explicit URI (per ``09-cli.md`` §1).
DEFAULT_URI = "sqlite+aiosqlite:///:memory:"

# Per-entity-kind in-progress states for ``count_with_in_flight_jobs``
# defaults — unused since DEC-035 #3 makes the caller pass the list, but
# the dispatch table for "which job_handle column" lives here. The
# Protocol does NOT enumerate which handle column is "the" handle for
# count purposes when an entity has multiple (Paper has fetcher /
# screener / planner); we treat the count predicate as ``ANY of the
# handle columns is non-null`` for those cases. The column list per kind
# is consumed only by ``count_with_in_flight_jobs``.
_HANDLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "cg": ("harness_job_handle",),
    "entry": ("implementation_job_handle",),
    "run": ("compute_job_handle",),
    "proposal": ("planner_job_handle",),
    "paper": ("fetcher_job_handle", "screener_job_handle", "planner_job_handle"),
}


# === Cursor encoding =========================================================
#
# Per DEC-035 #1: the cursor is plugin-internal; consumers treat it as
# opaque. We encode ``(created_at, id)`` for FIFO ordering as base64
# JSON. ``list_techniques`` orders by ``id`` only (techniques have no
# ``created_at`` column per ``01-data-model.md`` §5.7's TechniqueRef
# shape) — handled inline.


def _encode_cursor(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_cursor(cursor: str) -> dict[str, Any]:
    raw = base64.urlsafe_b64decode(cursor.encode("ascii"))
    decoded: object = json.loads(raw.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"cursor payload is not a dict: {decoded!r}")
    # ``decoded`` is now ``dict[Unknown, Unknown]`` to pyright; rebuild
    # with explicit string keys so the return-type narrows cleanly.
    return {str(k): v for k, v in decoded.items()}  # type: ignore[reportUnknownVariableType]


def _parse_dt(value: Any) -> datetime:
    """Parse a stored datetime value back to a tz-aware ``datetime``.

    SQLite returns ISO-8601 strings via SQLAlchemy's adapter for
    ``DateTime(timezone=True)``; pass-through for actual ``datetime``
    values (which is what newer SQLAlchemy versions return).
    """
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        # ``fromisoformat`` accepts ``+00:00`` and naive forms.
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    raise TypeError(f"unexpected datetime serialization: {type(value).__name__}: {value!r}")


# === Transaction implementations =============================================
#
# Per ``07`` §5.3 the ``Transaction`` Protocol mirrors the writes on the
# parent ``MetadataStore``. The :class:`_Transaction` object below
# delegates to the same row-builders used by :class:`SqliteStore` but
# binds them to a single :class:`AsyncConnection` so all writes land in
# one BEGIN/COMMIT block. The :class:`_TransactionContextManager`
# wraps SQLAlchemy's ``engine.begin()`` block; commit / rollback is
# automatic on context exit.


class _Transaction:
    """Connection-scoped writer for the :class:`Transaction` Protocol.

    Constructed by :class:`_TransactionContextManager`; never directly.
    """

    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def create_cg(self, cg: ComparisonGroupRecord) -> ComparisonGroupRecord:
        return await _do_create_cg(self._conn, cg)

    async def create_entry(self, entry: EntryRecord) -> EntryRecord:
        return await _do_create_entry(self._conn, entry)

    async def create_run(self, run: RunRecord) -> RunRecord:
        return await _do_create_run(self._conn, run)

    async def create_proposal(self, proposal: ProposalRecord) -> ProposalRecord:
        return await _do_create_proposal(self._conn, proposal)

    async def create_paper(self, paper: PaperRecord) -> PaperRecord:
        return await _do_create_paper(self._conn, paper)

    async def transition_cg_state(
        self,
        cg_id: str,
        expected_version: int,
        target_state: CGState,
        **fields: object,
    ) -> ComparisonGroupRecord:
        return await _do_transition(
            self._conn, "cg", cg_id, expected_version, target_state, ComparisonGroupRecord, fields
        )

    async def transition_entry_state(
        self,
        entry_id: str,
        expected_version: int,
        target_state: EntryState,
        **fields: object,
    ) -> EntryRecord:
        return await _do_transition(
            self._conn, "entry", entry_id, expected_version, target_state, EntryRecord, fields
        )

    async def transition_run_state(
        self,
        run_id: str,
        expected_version: int,
        target_state: RunState,
        **fields: object,
    ) -> RunRecord:
        return await _do_transition(
            self._conn, "run", run_id, expected_version, target_state, RunRecord, fields
        )

    async def transition_proposal_state(
        self,
        proposal_id: str,
        expected_version: int,
        target_state: ProposalState,
        **fields: object,
    ) -> ProposalRecord:
        return await _do_transition(
            self._conn,
            "proposal",
            proposal_id,
            expected_version,
            target_state,
            ProposalRecord,
            fields,
        )

    async def transition_paper_state(
        self,
        arxiv_id: str,
        expected_version: int,
        target_state: PaperState,
        **fields: object,
    ) -> PaperRecord:
        return await _do_transition(
            self._conn, "paper", arxiv_id, expected_version, target_state, PaperRecord, fields
        )


class _TransactionContextManager(AbstractAsyncContextManager[_Transaction]):
    """Wrapper around ``engine.begin()`` exposing :class:`_Transaction`.

    The Protocol shape is ``async with await store.transaction() as tx``,
    so :meth:`SqliteStore.transaction` returns one of these *eagerly*
    (no ``await`` until the ``async with``).
    """

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._begin_cm: AbstractAsyncContextManager[AsyncConnection] | None = None

    async def __aenter__(self) -> _Transaction:
        self._begin_cm = self._engine.begin()
        conn = await self._begin_cm.__aenter__()
        return _Transaction(conn)

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._begin_cm is not None:
            await self._begin_cm.__aexit__(exc_type, exc, tb)
        return None


# === Low-level write helpers shared by the store + transaction ===============
#
# These free functions take an explicit :class:`AsyncConnection` so they
# can be called from either a direct ``engine.begin()`` block (the
# store's outer methods) or from an active transaction (the
# :class:`_Transaction` mirror).


def _serialize_for_row(record: Any, exclude: set[str] | None = None) -> dict[str, Any]:
    """Convert a Pydantic record to a dict suitable for SQLAlchemy
    column binding.

    Inner :class:`JobHandle` objects ``model_dump`` to plain dicts.
    Datetime fields are kept as datetime objects (SQLAlchemy's
    ``DateTime`` adapter handles them).
    """
    raw = record.model_dump(mode="python")
    if exclude:
        for key in exclude:
            raw.pop(key, None)
    return raw


async def _do_create_cg(conn: AsyncConnection, cg: ComparisonGroupRecord) -> ComparisonGroupRecord:
    values = _serialize_for_row(cg)
    await conn.execute(insert(cgs_table).values(**values))
    return cg


async def _do_create_entry(conn: AsyncConnection, entry: EntryRecord) -> EntryRecord:
    values = _serialize_for_row(entry)
    await conn.execute(insert(entries_table).values(**values))
    return entry


async def _do_create_run(conn: AsyncConnection, run: RunRecord) -> RunRecord:
    values = _serialize_for_row(run)
    await conn.execute(insert(runs_table).values(**values))
    return run


async def _do_create_proposal(conn: AsyncConnection, proposal: ProposalRecord) -> ProposalRecord:
    values = _serialize_for_row(proposal)
    await conn.execute(insert(proposals_table).values(**values))
    return proposal


async def _do_create_paper(conn: AsyncConnection, paper: PaperRecord) -> PaperRecord:
    values = _serialize_for_row(paper)
    await conn.execute(insert(papers_table).values(**values))
    return paper


_RECORD_BY_KIND: dict[str, type[Any]] = {
    "cg": ComparisonGroupRecord,
    "entry": EntryRecord,
    "run": RunRecord,
    "proposal": ProposalRecord,
    "paper": PaperRecord,
}


async def _do_transition(
    conn: AsyncConnection,
    kind: Literal["cg", "entry", "run", "proposal", "paper"],
    entity_id: str,
    expected_version: int,
    target_state: str,
    record_type: type[Any],
    fields: dict[str, object],
) -> Any:
    """CAS state transition (§5.4).

    Translates to ``UPDATE table SET state = :s, version = version + 1,
    updated_at = :u, **fields WHERE pk = :id AND version = :expected
    RETURNING *``. On no-row-affected: looks up the actual version and
    raises :class:`ConflictError`.
    """
    table = ENTITY_TABLE[kind]
    pk = ENTITY_PK_COLUMN[kind]
    now = datetime.now(UTC)
    # Sanitize ``fields`` — only let through columns that exist on the
    # table (Pydantic models can validate their own keys, but ``**fields``
    # is variadic and surface for engine-driven atomic-set writes; an
    # unknown key would render an SQL error ``column does not exist``,
    # which obscures the real problem). Reject unknown keys explicitly.
    column_names = {c.name for c in table.columns}
    for key in fields:
        if key not in column_names:
            raise ValueError(
                f"transition_{kind}_state: unknown column {key!r} for table "
                f"{table.name!r}; allowed columns: {sorted(column_names)}"
            )
    pk_col = table.c[pk]
    version_col = table.c["version"]
    set_values: dict[str, Any] = {
        "state": target_state,
        "version": version_col + 1,
        "updated_at": now,
    }
    set_values.update(fields)
    stmt = (
        update(table)
        .where(pk_col == entity_id, version_col == expected_version)
        .values(**set_values)
        .returning(*table.columns)
    )
    result = await conn.execute(stmt)
    row = result.mappings().first()
    if row is None:
        actual_version_result = await conn.execute(select(version_col).where(pk_col == entity_id))
        actual = actual_version_result.scalar_one_or_none()
        raise ConflictError(
            entity_type=kind,
            entity_id=entity_id,
            expected_version=expected_version,
            actual_version=int(actual) if actual is not None else -1,
        )
    return row_to_record(record_type, row)


# === Scheduling-query predicate builders =====================================
#
# Each scheduling query name maps to a (Table, predicate) pair. Cursor
# pagination is generic over (table, predicate, ordering); the predicate
# captures the per-query WHERE-clause constraints.


def _predicate_ready_harness_build() -> tuple[Table, ColumnElement[bool], type[Any]]:
    return (
        cgs_table,
        and_(
            cgs_table.c.state == "draft",
            cgs_table.c.harness_job_handle.is_(None),
        ),
        ComparisonGroupRecord,
    )


def _predicate_in_flight_harness_build() -> tuple[Table, ColumnElement[bool], type[Any]]:
    return (
        cgs_table,
        and_(
            cgs_table.c.state == "implementing",
            cgs_table.c.harness_job_handle.is_not(None),
        ),
        ComparisonGroupRecord,
    )


def _predicate_ready_to_implement_entry() -> tuple[Table, ColumnElement[bool], type[Any]]:
    # Entry is non-baseline (has a technique_id), in 'pending', and parent
    # CG is in 'implementing'. Per ``07`` §5.6.3.
    parent_implementing = exists().where(
        and_(
            cgs_table.c.id == entries_table.c.cg_id,
            cgs_table.c.state == "implementing",
        )
    )
    return (
        entries_table,
        and_(
            entries_table.c.state == "pending",
            entries_table.c.is_baseline.is_(False),
            entries_table.c.technique_id.is_not(None),
            parent_implementing,
        ),
        EntryRecord,
    )


def _predicate_in_flight_entry_implementation() -> tuple[Table, ColumnElement[bool], type[Any]]:
    return (
        entries_table,
        entries_table.c.implementation_job_handle.is_not(None),
        EntryRecord,
    )


def _predicate_ready_review_and_run() -> tuple[Table, ColumnElement[bool], type[Any]]:
    # Per ``07`` §5.6.3: CGs in 'implemented' where all child entries have
    # reached a terminal implementation state (``implemented`` or
    # ``implementation_failed``) and ≥1 entry is ``implemented``.
    has_non_terminal_entry = exists().where(
        and_(
            entries_table.c.cg_id == cgs_table.c.id,
            entries_table.c.state.notin_(["implemented", "implementation_failed"]),
        )
    )
    has_implemented_entry = exists().where(
        and_(
            entries_table.c.cg_id == cgs_table.c.id,
            entries_table.c.state == "implemented",
        )
    )
    return (
        cgs_table,
        and_(
            cgs_table.c.state == "implemented",
            not_(has_non_terminal_entry),
            has_implemented_entry,
        ),
        ComparisonGroupRecord,
    )


def _predicate_ready_to_run() -> tuple[Table, ColumnElement[bool], type[Any]]:
    return (runs_table, runs_table.c.state == "pending", RunRecord)


def _predicate_in_flight_runs() -> tuple[Table, ColumnElement[bool], type[Any]]:
    return (
        runs_table,
        and_(
            runs_table.c.state.in_(["submitted", "running"]),
            runs_table.c.compute_job_handle.is_not(None),
        ),
        RunRecord,
    )


def _predicate_ready_to_evaluate() -> tuple[Table, ColumnElement[bool], type[Any]]:
    has_non_terminal_run = exists().where(
        and_(
            runs_table.c.cg_id == cgs_table.c.id,
            runs_table.c.state.notin_(["succeeded", "failed", "inconclusive"]),
        )
    )
    return (
        cgs_table,
        and_(cgs_table.c.state == "running", not_(has_non_terminal_run)),
        ComparisonGroupRecord,
    )


def _predicate_in_flight_evaluation() -> tuple[Table, ColumnElement[bool], type[Any]]:
    return (cgs_table, cgs_table.c.state == "evaluating", ComparisonGroupRecord)


def _predicate_ready_proposal_design() -> tuple[Table, ColumnElement[bool], type[Any]]:
    return (
        proposals_table,
        and_(
            proposals_table.c.state == "proposal_submitted",
            proposals_table.c.planner_job_handle.is_(None),
        ),
        ProposalRecord,
    )


def _predicate_in_flight_proposal_design() -> tuple[Table, ColumnElement[bool], type[Any]]:
    return (
        proposals_table,
        and_(
            proposals_table.c.state == "designing",
            proposals_table.c.planner_job_handle.is_not(None),
        ),
        ProposalRecord,
    )


def _predicate_pending_proposal_user_decision() -> tuple[Table, ColumnElement[bool], type[Any]]:
    return (
        proposals_table,
        and_(
            proposals_table.c.state == "designed",
            proposals_table.c.user_decision.is_(None),
        ),
        ProposalRecord,
    )


def _predicate_ready_paper_fetch() -> tuple[Table, ColumnElement[bool], type[Any]]:
    return (
        papers_table,
        and_(
            papers_table.c.state == "submitted",
            papers_table.c.fetcher_job_handle.is_(None),
        ),
        PaperRecord,
    )


def _predicate_in_flight_paper_fetch() -> tuple[Table, ColumnElement[bool], type[Any]]:
    return (
        papers_table,
        and_(
            papers_table.c.state == "fetching",
            papers_table.c.fetcher_job_handle.is_not(None),
        ),
        PaperRecord,
    )


def _predicate_ready_paper_screen() -> tuple[Table, ColumnElement[bool], type[Any]]:
    return (
        papers_table,
        and_(
            papers_table.c.state == "screening",
            papers_table.c.screener_job_handle.is_(None),
        ),
        PaperRecord,
    )


def _predicate_in_flight_paper_screen() -> tuple[Table, ColumnElement[bool], type[Any]]:
    return (
        papers_table,
        and_(
            papers_table.c.state == "screening",
            papers_table.c.screener_job_handle.is_not(None),
        ),
        PaperRecord,
    )


def _predicate_ready_paper_plan() -> tuple[Table, ColumnElement[bool], type[Any]]:
    return (
        papers_table,
        and_(
            papers_table.c.state == "planning",
            papers_table.c.planner_job_handle.is_(None),
        ),
        PaperRecord,
    )


def _predicate_in_flight_paper_plan() -> tuple[Table, ColumnElement[bool], type[Any]]:
    return (
        papers_table,
        and_(
            papers_table.c.state == "planning",
            papers_table.c.planner_job_handle.is_not(None),
        ),
        PaperRecord,
    )


def _predicate_partial_pending_promotion() -> tuple[Table, ColumnElement[bool], type[Any]]:
    # Per ``07`` §5.6.5 the promotion flag is user-set; ``PaperRecord``
    # does not yet carry an explicit flag column, so we surface
    # everything in ``state='partial'`` for tooling. When the flag
    # column lands on ``PaperRecord`` (Session C / DEC-032 follow-up),
    # this predicate gains ``papers.c.promote_pending == True``.
    return (papers_table, papers_table.c.state == "partial", PaperRecord)


# === SqliteStore =============================================================


class SqliteStore:
    """The reference :class:`MetadataStore` plugin (DEC-030, DEC-036).

    Construction takes a SQLAlchemy URI (default in-memory). Callers
    must ``await store.migrate()`` before issuing any other call —
    ``migrate()`` is idempotent (uses ``metadata.create_all()`` with
    ``checkfirst=True``).

    The companion ``Postgres`` plugin (Task 3.F1) reuses this code
    modulo dialect-specific bits (advisory locks for hot-contention
    lease edges per ``07`` §5.6.7; window-function-derived tenant
    fairness per ``07`` §5.5).
    """

    name: str = "sqlite"
    capabilities: MetadataStoreCapabilities = MetadataStoreCapabilities(
        is_tenant_aware=False,
        supports_transactions=True,
    )

    def __init__(self, uri: str = DEFAULT_URI) -> None:
        # ``future=True`` is the SQLAlchemy 2.0 default; the explicit
        # ``echo=False`` keeps SQL out of pytest captures.
        self._engine: AsyncEngine = create_async_engine(uri, echo=False, future=True)

    async def migrate(self) -> None:
        """Apply boot-time DDL (idempotent).

        Per DEC-036: ``metadata.create_all(checkfirst=True)`` —
        ``CREATE TABLE IF NOT EXISTS`` semantics on both SQLite and
        Postgres. Production deployments lift this into a versioned
        Alembic migration in Task 3.H2.
        """
        async with self._engine.begin() as conn:
            await conn.run_sync(metadata.create_all, checkfirst=True)

    async def dispose(self) -> None:
        """Tear down the engine — for test cleanup."""
        await self._engine.dispose()

    # === CRUD round-trip — pipeline tracking entities (§5.3) ================

    async def get_cg(self, cg_id: str) -> ComparisonGroupRecord | None:
        async with self._engine.connect() as conn:
            result = await conn.execute(select(cgs_table).where(cgs_table.c.id == cg_id))
            row = result.mappings().first()
            return row_to_record(ComparisonGroupRecord, row) if row else None

    async def list_entries_for_cg(
        self, cg_id: str, limit: int = 100, cursor: str | None = None
    ) -> CursorPage[EntryRecord]:
        return await self._paginate(
            entries_table,
            entries_table.c.cg_id == cg_id,
            EntryRecord,
            limit,
            cursor,
        )

    async def get_entry(self, entry_id: str) -> EntryRecord | None:
        async with self._engine.connect() as conn:
            result = await conn.execute(select(entries_table).where(entries_table.c.id == entry_id))
            row = result.mappings().first()
            return row_to_record(EntryRecord, row) if row else None

    async def list_runs_for_entry(
        self, entry_id: str, limit: int = 100, cursor: str | None = None
    ) -> CursorPage[RunRecord]:
        return await self._paginate(
            runs_table,
            runs_table.c.entry_id == entry_id,
            RunRecord,
            limit,
            cursor,
        )

    async def get_run(self, run_id: str) -> RunRecord | None:
        async with self._engine.connect() as conn:
            result = await conn.execute(select(runs_table).where(runs_table.c.id == run_id))
            row = result.mappings().first()
            return row_to_record(RunRecord, row) if row else None

    async def get_proposal(self, proposal_id: str) -> ProposalRecord | None:
        async with self._engine.connect() as conn:
            result = await conn.execute(
                select(proposals_table).where(proposals_table.c.id == proposal_id)
            )
            row = result.mappings().first()
            return row_to_record(ProposalRecord, row) if row else None

    async def list_cgs_for_proposal(
        self, proposal_id: str, limit: int = 100, cursor: str | None = None
    ) -> CursorPage[ComparisonGroupRecord]:
        return await self._paginate(
            cgs_table,
            cgs_table.c.proposal_id == proposal_id,
            ComparisonGroupRecord,
            limit,
            cursor,
        )

    async def get_paper(self, arxiv_id: str) -> PaperRecord | None:
        async with self._engine.connect() as conn:
            result = await conn.execute(
                select(papers_table).where(papers_table.c.arxiv_id == arxiv_id)
            )
            row = result.mappings().first()
            return row_to_record(PaperRecord, row) if row else None

    async def list_techniques_for_paper(
        self, arxiv_id: str, limit: int = 100, cursor: str | None = None
    ) -> CursorPage[TechniqueRef]:
        # Cursor anchors on ``id`` (techniques have no ``created_at``);
        # the lookup uses the denormalized ``fidelity_anchor_arxiv_id``
        # column populated by ``upsert_technique``.
        async with self._engine.connect() as conn:
            stmt: Select[Any] = (
                select(techniques_table)
                .where(
                    techniques_table.c.fidelity_anchor_kind == "paper",
                    techniques_table.c.fidelity_anchor_arxiv_id == arxiv_id,
                )
                .order_by(techniques_table.c.id)
                .limit(limit + 1)
            )
            if cursor is not None:
                anchor = _decode_cursor(cursor)
                stmt = stmt.where(techniques_table.c.id > anchor["id"])
            rows = (await conn.execute(stmt)).mappings().all()
            return _technique_page(rows, limit)

    async def create_cg(self, cg: ComparisonGroupRecord) -> ComparisonGroupRecord:
        async with self._engine.begin() as conn:
            return await _do_create_cg(conn, cg)

    async def create_entry(self, entry: EntryRecord) -> EntryRecord:
        async with self._engine.begin() as conn:
            return await _do_create_entry(conn, entry)

    async def create_run(self, run: RunRecord) -> RunRecord:
        async with self._engine.begin() as conn:
            return await _do_create_run(conn, run)

    async def create_proposal(self, proposal: ProposalRecord) -> ProposalRecord:
        async with self._engine.begin() as conn:
            return await _do_create_proposal(conn, proposal)

    async def create_paper(self, paper: PaperRecord) -> PaperRecord:
        async with self._engine.begin() as conn:
            return await _do_create_paper(conn, paper)

    # === Methodology-touching lookups (§5.3) ================================

    async def get_technique(self, technique_id: str) -> TechniqueRef | None:
        async with self._engine.connect() as conn:
            result = await conn.execute(
                select(techniques_table).where(techniques_table.c.id == technique_id)
            )
            row = result.mappings().first()
            return _technique_from_row(row) if row else None

    async def list_techniques(
        self, limit: int = 100, cursor: str | None = None
    ) -> CursorPage[TechniqueRef]:
        async with self._engine.connect() as conn:
            stmt: Select[Any] = (
                select(techniques_table).order_by(techniques_table.c.id).limit(limit + 1)
            )
            if cursor is not None:
                anchor = _decode_cursor(cursor)
                stmt = stmt.where(techniques_table.c.id > anchor["id"])
            rows = (await conn.execute(stmt)).mappings().all()
            return _technique_page(rows, limit)

    async def upsert_technique(self, technique: TechniqueRef) -> None:
        # SQLite's ON CONFLICT DO UPDATE — same shape on Postgres
        # (``ON CONFLICT (id) DO UPDATE``). SQLAlchemy renders
        # dialect-appropriate SQL via ``insert(...).on_conflict_do_update``
        # but that's a per-dialect import. For the SQLite reference we
        # use a portable read-then-update pattern; cost is negligible
        # for this registry-mirror surface (≪10K rows per ``07`` §5.3).
        anchor = technique.fidelity_anchor
        anchor_kind: str | None = None
        anchor_arxiv: str | None = None
        if anchor is not None:
            anchor_kind = anchor.kind
            if anchor.kind == "paper":
                anchor_arxiv = anchor.arxiv_id
        values: dict[str, Any] = {
            "id": technique.id,
            "name": technique.name,
            "description": technique.description,
            "category": technique.category,
            "compatible_factor_types": list(technique.compatible_factor_types),
            "standard": technique.standard,
            "fidelity_anchor": (
                technique.fidelity_anchor.model_dump(mode="json")
                if technique.fidelity_anchor is not None
                else None
            ),
            "fidelity_anchor_kind": anchor_kind,
            "fidelity_anchor_arxiv_id": anchor_arxiv,
            "affects_extension_points": list(technique.affects_extension_points),
            "implies_controlled": list(technique.implies_controlled),
            "parameter_schema": technique.parameter_schema,
        }
        async with self._engine.begin() as conn:
            existing = await conn.execute(
                select(techniques_table.c.id).where(techniques_table.c.id == technique.id)
            )
            if existing.first() is None:
                await conn.execute(insert(techniques_table).values(**values))
            else:
                await conn.execute(
                    update(techniques_table)
                    .where(techniques_table.c.id == technique.id)
                    .values(**values)
                )

    # === Conditional state transitions (§5.4) ===============================

    async def transition_cg_state(
        self,
        cg_id: str,
        expected_version: int,
        target_state: CGState,
        **fields: object,
    ) -> ComparisonGroupRecord:
        async with self._engine.begin() as conn:
            return await _do_transition(
                conn, "cg", cg_id, expected_version, target_state, ComparisonGroupRecord, fields
            )

    async def transition_entry_state(
        self,
        entry_id: str,
        expected_version: int,
        target_state: EntryState,
        **fields: object,
    ) -> EntryRecord:
        async with self._engine.begin() as conn:
            return await _do_transition(
                conn, "entry", entry_id, expected_version, target_state, EntryRecord, fields
            )

    async def transition_run_state(
        self,
        run_id: str,
        expected_version: int,
        target_state: RunState,
        **fields: object,
    ) -> RunRecord:
        async with self._engine.begin() as conn:
            return await _do_transition(
                conn, "run", run_id, expected_version, target_state, RunRecord, fields
            )

    async def transition_proposal_state(
        self,
        proposal_id: str,
        expected_version: int,
        target_state: ProposalState,
        **fields: object,
    ) -> ProposalRecord:
        async with self._engine.begin() as conn:
            return await _do_transition(
                conn,
                "proposal",
                proposal_id,
                expected_version,
                target_state,
                ProposalRecord,
                fields,
            )

    async def transition_paper_state(
        self,
        arxiv_id: str,
        expected_version: int,
        target_state: PaperState,
        **fields: object,
    ) -> PaperRecord:
        async with self._engine.begin() as conn:
            return await _do_transition(
                conn, "paper", arxiv_id, expected_version, target_state, PaperRecord, fields
            )

    # === Scheduling queries — CG-execution (§5.6.3) =========================

    async def get_ready_for_harness_build(
        self, limit: int = 100, cursor: str | None = None
    ) -> CursorPage[ComparisonGroupRecord]:
        return await self._paginate_predicate(_predicate_ready_harness_build(), limit, cursor)

    async def get_in_flight_harness_build(
        self, limit: int = 100, cursor: str | None = None
    ) -> CursorPage[ComparisonGroupRecord]:
        return await self._paginate_predicate(_predicate_in_flight_harness_build(), limit, cursor)

    async def get_ready_to_implement_entry(
        self, limit: int = 100, cursor: str | None = None
    ) -> CursorPage[EntryRecord]:
        return await self._paginate_predicate(_predicate_ready_to_implement_entry(), limit, cursor)

    async def get_in_flight_entry_implementation(
        self, limit: int = 100, cursor: str | None = None
    ) -> CursorPage[EntryRecord]:
        return await self._paginate_predicate(
            _predicate_in_flight_entry_implementation(), limit, cursor
        )

    async def get_ready_for_review_and_run(
        self, limit: int = 100, cursor: str | None = None
    ) -> CursorPage[ComparisonGroupRecord]:
        return await self._paginate_predicate(_predicate_ready_review_and_run(), limit, cursor)

    async def get_ready_to_run(
        self, limit: int = 100, cursor: str | None = None
    ) -> CursorPage[RunRecord]:
        return await self._paginate_predicate(_predicate_ready_to_run(), limit, cursor)

    async def get_in_flight_runs(
        self, limit: int = 100, cursor: str | None = None
    ) -> CursorPage[RunRecord]:
        return await self._paginate_predicate(_predicate_in_flight_runs(), limit, cursor)

    async def get_ready_to_evaluate(
        self, limit: int = 100, cursor: str | None = None
    ) -> CursorPage[ComparisonGroupRecord]:
        return await self._paginate_predicate(_predicate_ready_to_evaluate(), limit, cursor)

    async def get_in_flight_evaluation(
        self, limit: int = 100, cursor: str | None = None
    ) -> CursorPage[ComparisonGroupRecord]:
        return await self._paginate_predicate(_predicate_in_flight_evaluation(), limit, cursor)

    # === Scheduling queries — proposal pipeline (§5.6.4) ====================

    async def get_ready_for_proposal_design(
        self, limit: int = 100, cursor: str | None = None
    ) -> CursorPage[ProposalRecord]:
        return await self._paginate_predicate(_predicate_ready_proposal_design(), limit, cursor)

    async def get_in_flight_proposal_design(
        self, limit: int = 100, cursor: str | None = None
    ) -> CursorPage[ProposalRecord]:
        return await self._paginate_predicate(_predicate_in_flight_proposal_design(), limit, cursor)

    async def get_pending_proposal_user_decision(
        self, limit: int = 100, cursor: str | None = None
    ) -> CursorPage[ProposalRecord]:
        return await self._paginate_predicate(
            _predicate_pending_proposal_user_decision(), limit, cursor
        )

    # === Scheduling queries — paper-ingestion (§5.6.5) ======================

    async def get_ready_for_paper_fetch(
        self, limit: int = 100, cursor: str | None = None
    ) -> CursorPage[PaperRecord]:
        return await self._paginate_predicate(_predicate_ready_paper_fetch(), limit, cursor)

    async def get_in_flight_paper_fetch(
        self, limit: int = 100, cursor: str | None = None
    ) -> CursorPage[PaperRecord]:
        return await self._paginate_predicate(_predicate_in_flight_paper_fetch(), limit, cursor)

    async def get_ready_for_paper_screen(
        self, limit: int = 100, cursor: str | None = None
    ) -> CursorPage[PaperRecord]:
        return await self._paginate_predicate(_predicate_ready_paper_screen(), limit, cursor)

    async def get_in_flight_paper_screen(
        self, limit: int = 100, cursor: str | None = None
    ) -> CursorPage[PaperRecord]:
        return await self._paginate_predicate(_predicate_in_flight_paper_screen(), limit, cursor)

    async def get_ready_for_paper_plan(
        self, limit: int = 100, cursor: str | None = None
    ) -> CursorPage[PaperRecord]:
        return await self._paginate_predicate(_predicate_ready_paper_plan(), limit, cursor)

    async def get_in_flight_paper_plan(
        self, limit: int = 100, cursor: str | None = None
    ) -> CursorPage[PaperRecord]:
        return await self._paginate_predicate(_predicate_in_flight_paper_plan(), limit, cursor)

    async def get_partial_pending_promotion(
        self, limit: int = 100, cursor: str | None = None
    ) -> CursorPage[PaperRecord]:
        return await self._paginate_predicate(_predicate_partial_pending_promotion(), limit, cursor)

    # === Count aggregations (§5.6.6) ========================================

    async def count_in_state(
        self,
        entity_kind: Literal["cg", "entry", "run", "proposal", "paper"],
        states: list[str],
    ) -> int:
        if not states:
            return 0
        table = ENTITY_TABLE[entity_kind]
        async with self._engine.connect() as conn:
            stmt = select(func.count()).select_from(table).where(table.c.state.in_(states))
            result = await conn.execute(stmt)
            value = result.scalar_one()
            return int(value)

    async def count_with_in_flight_jobs(
        self,
        entity_kind: Literal["cg", "entry", "run", "proposal", "paper"],
        in_flight_states: list[str],
    ) -> int:
        if not in_flight_states:
            return 0
        table = ENTITY_TABLE[entity_kind]
        handle_cols = _HANDLE_COLUMNS[entity_kind]
        # ``ANY of the handle columns is non-null`` — for entities with
        # one handle (cg / entry / run / proposal) this collapses to a
        # single ``IS NOT NULL``; for paper it's the OR across the three
        # per-stage handles (per ``07`` §5.6.6's count predicate).
        handle_predicate: ColumnElement[bool] = or_(
            *[table.c[col].is_not(None) for col in handle_cols]
        )
        async with self._engine.connect() as conn:
            stmt = (
                select(func.count())
                .select_from(table)
                .where(table.c.state.in_(in_flight_states), handle_predicate)
            )
            result = await conn.execute(stmt)
            value = result.scalar_one()
            return int(value)

    # === Lease ops (§5.6.7) ==================================================

    async def acquire_lease(
        self,
        entity_kind: EntityKind,
        entity_id: str,
        lease_seconds: int,
        lease_holder_id: str,
    ) -> LeaseToken | None:
        table = ENTITY_TABLE[entity_kind]
        pk = ENTITY_PK_COLUMN[entity_kind]
        pk_col = table.c[pk]
        now = datetime.now(UTC)
        expires_at = now + timedelta(seconds=lease_seconds)
        nonce = str(uuid4())
        # Per DEC-035 #2: WHERE id = :id AND (lease_expires_at IS NULL
        # OR lease_expires_at < now). No version-CAS predicate. On 0
        # rows return None.
        stmt = (
            update(table)
            .where(
                pk_col == entity_id,
                or_(
                    table.c.lease_expires_at.is_(None),
                    table.c.lease_expires_at < now,
                ),
            )
            .values(
                leased_by=lease_holder_id,
                lease_expires_at=expires_at,
                lease_nonce=nonce,
            )
            .returning(pk_col)
        )
        async with self._engine.begin() as conn:
            result = await conn.execute(stmt)
            matched = result.first()
            if matched is None:
                return None
        return LeaseToken(
            entity_kind=entity_kind,
            entity_id=entity_id,
            acquired_at=now,
            expires_at=expires_at,
            lease_holder_id=lease_holder_id,
            nonce=nonce,
        )

    async def release_lease(self, lease_token: LeaseToken) -> None:
        table = ENTITY_TABLE[lease_token.entity_kind]
        pk = ENTITY_PK_COLUMN[lease_token.entity_kind]
        pk_col = table.c[pk]
        # Idempotent: WHERE pk = :id AND lease_nonce = :n. 0 rows is a
        # silent no-op (per ``07`` §5.6.7 / DEC-035 #2).
        stmt = (
            update(table)
            .where(pk_col == lease_token.entity_id, table.c.lease_nonce == lease_token.nonce)
            .values(leased_by=None, lease_expires_at=None, lease_nonce=None)
        )
        async with self._engine.begin() as conn:
            await conn.execute(stmt)

    async def extend_lease(self, lease_token: LeaseToken, additional_seconds: int) -> LeaseToken:
        table = ENTITY_TABLE[lease_token.entity_kind]
        pk = ENTITY_PK_COLUMN[lease_token.entity_kind]
        pk_col = table.c[pk]
        new_expires = lease_token.expires_at + timedelta(seconds=additional_seconds)
        new_nonce = str(uuid4())
        stmt = (
            update(table)
            .where(pk_col == lease_token.entity_id, table.c.lease_nonce == lease_token.nonce)
            .values(lease_expires_at=new_expires, lease_nonce=new_nonce)
            .returning(pk_col)
        )
        async with self._engine.begin() as conn:
            result = await conn.execute(stmt)
            if result.first() is None:
                # Per DEC-035 #2: lease lost (expired and reacquired).
                raise LeaseLostError(lease_token.entity_kind, lease_token.entity_id)
        return LeaseToken(
            entity_kind=lease_token.entity_kind,
            entity_id=lease_token.entity_id,
            acquired_at=lease_token.acquired_at,
            expires_at=new_expires,
            lease_holder_id=lease_token.lease_holder_id,
            nonce=new_nonce,
        )

    # === Transactional grouping (§5.3) =======================================

    async def transaction(self) -> AbstractAsyncContextManager[_Transaction]:
        return _TransactionContextManager(self._engine)

    # === Internal pagination helpers ========================================

    async def _paginate(
        self,
        table: Table,
        predicate: ColumnElement[bool],
        record_type: type[Any],
        limit: int,
        cursor: str | None,
    ) -> CursorPage[Any]:
        return await self._paginate_predicate((table, predicate, record_type), limit, cursor)

    async def _paginate_predicate(
        self,
        predicate_spec: tuple[Table, ColumnElement[bool], type[Any]],
        limit: int,
        cursor: str | None,
    ) -> CursorPage[Any]:
        table, predicate, record_type = predicate_spec
        pk_name = "arxiv_id" if "arxiv_id" in {c.name for c in table.columns} else "id"
        pk_col = table.c[pk_name]
        created_at_col = table.c["created_at"]
        # ``created_at, id`` ordering anchors a stable cursor (per DEC-035
        # #1's reasoning). Fetch ``limit + 1`` rows so we can detect
        # whether a next page exists without a separate COUNT.
        stmt: Select[Any] = (
            select(table).where(predicate).order_by(created_at_col, pk_col).limit(limit + 1)
        )
        if cursor is not None:
            anchor = _decode_cursor(cursor)
            anchor_dt = _parse_dt(anchor["created_at"])
            anchor_pk = anchor["id"]
            # ``(created_at, id) > (anchor_created_at, anchor_id)``
            # rendered as a row-tuple comparison fallback. SQLite has
            # historic gaps in row-tuple comparisons; expand it.
            stmt = stmt.where(
                or_(
                    created_at_col > anchor_dt,
                    and_(created_at_col == anchor_dt, pk_col > anchor_pk),
                )
            )
        async with self._engine.connect() as conn:
            result = await conn.execute(stmt)
            rows = result.mappings().all()
        items: list[Any] = []
        for row in rows[:limit]:
            items.append(row_to_record(record_type, row))
        next_cursor: str | None = None
        if len(rows) > limit:
            last = rows[limit - 1]
            next_cursor = _encode_cursor(
                {
                    "created_at": _parse_dt(last["created_at"]).isoformat(),
                    "id": last[pk_name],
                }
            )
        return CursorPage(items=items, next_cursor=next_cursor)


# === Technique helpers (registry mirror) =====================================


def _technique_from_row(row: Any) -> TechniqueRef:
    """Hydrate a :class:`TechniqueRef` from a ``techniques`` row mapping."""
    payload: dict[str, Any] = {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "category": row["category"],
        "compatible_factor_types": row["compatible_factor_types"],
        "standard": row["standard"],
        "fidelity_anchor": row["fidelity_anchor"],
        "affects_extension_points": row["affects_extension_points"],
        "implies_controlled": row["implies_controlled"] or [],
        "parameter_schema": row["parameter_schema"],
    }
    return TechniqueRef.model_validate(payload)


def _technique_page(rows: Sequence[Any], limit: int) -> CursorPage[TechniqueRef]:
    items = [_technique_from_row(r) for r in rows[:limit]]
    next_cursor: str | None = None
    if len(rows) > limit:
        next_cursor = _encode_cursor({"id": rows[limit - 1]["id"]})
    return CursorPage[TechniqueRef](items=items, next_cursor=next_cursor)


# Suppress unused-import noise from helpers used only for transaction-spec
# narrowing — pyright sees these via the `__all__` re-export below but
# the smaller helper names live as module-private.
_ = (
    Column,
    delete,
    factor_models_table,
    agent_sessions_table,
    run_costs_table,
    transition_log_table,
    cast,
)


__all__ = ["SqliteStore"]
