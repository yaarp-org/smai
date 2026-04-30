""":class:`PostgresStore` — Postgres reference :class:`MetadataStore` plugin.

Per ``designs/smai/07-plugin-interfaces.md`` §5, DEC-029 (Protocol defs in
smai-core), DEC-030 (SQL-shape only), DEC-035 (cursor pagination, lease
nonce, caller-resolved pool→states), DEC-036 (SQLAlchemy 2.0 async Core;
schema shared across plugins; Alembic-driven migrations land in
Task 3.H2).

The Postgres plugin imports the declarative schema from
:mod:`smai_orchestrator.migrations` (per Task 3.H2's schema lift —
prior to that, the schema lived in ``smai_store_sqlite._schema`` and
this plugin cross-imported it). Dialect-agnostic Pydantic ↔ row
helpers come from ``smai_store_sqlite._serde`` (the row-shape is
substrate-shared per DEC-036). SQLAlchemy renders Postgres-specific
SQL (``TIMESTAMP WITH TIME ZONE``, ``JSONB``-via-``JSON``, native
``RETURNING``, ``ON CONFLICT DO UPDATE``) from the same ``MetaData``
declaration at engine-bind time.

What this plugin adds over the shared shape:

* **``pg_try_advisory_xact_lock`` fast path** on :meth:`acquire_lease`
  (per §5.6.7) — keyed on a 64-bit hash of ``(entity_kind, entity_id)``
  to serialize contenders before the row UPDATE. Reduces wasted
  UPDATEs on hot-contention edges (e.g., dozens of workers polling for
  runs to dispatch when the run pool just opened slots). The CAS-via-
  ``nonce`` semantics on the row UPDATE are unchanged; the advisory
  lock is implementation-internal and does not leak through the
  Protocol.
* **``gen_random_uuid()`` for nonce generation** — Postgres-native;
  the SQLite plugin generates nonces in Python via ``uuid4()``. Both
  produce the same shape token surface (a 36-char string).

What this plugin adds with the **opt-in** ``tenant_aware=True`` mode
(Task 3.G2):

* **Tenant-aware scheduling.** Per §5.5 / §5.6.8 the default OSS
  PostgresStore reports ``is_tenant_aware=False`` and returns
  scheduling-query results in FIFO order. Constructing
  ``PostgresStore(tenant_aware=True)`` flips ``capabilities.is_tenant_aware``
  to ``True``, runs the opt-in ``tenant_aware`` Alembic branch (adding
  ``tenant_id VARCHAR(64)`` + composite indexes to every pipeline-
  tracking table), and overrides ``_paginate_predicate`` with the
  ``ROW_NUMBER() OVER (PARTITION BY tenant_id ORDER BY created_at, id)``
  window-function shape — interleaving (``fair_scheduling="round_robin"``)
  or weight-modulating (``fair_scheduling="weighted"``) across tenants.
  Single-tenant deployments leave the flag at the default
  ``tenant_aware=False`` and pay zero cost (the schema stays identical
  to the canonical OSS shape; the queries take the FIFO path
  unchanged). See the Task 3.G2 status note for the design carry-
  forward, including the closed ``AuroraStore`` subclass relationship.

Connection pooling. SQLAlchemy's :class:`AsyncEngine` ships with a
built-in pool (``QueuePool`` defaults: ``pool_size=5``,
``max_overflow=10``). Production deployments typically tune this via
constructor options — pass ``pool_size``, ``max_overflow``,
``pool_pre_ping``, ``pool_recycle`` through ``engine_kwargs`` (the
``PostgresStore.__init__`` extra-kwargs argument). Default is
``pool_pre_ping=True`` to detect stale connections after a Postgres
restart.
"""

from __future__ import annotations

import base64
import hashlib
import json
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal
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

# Per DEC-036 / Task 3.H2: the declarative schema is the substrate
# shared between every :class:`MetadataStore` plugin and lives in
# ``smai_orchestrator.migrations``. SQLAlchemy renders dialect-specific
# SQL from the single declaration at engine-bind time.
from smai_orchestrator.migrations import (
    ENTITY_PK_COLUMN,
    ENTITY_TABLE,
    TENANT_AWARE_BRANCH,
    cgs_table,
    entries_table,
    metadata,
    papers_table,
    proposals_table,
    runs_table,
    techniques_table,
    upgrade_to_head,
)
from smai_orchestrator.migrations.serde import row_to_record
from sqlalchemy import (
    ColumnElement,
    Float,
    Select,
    String,
    Table,
    and_,
    case,
    column,
    exists,
    func,
    insert,
    literal,
    not_,
    or_,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    create_async_engine,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

# Default URL points at the dockerized fixture under
# ``plugins/smai-store-postgres/compose.yaml``. Production deployments
# pass an explicit URL per ``09-cli.md`` §1.
DEFAULT_URI = "postgresql+asyncpg://smai:smai@localhost:5433/smai_test"


# Per-entity-kind in-progress states for ``count_with_in_flight_jobs``.
# Mirrors the SQLite plugin — caller resolves the state list per
# DEC-035 #3; the dispatch table here just maps each kind to the set of
# JSON columns whose non-null-ness defines "has an in-flight job."
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
# opaque. Same encoding as the SQLite plugin so the cross-plugin
# conformance fixtures share a shape — base64-encoded JSON of
# ``(created_at, id)`` for the FIFO scheduling queries, ``id`` only for
# the technique-registry list.


def _encode_cursor(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_cursor(cursor: str) -> dict[str, Any]:
    raw = base64.urlsafe_b64decode(cursor.encode("ascii"))
    decoded: object = json.loads(raw.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"cursor payload is not a dict: {decoded!r}")
    return {str(k): v for k, v in decoded.items()}  # type: ignore[reportUnknownVariableType]


def _parse_dt(value: Any) -> datetime:
    """Parse a stored datetime value back to a tz-aware ``datetime``.

    Postgres's ``TIMESTAMP WITH TIME ZONE`` round-trips as a tz-aware
    ``datetime`` directly; the cursor-decode path may also surface ISO
    strings, hence the dual handling.
    """
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    raise TypeError(f"unexpected datetime serialization: {type(value).__name__}: {value!r}")


# === Advisory-lock key derivation ============================================
#
# Per ``07-plugin-interfaces.md`` §5.6.7: "the Postgres implementation
# MAY use ``pg_try_advisory_xact_lock(...)`` keyed on a hash of
# ``(entity_kind, entity_id)`` as an additional fast-path serialization
# layer." Postgres's advisory-lock keys are 64-bit signed integers; we
# derive a stable signed-64-bit int from the SHA-256 of
# ``"<kind>:<entity_id>"`` and fold to two's-complement signed range.


def _advisory_lock_key(entity_kind: str, entity_id: str) -> int:
    """Derive a stable signed 64-bit key for ``pg_try_advisory_xact_lock``.

    Same ``(kind, id)`` pair always maps to the same key; different
    pairs almost always map to different keys (sha-256 collision domain).
    Returned value is in the signed-64-bit range so it round-trips cleanly
    through Postgres's ``bigint`` parameter binding without surprise.
    """
    digest = hashlib.sha256(f"{entity_kind}:{entity_id}".encode()).digest()
    # Take the first 8 bytes; treat as big-endian unsigned, then fold to
    # signed-64-bit range. ``int.from_bytes`` returns ``int`` (Python's
    # arbitrary-precision); the modulo + offset clips into
    # ``[-2**63, 2**63 - 1]``.
    unsigned = int.from_bytes(digest[:8], "big", signed=False)
    if unsigned >= 2**63:
        return unsigned - 2**64
    return unsigned


# === Transaction implementations =============================================
#
# Per ``07`` §5.3 the ``Transaction`` Protocol mirrors the writes on the
# parent ``MetadataStore``. :class:`_Transaction` delegates to the same
# row-builders as :class:`PostgresStore` but binds them to a single
# :class:`AsyncConnection` so all writes land in one BEGIN/COMMIT
# block. Same shape as the SQLite reference's ``_Transaction`` —
# different module so the per-plugin imports don't loop.


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
    so :meth:`PostgresStore.transaction` returns one of these eagerly
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


def _serialize_for_row(record: Any, exclude: set[str] | None = None) -> dict[str, Any]:
    """Convert a Pydantic record to a dict suitable for SQLAlchemy column binding."""
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

    Postgres's ``RETURNING`` is the canonical fast path here — same
    shape SQLite emulates via SQLAlchemy.
    """
    table = ENTITY_TABLE[kind]
    pk = ENTITY_PK_COLUMN[kind]
    now = datetime.now(UTC)
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
# Same predicate shapes as the SQLite reference. Kept here as plugin-local
# constants (not imported from sqlite) because the SQLite plugin's
# ``_store.py`` defines them as module-private and importing those would
# couple the two plugins more tightly than the ``_schema`` import already
# does. The duplication is small and the predicates change rarely — when
# they do, both plugins update in lockstep (the conformance suite catches
# drift).


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


def _predicate_proposals_at_human_gate() -> tuple[Table, ColumnElement[bool], type[Any]]:
    """All proposals in state ``designed`` regardless of decision flag.

    Per ``08`` §2.2 / W1.a reconciliation: the worker's three
    ``dispatch_time`` edges out of ``designed`` need to evaluate
    ``user_decision`` each cycle, so the predicate must include
    decided-but-not-yet-transitioned proposals.
    """
    return (
        proposals_table,
        proposals_table.c.state == "designed",
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
    return (papers_table, papers_table.c.state == "partial", PaperRecord)


# === Tenant-fair scheduling (opt-in tenant_aware=True mode) ==================
#
# Per ``07-plugin-interfaces.md`` §5.5 / §5.6.8: tenancy-aware fairness
# ordering is an *implementation* concern. The default OSS PostgresStore
# (``tenant_aware=False``) reports ``is_tenant_aware=False`` and returns
# FIFO ordering by ``(created_at, id)``. Setting
# ``PostgresStore(tenant_aware=True)`` (Task 3.G2) opts in to the
# tenant-aware schema (the ``tenant_aware`` Alembic branch adds a
# ``tenant_id`` column + composite index to every pipeline-tracking
# table) and the window-function ordering shape:
#
#     SELECT *,
#            ROW_NUMBER() OVER (
#                PARTITION BY tenant_id
#                ORDER BY created_at, id
#            ) AS tenant_rank
#       FROM <table>
#      WHERE <predicate>
#      ORDER BY tenant_rank, tenant_id, created_at, id
#      LIMIT :limit
#
# For ``fair_scheduling="weighted"`` the ``ROW_NUMBER`` is divided by a
# per-tenant weight (a CASE expression built from
# :attr:`fair_scheduling_weights`); see :meth:`_tenant_fair_rank_expr`
# for the math + ordering tie-break.
#
# The closed ``AuroraStore`` plugin (post-M4 per DEC-027) subclasses
# :class:`PostgresStore` and overrides ``_paginate_predicate`` further
# (e.g., per-tenant priority weights, fairness windowing). The
# tenant-aware schema produced by 0002 is the substrate AuroraStore
# inherits.


# === PostgresStore ===========================================================


class PostgresStore:
    """Reference Postgres :class:`MetadataStore` plugin (DEC-030, DEC-036).

    Construction takes a SQLAlchemy URL targeting Postgres via asyncpg
    (``postgresql+asyncpg://...``). Callers must ``await store.migrate()``
    before issuing any other call — ``migrate()`` is idempotent.

    Implementation-internal Postgres-specific paths:

    * Advisory-lock fast path on :meth:`acquire_lease` (per §5.6.7).
    * ``gen_random_uuid()`` for nonce generation in advisory-lock-mode
      acquire (per the §5.6.7 illustrative SQL); Python-side ``uuid4()``
      is the fallback when ``use_advisory_locks=False``.

    Per ``07-plugin-interfaces.md`` §5.5 / §5.6.8 the default plugin
    reports ``is_tenant_aware=False`` and returns FIFO scheduling-query
    results. The opt-in ``tenant_aware=True`` mode (Task 3.G2):

    * flips ``capabilities.is_tenant_aware`` to ``True``,
    * runs the ``tenant_aware`` Alembic branch (revision
      ``0002_tenant_aware_schema``) at boot, adding ``tenant_id``
      columns + composite indexes to every pipeline-tracking table,
    * routes scheduling-query pagination through a window-function
      ordering shape that interleaves (``fair_scheduling="round_robin"``)
      or weight-modulates (``fair_scheduling="weighted"``) candidates
      across tenants.

    Default OSS deployments leave ``tenant_aware=False`` and pay zero
    cost — the schema and queries are unchanged from the canonical OSS
    shape.
    """

    name: str = "postgres"

    def __init__(
        self,
        uri: str = DEFAULT_URI,
        *,
        use_advisory_locks: bool = True,
        tenant_aware: bool = False,
        fair_scheduling: Literal["off", "round_robin", "weighted"] = "off",
        fair_scheduling_weights: dict[str, float] | None = None,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Construct a Postgres-backed :class:`MetadataStore` plugin.

        ``uri``: A SQLAlchemy URL with the asyncpg driver, e.g.
        ``postgresql+asyncpg://user:pw@host:5432/db``.

        ``use_advisory_locks``: When True (default), :meth:`acquire_lease`
        wraps the row UPDATE with ``pg_try_advisory_xact_lock`` for the
        hot-contention fast path per §5.6.7. Implementation-internal —
        the Protocol's CAS-via-``nonce`` semantics hold either way.
        Tests that want to exercise the fallback path (no advisory lock)
        pass ``False``.

        ``tenant_aware`` (Task 3.G2 / `07` §5.5 / §5.6.8): When ``True``,
        :meth:`migrate` upgrades to the ``tenant_aware`` Alembic branch
        head (which depends on ``0001_initial_schema``, so the upgrade
        runs both 0001 and 0002 in order against a fresh database; on
        an already-canonical-OSS database the upgrade adds 0002's
        ``tenant_id`` columns + indexes idempotently). The plugin then
        reports ``is_tenant_aware=True`` and routes scheduling-query
        pagination through the window-function ordering shape per
        :meth:`_tenant_fair_rank_expr`. ``False`` (default): canonical
        OSS shape — single ``upgrade_to_head`` against the default
        branch, FIFO scheduling-query ordering. **Operators must apply
        ``smai migrate --upgrade-to=tenant_aware`` against an existing
        deployment before flipping this flag** if they prefer the
        explicit migration path; constructor-driven boot-time migrate is
        idempotent and works equally.

        ``fair_scheduling`` (Task 3.G2 / `05` §6 / `07` §5.5): The
        scheduling-query ordering policy. ``"off"`` is FIFO regardless
        of ``tenant_aware``; ``"round_robin"`` and ``"weighted"`` only
        apply when ``tenant_aware=True`` (silently ignored otherwise per
        the brief's "single-tenant deployments pay zero cost" contract).
        ``EngineConfig.fair_scheduling`` is the engine-side documented
        surface; operators set both values consistently via the same
        config-layering pipeline (env → smai.yaml → flags).

        ``fair_scheduling_weights``: Per-tenant weight map for
        ``fair_scheduling="weighted"`` (`05` §6). Tenant id → weight
        (positive float). A higher weight schedules that tenant's rows
        earlier in the dispatch queue. Empty / ``None`` is treated as
        uniform weights (== ``"round_robin"`` semantically). Tenants
        absent from the map default to weight ``1.0``. See
        :meth:`_tenant_fair_rank_expr` for the math.

        ``engine_kwargs``: Extra kwargs passed to
        ``create_async_engine``. Useful for production deployments that
        want to override the SQLAlchemy default pool settings —
        ``pool_size``, ``max_overflow``, ``pool_pre_ping``,
        ``pool_recycle``. Defaults: SQLAlchemy's own defaults plus
        ``pool_pre_ping=True`` (detect stale connections after a
        Postgres restart).
        """
        kwargs: dict[str, Any] = {"echo": False, "future": True, "pool_pre_ping": True}
        if engine_kwargs:
            kwargs.update(engine_kwargs)
        self._engine: AsyncEngine = create_async_engine(uri, **kwargs)
        self._use_advisory_locks: bool = use_advisory_locks
        self._tenant_aware: bool = tenant_aware
        self._fair_scheduling: Literal["off", "round_robin", "weighted"] = fair_scheduling
        self._fair_scheduling_weights: dict[str, float] = (
            dict(fair_scheduling_weights) if fair_scheduling_weights else {}
        )
        self.capabilities: MetadataStoreCapabilities = MetadataStoreCapabilities(
            is_tenant_aware=tenant_aware,
            supports_transactions=True,
            supports_leasing=True,
        )

    async def migrate(self) -> None:
        """Apply boot-time DDL via Alembic upgrade-to-head.

        Per Task 3.H2 / DEC-036: drives the shared Alembic env at
        :mod:`smai_orchestrator.migrations` against this plugin's
        :class:`AsyncEngine`. Idempotent — re-running against a
        head-stamped database is a no-op (Alembic consults
        ``alembic_version``).

        Branch selection (Task 3.G2):

        * ``tenant_aware=False`` (default): upgrades to the ``default``
          branch's head (``0001_initial_schema``).
        * ``tenant_aware=True``: upgrades to the ``tenant_aware``
          branch's head (``0002_tenant_aware_schema``); Alembic walks
          the ``depends_on=0001`` chain so 0001 runs first then 0002
          adds ``tenant_id`` + indexes.
        """
        branch = TENANT_AWARE_BRANCH if self._tenant_aware else None
        await upgrade_to_head(self._engine, branch=branch)

    async def drop_all(self) -> None:
        """Drop every table — for test cleanup and fixture rebuilds.

        Per Task 3.H2: also drops Alembic's ``alembic_version`` row /
        table so a subsequent ``migrate()`` call rebuilds the schema
        from scratch (otherwise the lingering version row makes
        Alembic skip the ``upgrade head`` pass and leaves the database
        empty-but-stamped).
        """
        async with self._engine.begin() as conn:
            await conn.run_sync(metadata.drop_all, checkfirst=True)
            await conn.execute(text("DROP TABLE IF EXISTS alembic_version"))

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
        return await self._paginate_predicate(
            (entries_table, entries_table.c.cg_id == cg_id, EntryRecord), limit, cursor
        )

    async def get_entry(self, entry_id: str) -> EntryRecord | None:
        async with self._engine.connect() as conn:
            result = await conn.execute(select(entries_table).where(entries_table.c.id == entry_id))
            row = result.mappings().first()
            return row_to_record(EntryRecord, row) if row else None

    async def list_runs_for_entry(
        self, entry_id: str, limit: int = 100, cursor: str | None = None
    ) -> CursorPage[RunRecord]:
        return await self._paginate_predicate(
            (runs_table, runs_table.c.entry_id == entry_id, RunRecord), limit, cursor
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
        return await self._paginate_predicate(
            (cgs_table, cgs_table.c.proposal_id == proposal_id, ComparisonGroupRecord),
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
        """Postgres-native ``INSERT ... ON CONFLICT (id) DO UPDATE``.

        SQLite uses a portable read-then-update pattern; Postgres has the
        canonical UPSERT shape via ``sqlalchemy.dialects.postgresql.insert``.
        Single round-trip on every call.
        """
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
        ins = pg_insert(techniques_table).values(**values)
        update_cols = {col: ins.excluded[col] for col in values if col != "id"}
        stmt = ins.on_conflict_do_update(index_elements=[techniques_table.c.id], set_=update_cols)
        async with self._engine.begin() as conn:
            await conn.execute(stmt)

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

    async def get_proposals_at_human_gate(
        self, limit: int = 100, cursor: str | None = None
    ) -> CursorPage[ProposalRecord]:
        return await self._paginate_predicate(_predicate_proposals_at_human_gate(), limit, cursor)

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
        """Acquire an opportunistic lease on ``(entity_kind, entity_id)``.

        Postgres fast path (default ``use_advisory_locks=True``):

        1. ``pg_try_advisory_xact_lock(:key)`` — non-blocking; returns
           ``False`` immediately if another transaction holds the
           advisory lock for this ``(kind, id)``. Same-transaction
           release is automatic at COMMIT.
        2. If acquired: the same UPDATE shape as the SQLite plugin —
           ``WHERE pk = :id AND (lease_expires_at IS NULL OR
           lease_expires_at < now())``.
        3. ``RETURNING`` surfaces the new ``lease_nonce`` (the canonical
           gating signal per §5.6.7).

        Fallback path (``use_advisory_locks=False``): drop step 1; rely
        on the row UPDATE alone. Useful for tests that want to exercise
        the SQLite-equivalent path.

        Returns ``None`` (not a raise) when another worker holds the
        lease — per §5.6.7, lease-acquire failure is normal in the poll
        loop.
        """
        table = ENTITY_TABLE[entity_kind]
        pk = ENTITY_PK_COLUMN[entity_kind]
        pk_col = table.c[pk]
        now = datetime.now(UTC)
        expires_at = now + timedelta(seconds=lease_seconds)
        nonce = str(uuid4())
        async with self._engine.begin() as conn:
            if self._use_advisory_locks:
                lock_key = _advisory_lock_key(entity_kind, entity_id)
                # ``pg_try_advisory_xact_lock`` is non-blocking; returns
                # bool. Lock is auto-released at COMMIT (no manual
                # ``pg_advisory_unlock`` needed).
                lock_result = await conn.execute(
                    text("SELECT pg_try_advisory_xact_lock(:k)").bindparams(k=lock_key)
                )
                acquired = bool(lock_result.scalar_one())
                if not acquired:
                    return None
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
        # Per §5.6.7 / DEC-035 #2: idempotent; nonce-only gate; 0 rows is
        # a silent no-op (stale token or unknown entity).
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

    # === Internal pagination helper =========================================

    def _tenant_fair_rank_expr(self, table: Table, pk_col: ColumnElement[Any]) -> Any:
        """Build the tenant-fair effective-rank expression for ``table``.

        Per Task 3.G2 / `07` §5.5: for ``fair_scheduling="round_robin"``
        the effective rank is just ``ROW_NUMBER() OVER (PARTITION BY
        tenant_id ORDER BY created_at, id)`` cast to float. For
        ``fair_scheduling="weighted"`` the row-number is divided by a
        per-tenant weight from :attr:`fair_scheduling_weights` (rendered
        as a SQL ``CASE`` expression keyed on ``tenant_id``); tenants
        absent from the map default to weight ``1.0``. Lower effective
        rank schedules earlier — so a tenant with weight ``2.0`` gets
        ranks ``0.5, 1.0, 1.5, ...`` and is interleaved twice as densely
        as a tenant with weight ``1.0`` whose ranks are ``1.0, 2.0,
        3.0, ...``.

        Worked example: weights ``{"tenant_a": 2.0, "tenant_b": 1.0}``
        with three rows per tenant produces the order::

            a1 (0.5), a2 (1.0), b1 (1.0), a3 (1.5), b2 (2.0), b3 (3.0)

        The tie-break for equal effective ranks is ``tenant_id`` ASC
        then ``created_at`` ASC then ``<pk>`` ASC — making the ordering
        fully deterministic.

        Returns the rank expression (a SQLAlchemy column expression
        usable in ``.order_by()`` / cursor predicates). Caller selects
        ``tenant_id`` separately.

        Implementation note: ``tenant_id`` is added to the table by the
        opt-in 0002 Alembic revision but is NOT declared on the shared
        :class:`Table` object in :mod:`smai_orchestrator.migrations` —
        keeping it out of the shared schema preserves the "OSS pays
        zero cost" contract. Here we reference it via
        :func:`sqlalchemy.column` (an unbound column reference resolved
        at SQL render time) so the expression compiles correctly
        regardless of whether the column is declared on the Table.
        """
        tenant_col = column("tenant_id", String(64))
        created_at_col = table.c["created_at"]
        row_number = func.row_number().over(
            partition_by=tenant_col,
            order_by=(created_at_col, pk_col),
        )
        # Cast to float so the divide-by-weight branch returns a
        # consistent type across modes; Postgres FLOAT keeps the
        # comparison lex-clean against cursor-encoded floats.
        rank_as_float = func.cast(row_number, Float)
        if self._fair_scheduling == "weighted" and self._fair_scheduling_weights:
            # Build the per-tenant weight CASE: WHEN tenant_id = 'X'
            # THEN <weight> ... ELSE 1.0. Tenants absent from the map
            # take weight 1.0 — same as round-robin for those rows.
            weight_expr = case(
                {
                    tenant: literal(float(weight))
                    for tenant, weight in self._fair_scheduling_weights.items()
                },
                value=tenant_col,
                else_=literal(1.0),
            )
            return rank_as_float / weight_expr
        return rank_as_float

    async def _paginate_predicate(
        self,
        predicate_spec: tuple[Table, ColumnElement[bool], type[Any]],
        limit: int,
        cursor: str | None,
    ) -> CursorPage[Any]:
        """Cursor-paginated scheduling-query result page.

        Default mode (``tenant_aware=False`` or ``fair_scheduling="off"``):
        FIFO ordering by ``(created_at, <pk>)`` — the canonical OSS
        single-tenant shape per `07` §5.6.10. The closed
        ``AuroraStore`` plugin (post-M4 per DEC-027) subclasses this
        method to layer additional tenant-priority logic on top.

        Tenant-fair mode (``tenant_aware=True`` and
        ``fair_scheduling != "off"``): renders the
        :meth:`_tenant_fair_rank_expr` window-function CTE, ordering by
        ``(effective_rank, tenant_id, created_at, <pk>)``. Cursor
        encoding switches to the rank-tuple shape so pagination across
        pages stays well-defined under steady-state writes.
        """
        table, predicate, record_type = predicate_spec
        pk_name = "arxiv_id" if "arxiv_id" in {c.name for c in table.columns} else "id"
        pk_col = table.c[pk_name]
        if self._is_tenant_fair_mode():
            return await self._paginate_predicate_tenant_fair(
                table=table,
                predicate=predicate,
                record_type=record_type,
                pk_name=pk_name,
                pk_col=pk_col,
                limit=limit,
                cursor=cursor,
            )
        created_at_col = table.c["created_at"]
        stmt: Select[Any] = (
            select(table).where(predicate).order_by(created_at_col, pk_col).limit(limit + 1)
        )
        if cursor is not None:
            anchor = _decode_cursor(cursor)
            anchor_dt = _parse_dt(anchor["created_at"])
            anchor_pk = anchor["id"]
            # Postgres natively supports row-tuple comparison
            # (``(created_at, id) > (a, b)``); the expanded form below
            # matches the SQLite plugin's portable shape so cursor
            # tokens encoded by either plugin decode the same way.
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

    def _is_tenant_fair_mode(self) -> bool:
        """``True`` when scheduling-query results should follow the
        tenant-fair ordering (`07` §5.5 / §5.6.8).

        Both flags must align: the schema needs a ``tenant_id`` column
        (``tenant_aware=True``) and the policy must be on
        (``fair_scheduling != "off"``). Single-tenant deployments
        (``tenant_aware=False``) silently take the FIFO path regardless
        of ``fair_scheduling`` — per the brief's "single-tenant
        deployments pay zero cost" contract.
        """
        return self._tenant_aware and self._fair_scheduling != "off"

    async def _paginate_predicate_tenant_fair(  # noqa: PLR0913
        self,
        *,
        table: Table,
        predicate: ColumnElement[bool],
        record_type: type[Any],
        pk_name: str,
        pk_col: ColumnElement[Any],
        limit: int,
        cursor: str | None,
    ) -> CursorPage[Any]:
        """Render the tenant-fair window-function CTE for ``table``.

        See :meth:`_tenant_fair_rank_expr` for the rank formula. The
        cursor encodes ``(effective_rank, tenant_id, created_at, pk)``
        so cross-page iteration filters lex-greater than the last seen
        anchor.
        """
        rank_expr = self._tenant_fair_rank_expr(table, pk_col).label("effective_rank")
        # ``tenant_id`` is added by the opt-in 0002 Alembic revision but
        # is not declared on the shared :class:`Table` object — see
        # :meth:`_tenant_fair_rank_expr`'s implementation note. Pull it
        # in via :func:`column` so the CTE projects it for the cursor
        # tie-break + ORDER BY.
        tenant_col_ref = column("tenant_id", String(64)).label("tenant_id")
        # Build a CTE so the rank expression is a real column we can
        # filter / sort on twice without re-rendering the window
        # function.
        ranked_cte = (
            select(table, tenant_col_ref, rank_expr).where(predicate).cte(name="tenant_ranked")
        )
        ranked_rank = ranked_cte.c["effective_rank"]
        ranked_tenant = ranked_cte.c["tenant_id"]
        ranked_created = ranked_cte.c["created_at"]
        ranked_pk = ranked_cte.c[pk_name]
        stmt: Select[Any] = (
            select(ranked_cte)
            .order_by(ranked_rank, ranked_tenant, ranked_created, ranked_pk)
            .limit(limit + 1)
        )
        if cursor is not None:
            anchor = _decode_cursor(cursor)
            anchor_rank = float(anchor["rank"])
            anchor_tenant = anchor["tenant_id"]
            anchor_dt = _parse_dt(anchor["created_at"])
            anchor_pk = anchor[pk_name]
            # Lexicographic "greater than" on the 4-tuple ordering key,
            # expanded into chained ANDs / ORs since SQLAlchemy doesn't
            # render multi-column row-value tuple comparisons portably
            # across all the cases we hit (NULL-safe handling for
            # tenant_id varies by dialect; the expanded form is
            # explicit).
            stmt = stmt.where(
                or_(
                    ranked_rank > anchor_rank,
                    and_(
                        ranked_rank == anchor_rank,
                        ranked_tenant > anchor_tenant,
                    ),
                    and_(
                        ranked_rank == anchor_rank,
                        ranked_tenant == anchor_tenant,
                        ranked_created > anchor_dt,
                    ),
                    and_(
                        ranked_rank == anchor_rank,
                        ranked_tenant == anchor_tenant,
                        ranked_created == anchor_dt,
                        ranked_pk > anchor_pk,
                    ),
                )
            )
        async with self._engine.connect() as conn:
            result = await conn.execute(stmt)
            rows = result.mappings().all()
        # The CTE includes two non-record columns (``tenant_id`` is on
        # the row but absent from the Pydantic ``record_type`` since the
        # OSS records are tenancy-agnostic per `07` §5.6.8;
        # ``effective_rank`` is the rank we computed). Project the row
        # mapping down to just the declared :class:`Table` columns
        # before ``row_to_record`` so Pydantic's ``extra="forbid"``
        # discipline doesn't raise.
        record_columns: set[str] = {c.name for c in table.columns}
        items: list[Any] = []
        for row in rows[:limit]:
            projected = {k: v for k, v in row.items() if k in record_columns}
            items.append(row_to_record(record_type, projected))
        next_cursor: str | None = None
        if len(rows) > limit:
            last = rows[limit - 1]
            next_cursor = _encode_cursor(
                {
                    "rank": float(last["effective_rank"]),
                    "tenant_id": last["tenant_id"],
                    "created_at": _parse_dt(last["created_at"]).isoformat(),
                    pk_name: last[pk_name],
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


__all__ = ["PostgresStore"]
