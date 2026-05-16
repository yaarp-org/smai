# SMAI metadata-schema migrations

Per Task 3.H2 / DEC-036: SMAI v2's :class:`MetadataStore` plugins
share a single SQLAlchemy 2.0 Core schema and a single Alembic env.
Both reference plugins (``smai-store-sqlite`` and ``smai-store-postgres``)
import the schema from this package and call :func:`upgrade_to_head`
at boot.

This document covers the operational story: how to run a migration,
the rollback policy, retention defaults, and the developer flow for
adding the next revision.

---

## Running migrations

### `smai dev`

`smai dev` boots in-band and calls ``MetadataStore.migrate()`` during
plugin instantiation. Both plugins implement ``migrate()`` as
`upgrade_to_head` against their async engine. Idempotent: running
`smai dev` against an empty database creates the schema; running it
against a current-version database is a no-op.

### `smai start` (production)

`smai start` is `smai dev`'s production sibling (Task 3.G3). The
production deployment pattern is:

```bash
smai migrate --check    # exits 0 iff the schema is at head
smai migrate            # runs Alembic upgrade head
smai start              # boots the worker
```

The pre-flight `--check` fails the deployment fast if a migration is
needed but hasn't been run; this is the "no surprise migrations on
worker boot" invariant. `smai dev` skips the check (laptop ergonomic
— laptop deployments accept the implicit upgrade-on-boot).

### `smai migrate`

The verb's three flags (per `09-cli.md` §1):

* `smai migrate` — equivalent to ``alembic upgrade head``. Idempotent.
* `smai migrate --dry-run` — equivalent to ``alembic upgrade head --sql``.
  Renders the SQL to stdout without executing it; useful for review
  before a prod deployment.
* `smai migrate --check` — exits ``0`` if the schema is at head, ``1``
  otherwise (with the version delta on stderr). Useful for `smai start`
  pre-flight and for CI gates that want to verify migrations are
  applied.
* `smai migrate --prune` — runs the retention sweep (see below).

---

## Rollback policy (design-time deferral)

**v2 does NOT implement automatic schema rollback.** Calling
``alembic downgrade`` (or ``smai migrate --downgrade``, which is not
shipped) raises :class:`NotImplementedError` from the affected
revision.

The rationale, in three points:

1. **Backup-first recovery.** Production deployments take regular
   database backups (Postgres point-in-time recovery / SQLite
   filesystem snapshots). A "bad migration" recovers from backup —
   ``UPDATE``s applied between the migration and the rollback are
   recoverable from the audit log (``transition_log``) but not from
   the schema itself, so the down-migration would not be lossless
   anyway. Backup restore is the lossless path.
2. **Migration shape stays simple.** Because down-migrations are not
   maintained, every revision is purely additive (new columns,
   new tables, new indexes). Schema-shape changes that would require
   a destructive down-migration (rename, drop, type change) are
   instead expressed as a sequence of additive revisions: add the
   new column, double-write, backfill, switch readers, drop the old
   column in a later revision. This pattern is more verbose but
   roll-forward-safe; the v3 work that lifts schema-shape changes
   into the design will revisit if needed.
3. **Forward-fix discipline.** A bad migration in v2 ships a
   forward-fix revision rather than a rollback. The audit log keeps
   the history; the operator's runbook documents the recovery flow.

If a v3 deployment shape needs explicit rollback, the path is:

* Implement ``downgrade()`` in each revision (currently ``raise``).
* Add a ``smai migrate --downgrade <target>`` flag to the CLI.
* Update the operator runbook with rollback decision criteria.

These are tracked as v3 work; v2 ships the documented forward-only
shape.

---

## Retention defaults (DEC-033 #1, #2)

Three tables grow linearly in time:

* `transition_log` — every state transition writes a row. Audit
  trail; consumed by operator queries (per `05-orchestrator.md` §9
  OQ5).
* `agent_sessions` — every agent turn writes a row. Token-shaped
  cost ledger.
* `run_costs` — every completed run writes a row. GPU-shaped cost
  ledger.

Default retention windows, set in
:data:`smai_orchestrator.migrations.runner.DEFAULT_RETENTION_DAYS`:

| Table | Days | Reasoning |
|-------|------|-----------|
| `transition_log` | 90 | One quarter — enough for incident investigation past a release boundary. |
| `agent_sessions` | 180 | Two quarters — covers semi-annual cost retros + the end-of-year accounting cycle. |
| `run_costs` | 365 | One year — annual GPU-spend review. |

Operators override per deployment via
:attr:`EngineConfig.retention_policies`:

```yaml
engine:
  retention_policies:
    transition_log: 30
    agent_sessions: 90
    run_costs: 730
```

Setting a value to ``0`` disables retention for that table (no rows
deleted regardless of age). Tables not in :data:`RETENTION_TABLES` are
rejected with :class:`ValueError` — the validator catches typos at
load time.

The retention sweep is **never automatic at boot**. Operators run it
explicitly via:

```bash
smai migrate --prune
```

This prints per-table deletion counts on stdout. Schedule it via cron
or a Kubernetes CronJob; production cadence is typically nightly.

---

## Adding a new revision

The standard Alembic flow with one wrinkle (the env is async-aware
but Alembic's autogenerate is sync-only):

1. Update the SQLAlchemy declarations in :mod:`.metadata`.
2. Run autogenerate against a synchronous handle to a current-version
   database. Example for SQLite:

   ```bash
   uv run alembic -c packages/smai-orchestrator/src/smai_orchestrator/migrations/alembic.ini \
       --raiseerr revision --autogenerate -m "<short message>"
   ```

   Note: Alembic's autogenerate connects via the configured
   ``sqlalchemy.url`` synchronously. Use a sync URL for autogenerate
   (e.g., ``sqlite:///tmp/test.db`` or
   ``postgresql://...``); the generated revision is dialect-portable
   so the async-driver path picks it up at runtime.

3. Hand-edit the generated revision:

   * Keep ``upgrade()`` additive and set ``downgrade()`` to ``raise
     NotImplementedError(...)`` per the forward-only policy (DEC-036).
   * Use the **dual-mode add-column pattern** established by revisions
     0003 / 0004. The 0001 baseline applies the schema via
     ``MetaData.create_all(checkfirst=True)`` against the *canonical*
     SQLAlchemy declarations in :mod:`.metadata`, which already include
     your new column once step 1 is done. So on a fresh database 0001
     creates the column and a naive ``op.add_column`` in your revision
     then errors on the duplicate. Guard it:

     ```python
     if context.is_offline_mode():
         # MockConnection can't be introspected; emit the ADD COLUMN
         # unconditionally so --sql output covers the legacy upgrade path.
         op.add_column("<table>", sa.Column("<col>", ...))
         return
     inspector = sa.inspect(op.get_bind())
     existing = {c["name"] for c in inspector.get_columns("<table>")}
     if "<col>" not in existing:
         op.add_column("<table>", sa.Column("<col>", ...))
     ```

     `versions/0003_cg_symbolic_id.py` is the reference shape.

4. Add a unit test in `tests/test_migrations/` that runs the new
   revision against an in-memory SQLite + asserts the resulting
   schema matches what the new declarations describe.

5. Run the conformance suite against both reference plugins (SQLite
   in-process; Postgres against the docker-compose fixture). The
   suite catches schema-shape regressions across both dialects.

---

## Branch graph (Task 3.G2: opt-in tenant_aware extension)

The Alembic env carries **two branches**:

| Branch label | Head revision | Applied by |
|--------------|---------------|------------|
| `default` | `0004_cg_impl_phase_attempt` | Every reference plugin's `migrate()` at boot; `smai migrate` (no flag); `smai dev`. |
| `tenant_aware` | `0002_tenant_aware_schema` | `PostgresStore(tenant_aware=True).migrate()`; `smai migrate --upgrade-to=tenant_aware`. |

The `tenant_aware` branch is a **separate-root** revision with
`depends_on="0001_initial_schema"`. Alembic enforces the dependency:
upgrading to `tenant_aware@head` against an unstamped database applies
0001 first then 0002; against a 0001-stamped database it applies only
0002. The shape keeps the OSS canonical schema (the `default` branch)
decoupled from the opt-in extension.

### Default-branch revisions

The `default` branch is a linear chain `0001 → 0003 → 0004` (0002 is
the off-branch tenant_aware extension above):

| Revision | Adds |
|----------|------|
| `0001_initial_schema` | The full v1 schema declared in :mod:`.metadata` (applied via `create_all(checkfirst=True)`). |
| `0003_cg_symbolic_id` | Nullable `cgs.symbolic_id VARCHAR(128)` — preserves the planner's human-readable CG draft id after the round-9 fix made the primary CG `id` a fresh ULID-shaped string. |
| `0004_cg_impl_phase_attempt` | `cgs.implementation_phase_attempt INTEGER` (default 0) — the CG-level retry counter the round-10 `RetryPolicy` on the `implementing` dispatch state reads. |

Both 0003 and 0004 use the dual-mode add-column pattern documented in
"Adding a new revision" above (offline-unconditional / online
inspect-and-skip) so they are no-ops on a fresh database that 0001's
`create_all` already brought to the canonical shape.

### What 0002 adds

A nullable `tenant_id VARCHAR(64)` column on each pipeline-tracking
table (`cgs`, `entries`, `runs`, `proposals`, `papers`) plus a composite
index `(tenant_id, created_at, <pk>)` on each. The column anchors the
`ROW_NUMBER() OVER (PARTITION BY tenant_id ORDER BY created_at, id)`
window-function ordering the `PostgresStore(tenant_aware=True)` queries
emit when `fair_scheduling != "off"`.

Operators populating `tenant_id` is their concern — the OSS plugin
treats NULL as a single `<no-tenant>` partition. The closed
`AuroraStore` plugin (post-M4 per DEC-027) inherits this schema and
layers tenant-priority semantics on top.

### Running the tenant-aware migration

Either:

```bash
# Operator-driven explicit upgrade — preferred for production.
smai migrate --upgrade-to=tenant_aware
```

…or just construct the plugin with `tenant_aware=True` and let
`migrate()` do it:

```python
store = PostgresStore(uri=..., tenant_aware=True, fair_scheduling="round_robin")
await store.migrate()  # upgrades to tenant_aware@head idempotently
```

The default `smai migrate` (no flag) targets `default@head` only —
running it against a tenant-aware-stamped database is a no-op (0002 is
on a separate branch the default upgrade chain never visits).

### Rolling back from tenant_aware

Per the forward-only policy: restore from a backup taken before the
0002 upgrade. Alternatively, flipping `PostgresStore(tenant_aware=
False)` against an already-extended schema is operationally fine — the
`tenant_id` column / index sit unused and the queries take the FIFO
path — but the Alembic version row pins `0002_tenant_aware_schema`, so
`smai migrate --check` against the default branch would flag drift.
