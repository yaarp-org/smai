# smai-store-postgres

MetadataStore plugin: PostgreSQL reference production implementation.

Per `designs/smai/07-plugin-interfaces.md` §5, DEC-030 (SQL-shape only),
DEC-036 (SQLAlchemy 2.0 async Core; schema shared with the SQLite
reference; Alembic deferred to Task 3.H2).

The plugin reuses the declarative `MetaData` / `Table` set from
`smai-store-sqlite` and swaps the driver to `asyncpg`. SQLAlchemy
renders Postgres-specific SQL (`TIMESTAMP WITH TIME ZONE`, native
`RETURNING`, `ON CONFLICT DO UPDATE`) at engine-bind time. Two
Postgres-specific fast paths layer on top of the shared shape:

- **`pg_try_advisory_xact_lock`** on `acquire_lease` (per §5.6.7) —
  reduces wasted UPDATEs on hot-contention edges. Implementation-
  internal: the CAS-via-`nonce` semantics on the row UPDATE are
  unchanged.
- **`gen_random_uuid()` / Postgres-native UPSERT** for
  `upsert_technique`'s `INSERT ... ON CONFLICT (id) DO UPDATE` — single
  round-trip on every call.

Per §5.5 / §5.6.8 the OSS plugin reports `is_tenant_aware=False`.
Tenant-fair scheduling lives in the closed `AuroraStore` plugin.

## Running tests

The plugin's tests are integration-shaped: they require a running
Postgres instance. The repo ships a Docker Compose fixture pinned to
`postgres:17-alpine`:

```bash
docker compose -f plugins/smai-store-postgres/compose.yaml up -d
uv run pytest plugins/smai-store-postgres/tests/
docker compose -f plugins/smai-store-postgres/compose.yaml down -v
```

When no Postgres is reachable, every test in this directory cleanly
skips — so contributors without Docker still see a green
`uv run pytest`. The skip is gated on `SMAI_POSTGRES_TEST_URL` being
unset AND a TCP probe to `localhost:5433` failing.

To point the suite at a non-default Postgres (different host / port /
credentials):

```bash
export SMAI_POSTGRES_TEST_URL='postgresql+asyncpg://user:pw@host:port/db'
uv run pytest plugins/smai-store-postgres/tests/
```
