# smai-store-postgres

MetadataStore plugin: PostgreSQL reference production implementation.
Backed by SQLAlchemy 2.0 async Core + `asyncpg`. The plugin reuses the
declarative `MetaData` / `Table` set from `smai-store-sqlite`, swaps the
driver to `asyncpg`, and adds two Postgres-specific fast paths:

- **`pg_try_advisory_xact_lock`** on `acquire_lease`: reduces wasted
  UPDATEs on hot-contention edges. The CAS-via-`nonce` semantics on the row
  UPDATE are unchanged.
- **Postgres-native UPSERT** for `upsert_technique`'s
  `INSERT ... ON CONFLICT (id) DO UPDATE`: single round-trip on every call.

The OSS plugin reports `is_tenant_aware=False`.

## Configuration

```yaml
plugins:
  metadata_store: postgres
  metadata_store_config:
    uri: "postgresql+asyncpg://user:pw@host:5432/smai"   # required
    use_advisory_locks: true     # default
    tenant_aware: false          # default
    fair_scheduling: "off"       # default
    # fair_scheduling_weights: {}
    # engine_kwargs: { pool_size: 10 }
```

The `uri` embeds credentials; there is no separate credential argument.

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
skips, so contributors without Docker still see a green
`uv run pytest`. The skip is gated on `SMAI_POSTGRES_TEST_URL` being
unset AND a TCP probe to `localhost:5433` failing.

To point the suite at a non-default Postgres (different host / port /
credentials):

```bash
export SMAI_POSTGRES_TEST_URL='postgresql+asyncpg://user:pw@host:port/db'
uv run pytest plugins/smai-store-postgres/tests/
```
