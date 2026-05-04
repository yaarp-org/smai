"""MetadataStore plugin: PostgreSQL reference production implementation.

Per ``designs/smai/07-plugin-interfaces.md`` §5, DEC-030 (SQL-shape only),
DEC-036 (SQLAlchemy 2.0 async Core; Alembic deferred to Task 3.H2). The
plugin reuses the declarative ``MetaData`` / ``Table`` set declared in
``smai-store-sqlite`` (DEC-036's "the same Core schema") and swaps the
driver to asyncpg so SQLAlchemy renders dialect-specific SQL at
engine-bind time.

The Postgres dialect adds two implementation-internal fast paths over
the shared SQLite/Postgres CAS shape:

* **``pg_try_advisory_xact_lock``** on the lease-acquire predicate
  (per ``07-plugin-interfaces.md`` §5.6.7) — reduces wasted UPDATEs
  on hot-contention edges. Implementation-internal: the
  CAS-via-``nonce`` semantics on the row UPDATE are unchanged; the
  advisory lock just serializes contenders before they touch the row.
* **``gen_random_uuid()``** for nonce generation — Postgres-native;
  the SQLite plugin generates nonces in Python via ``uuid4()``. Both
  produce the same shape token surface.

Window-function-derived tenant-fair scheduling is documented as a
seam for the closed ``AuroraStore`` plugin; the OSS reference
plugin reports ``is_tenant_aware=False`` per ``07`` §5.5 / §5.6.8 and
returns scheduling-query results in the default FIFO order.

Entry point::

    [project.entry-points."smai.metadata_stores"]
    postgres = "smai_store_postgres:PostgresStore"
"""

from smai_store_postgres._event_channel import (
    NOTIFY_CHANNEL,
    PayloadTooLargeError,
    PgNotifyEventChannel,
)
from smai_store_postgres._store import PostgresStore

__all__ = [
    "NOTIFY_CHANNEL",
    "PayloadTooLargeError",
    "PgNotifyEventChannel",
    "PostgresStore",
]
