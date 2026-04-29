"""Shared fixtures for the Postgres plugin's test modules.

The Postgres plugin's tests are integration-shaped (per
``07-plugin-interfaces.md`` §8 / implementation_plan §4.1's CI-lane
discipline): they depend on a running Postgres instance reachable at
the URL from the ``SMAI_POSTGRES_TEST_URL`` env var (the default
``compose.yaml`` provides ``postgresql+asyncpg://smai:smai@localhost:5433/smai_test``).

When the env var is unset OR the database is not reachable, every test
in this directory cleanly skips — so contributors without Docker still
see a green ``uv run pytest``. The CI-side wiring is documented in
``README.md``.
"""

from __future__ import annotations

import os
import socket
from collections.abc import AsyncGenerator
from urllib.parse import urlparse

import pytest
from smai_store_postgres import PostgresStore

POSTGRES_URL_ENV: str = "SMAI_POSTGRES_TEST_URL"
DEFAULT_TEST_URL: str = "postgresql+asyncpg://smai:smai@localhost:5433/smai_test"


def resolve_postgres_url() -> str | None:
    """Return the Postgres test URL or None if not configured.

    Resolution order:

    1. ``$SMAI_POSTGRES_TEST_URL`` if set.
    2. The compose.yaml default URL — but only if a TCP probe shows the
       port is open. The probe avoids a slow asyncpg connection error
       in environments where Docker isn't running.
    """
    explicit = os.environ.get(POSTGRES_URL_ENV)
    if explicit:
        return explicit
    parsed = urlparse(DEFAULT_TEST_URL.replace("+asyncpg", ""))
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.2)
    try:
        sock.connect((host, port))
    except OSError:
        return None
    finally:
        sock.close()
    return DEFAULT_TEST_URL


@pytest.fixture
def postgres_url() -> str:
    """Per-test Postgres URL; skip if no DB reachable.

    Function-scoped (rather than session-scoped) so a test that mutates
    ``SMAI_POSTGRES_TEST_URL`` between cases re-resolves cleanly.
    """
    url = resolve_postgres_url()
    if url is None:
        pytest.skip(
            f"No Postgres test database reachable. Set {POSTGRES_URL_ENV} or run "
            "`docker compose -f plugins/smai-store-postgres/compose.yaml up -d`."
        )
    return url


@pytest.fixture
async def fresh_store(postgres_url: str) -> AsyncGenerator[PostgresStore, None]:
    """Per-test :class:`PostgresStore` against a freshly migrated schema.

    Each test gets a clean schema (drop_all + migrate around the
    yield). Heavier than SQLite's ``:memory:`` per-test isolation (each
    call is a real network round-trip) but still practical at v1 scale —
    the conformance suite is ~25 tests; a full run against a local
    Docker fixture takes a few seconds.
    """
    store = PostgresStore(uri=postgres_url)
    await store.drop_all()
    await store.migrate()
    try:
        yield store
    finally:
        await store.drop_all()
        await store.dispose()
