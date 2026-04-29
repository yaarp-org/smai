"""Conformance test wiring for :class:`PostgresStore`.

Per Task 3.F1 / ``07-plugin-interfaces.md`` §5.8: subclass
:class:`MetadataStoreConformance` so the universal contract suite runs
against the real Postgres plugin against a Docker-compose fixture.

This module is integration-shaped: every test cleanly skips when no
Postgres is reachable (see :mod:`conftest`), so contributors without
Docker still see a green ``uv run pytest``. CI runs this in a dedicated
Postgres lane (see ``README.md`` for the recommended workflow shape).

The conformance base class's default ``store`` fixture calls
``make_store()`` once per test. We override the ``store`` fixture
directly (rather than ``make_store``) so we can accept the
``postgres_url`` fixture from :mod:`conftest`, drop+migrate around the
yield, and dispose the engine on teardown.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
from smai_core.plugins import MetadataStore
from smai_core.plugins.conformance import MetadataStoreConformance
from smai_store_postgres import PostgresStore


class TestPostgresStoreConformance(MetadataStoreConformance):
    """Run the universal :class:`MetadataStore` contract suite against
    :class:`PostgresStore` (Docker-compose fixture)."""

    @pytest.fixture
    async def store(  # type: ignore[override]
        self, postgres_url: str
    ) -> AsyncGenerator[MetadataStore, None]:
        s = PostgresStore(uri=postgres_url)
        await s.drop_all()
        await s.migrate()
        try:
            yield s
        finally:
            await s.drop_all()
            await s.dispose()
