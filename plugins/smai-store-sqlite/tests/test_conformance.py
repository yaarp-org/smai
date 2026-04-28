"""Conformance test wiring for :class:`SqliteStore`.

Per Task 2.A2: subclass :class:`MetadataStoreConformance` (per
``07-plugin-interfaces.md`` §5.8) so the 22 universal contract tests
run against the real plugin. The base ships in
``smai_core.plugins.conformance`` (Task 1.9, behind the
``[conformance]`` extra dep — so this file imports cleanly only after
``uv sync --extra conformance``).

Each test gets a fresh in-memory SQLite database via
``make_store()`` — :memory: + ``await store.migrate()`` is the
canonical "fresh schema, isolated state" pattern called out in the
base's docstring.
"""

from __future__ import annotations

from smai_core.plugins.conformance import MetadataStoreConformance
from smai_store_sqlite import SqliteStore


class TestSqliteStoreConformance(MetadataStoreConformance):
    """Run the universal :class:`MetadataStore` contract suite against
    :class:`SqliteStore`."""

    async def make_store(self) -> SqliteStore:
        # ``cache=shared`` is intentionally omitted — each test wants a
        # fully isolated DB so cross-test state cannot leak. The
        # default ``:memory:`` URL gives one per call.
        store = SqliteStore(uri="sqlite+aiosqlite:///:memory:")
        await store.migrate()
        return store
