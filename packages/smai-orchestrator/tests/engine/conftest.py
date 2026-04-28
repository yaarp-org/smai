"""Pytest fixtures shared across engine tests.

Auto-discovered by pytest. Fixtures here yield real plugin instances
(SqliteStore, FakeCompute, FakeArtifactStore); helper classes / builders
live in :mod:`_helpers` and are imported directly by test modules.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from smai_store_sqlite import SqliteStore

# Make ``_helpers.py`` importable from sibling test files without using
# package-qualified paths (pytest's ``importlib`` import-mode does not
# add the test directory to ``sys.path``).
sys.path.insert(0, str(Path(__file__).parent))

from _helpers import FakeArtifactStore, FakeCompute  # noqa: E402


@pytest_asyncio.fixture
async def sqlite_store() -> AsyncIterator[SqliteStore]:
    """Construct an in-memory :class:`SqliteStore` per test (Task 2.A2).

    The plugin's ``migrate()`` is idempotent; ``:memory:`` keeps tests
    isolated and fast.
    """
    store = SqliteStore("sqlite+aiosqlite:///:memory:")
    await store.migrate()
    try:
        yield store
    finally:
        await store.dispose()


@pytest.fixture
def fake_compute() -> FakeCompute:
    return FakeCompute()


@pytest.fixture
def fake_artifact_store() -> FakeArtifactStore:
    return FakeArtifactStore()
