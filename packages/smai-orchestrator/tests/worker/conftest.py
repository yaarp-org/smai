"""Pytest fixtures shared across worker tests.

Auto-discovered by pytest. Re-mounts the engine test directory on
``sys.path`` so the ``_helpers`` module ships with FakeClock /
FakeMonotonic / FakeCompute / FakeArtifactStore / make_gate /
make_dispatch / make_job_handle / make_job_status — the same
substrate the engine tests use, kept in one place.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from smai_artifacts_localfs import LocalFsStore
from smai_store_sqlite import SqliteStore

_ENGINE_TESTS_DIR = Path(__file__).resolve().parent.parent / "engine"
sys.path.insert(0, str(_ENGINE_TESTS_DIR))


@pytest_asyncio.fixture
async def sqlite_store() -> AsyncIterator[SqliteStore]:
    """In-memory :class:`SqliteStore` per test (shared with engine tests)."""
    store = SqliteStore("sqlite+aiosqlite:///:memory:")
    await store.migrate()
    try:
        yield store
    finally:
        await store.dispose()


@pytest.fixture
def localfs_store(tmp_path: Path) -> LocalFsStore:
    """File-backed :class:`LocalFsStore` rooted at ``tmp_path`` per test."""
    return LocalFsStore(tmp_path)
