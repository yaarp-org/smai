"""Pytest fixtures shared across checkpointer tests.

Auto-discovered by pytest. Re-mounts the engine test directory on
``sys.path`` so the ``_helpers`` module is importable here too (for
FakeClock / FakeArtifactStore / FakeMonotonic).
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
    store = SqliteStore("sqlite+aiosqlite:///:memory:")
    await store.migrate()
    try:
        yield store
    finally:
        await store.dispose()


@pytest.fixture
def localfs_store(tmp_path: Path) -> LocalFsStore:
    return LocalFsStore(tmp_path)
