"""Pytest fixtures for the specs sub-package tests.

Re-mounts the engine test directory on ``sys.path`` so the shared
``_helpers`` module (FakeCompute / FakeArtifactStore / FakeMonotonic /
FakeClock / make_dispatch / make_gate / make_job_handle /
make_job_status) is importable here.

Per-test isolation: every test that touches the registry uses the
``reset_spec_registry`` autouse fixture below.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
import pytest_asyncio
from smai_artifacts_localfs import LocalFsStore
from smai_orchestrator.runtime import reset_registry
from smai_store_sqlite import SqliteStore

_SPECS_TESTS_DIR = Path(__file__).resolve().parent
_ENGINE_TESTS_DIR = Path(__file__).resolve().parent.parent / "engine"
# smai-agents tests directory carries shared model-response builders
# (`_agent_helpers`) consumed by `_e1_fakes` for proposal-spec tests.
_AGENTS_TESTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "smai-agents" / "tests"
sys.path.insert(0, str(_SPECS_TESTS_DIR))
sys.path.insert(0, str(_ENGINE_TESTS_DIR))
sys.path.insert(0, str(_AGENTS_TESTS_DIR))


@pytest.fixture(autouse=True)
def reset_spec_registry() -> Iterator[None]:
    """Clear the spec registry before and after each test."""
    reset_registry()
    try:
        yield
    finally:
        reset_registry()


@pytest_asyncio.fixture
async def sqlite_store() -> AsyncIterator[SqliteStore]:
    """In-memory :class:`SqliteStore` per test."""
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
