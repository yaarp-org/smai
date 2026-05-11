"""``SqliteStore.__init__`` URI handling — file-backed paths get
``~`` expanded and their parent directory created so verbs that use the
``smai.yaml`` value verbatim (``smai migrate`` / ``smai start`` /
``smai verify``) don't fail with "unable to open database file".

In-memory URIs (``smai dev`` default) are left alone.
"""

from __future__ import annotations

import asyncio

from smai_store_sqlite import SqliteStore


def test_file_backed_uri_creates_parent_directory(tmp_path) -> None:
    db_path = tmp_path / "nested" / "dir" / "state.db"
    assert not db_path.parent.exists()
    store = SqliteStore(uri=f"sqlite+aiosqlite:///{db_path}")
    assert db_path.parent.is_dir()

    async def _exercise() -> None:
        # The store is usable end-to-end against the freshly-created path.
        await store.migrate()
        await store.dispose()

    asyncio.run(_exercise())
    assert db_path.exists()


def test_user_home_tilde_is_expanded(tmp_path, monkeypatch) -> None:
    # Point ``~`` at a temp dir so we don't touch the real home.
    monkeypatch.setenv("HOME", str(tmp_path))
    store = SqliteStore(uri="sqlite+aiosqlite:///~/smai-test/state.db")
    try:
        assert (tmp_path / "smai-test").is_dir()
    finally:
        asyncio.run(store.dispose())


def test_in_memory_uri_is_left_alone(tmp_path, monkeypatch) -> None:
    # No directories should be created for an in-memory store.
    monkeypatch.chdir(tmp_path)
    store = SqliteStore(uri="sqlite+aiosqlite:///:memory:")
    try:
        assert list(tmp_path.iterdir()) == []
    finally:
        asyncio.run(store.dispose())


def test_default_uri_is_in_memory(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    store = SqliteStore()
    try:
        assert list(tmp_path.iterdir()) == []
    finally:
        asyncio.run(store.dispose())
