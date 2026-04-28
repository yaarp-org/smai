"""Plugin-specific tests for :class:`LocalFsStore` beyond the universal
:class:`ArtifactStoreConformance` suite.

Covers behavior the conformance base does not assert on — atomic
``put`` under concurrent writers, env-var root resolution, and
``file://`` URL fetchability.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import pytest
from smai_artifacts_localfs import LocalFsStore


def test_root_defaults_to_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMAI_ARTIFACTS_ROOT", str(tmp_path / "from-env"))
    store = LocalFsStore()
    assert store.root == (tmp_path / "from-env").resolve()


def test_key_escape_rejected(tmp_path: Path) -> None:
    store = LocalFsStore(root=tmp_path / "artifacts")
    with pytest.raises(ValueError):

        async def _go() -> None:
            await store.put("../escape", b"x")

        asyncio.run(_go())


async def test_concurrent_put_does_not_corrupt(tmp_path: Path) -> None:
    store = LocalFsStore(root=tmp_path / "artifacts")
    key = "concurrent/key"
    payload_a = b"A" * 4096
    payload_b = b"B" * 4096
    # Issue many concurrent overwrites with two distinct payloads. Atomic
    # rename guarantees the final bytes equal one of the two payloads in
    # full — never a mixed / truncated value.
    tasks = [store.put(key, payload_a if i % 2 == 0 else payload_b) for i in range(50)]
    await asyncio.gather(*tasks)
    final = await store.get(key)
    assert final in (payload_a, payload_b)


async def test_url_for_file_uri_is_fetchable(tmp_path: Path) -> None:
    store = LocalFsStore(root=tmp_path / "artifacts")
    key = "fetch/me"
    payload = b"hello, file scheme"
    await store.put(key, payload)
    url = await store.url_for(key)
    parsed = urlparse(url)
    assert parsed.scheme == "file"
    assert urlopen(url).read() == payload  # noqa: S310 - file scheme, test-only


async def test_list_walks_nested_files(tmp_path: Path) -> None:
    store = LocalFsStore(root=tmp_path / "artifacts")
    keys = ["a/b/c.bin", "a/b/d.bin", "a/e.bin", "z/other.bin"]
    for k in keys:
        await store.put(k, b"x")
    seen: list[str] = []
    iterator = await store.list("a/")
    async for k in iterator:
        seen.append(k)
    assert set(seen) == {"a/b/c.bin", "a/b/d.bin", "a/e.bin"}
