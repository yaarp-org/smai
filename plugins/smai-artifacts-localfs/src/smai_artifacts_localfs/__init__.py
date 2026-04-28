"""ArtifactStore plugin: local filesystem reference implementation.

Per ``designs/smai/07-plugin-interfaces.md`` §6 and
``designs/smai/implementation_plan.md`` §3.3 Task 2.A3.

Stores artifact bytes under a configurable root directory
(``SMAI_ARTIFACTS_ROOT`` env var, defaulting to ``~/.smai/artifacts``).
This is the reference plugin used by ``smai dev``; the BYO-bucket S3
sibling lands in Task 3.F2.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal

from smai_core.plugins import (
    ArtifactNotFound,
    ArtifactStoreCapabilities,
)

_ENV_VAR = "SMAI_ARTIFACTS_ROOT"


def _default_root() -> Path:
    """Resolve the default artifact root lazily so tests can monkeypatch
    ``$HOME`` without paying for module-import-time path capture.
    """
    return Path.home() / ".smai" / "artifacts"


async def _async_iter_keys(keys: list[str]) -> AsyncIterator[str]:
    """Async-generator helper backing :meth:`LocalFsStore.list`.

    Module-level so :meth:`LocalFsStore.list` can ``return`` (not
    ``yield``) the async iterator — the verbatim spec shape from
    ``07-plugin-interfaces.md`` §6.2 is ``async def list ->
    AsyncIterator[str]``, where the awaitable resolves to an async
    iterator the consumer then iterates.
    """
    for key in keys:
        yield key


class LocalFsStore:
    """Local-filesystem :class:`ArtifactStore` reference implementation.

    Concurrency: ``put`` is atomic (temp file + ``os.replace``) so
    competing writers cannot leave a partially written value at a key.
    ``delete`` is idempotent on missing keys (matches S3 / cloud blob
    store convention).
    """

    name: str = "localfs"
    capabilities: ArtifactStoreCapabilities = ArtifactStoreCapabilities(
        supports_presigned_urls=True,
        max_object_size_bytes=None,
    )

    def __init__(self, root: Path | str | None = None) -> None:
        if root is None:
            env_value = os.environ.get(_ENV_VAR)
            root = Path(env_value) if env_value else _default_root()
        self._root: Path = Path(root).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def _path_for(self, key: str) -> Path:
        candidate = (self._root / key).resolve()
        try:
            candidate.relative_to(self._root)
        except ValueError as exc:
            raise ValueError(f"key {key!r} escapes artifact root") from exc
        return candidate

    async def put(
        self,
        key: str,
        data: bytes,
        content_type: str | None = None,
    ) -> None:
        path = self._path_for(key)
        await asyncio.to_thread(self._atomic_write, path, data)

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            raise

    async def get(self, key: str) -> bytes:
        path = self._path_for(key)
        try:
            return await asyncio.to_thread(path.read_bytes)
        except FileNotFoundError as exc:
            raise ArtifactNotFound(key) from exc

    async def exists(self, key: str) -> bool:
        path = self._path_for(key)
        return await asyncio.to_thread(path.is_file)

    async def list(self, prefix: str) -> AsyncIterator[str]:
        keys = await asyncio.to_thread(self._collect_under_prefix, prefix)
        return _async_iter_keys(keys)

    def _collect_under_prefix(self, prefix: str) -> list[str]:
        # Walk the deepest existing directory whose path is a string-prefix of
        # ``prefix``, then string-filter relative paths against ``prefix``.
        # Trailing-slash and mid-name prefixes both work.
        parts = prefix.split("/")
        walk_root = self._root
        for part in parts[:-1]:
            if not part:
                continue
            candidate = walk_root / part
            if candidate.is_dir():
                walk_root = candidate
            else:
                break
        if not walk_root.is_dir():
            return []
        results: list[str] = []
        for path in walk_root.rglob("*"):
            if path.is_file():
                rel = path.relative_to(self._root).as_posix()
                if rel.startswith(prefix):
                    results.append(rel)
        results.sort()
        return results

    async def delete(self, key: str) -> None:
        path = self._path_for(key)
        await asyncio.to_thread(self._unlink_if_exists, path)

    @staticmethod
    def _unlink_if_exists(path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            return

    async def url_for(
        self,
        key: str,
        expires_in: int = 3600,
        method: Literal["GET", "PUT"] = "GET",
    ) -> str:
        # ``expires_in`` and ``method`` are accepted for Protocol parity but
        # ignored — local-fs paths have no presigned-URL semantics.
        del expires_in, method
        path = self._path_for(key)
        return path.as_uri()


__all__ = ["LocalFsStore"]
