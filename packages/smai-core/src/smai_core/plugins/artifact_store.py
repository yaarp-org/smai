""":class:`ArtifactStore` Protocol — blob storage with optional presigned-URL
semantics.

Per ``designs/smai/07-plugin-interfaces.md`` §6 and DEC-028.

The Protocol does not know anything about artifact semantics — keys are
opaque strings, values are opaque bytes. Higher-layer code owns the key
conventions (v1's are catalogued in ``CLAUDE.md`` "S3 Artifact Layout"
and serve as a reference; v2 conventions are owned by ``05-orchestrator.md``
and the agents per their dispatch payloads).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

# === Capability flags (§6.2) =================================================


class ArtifactStoreCapabilities(BaseModel):
    """Static per-plugin capability flags (§6.2).

    ``supports_presigned_urls``:
        ``True`` for S3-shaped backends; ``False`` for ``LocalFsStore``.

    ``max_object_size_bytes``:
        ``None`` means unbounded (``LocalFsStore``); ``5 * 1024**3`` for
        S3 single-PUT, etc.
    """

    model_config = ConfigDict(extra="forbid")

    supports_presigned_urls: bool
    max_object_size_bytes: int | None


# === Error contract (§6.3) ===================================================


class ArtifactStoreError(Exception):
    """Base class for all :class:`ArtifactStore` errors (§6.3)."""


class ArtifactNotFound(ArtifactStoreError):
    """Raised by :meth:`ArtifactStore.get` when no value exists at
    ``key`` (§6.3).

    ``delete`` is idempotent and never raises this — same convention as
    S3 / cloud blob stores.
    """

    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(f"artifact not found: {key!r}")


class PresignedUrlsUnsupported(ArtifactStoreError):
    """Raised by :meth:`ArtifactStore.url_for` when the plugin does not
    support presigned URLs (§6.3).

    The capability flag ``capabilities.supports_presigned_urls``
    indicates this in advance.
    """


class ArtifactTooLarge(ArtifactStoreError):
    """Raised by :meth:`ArtifactStore.put` when ``data`` exceeds the
    plugin's ``capabilities.max_object_size_bytes`` (§6.3)."""

    def __init__(self, key: str, size: int, limit: int) -> None:
        self.key = key
        self.size = size
        self.limit = limit
        super().__init__(
            f"artifact {key!r}: size {size} bytes exceeds plugin limit {limit} bytes",
        )


# === Protocol shape (§6.2) ===================================================


@runtime_checkable
class ArtifactStore(Protocol):
    """Blob storage with optional presigned-URL semantics (§6.2).

    Plugins register via::

        [project.entry-points."smai.artifact_stores"]
        <name> = "<module>:<class>"
    """

    name: str
    capabilities: ArtifactStoreCapabilities

    async def put(
        self,
        key: str,
        data: bytes,
        content_type: str | None = None,
    ) -> None:
        """Store ``data`` under ``key`` (§6.2).

        Overwrites any existing value at the same key. ``content_type``
        is advisory — plugins MAY surface it via store-native metadata
        (S3 ``Content-Type`` header, etc.); the Protocol does not
        require it to be retrievable.
        """
        ...

    async def get(self, key: str) -> bytes:
        """Retrieve the bytes stored under ``key`` (§6.2).

        MUST raise :class:`ArtifactNotFound` if no value exists at
        ``key``.
        """
        ...

    async def exists(self, key: str) -> bool:
        """Return ``True`` iff a value exists at ``key`` (§6.2).

        No data transferred.
        """
        ...

    async def list(self, prefix: str) -> AsyncIterator[str]:
        """Yield keys whose names begin with ``prefix`` (§6.2).

        Order is plugin-defined (typically lexicographic). Streaming
        — no all-at-once materialization for very large prefixes.

        Per the spec's verbatim signature: an ``async def`` returning
        an :class:`AsyncIterator`. Implementations return (not yield)
        an async iterator — typically a thin async-generator-backed
        wrapper, so consumers write ``async for k in await
        store.list(prefix):``.
        """
        ...

    async def delete(self, key: str) -> None:
        """Delete the value at ``key`` (§6.2).

        No-op if the key doesn't exist (do NOT raise
        :class:`ArtifactNotFound` — delete is idempotent).
        """
        ...

    async def url_for(
        self,
        key: str,
        expires_in: int = 3600,
        method: Literal["GET", "PUT"] = "GET",
    ) -> str:
        """Return a presigned URL for direct client access to ``key``
        (§6.2).

        Used by the hosted backend's UI to stream large artifacts
        without proxying through the API tier.

        Plugins that don't support presigned URLs (``LocalFsStore`` in
        v1) MUST raise :class:`PresignedUrlsUnsupported`. The
        capability flag ``capabilities.supports_presigned_urls``
        indicates this in advance.
        """
        ...
