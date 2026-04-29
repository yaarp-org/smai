"""ArtifactStore plugin: BYO-bucket S3 implementation.

Per ``designs/smai/07-plugin-interfaces.md`` §6 and
``designs/smai/implementation_plan.md`` §3.4 Task 3.F2.

Stores artifact bytes in a caller-provided S3 bucket. Mirrors the shape
of the :class:`LocalFsStore` reference plugin (Task 2.A3) and pairs with
it as the production-grade sibling — :class:`LocalFsStore` is the
``smai dev`` default; :class:`S3Store` is what ``smai start`` reaches
for in production.

Driver choice: synchronous ``boto3`` wrapped with
:func:`asyncio.to_thread` at the plugin's async surface — the same
pattern as ``smai-llm-bedrock``. Going via ``aioboto3`` was considered
(the brief in the task plan suggests it) but was rejected because it
forces an ``aiobotocore``/``boto3`` version-pin matrix that complicates
the workspace install footprint without buying a real performance win
at v1 scale (artifact reads/writes are bounded by S3 round-trips, not
in-process scheduling).

Bucket discipline: BYO-bucket (the bucket exists at construction time;
the plugin does not auto-create or auto-discover). Single bucket per
store instance — cross-bucket / cross-account routing is explicitly out
of scope (implementation_plan §6 Q10). The optional ``prefix``
constructor argument is a key-namespacing convenience that lets a
single bucket host multiple deployments.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Literal, cast

from smai_core.plugins import (
    ArtifactNotFound,
    ArtifactStoreCapabilities,
    ArtifactTooLarge,
)

# --- module-level constants -------------------------------------------------

# S3 single-PUT object size limit. Multipart upload would lift this; per
# the task brief that is deferred (no v1 caller hits the ceiling in the
# CG-execution / proposal / paper-ingestion artifact set).
_S3_SINGLE_PUT_LIMIT_BYTES: int = 5 * 1024**3  # 5 GiB

# The Protocol default for ``url_for(... expires_in=3600 ...)`` per
# ``07-plugin-interfaces.md`` §6.2. Re-declared here so the constructor
# argument and Protocol-method signature default agree by construction.
_DEFAULT_EXPIRES_IN: int = 3600


async def _async_iter_keys(keys: list[str]) -> AsyncIterator[str]:
    """Async-generator helper backing :meth:`S3Store.list`.

    Module-level so :meth:`S3Store.list` can ``return`` (not ``yield``)
    the async iterator — the verbatim spec shape from
    ``07-plugin-interfaces.md`` §6.2 is ``async def list ->
    AsyncIterator[str]``, where the awaitable resolves to an async
    iterator the consumer then iterates.
    """
    for key in keys:
        yield key


def _build_s3_client(region: str | None) -> Any:
    """Construct a real ``s3`` boto3 client.

    Lazily imported so the plugin module is importable in environments
    without boto3 installed (e.g., ``pyright`` on a CI runner that
    hasn't synced workspace dependencies yet).

    SigV4 is forced via ``signature_version="s3v4"`` so presigned URLs
    use the modern ``X-Amz-Expires`` query-parameter shape (SigV2 emits
    ``Expires=<unix-timestamp>`` instead, which several AWS regions
    have stopped accepting). A region-aware default is required for
    SigV4 to compute the canonical request, so we pass through the
    constructor's ``region`` argument when provided; without it the
    boto3 default chain decides.
    """
    try:
        import boto3  # noqa: PLC0415  # pyright: ignore[reportMissingTypeStubs]
        from botocore.config import (
            Config,  # noqa: PLC0415  # pyright: ignore[reportMissingTypeStubs]
        )
    except ImportError as exc:  # pragma: no cover - declared dep
        raise RuntimeError(
            "smai-artifacts-s3 requires boto3; install with `pip install smai-artifacts-s3`"
        ) from exc
    factory = cast("Any", boto3).client
    config = cast("Any", Config)(signature_version="s3v4")
    if region is None:
        return factory("s3", config=config)
    return factory("s3", region_name=region, config=config)


def _is_not_found(exc: BaseException) -> bool:
    """True iff ``exc`` is a botocore :class:`ClientError` for a missing key.

    S3 returns ``404`` with code ``"NoSuchKey"`` on ``get_object`` and
    a bare ``"404"`` on ``head_object`` (the head response carries no
    ``NoSuchKey`` body). Both are normalized to "missing" here.
    """
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    error = response.get("Error")
    if not isinstance(error, dict):
        return False
    code = error.get("Code")
    return code in ("404", "NoSuchKey", "NotFound")


class S3Store:
    """BYO-bucket S3 :class:`ArtifactStore` implementation.

    Constructor::

        S3Store("my-bucket")
        S3Store("my-bucket", region="us-east-1", prefix="prod/")
        S3Store(
            "my-bucket",
            region="us-east-1",
            presigned_url_expiry_seconds=900,  # 15 minutes
        )

    The ``client`` keyword argument is a test seam — production callers
    leave it None and the plugin builds a stock boto3 ``s3`` client from
    the default credential chain. Tests can pass a moto-mocked client
    directly.

    Per ``07-plugin-interfaces.md`` §6.3 / §6.4:

    * ``get`` of a missing key raises :class:`ArtifactNotFound`.
    * ``delete`` is idempotent on missing keys (S3's ``delete_object``
      already is — no extra plumbing required).
    * ``url_for`` returns a real presigned URL with ``expires_in``
      honored via ``X-Amz-Expires``.
    """

    name: str = "s3"
    capabilities: ArtifactStoreCapabilities

    def __init__(
        self,
        bucket: str,
        *,
        region: str | None = None,
        prefix: str = "",
        presigned_url_expiry_seconds: int = _DEFAULT_EXPIRES_IN,
        max_object_size_bytes: int | None = _S3_SINGLE_PUT_LIMIT_BYTES,
        client: Any = None,
    ) -> None:
        self._bucket = bucket
        self._region = region
        self._prefix = prefix
        self._presigned_url_expiry_seconds = presigned_url_expiry_seconds
        self._max_object_size_bytes = max_object_size_bytes
        self.capabilities = ArtifactStoreCapabilities(
            supports_presigned_urls=True,
            max_object_size_bytes=max_object_size_bytes,
        )
        self._client: Any = client if client is not None else _build_s3_client(region)

    @property
    def bucket(self) -> str:
        return self._bucket

    @property
    def prefix(self) -> str:
        return self._prefix

    def _full_key(self, key: str) -> str:
        """Map a caller-visible key to its S3 object key (with the
        store-level prefix applied)."""
        return f"{self._prefix}{key}" if self._prefix else key

    def _strip_prefix(self, full_key: str) -> str:
        """Inverse of :meth:`_full_key` for :meth:`list` output."""
        if self._prefix and full_key.startswith(self._prefix):
            return full_key[len(self._prefix) :]
        return full_key

    async def put(
        self,
        key: str,
        data: bytes,
        content_type: str | None = None,
    ) -> None:
        # Proactive size check — fail fast without paying the upload
        # round-trip when the caller hands us a payload we already know
        # would be rejected (S3's hard 5 GiB single-PUT ceiling, or a
        # smaller bucket-policy limit configured at construction).
        if self._max_object_size_bytes is not None and len(data) > self._max_object_size_bytes:
            raise ArtifactTooLarge(key, len(data), self._max_object_size_bytes)
        full_key = self._full_key(key)
        kwargs: dict[str, Any] = {"Bucket": self._bucket, "Key": full_key, "Body": data}
        if content_type is not None:
            kwargs["ContentType"] = content_type
        await asyncio.to_thread(lambda: self._client.put_object(**kwargs))

    async def get(self, key: str) -> bytes:
        full_key = self._full_key(key)
        try:
            response = await asyncio.to_thread(
                lambda: self._client.get_object(Bucket=self._bucket, Key=full_key)
            )
        except Exception as exc:
            if _is_not_found(exc):
                raise ArtifactNotFound(key) from exc
            raise
        body = response["Body"]
        return await asyncio.to_thread(body.read)

    async def exists(self, key: str) -> bool:
        full_key = self._full_key(key)
        try:
            await asyncio.to_thread(
                lambda: self._client.head_object(Bucket=self._bucket, Key=full_key)
            )
        except Exception as exc:
            if _is_not_found(exc):
                return False
            raise
        return True

    async def list(self, prefix: str) -> AsyncIterator[str]:
        full_prefix = self._full_key(prefix)
        keys = await asyncio.to_thread(self._list_all_keys, full_prefix)
        return _async_iter_keys([self._strip_prefix(k) for k in keys])

    def _list_all_keys(self, full_prefix: str) -> list[str]:
        """Walk every page of ``list_objects_v2`` for the given prefix.

        boto3's paginator handles the continuation-token loop for us;
        we materialize the keys into a list and hand them to the async
        iterator wrapper. At v1 artifact-volume scale this is fine —
        the prefixes the orchestrator scans are bounded (one CG /
        proposal worth of artifacts at a time, not whole-bucket
        scans).
        """
        paginator = self._client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self._bucket, Prefix=full_prefix):
            for obj in page.get("Contents", ()):
                keys.append(obj["Key"])
        return keys

    async def delete(self, key: str) -> None:
        full_key = self._full_key(key)
        # S3's ``delete_object`` is idempotent on missing keys — no
        # ArtifactNotFound bubble-up needed (matches the §6.3 contract).
        await asyncio.to_thread(
            lambda: self._client.delete_object(Bucket=self._bucket, Key=full_key)
        )

    async def url_for(
        self,
        key: str,
        expires_in: int = _DEFAULT_EXPIRES_IN,
        method: Literal["GET", "PUT"] = "GET",
    ) -> str:
        """Return a presigned URL for direct client access to ``key``.

        Reconciliation of the two expiry knobs:

        * ``presigned_url_expiry_seconds`` (constructor) — the per-store
          default. Useful when one component owns the lifetime policy
          for every URL the store hands out (e.g., the dashboard wants
          15-minute URLs across the board).
        * ``expires_in`` (Protocol method arg) — per-call override.

        When the caller leaves ``expires_in`` at the Protocol default
        (3600), the store's configured default applies. An explicit
        ``expires_in`` always wins. The corner case (caller passes 3600
        when the store's configured default differs) collapses to the
        store's default — explicitly documented; callers who genuinely
        want a 1-hour URL on a non-default store should pass any other
        value or set the store's default to 3600.
        """
        full_key = self._full_key(key)
        effective_expires = (
            self._presigned_url_expiry_seconds if expires_in == _DEFAULT_EXPIRES_IN else expires_in
        )
        client_method = "get_object" if method == "GET" else "put_object"
        url: str = await asyncio.to_thread(
            lambda: self._client.generate_presigned_url(
                ClientMethod=client_method,
                Params={"Bucket": self._bucket, "Key": full_key},
                ExpiresIn=effective_expires,
            )
        )
        return url


__all__ = ["S3Store"]
