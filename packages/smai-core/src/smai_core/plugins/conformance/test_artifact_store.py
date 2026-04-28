""":class:`ArtifactStoreConformance` — parameterizable test base for
:class:`ArtifactStore` plugins.

Per ``designs/smai/07-plugin-interfaces.md`` §6.4 (the per-Protocol
conformance enumeration) and §8 (cross-cutting discipline).

Subclass and override :meth:`ArtifactStoreConformance.make_store` to
run the suite against your plugin::

    from smai_core.plugins.conformance import ArtifactStoreConformance
    from smai_artifacts_localfs import LocalFsStore

    class TestLocalFsConformance(ArtifactStoreConformance):
        def make_store(self) -> ArtifactStore:
            return LocalFsStore(root="/tmp/conformance")

The ``async def list -> AsyncIterator[str]`` shape (verbatim from the
spec) is exercised here by an ``async for k in await store.list(...)``
call site — implementations return (not yield) the async iterator.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from smai_core.plugins import (
    ArtifactNotFound,
    ArtifactStore,
    ArtifactStoreCapabilities,
    ArtifactTooLarge,
    PresignedUrlsUnsupported,
)
from smai_core.plugins.conformance._common import (
    skip_no_factory_override,
)


class ArtifactStoreConformance:
    """Universal contract suite for :class:`ArtifactStore` implementations.

    The contract methods correspond 1:1 to the rules in
    ``07-plugin-interfaces.md`` §6.4:

    * :meth:`test_put_then_get_round_trip` — round-trip
    * :meth:`test_overwrite` — ``put`` over an existing key replaces
    * :meth:`test_exists_post_put` — ``exists`` reflects ``put``
    * :meth:`test_list_returns_keys_under_prefix` — list filtering by
      prefix; exercises the ``async def list -> AsyncIterator[str]``
      shape
    * :meth:`test_delete_idempotency` — delete of a missing key is a no-op
    * :meth:`test_delete_removes` — delete actually removes
    * :meth:`test_get_missing_raises_not_found` — not-found shape with
      key on the exception payload
    * :meth:`test_url_for_supported` / :meth:`test_url_for_unsupported`
      — presigned-URL capability honesty
    * :meth:`test_max_object_size_honored` — large-object handling per
      capability
    * :meth:`test_varied_byte_sizes_round_trip` — fixture sizes
      (1 byte, 1 KiB, 1 MiB) per §6.4
    """

    # ---- factory hook ------------------------------------------------------

    def make_store(self) -> ArtifactStore:
        """Return a configured :class:`ArtifactStore` instance.

        Override this in your subclass. The default skip-collects.
        """
        skip_no_factory_override(type(self).__name__, "make_store")
        raise AssertionError("unreachable")  # pragma: no cover

    @pytest.fixture
    def fixture_key_namespace(self) -> str:
        """A fixture-only key prefix to avoid colliding with deployed data
        when the suite is run against a live backend (per §6.4)."""
        return "smai-conformance/"

    # ---- fixtures ----------------------------------------------------------

    @pytest.fixture
    def store(self) -> ArtifactStore:
        return self.make_store()

    # ---- §6.4 contract assertions ------------------------------------------

    async def test_put_then_get_round_trip(
        self, store: ArtifactStore, fixture_key_namespace: str
    ) -> None:
        """``put`` then ``get`` returns the same bytes."""
        key = f"{fixture_key_namespace}round-trip"
        data = b"hello, conformance"
        await store.put(key, data)
        try:
            assert await store.get(key) == data
        finally:
            await store.delete(key)

    async def test_overwrite(self, store: ArtifactStore, fixture_key_namespace: str) -> None:
        """``put`` over an existing key replaces the value."""
        key = f"{fixture_key_namespace}overwrite"
        await store.put(key, b"first")
        await store.put(key, b"second")
        try:
            assert await store.get(key) == b"second"
        finally:
            await store.delete(key)

    async def test_exists_post_put(self, store: ArtifactStore, fixture_key_namespace: str) -> None:
        """``exists`` is ``True`` after ``put`` and ``False`` after ``delete``."""
        key = f"{fixture_key_namespace}exists"
        assert await store.exists(key) is False
        await store.put(key, b"x")
        try:
            assert await store.exists(key) is True
        finally:
            await store.delete(key)
        assert await store.exists(key) is False

    async def test_list_returns_keys_under_prefix(
        self, store: ArtifactStore, fixture_key_namespace: str
    ) -> None:
        """Keys written under a prefix appear in the list iterator; keys
        outside the prefix do not.

        Exercises the verbatim spec shape: ``async def list ->
        AsyncIterator[str]`` — implementations return (not yield) an
        async iterator, so consumers write
        ``async for k in await store.list(prefix):``.
        """
        prefix = f"{fixture_key_namespace}list-prefix/"
        outside = f"{fixture_key_namespace}other/key"
        inside = [f"{prefix}a", f"{prefix}b", f"{prefix}nested/c"]
        try:
            for k in inside:
                await store.put(k, b"x")
            await store.put(outside, b"x")
            iterator: AsyncIterator[str] = await store.list(prefix)
            seen: set[str] = set()
            async for k in iterator:
                seen.add(k)
            assert set(inside).issubset(seen)
            assert outside not in seen
        finally:
            for k in (*inside, outside):
                await store.delete(k)

    async def test_delete_idempotency(
        self, store: ArtifactStore, fixture_key_namespace: str
    ) -> None:
        """Delete of a missing key is a no-op (no raise)."""
        key = f"{fixture_key_namespace}missing"
        # No raise, no return value to check.
        await store.delete(key)
        await store.delete(key)

    async def test_delete_removes(self, store: ArtifactStore, fixture_key_namespace: str) -> None:
        """``delete`` removes the value; subsequent ``get`` raises."""
        key = f"{fixture_key_namespace}delete-removes"
        await store.put(key, b"data")
        await store.delete(key)
        with pytest.raises(ArtifactNotFound):
            await store.get(key)

    async def test_get_missing_raises_not_found(
        self, store: ArtifactStore, fixture_key_namespace: str
    ) -> None:
        """``get`` of a missing key raises :class:`ArtifactNotFound` with
        the key in the exception payload."""
        key = f"{fixture_key_namespace}does-not-exist"
        with pytest.raises(ArtifactNotFound) as excinfo:
            await store.get(key)
        assert excinfo.value.key == key

    async def test_url_for_supported(
        self, store: ArtifactStore, fixture_key_namespace: str
    ) -> None:
        """Presigned-URL round-trip (gated on ``capabilities.supports_presigned_urls``).

        Per §6.4: fetched URL serves the artifact bytes; expired URL
        returns 403. The bases assert only that the URL is returned and
        is a valid string — substrate-specific HTTP fetch validation is
        the plugin's responsibility (the per-PR conformance gate doesn't
        make network calls).
        """
        if not store.capabilities.supports_presigned_urls:
            pytest.skip("plugin reports supports_presigned_urls=False")
        key = f"{fixture_key_namespace}url-for"
        await store.put(key, b"x")
        try:
            url = await store.url_for(key)
            assert isinstance(url, str) and url
        finally:
            await store.delete(key)

    async def test_url_for_unsupported(
        self, store: ArtifactStore, fixture_key_namespace: str
    ) -> None:
        """Plugins without presigned-URL support raise
        :class:`PresignedUrlsUnsupported`."""
        if store.capabilities.supports_presigned_urls:
            pytest.skip("plugin reports supports_presigned_urls=True")
        key = f"{fixture_key_namespace}no-url"
        await store.put(key, b"x")
        try:
            with pytest.raises(PresignedUrlsUnsupported):
                await store.url_for(key)
        finally:
            await store.delete(key)

    async def test_max_object_size_honored(
        self, store: ArtifactStore, fixture_key_namespace: str
    ) -> None:
        """Large-object handling — ``put`` of an object exceeding
        ``capabilities.max_object_size_bytes`` raises
        :class:`ArtifactTooLarge`. Skipped when the plugin sets the
        limit to ``None`` (unbounded)."""
        limit = store.capabilities.max_object_size_bytes
        if limit is None:
            pytest.skip("plugin reports max_object_size_bytes=None (unbounded)")
        key = f"{fixture_key_namespace}too-large"
        too_big = b"x" * (limit + 1)
        with pytest.raises(ArtifactTooLarge):
            await store.put(key, too_big)

    @pytest.mark.parametrize(
        "size_bytes",
        [1, 1024, 1024 * 1024],  # per §6.4: 1 byte, 1 KiB, 1 MiB
        ids=["1B", "1KiB", "1MiB"],
    )
    async def test_varied_byte_sizes_round_trip(
        self,
        store: ArtifactStore,
        fixture_key_namespace: str,
        size_bytes: int,
    ) -> None:
        """Round-trip over fixture sizes from §6.4."""
        # Skip sizes that exceed the plugin's documented limit.
        limit = store.capabilities.max_object_size_bytes
        if limit is not None and size_bytes > limit:
            pytest.skip(f"size {size_bytes} exceeds plugin max {limit}")
        key = f"{fixture_key_namespace}sized-{size_bytes}"
        data = bytes(size_bytes)
        await store.put(key, data)
        try:
            assert await store.get(key) == data
        finally:
            await store.delete(key)

    def test_capabilities_are_capabilities_model(self, store: ArtifactStore) -> None:
        """Capability attribute is the right Pydantic model — guard against
        plugins that forget to instantiate it."""
        assert isinstance(store.capabilities, ArtifactStoreCapabilities)
