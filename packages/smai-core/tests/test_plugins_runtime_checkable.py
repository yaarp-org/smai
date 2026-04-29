"""``runtime_checkable`` smoke tests for the four plugin Protocols.

Per Task 1.8 — for each Protocol, write a minimal conforming-class fixture
that satisfies the attribute presence check, and a non-conforming fixture
that doesn't, and assert ``isinstance`` returns the right value.

Note that ``runtime_checkable`` only checks attribute *presence*, not
signatures — that's expected; the conformance test bases (Task 1.9) are
the actual signature contract.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from typing import Any

from smai_core.plugins import (
    ArtifactStore,
    ArtifactStoreCapabilities,
    Compute,
    ComputeCapabilities,
    CursorPage,
    JobHandle,
    JobStatus,
    LeaseToken,
    LlmCapabilities,
    LlmProvider,
    MetadataStore,
    MetadataStoreCapabilities,
    ModelResponse,
    NormalizedMessage,
    Transaction,
)

# ---------- LlmProvider ------------------------------------------------------


class _ConformingLlm:
    name = "ok"
    capabilities = LlmCapabilities(
        supports_caching=False,
        context_window=200_000,
        max_output_tokens=4096,
        model_id="test-model",
    )

    async def call(  # noqa: D401, PLR0913 - signature shape match only
        self,
        system: str,
        messages: list[NormalizedMessage],
        tools: object = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        cache_config: object = None,
    ) -> ModelResponse:
        raise NotImplementedError


class _NonConformingLlm:
    """Missing the ``call`` method → fails ``isinstance``."""

    name = "no-call"
    capabilities = LlmCapabilities(
        supports_caching=False,
        context_window=1,
        max_output_tokens=1,
        model_id="x",
    )


def test_llm_provider_conforming_isinstance_true() -> None:
    assert isinstance(_ConformingLlm(), LlmProvider) is True


def test_llm_provider_non_conforming_isinstance_false() -> None:
    assert isinstance(_NonConformingLlm(), LlmProvider) is False


# ---------- MetadataStore ----------------------------------------------------


def _make_metadata_store_capabilities() -> MetadataStoreCapabilities:
    return MetadataStoreCapabilities(is_tenant_aware=False)


_METADATA_STORE_METHODS = (
    "get_cg",
    "list_entries_for_cg",
    "get_entry",
    "list_runs_for_entry",
    "get_run",
    "get_proposal",
    "list_cgs_for_proposal",
    "get_paper",
    "list_techniques_for_paper",
    "create_cg",
    "create_entry",
    "create_run",
    "create_proposal",
    "create_paper",
    "get_technique",
    "list_techniques",
    "upsert_technique",
    "transition_cg_state",
    "transition_entry_state",
    "transition_run_state",
    "transition_proposal_state",
    "transition_paper_state",
    "get_ready_for_harness_build",
    "get_in_flight_harness_build",
    "get_ready_to_implement_entry",
    "get_in_flight_entry_implementation",
    "get_ready_for_review_and_run",
    "get_ready_to_run",
    "get_in_flight_runs",
    "get_ready_to_evaluate",
    "get_in_flight_evaluation",
    "get_ready_for_proposal_design",
    "get_in_flight_proposal_design",
    "get_proposals_at_human_gate",
    "get_ready_for_paper_fetch",
    "get_in_flight_paper_fetch",
    "get_ready_for_paper_screen",
    "get_in_flight_paper_screen",
    "get_ready_for_paper_plan",
    "get_in_flight_paper_plan",
    "get_partial_pending_promotion",
    "count_in_state",
    "count_with_in_flight_jobs",
    "acquire_lease",
    "release_lease",
    "extend_lease",
    "transaction",
)


class _ConformingStore:
    """Stub that provides every attribute named on
    :class:`MetadataStore` as a no-op coroutine. ``runtime_checkable``
    only checks attribute presence, not signatures."""

    name = "stub"
    capabilities = _make_metadata_store_capabilities()


async def _noop(*args: Any, **kwargs: Any) -> Any:
    return None


for _method in _METADATA_STORE_METHODS:
    setattr(_ConformingStore, _method, _noop)


class _NonConformingStore:
    name = "missing-methods"
    capabilities = _make_metadata_store_capabilities()
    # Deliberately omits every method.


def test_metadata_store_conforming_isinstance_true() -> None:
    assert isinstance(_ConformingStore(), MetadataStore) is True


def test_metadata_store_non_conforming_isinstance_false() -> None:
    assert isinstance(_NonConformingStore(), MetadataStore) is False


def test_transaction_conforming_isinstance_true() -> None:
    class _Tx:
        async def create_cg(self, *a: Any, **kw: Any) -> Any: ...
        async def create_entry(self, *a: Any, **kw: Any) -> Any: ...
        async def create_run(self, *a: Any, **kw: Any) -> Any: ...
        async def create_proposal(self, *a: Any, **kw: Any) -> Any: ...
        async def create_paper(self, *a: Any, **kw: Any) -> Any: ...
        async def transition_cg_state(self, *a: Any, **kw: Any) -> Any: ...
        async def transition_entry_state(self, *a: Any, **kw: Any) -> Any: ...
        async def transition_run_state(self, *a: Any, **kw: Any) -> Any: ...
        async def transition_proposal_state(self, *a: Any, **kw: Any) -> Any: ...
        async def transition_paper_state(self, *a: Any, **kw: Any) -> Any: ...

    assert isinstance(_Tx(), Transaction) is True


def test_transaction_non_conforming_isinstance_false() -> None:
    class _MissingTx:
        async def create_cg(self, *a: Any, **kw: Any) -> Any: ...

        # missing the rest

    assert isinstance(_MissingTx(), Transaction) is False


# ---------- ArtifactStore ----------------------------------------------------


class _ConformingArtifactStore:
    name = "ok"
    capabilities = ArtifactStoreCapabilities(
        supports_presigned_urls=False,
        max_object_size_bytes=None,
    )

    async def put(self, key: str, data: bytes, content_type: str | None = None) -> None: ...
    async def get(self, key: str) -> bytes:
        return b""

    async def exists(self, key: str) -> bool:
        return False

    async def list(self, prefix: str) -> AsyncIterator[str]:
        async def _empty() -> AsyncIterator[str]:
            if False:  # pragma: no cover
                yield ""

        return _empty()

    async def delete(self, key: str) -> None: ...
    async def url_for(
        self,
        key: str,
        expires_in: int = 3600,
        method: str = "GET",
    ) -> str:
        return ""


class _NonConformingArtifactStore:
    name = "missing"
    capabilities = ArtifactStoreCapabilities(
        supports_presigned_urls=False,
        max_object_size_bytes=None,
    )

    async def put(self, key: str, data: bytes, content_type: str | None = None) -> None: ...

    # missing get, exists, list, delete, url_for


def test_artifact_store_conforming_isinstance_true() -> None:
    assert isinstance(_ConformingArtifactStore(), ArtifactStore) is True


def test_artifact_store_non_conforming_isinstance_false() -> None:
    assert isinstance(_NonConformingArtifactStore(), ArtifactStore) is False


# ---------- Compute ----------------------------------------------------------


class _ConformingCompute:
    name = "ok"
    capabilities = ComputeCapabilities(
        supports_gpu=False,
        max_timeout_seconds=3600,
    )

    async def submit(
        self,
        image: str,
        command: list[str],
        env: dict[str, str],
        gpu: bool = False,
        timeout_seconds: int = 3600,
        **plugin_options: Any,
    ) -> JobHandle:
        return JobHandle(plugin="ok", handle="x")

    async def status(self, handle: JobHandle) -> JobStatus:
        return JobStatus(
            state="succeeded",
            exit_code=0,
            started_at=None,
            finished_at=None,
            failure_reason=None,
        )

    async def logs(self, handle: JobHandle) -> str:
        return ""

    async def cancel(self, handle: JobHandle) -> None: ...


class _NonConformingCompute:
    name = "missing"
    capabilities = ComputeCapabilities(
        supports_gpu=False,
        max_timeout_seconds=1,
    )

    async def submit(self, *a: Any, **kw: Any) -> Any: ...

    # missing status, logs, cancel


def test_compute_conforming_isinstance_true() -> None:
    assert isinstance(_ConformingCompute(), Compute) is True


def test_compute_non_conforming_isinstance_false() -> None:
    assert isinstance(_NonConformingCompute(), Compute) is False


# ---------- Helpers used in other tests --------------------------------------


def test_async_context_manager_used_by_metadata_store_transaction() -> None:
    """Sanity check: the ``transaction()`` annotation type is importable
    from contextlib (no third-party-dep cost)."""
    assert AbstractAsyncContextManager is not None


def test_cursor_page_and_lease_token_are_modules_imported() -> None:
    assert CursorPage is not None
    assert LeaseToken is not None
