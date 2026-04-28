"""Shared fixtures and stub helpers for the four conformance test bases.

Per ``designs/smai/07-plugin-interfaces.md`` §8 (cross-cutting discipline)
and Task 1.9 of the implementation plan.

This module pulls in :mod:`pytest` at import time — the four conformance
test classes (``test_<interface>.py`` siblings) reference
``@pytest.fixture`` and ``@pytest.mark.parametrize`` at class-definition
time, so a lazy import doesn't help. Per the package ``__init__``'s
documentation, pytest is an *optional* runtime dep on ``smai-core``;
importing this package without ``smai-core[conformance]`` installed
raises :class:`ImportError` early with a clear message.

Why a single ``_common.py``: the four bases share the
:func:`time_provider` fixture (the deferred wall-clock seam per Task 2.C1),
the ``_unimplemented_*`` stub helpers used by Task 1.9's own structural
tests (and by any future plugin task that wants to verify its bases
collect cleanly without needing a real plugin), and the documentation of
the override pattern. Sharing keeps the per-Protocol files focused on the
contract methods alone.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel

from smai_core.plugins import (
    ArtifactStoreCapabilities,
    ComputeCapabilities,
    JobHandle,
    JobStatus,
    LlmCapabilities,
    MetadataStoreCapabilities,
    ModelResponse,
    NormalizedMessage,
)

# === The wall-clock seam (deferred to Task 2.C1; §6 Q4) ======================

# Type alias for the time provider — a no-arg callable returning a float
# in the ``time.monotonic()`` style. Subclasses override the fixture below
# with a fake clock once Task 2.C1's ``EngineConfig``-driven seam ships.
TimeProvider = Callable[[], float]


# === Stub fixtures for skip-collection / inheritance smoke tests =============
#
# These stubs are Protocol-conforming objects whose methods raise
# :class:`NotImplementedError`. Used internally by Task 1.9's own
# structural tests to verify that a subclass that overrides the
# ``make_<interface>()`` factory inherits and runs the contract methods
# (those tests assert the stubs raise ``NotImplementedError`` when the
# base's contract method calls into them — proving the inheritance is
# wired correctly).
#
# Phase 2's plugin tasks (2.A1 / 2.A2 / 2.A3 / 2.A4) replace
# ``make_<interface>()`` with a real fixture that returns the plugin's
# instance; the bases' contract methods then exercise real behavior.


class _UnimplementedLlmProvider:
    """Protocol-conforming stub whose ``call`` raises :class:`NotImplementedError`."""

    name = "stub-llm"
    capabilities = LlmCapabilities(
        supports_caching=False,
        context_window=200_000,
        max_output_tokens=4096,
        model_id="stub-model",
    )

    async def call(  # noqa: PLR0913 - signature shape must match Protocol
        self,
        system: str,
        messages: list[NormalizedMessage],
        tools: object = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        cache_config: object = None,
    ) -> ModelResponse:
        raise NotImplementedError("stub provider")


class _UnimplementedMetadataStore:
    """Protocol-conforming stub for :class:`MetadataStore`.

    Every method raises :class:`NotImplementedError`. ``runtime_checkable``
    only checks attribute presence, so adding the methods as ``_unimpl``
    coroutines is sufficient.
    """

    name = "stub-store"
    capabilities = MetadataStoreCapabilities(is_tenant_aware=False)


_METADATA_STORE_METHODS: tuple[str, ...] = (
    # CRUD (14)
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
    # methodology lookups (3)
    "get_technique",
    "list_techniques",
    "upsert_technique",
    # transitions (5)
    "transition_cg_state",
    "transition_entry_state",
    "transition_run_state",
    "transition_proposal_state",
    "transition_paper_state",
    # CG-execution scheduling queries (10)
    "get_ready_for_harness_build",
    "get_in_flight_harness_build",
    "get_ready_to_implement_entry",
    "get_in_flight_entry_implementation",
    "get_ready_for_review_and_run",
    "get_ready_to_run",
    "get_in_flight_runs",
    "get_ready_to_evaluate",
    "get_in_flight_evaluation",
    # proposal scheduling queries (3)
    "get_ready_for_proposal_design",
    "get_in_flight_proposal_design",
    "get_pending_proposal_user_decision",
    # paper-ingestion scheduling queries (6)
    "get_ready_for_paper_fetch",
    "get_in_flight_paper_fetch",
    "get_ready_for_paper_screen",
    "get_in_flight_paper_screen",
    "get_ready_for_paper_plan",
    "get_in_flight_paper_plan",
    "get_partial_pending_promotion",
    # counts (2)
    "count_in_state",
    "count_with_in_flight_jobs",
    # leases (3)
    "acquire_lease",
    "release_lease",
    "extend_lease",
    # transactional grouping (1)
    "transaction",
)


# Subset of scheduling-query method names that are parameterized over in the
# MetadataStore conformance suite (per §5.6.10 — covers every ``get_ready_for_*``
# / ``get_in_flight_*`` / ``get_pending_*`` / ``get_partial_*`` query).
SCHEDULING_QUERY_METHODS: tuple[str, ...] = tuple(
    m
    for m in _METADATA_STORE_METHODS
    if m.startswith(("get_ready_", "get_in_flight_", "get_pending_", "get_partial_"))
)


async def _unimpl(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("stub metadata store")


for _method in _METADATA_STORE_METHODS:
    setattr(_UnimplementedMetadataStore, _method, _unimpl)


class _UnimplementedTransaction:
    """Protocol-conforming :class:`Transaction` stub."""


_TRANSACTION_METHODS: tuple[str, ...] = (
    "create_cg",
    "create_entry",
    "create_run",
    "create_proposal",
    "create_paper",
    "transition_cg_state",
    "transition_entry_state",
    "transition_run_state",
    "transition_proposal_state",
    "transition_paper_state",
)

for _method in _TRANSACTION_METHODS:
    setattr(_UnimplementedTransaction, _method, _unimpl)


class _UnimplementedArtifactStore:
    """Protocol-conforming :class:`ArtifactStore` stub.

    The ``async def list -> AsyncIterator[str]`` signature is preserved
    verbatim from the spec — implementations return (not yield) an async
    iterator. Per ``07`` §6.2's verbatim signature retention.
    """

    name = "stub-artifacts"
    capabilities = ArtifactStoreCapabilities(
        supports_presigned_urls=False,
        max_object_size_bytes=None,
    )

    async def put(self, key: str, data: bytes, content_type: str | None = None) -> None:
        raise NotImplementedError("stub artifact store")

    async def get(self, key: str) -> bytes:
        raise NotImplementedError("stub artifact store")

    async def exists(self, key: str) -> bool:
        raise NotImplementedError("stub artifact store")

    async def list(self, prefix: str) -> AsyncIterator[str]:
        raise NotImplementedError("stub artifact store")

    async def delete(self, key: str) -> None:
        raise NotImplementedError("stub artifact store")

    async def url_for(
        self,
        key: str,
        expires_in: int = 3600,
        method: str = "GET",
    ) -> str:
        raise NotImplementedError("stub artifact store")


class _UnimplementedCompute:
    """Protocol-conforming :class:`Compute` stub."""

    name = "stub-compute"
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
        **plugin_options: object,
    ) -> JobHandle:
        raise NotImplementedError("stub compute")

    async def status(self, handle: JobHandle) -> JobStatus:
        raise NotImplementedError("stub compute")

    async def logs(self, handle: JobHandle) -> str:
        raise NotImplementedError("stub compute")

    async def cancel(self, handle: JobHandle) -> None:
        raise NotImplementedError("stub compute")


# === Skip-collection helper =================================================
#
# Used by the four base classes' ``make_<interface>()`` factory methods.
# A subclass that doesn't override ``make_<interface>()`` reaches this
# branch, and pytest skips the class with a clear message.


_SKIP_MESSAGE_TEMPLATE = (
    "{class_name}: override `{factory_name}()` in your subclass to "
    "exercise the conformance suite against your plugin implementation."
)


def skip_no_factory_override(class_name: str, factory_name: str) -> None:
    """Skip the test with a clear message when the factory wasn't overridden."""
    pytest.skip(_SKIP_MESSAGE_TEMPLATE.format(class_name=class_name, factory_name=factory_name))


# === Pipeline-record-stub construction helpers ==============================
#
# The pipeline-tracking record types (``ComparisonGroupRecord`` etc.) are
# Pydantic stubs with ``extra="allow"`` and no declared fields per
# ``07-plugin-interfaces.md`` §3.1 / DEC-029 (carry-forward from Task 1.8).
# Constructing them with kwargs (``ComparisonGroupRecord(id="x", state=...)``)
# trips pyright-strict's ``reportCallIssue`` because the class has no
# declared parameters even though Pydantic accepts the call at runtime.
#
# Resolution: use ``model_validate({...})`` — the dict path is unconditionally
# typed and works for any extra-allow Pydantic stub. The helpers below are
# narrow wrappers so the bases stay readable.

_R = TypeVar("_R", bound=BaseModel)


def make_record(record_type: type[_R], **fields: object) -> _R:
    """Construct a stub pipeline-record from kwargs.

    Goes through ``model_validate`` so pyright-strict accepts the call —
    direct kwarg construction would trip ``reportCallIssue`` (the stub
    has ``extra="allow"`` but no declared fields).
    """
    return record_type.model_validate(dict(fields))


__all__ = [
    "SCHEDULING_QUERY_METHODS",
    "TimeProvider",
    "_UnimplementedArtifactStore",
    "_UnimplementedCompute",
    "_UnimplementedLlmProvider",
    "_UnimplementedMetadataStore",
    "_UnimplementedTransaction",
    "make_record",
    "skip_no_factory_override",
]
