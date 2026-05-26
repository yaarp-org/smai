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
from datetime import UTC, datetime
from pathlib import Path
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
    WorkspaceHandle,
)

# The pipeline-tracking record types live in ``smai-orchestrator`` per
# Task 1.10 / ``01-data-model.md`` §5.1. The conformance subpackage is
# opt-in test infrastructure (gated behind the ``[conformance]`` extra
# dep); test callers import the record types directly from
# ``smai_orchestrator.entities.tracking``. :func:`make_record` below
# dispatches on the passed ``record_type.__name__`` so it does not need
# the symbols imported here. ``tools/check_deps.py`` exempts this
# subdirectory from rule 2 to admit the cross-boundary import where it
# is needed (in :mod:`test_metadata_store`).

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

    async def credentials_for_subprocess(self) -> dict[str, str]:
        # Sub-PR F: second Protocol method (substrate-dispatch surface).
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
    "get_proposals_at_human_gate",
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
# / ``get_in_flight_*`` / ``get_pending_*`` / ``get_partial_*`` / ``get_proposals_*``
# query). The ``get_proposals_*`` prefix admits human-gate parking queries whose
# semantics ("proposals in a state pending the human gate") don't fit the
# action-noun shapes of the other prefixes — see ``08`` §2.2 / Task 3.E1.
SCHEDULING_QUERY_METHODS: tuple[str, ...] = tuple(
    m
    for m in _METADATA_STORE_METHODS
    if m.startswith(
        ("get_ready_", "get_in_flight_", "get_pending_", "get_partial_", "get_proposals_")
    )
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

    async def stage_workspace(self, local_path: Path) -> WorkspaceHandle:
        raise NotImplementedError("stub compute")

    async def harvest_workspace(self, handle: WorkspaceHandle, local_path: Path) -> None:
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


# === Consistency-of-use helpers (round-20 step-back HIGH-#1) ================
#
# Round-20's load-bearing observation: ``Compute.logs(handle)`` exists on
# the Protocol, but agent-side dispatch callers (e.g.,
# ``run_experiment._harvest_result``) failed to call it on failure, so
# the container's stderr never reached the agent loop. The fix is
# consistency-of-use across all dispatch sites, not a new Protocol
# method. This helper makes that consistency testable at the dispatch
# call site — a unit test for any dispatcher asserts the helper
# returns non-empty content from a synthetic failing handle, pinning
# the convention.
#
# Importable from ``smai_core.plugins.conformance`` (re-exported by the
# package ``__init__``).


async def assert_logs_on_failure(compute: Any, failing_handle: JobHandle) -> str:
    """Assert ``compute.logs(failing_handle)`` returns non-empty content.

    Round-20 step-back HIGH-#1: container stderr was silently dropped
    by the agent-side dispatch path because callers did not invoke
    ``Compute.logs()`` on terminal-failure states. The cure is
    consistency-of-use across every dispatcher that ships under the
    refactor's unified factory (Step 2). This helper is the test-time
    pin for that convention.

    Returns the logs string so call sites can additionally assert on
    its contents.
    """
    logs = await compute.logs(failing_handle)
    assert logs, (
        "Compute.logs(handle) returned empty content on failure; "
        "dispatch callers must surface container stderr (round-20 "
        "step-back HIGH-#1)."
    )
    return logs


# === Pipeline-record fixture construction helpers ==========================
#
# The pipeline-tracking record types live in ``smai_orchestrator.entities.
# tracking`` (Task 1.10 / ``01-data-model.md`` §5). They have
# ``model_config = extra="forbid"`` and many required fields (``id``,
# ``proposal_id``, ``experiment_definition_id``, ``created_at``, etc.).
#
# :func:`make_record` provides sensible defaults for the required fields
# of each known record type so conformance tests can construct fixtures
# with terse overrides (``make_record(ComparisonGroupRecord, id="cg_x",
# state="draft")``).
#
# A test caller passing an unknown record type (or omitting a required
# field for which no default is known) gets the same Pydantic
# ``ValidationError`` they would get constructing the model directly.

_R = TypeVar("_R", bound=BaseModel)


def _epoch() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


def _defaults_for(record_type: type[BaseModel]) -> dict[str, object]:
    """Return the default field set for ``record_type`` (Task 1.10).

    Empty dict for record types we don't recognize — the caller's kwargs
    must fill in every required field.
    """
    name = record_type.__name__
    base: dict[str, object] = {
        "version": 1,
        "created_at": _epoch(),
        "updated_at": _epoch(),
    }
    if name == "ComparisonGroupRecord":
        return base | {
            "id": "cg_default",
            "proposal_id": "prop_default",
            "experiment_definition_id": "cg_default",
            "state": "draft",
        }
    if name == "EntryRecord":
        return base | {
            "id": "entry_default",
            "cg_id": "cg_default",
            "technique_id": None,
            "is_baseline": True,
            "entry_id": "entry_default",
            "state": "pending",
        }
    if name == "RunRecord":
        return base | {
            "id": "run_default",
            "cg_id": "cg_default",
            "entry_id": "entry_default",
            "seed": 0,
            "state": "pending",
        }
    if name == "ProposalRecord":
        return base | {
            "id": "prop_default",
            "submission_kind": "novel_technique",
            "state": "proposal_submitted",
        }
    if name == "PaperRecord":
        return base | {
            "arxiv_id": "2501.00001",
            "state": "submitted",
        }
    if name == "FactorModelRecord":
        return base | {
            "id": "fm_default",
            "factor_model_id": "fm_default",
            "research_question": "default research question",
        }
    return {}


def make_record(record_type: type[_R], **fields: object) -> _R:
    """Construct a pipeline-tracking record fixture (Task 1.10).

    Per-record defaults from :func:`_defaults_for` cover required fields
    so conformance tests can override only the fields they care about::

        cg = make_record(ComparisonGroupRecord, id="cg_round_trip", state="draft")

    Goes through ``model_validate`` so pyright-strict accepts the call —
    direct kwarg construction trips ``reportCallIssue`` because the merge
    of defaults + overrides is not visible to the static checker.
    """
    payload = _defaults_for(record_type) | dict(fields)
    return record_type.model_validate(payload)


__all__ = [
    "SCHEDULING_QUERY_METHODS",
    "TimeProvider",
    "_UnimplementedArtifactStore",
    "_UnimplementedCompute",
    "_UnimplementedLlmProvider",
    "_UnimplementedMetadataStore",
    "_UnimplementedTransaction",
    "assert_logs_on_failure",
    "make_record",
    "skip_no_factory_override",
]
