"""Structural tests for the four conformance test bases (Task 1.9).

These are tests OF the bases themselves (not tests THAT USE the bases —
those live in plugin packages, Phase 2). They verify:

* the bases are importable from :mod:`smai_core.plugins.conformance`
* each base exposes the expected number of test methods per the canonical
  enumerations in ``07-plugin-interfaces.md`` §4.7 / §5.8 / §6.4 / §7.5
* skip-collection works when ``make_<interface>()`` is not overridden
* a subclass that overrides the factory inherits and runs the contract
  methods (mechanically — the stub fixtures raise
  :class:`NotImplementedError` to confirm the call reached them)
* the :func:`time_provider` fixture is overrideable
* the :class:`CursorPage` parameterization on :class:`MetadataStore`
  scheduling queries resolves cleanly
"""

from __future__ import annotations

import inspect
import time

import pytest
from smai_core.plugins import (
    ArtifactStore,
    Compute,
    CursorPage,
    LlmProvider,
    MetadataStore,
)
from smai_core.plugins.conformance import (
    ArtifactStoreConformance,
    ComputeConformance,
    LlmProviderConformance,
    MetadataStoreConformance,
)
from smai_core.plugins.conformance._common import (
    SCHEDULING_QUERY_METHODS,
    _UnimplementedArtifactStore,
    _UnimplementedCompute,
    _UnimplementedLlmProvider,
    _UnimplementedMetadataStore,
    _UnimplementedTransaction,
)
from smai_core.plugins.metadata_store._records import (
    ComparisonGroupRecord,
    EntryRecord,
    PaperRecord,
    ProposalRecord,
    RunRecord,
)

# ---------- import tests ------------------------------------------------------


def test_llm_provider_conformance_importable() -> None:
    assert LlmProviderConformance is not None
    assert inspect.isclass(LlmProviderConformance)


def test_metadata_store_conformance_importable() -> None:
    assert MetadataStoreConformance is not None
    assert inspect.isclass(MetadataStoreConformance)


def test_artifact_store_conformance_importable() -> None:
    assert ArtifactStoreConformance is not None
    assert inspect.isclass(ArtifactStoreConformance)


def test_compute_conformance_importable() -> None:
    assert ComputeConformance is not None
    assert inspect.isclass(ComputeConformance)


def test_unimplemented_stubs_importable_internally() -> None:
    """Internal stub helpers (used by structural tests only) load."""
    assert _UnimplementedLlmProvider is not None
    assert _UnimplementedMetadataStore is not None
    assert _UnimplementedArtifactStore is not None
    assert _UnimplementedCompute is not None
    assert _UnimplementedTransaction is not None


# ---------- method-count enumeration -----------------------------------------


def _test_methods(cls: type) -> list[str]:
    """Names of test methods declared on ``cls`` (test_*)."""
    return sorted(
        name
        for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
        if name.startswith("test_")
    )


# Per ``07-plugin-interfaces.md`` §4.7. The 11 LlmProvider contract methods
# enumerated in the conformance base file's docstring.
_LLM_EXPECTED_METHODS = {
    "test_normalized_round_trip",
    "test_capability_caching_honesty",
    "test_capability_tool_use_honesty",
    "test_capabilities_immutable",
    "test_rate_limit_error_class",
    "test_unavailable_error_class",
    "test_invalid_request_error_class",
    "test_auth_error_class",
    "test_transient_retry_once",
    "test_tool_use_round_trip",
    "test_input_messages_not_mutated",
}


def test_llm_provider_method_count() -> None:
    methods = set(_test_methods(LlmProviderConformance))
    assert methods == _LLM_EXPECTED_METHODS, (
        f"unexpected LlmProvider conformance methods: "
        f"missing={_LLM_EXPECTED_METHODS - methods}, "
        f"extra={methods - _LLM_EXPECTED_METHODS}"
    )


# Per ``07-plugin-interfaces.md`` §6.4. 9 ArtifactStore contract methods +
# 1 capability-sanity check = 10.
_ARTIFACT_EXPECTED_METHODS = {
    "test_put_then_get_round_trip",
    "test_overwrite",
    "test_exists_post_put",
    "test_list_returns_keys_under_prefix",
    "test_delete_idempotency",
    "test_delete_removes",
    "test_get_missing_raises_not_found",
    "test_url_for_supported",
    "test_url_for_unsupported",
    "test_max_object_size_honored",
    "test_varied_byte_sizes_round_trip",
    "test_capabilities_are_capabilities_model",
}


def test_artifact_store_method_count() -> None:
    methods = set(_test_methods(ArtifactStoreConformance))
    assert methods == _ARTIFACT_EXPECTED_METHODS, (
        f"missing={_ARTIFACT_EXPECTED_METHODS - methods}, "
        f"extra={methods - _ARTIFACT_EXPECTED_METHODS}"
    )


# Per ``07-plugin-interfaces.md`` §7.5. 7 Compute contract methods +
# 1 capability-sanity check = 8.
_COMPUTE_EXPECTED_METHODS = {
    "test_submit_and_poll_succeeds",
    "test_submit_and_fail_nonzero_exit",
    "test_timeout_fires",
    "test_cancel_terminates",
    "test_job_handle_reconnection",
    "test_logs_returns_stdout",
    "test_invalid_image_raises",
    "test_capabilities_are_capabilities_model",
}


def test_compute_method_count() -> None:
    methods = set(_test_methods(ComputeConformance))
    assert methods == _COMPUTE_EXPECTED_METHODS, (
        f"missing={_COMPUTE_EXPECTED_METHODS - methods}, "
        f"extra={methods - _COMPUTE_EXPECTED_METHODS}"
    )


# Per ``07-plugin-interfaces.md`` §5.8 + §5.6.10. The MetadataStore contract
# method set (each method is a test entry-point; some are parameterized over
# scheduling-query method names or entity kinds, multiplying the collected
# test count beyond this list's size — see test_metadata_store_collection_count).
_METADATA_EXPECTED_METHODS = {
    "test_crud_round_trip_cg",
    "test_crud_round_trip_entry",
    "test_crud_round_trip_run",
    "test_crud_round_trip_proposal",
    "test_crud_round_trip_paper",
    "test_transition_succeeds_on_matching_version",
    "test_transition_raises_conflict_on_stale_version",
    "test_transaction_commits_on_clean_exit",
    "test_transaction_rolls_back_on_exception",
    "test_scheduling_query_returns_cursor_page",
    "test_scheduling_query_pagination_terminates",
    "test_count_in_state_multi_state_input",
    "test_count_with_in_flight_jobs_predicate",
    "test_acquire_then_acquire_fails",
    "test_release_then_acquire_succeeds",
    "test_extend_still_held_blocks_acquire",
    "test_acquire_expire_reacquire",
    "test_extend_after_loss_raises",
    "test_stale_token_release_noop",
    "test_unknown_token_release_noop",
    "test_tenant_fairness_contract",
    "test_capabilities_are_capabilities_model",
}


def test_metadata_store_method_count() -> None:
    methods = set(_test_methods(MetadataStoreConformance))
    assert methods == _METADATA_EXPECTED_METHODS, (
        f"missing={_METADATA_EXPECTED_METHODS - methods}, "
        f"extra={methods - _METADATA_EXPECTED_METHODS}"
    )


def test_scheduling_query_method_enumeration() -> None:
    """The parameterized scheduling-query enumeration in
    :data:`SCHEDULING_QUERY_METHODS` matches the canonical
    enumeration in §5.6.3 / §5.6.4 / §5.6.5 (19 methods)."""
    expected = {
        # CG-execution (9)
        "get_ready_for_harness_build",
        "get_in_flight_harness_build",
        "get_ready_to_implement_entry",
        "get_in_flight_entry_implementation",
        "get_ready_for_review_and_run",
        "get_ready_to_run",
        "get_in_flight_runs",
        "get_ready_to_evaluate",
        "get_in_flight_evaluation",
        # proposal (3)
        "get_ready_for_proposal_design",
        "get_in_flight_proposal_design",
        "get_pending_proposal_user_decision",
        # paper-ingestion (7)
        "get_ready_for_paper_fetch",
        "get_in_flight_paper_fetch",
        "get_ready_for_paper_screen",
        "get_in_flight_paper_screen",
        "get_ready_for_paper_plan",
        "get_in_flight_paper_plan",
        "get_partial_pending_promotion",
    }
    assert set(SCHEDULING_QUERY_METHODS) == expected
    assert len(SCHEDULING_QUERY_METHODS) == 19


# ---------- skip-collection on no override -----------------------------------


class _CollectsLlm(LlmProviderConformance):
    """Subclass with no ``make_provider`` override → tests skip-collect."""


class _CollectsMetadata(MetadataStoreConformance):
    """Subclass with no ``make_store`` override → tests skip-collect."""


class _CollectsArtifacts(ArtifactStoreConformance):
    """Subclass with no ``make_store`` override → tests skip-collect."""


class _CollectsCompute(ComputeConformance):
    """Subclass with no ``make_compute`` override → tests skip-collect."""


def test_llm_provider_skip_collection_no_override() -> None:
    """Calling the un-overridden factory raises ``Skipped``."""
    with pytest.raises(pytest.skip.Exception) as excinfo:
        _CollectsLlm().make_provider()
    assert "make_provider" in str(excinfo.value)


def test_metadata_store_skip_collection_no_override() -> None:
    import asyncio

    with pytest.raises(pytest.skip.Exception) as excinfo:
        # make_store is async — but the skip fires synchronously inside the
        # coroutine builder. Drive the coroutine and let the skip propagate.
        asyncio.run(_CollectsMetadata().make_store())
    assert "make_store" in str(excinfo.value)


def test_artifact_store_skip_collection_no_override() -> None:
    with pytest.raises(pytest.skip.Exception) as excinfo:
        _CollectsArtifacts().make_store()
    assert "make_store" in str(excinfo.value)


def test_compute_skip_collection_no_override() -> None:
    with pytest.raises(pytest.skip.Exception) as excinfo:
        _CollectsCompute().make_compute()
    assert "make_compute" in str(excinfo.value)


# ---------- subclass-with-stub-override mechanically inherits ----------------


class _LlmSubclassWithStub(LlmProviderConformance):
    """Subclass that returns the unimplemented stub — proves inheritance
    is wired (the contract method calls into the stub, which raises
    :class:`NotImplementedError`)."""

    def make_provider(self) -> LlmProvider:  # type: ignore[override]
        return _UnimplementedLlmProvider()  # type: ignore[return-value]


class _ArtifactSubclassWithStub(ArtifactStoreConformance):
    def make_store(self) -> ArtifactStore:  # type: ignore[override]
        return _UnimplementedArtifactStore()  # type: ignore[return-value]


class _ComputeSubclassWithStub(ComputeConformance):
    def make_compute(self) -> Compute:  # type: ignore[override]
        return _UnimplementedCompute()  # type: ignore[return-value]


class _MetadataSubclassWithStub(MetadataStoreConformance):
    async def make_store(self) -> MetadataStore:  # type: ignore[override]
        return _UnimplementedMetadataStore()  # type: ignore[return-value]


async def test_llm_subclass_inherits_and_invokes_stub() -> None:
    """The bases really do inherit and call into the make_provider() stub.

    The stub raises :class:`NotImplementedError` on call, so the contract
    method that exercises ``provider.call(...)`` raises in turn —
    confirming the inheritance + fixture wiring works.
    """
    instance = _LlmSubclassWithStub()
    provider = instance.make_provider()
    with pytest.raises(NotImplementedError):
        await instance.test_normalized_round_trip(
            provider=provider,
            fixture_messages=[],
        )


async def test_artifact_subclass_inherits_and_invokes_stub() -> None:
    instance = _ArtifactSubclassWithStub()
    store = instance.make_store()
    with pytest.raises(NotImplementedError):
        await instance.test_put_then_get_round_trip(store=store, fixture_key_namespace="x/")


async def test_compute_subclass_inherits_and_invokes_stub() -> None:
    instance = _ComputeSubclassWithStub()
    compute = instance.make_compute()
    with pytest.raises(NotImplementedError):
        await instance.test_submit_and_poll_succeeds(
            compute=compute,
            fixture_image="python:3.12-slim",
            time_provider=time.monotonic,
            poll_budget_seconds=10.0,
        )


async def test_metadata_subclass_inherits_and_invokes_stub() -> None:
    instance = _MetadataSubclassWithStub()
    store = await instance.make_store()
    with pytest.raises(NotImplementedError):
        await instance.test_crud_round_trip_cg(store=store)


# ---------- time_provider override (deferred wall-clock seam, §6 Q4) --------


class _CustomClockLlmConformance(LlmProviderConformance):
    """A subclass that overrides ``time_provider`` with a deterministic clock —
    proves the seam stays open for Task 2.C1 to wire its
    ``EngineConfig``-driven fake clock through."""

    _ticks: list[float] = [0.0, 1.0, 2.0]

    @pytest.fixture
    def time_provider(self):  # type: ignore[override]
        ticks = list(self._ticks)

        def fake() -> float:
            return ticks.pop(0) if ticks else 99.0

        return fake


def _is_pytest_fixture(obj: object) -> bool:
    """Return True iff ``obj`` is a pytest fixture (decorated with ``@pytest.fixture``).

    Across pytest versions the marker name has changed (``_pytestfixturefunction``,
    ``__pytest_wrapped__`` on a ``FixtureFunctionDefinition``, etc.). Use the
    repr (or class name) as a robust heuristic.
    """
    return (
        hasattr(obj, "_pytestfixturefunction")
        or hasattr(obj, "__pytest_wrapped__")
        or "fixture" in type(obj).__name__.lower()
        or "fixture" in repr(obj).lower()
    )


def test_llm_time_provider_overrideable() -> None:
    """The ``time_provider`` fixture exists on the base and is replaceable
    via subclass override (the deferred Task 2.C1 seam)."""
    base_fixture = LlmProviderConformance.__dict__["time_provider"]
    sub_fixture = _CustomClockLlmConformance.__dict__["time_provider"]
    assert base_fixture is not sub_fixture
    assert _is_pytest_fixture(base_fixture)
    assert _is_pytest_fixture(sub_fixture)


def test_compute_time_provider_overrideable() -> None:
    """Same check for :class:`ComputeConformance`."""

    class _Sub(ComputeConformance):
        @pytest.fixture
        def time_provider(self):  # type: ignore[override]
            return lambda: 0.0

    assert ComputeConformance.__dict__["time_provider"] is not _Sub.__dict__["time_provider"]


def test_metadata_time_provider_overrideable() -> None:
    """Same check for :class:`MetadataStoreConformance`."""

    class _Sub(MetadataStoreConformance):
        @pytest.fixture
        def time_provider(self):  # type: ignore[override]
            return lambda: 0.0

    assert MetadataStoreConformance.__dict__["time_provider"] is not _Sub.__dict__["time_provider"]


# ---------- CursorPage[T] parameterization (carry-forward 1.8 → 1.9) ---------


def test_cursor_page_parameterizes_for_each_record_type() -> None:
    """Per the carry-forward note from Task 1.8: ``CursorPage[T]`` resolves
    cleanly under pyright strict for each pipeline-tracking record type
    on the scheduling-query return type."""
    # Construct each parameterized form to verify Pydantic accepts them.
    cg_page = CursorPage[ComparisonGroupRecord](items=[], next_cursor=None)
    entry_page = CursorPage[EntryRecord](items=[], next_cursor=None)
    run_page = CursorPage[RunRecord](items=[], next_cursor=None)
    proposal_page = CursorPage[ProposalRecord](items=[], next_cursor=None)
    paper_page = CursorPage[PaperRecord](items=[], next_cursor=None)
    for page in (cg_page, entry_page, run_page, proposal_page, paper_page):
        assert page.items == []
        assert page.next_cursor is None
        assert page.total is None


def test_cursor_page_round_trips_pipeline_record_stub_extras() -> None:
    """Carry-forward from 1.8: ``_PipelineRecordStub.extra='allow'`` lets
    conformance fixtures construct stub records with arbitrary fields and
    round-trip them through :class:`CursorPage`."""
    record = ComparisonGroupRecord.model_validate(
        {
            "id": "cg_x",
            "state": "draft",
            "version": 1,
            "extra_field_one": "ok",
            "nested": {"a": 1, "b": [2, 3]},
        }
    )
    page = CursorPage[ComparisonGroupRecord](items=[record], next_cursor="opaque-token")
    assert len(page.items) == 1
    assert page.next_cursor == "opaque-token"
    # The arbitrary fields survive round-trip.
    dumped = page.model_dump()
    assert dumped["items"][0]["extra_field_one"] == "ok"
    assert dumped["items"][0]["nested"] == {"a": 1, "b": [2, 3]}
