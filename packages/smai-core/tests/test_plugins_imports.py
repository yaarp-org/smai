"""Import tests for plugin Protocol definitions.

Per Task 1.8 — Protocols and supporting types are importable from
``smai_core.plugins`` and re-exported from ``smai_core``.
"""

from __future__ import annotations


def test_imports_from_smai_core_plugins() -> None:
    """All runtime-importable Protocols and supporting types resolve from
    ``smai_core.plugins``.

    Pipeline-tracking record types (``ComparisonGroupRecord`` etc.) are
    NOT in this set — they live in ``smai-orchestrator`` per Task 1.10 /
    ``01-data-model.md`` §5.1 and are TYPE_CHECKING-only re-exports
    here. See :func:`test_record_types_importable_from_smai_orchestrator`
    below.
    """
    from smai_core.plugins import (  # noqa: F401
        ArtifactNotFound,
        ArtifactStore,
        ArtifactStoreCapabilities,
        ArtifactStoreError,
        ArtifactTooLarge,
        CacheConfig,
        Compute,
        ComputeCapabilities,
        ComputeError,
        ComputeUnavailable,
        ConflictError,
        CursorPage,
        EntityKind,
        JobHandle,
        JobImageInvalid,
        JobNotFound,
        JobState,
        JobStatus,
        LeaseLostError,
        LeaseToken,
        LlmCapabilities,
        LlmProvider,
        LlmProviderAuthError,
        LlmProviderError,
        LlmProviderInvalidRequest,
        LlmProviderRateLimited,
        LlmProviderUnavailable,
        MetadataStore,
        MetadataStoreCapabilities,
        MetadataStoreError,
        ModelResponse,
        NormalizedContent,
        NormalizedMessage,
        PresignedUrlsUnsupported,
        StopReason,
        TextContent,
        TokenUsage,
        ToolDefinition,
        ToolResultContent,
        ToolUseContent,
        Transaction,
    )


def test_record_types_importable_from_smai_orchestrator() -> None:
    """Per Task 1.10: pipeline-tracking record types live in
    ``smai_orchestrator.entities.tracking``; runtime construction goes
    through that path."""
    from smai_orchestrator.entities.tracking import (  # noqa: F401
        CGState,
        ComparisonGroupRecord,
        EntryRecord,
        EntryState,
        FactorModelRecord,
        PaperRecord,
        PaperState,
        ProposalRecord,
        ProposalState,
        RunRecord,
        RunState,
    )


def test_reexports_from_smai_core() -> None:
    """The four Protocols + key types are re-exported at the top level
    as runtime attributes.

    Pipeline-tracking record types are listed in ``smai_core.__all__``
    for typing convenience but are NOT runtime attributes (Task 1.10:
    they live in ``smai-orchestrator`` and are TYPE_CHECKING-only
    re-exports). See :func:`test_record_types_in_all_but_not_runtime`
    for the typing-side assertion.
    """
    import smai_core

    expected_names = {
        # Protocols
        "LlmProvider",
        "MetadataStore",
        "ArtifactStore",
        "Compute",
        "Transaction",
        # capability flags
        "LlmCapabilities",
        "MetadataStoreCapabilities",
        "ArtifactStoreCapabilities",
        "ComputeCapabilities",
        # normalized types
        "NormalizedMessage",
        "NormalizedContent",
        "TextContent",
        "ToolUseContent",
        "ToolResultContent",
        "ToolDefinition",
        "ModelResponse",
        "TokenUsage",
        "StopReason",
        "CacheConfig",
        # job types
        "JobHandle",
        "JobState",
        "JobStatus",
        # shared
        "CursorPage",
        "LeaseToken",
        "EntityKind",
        # errors
        "LlmProviderError",
        "LlmProviderRateLimited",
        "LlmProviderUnavailable",
        "LlmProviderInvalidRequest",
        "LlmProviderAuthError",
        "MetadataStoreError",
        "ConflictError",
        "LeaseLostError",
        "ArtifactStoreError",
        "ArtifactNotFound",
        "PresignedUrlsUnsupported",
        "ArtifactTooLarge",
        "ComputeError",
        "ComputeUnavailable",
        "JobNotFound",
        "JobImageInvalid",
    }
    for name in expected_names:
        assert hasattr(smai_core, name), f"smai_core missing re-export: {name}"
        assert name in smai_core.__all__, f"smai_core.__all__ missing: {name}"


def test_record_types_in_all_but_not_runtime() -> None:
    """Pipeline-tracking record types are advertised in ``smai_core.__all__``
    for typing convenience but NOT importable at runtime from smai_core
    (Task 1.10 — they live in ``smai-orchestrator``)."""
    import smai_core

    record_typing_names = {
        "ComparisonGroupRecord",
        "EntryRecord",
        "RunRecord",
        "ProposalRecord",
        "PaperRecord",
        "FactorModelRecord",
        "CGState",
        "EntryState",
        "RunState",
        "ProposalState",
        "PaperState",
    }
    for name in record_typing_names:
        assert name in smai_core.__all__, f"smai_core.__all__ missing typing alias: {name}"
        # NOT a runtime attribute — TYPE_CHECKING-only re-export.
        assert not hasattr(smai_core, name), (
            f"{name} should be TYPE_CHECKING-only on smai_core (per Task 1.10), "
            f"but it is exposed as a runtime attribute"
        )


def test_runtime_checkable_decorator_applied() -> None:
    """Each Protocol is decorated with ``@runtime_checkable`` — verified
    by attempting an :class:`isinstance` check (which raises
    ``TypeError`` if the Protocol is *not* runtime-checkable)."""
    from smai_core.plugins import (
        ArtifactStore,
        Compute,
        LlmProvider,
        MetadataStore,
        Transaction,
    )

    class _Empty:
        pass

    obj = _Empty()
    # No raise = Protocol is runtime-checkable. Both should return False
    # since _Empty does not satisfy the Protocol.
    assert isinstance(obj, LlmProvider) is False
    assert isinstance(obj, MetadataStore) is False
    assert isinstance(obj, ArtifactStore) is False
    assert isinstance(obj, Compute) is False
    assert isinstance(obj, Transaction) is False
