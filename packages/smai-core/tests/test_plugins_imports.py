"""Import tests for plugin Protocol definitions.

Per Task 1.8 — Protocols and supporting types are importable from
``smai_core.plugins`` and re-exported from ``smai_core``.
"""

from __future__ import annotations


def test_imports_from_smai_core_plugins() -> None:
    """All Protocols and supporting types resolve from
    ``smai_core.plugins``."""
    from smai_core.plugins import (  # noqa: F401
        ArtifactNotFound,
        ArtifactStore,
        ArtifactStoreCapabilities,
        ArtifactStoreError,
        ArtifactTooLarge,
        CacheConfig,
        CGState,
        ComparisonGroupRecord,
        Compute,
        ComputeCapabilities,
        ComputeError,
        ComputeUnavailable,
        ConflictError,
        CursorPage,
        EntityKind,
        EntryRecord,
        EntryState,
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
        PaperRecord,
        PaperState,
        PresignedUrlsUnsupported,
        ProposalRecord,
        ProposalState,
        RunRecord,
        RunState,
        StopReason,
        TextContent,
        TokenUsage,
        ToolDefinition,
        ToolResultContent,
        ToolUseContent,
        Transaction,
    )


def test_reexports_from_smai_core() -> None:
    """The four Protocols + key types are re-exported at the top level."""
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
        # records
        "ComparisonGroupRecord",
        "EntryRecord",
        "RunRecord",
        "ProposalRecord",
        "PaperRecord",
        "CGState",
        "EntryState",
        "RunState",
        "ProposalState",
        "PaperState",
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
