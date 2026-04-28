"""Plugin Protocol definitions — the four pipeline-layer plugin contracts.

Per ``designs/smai/07-plugin-interfaces.md`` §3-§7, DEC-028 (four
interfaces, not five), and DEC-029 (Protocol definitions live in
``smai-core``; concrete implementations live in plugin packages).

The four interfaces:

* :class:`LlmProvider` — agent / LLM seam (§4)
* :class:`MetadataStore` — pipeline-tracking entity persistence (§5)
* :class:`ArtifactStore` — opaque-blob persistence (§6)
* :class:`Compute` — containerized job lifecycle (§7)

Tier B integrators that consume only the methodology layer never import
from this package — :class:`MetadataStore`, :class:`ArtifactStore`, and
:class:`Compute` are pipeline-layer concerns. Tier A integrators (the
in-tree CLI, the hosted backend) import these Protocols and the plugin
packages register implementations via the ``smai.<namespace>`` entry
points.
"""

from smai_core.plugins._common import CursorPage, EntityKind, LeaseToken
from smai_core.plugins.artifact_store import (
    ArtifactNotFound,
    ArtifactStore,
    ArtifactStoreCapabilities,
    ArtifactStoreError,
    ArtifactTooLarge,
    PresignedUrlsUnsupported,
)
from smai_core.plugins.compute import (
    Compute,
    ComputeCapabilities,
    ComputeError,
    ComputeUnavailable,
    JobHandle,
    JobImageInvalid,
    JobNotFound,
    JobState,
    JobStatus,
)
from smai_core.plugins.llm_provider import (
    CacheConfig,
    LlmCapabilities,
    LlmProvider,
    LlmProviderAuthError,
    LlmProviderError,
    LlmProviderInvalidRequest,
    LlmProviderRateLimited,
    LlmProviderUnavailable,
    ModelResponse,
    NormalizedContent,
    NormalizedMessage,
    StopReason,
    TextContent,
    TokenUsage,
    ToolDefinition,
    ToolResultContent,
    ToolUseContent,
)
from smai_core.plugins.metadata_store import (
    CGState,
    ComparisonGroupRecord,
    ConflictError,
    EntryRecord,
    EntryState,
    LeaseLostError,
    MetadataStore,
    MetadataStoreCapabilities,
    MetadataStoreError,
    PaperRecord,
    PaperState,
    ProposalRecord,
    ProposalState,
    RunRecord,
    RunState,
    Transaction,
)

__all__ = [
    # _common
    "CursorPage",
    "EntityKind",
    "LeaseToken",
    # artifact_store
    "ArtifactNotFound",
    "ArtifactStore",
    "ArtifactStoreCapabilities",
    "ArtifactStoreError",
    "ArtifactTooLarge",
    "PresignedUrlsUnsupported",
    # compute
    "Compute",
    "ComputeCapabilities",
    "ComputeError",
    "ComputeUnavailable",
    "JobHandle",
    "JobImageInvalid",
    "JobNotFound",
    "JobState",
    "JobStatus",
    # llm_provider
    "CacheConfig",
    "LlmCapabilities",
    "LlmProvider",
    "LlmProviderAuthError",
    "LlmProviderError",
    "LlmProviderInvalidRequest",
    "LlmProviderRateLimited",
    "LlmProviderUnavailable",
    "ModelResponse",
    "NormalizedContent",
    "NormalizedMessage",
    "StopReason",
    "TextContent",
    "TokenUsage",
    "ToolDefinition",
    "ToolResultContent",
    "ToolUseContent",
    # metadata_store
    "CGState",
    "ComparisonGroupRecord",
    "ConflictError",
    "EntryRecord",
    "EntryState",
    "LeaseLostError",
    "MetadataStore",
    "MetadataStoreCapabilities",
    "MetadataStoreError",
    "PaperRecord",
    "PaperState",
    "ProposalRecord",
    "ProposalState",
    "RunRecord",
    "RunState",
    "Transaction",
]
