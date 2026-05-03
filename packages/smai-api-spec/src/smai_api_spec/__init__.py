"""Shared API contract for SMAI v2.

Per DEC-037: this package is depended on by both SMAI's OSS HTTP API
(``smai-api``, Phase 4 task K1) and the Yaarp v2 hosted-product HTTP
API. Pure data — Pydantic request/response models, URL path constants,
the error taxonomy, and the SSE event payload types. Runtime
dependencies are limited to ``pydantic``.

See the top-level :mod:`smai_api_spec.paths` for the URL constants
both implementations import; per-resource modules
(:mod:`~smai_api_spec.proposals`, :mod:`~smai_api_spec.papers`, ...)
hold the request and response Pydantic shapes.
"""

from __future__ import annotations

from smai_api_spec import (
    comparison_groups,
    entries,
    errors,
    events,
    experiments,
    pagination,
    papers,
    paths,
    proposals,
    runs,
    system,
)
from smai_api_spec._common import (
    APIBaseModel,
    BaseAuditedResponse,
    CGState,
    EntityKind,
    EntryState,
    PaperState,
    ProposalState,
    RunState,
    ScreenDecision,
    SubmissionKind,
    UserDecision,
)
from smai_api_spec.comparison_groups import (
    AgentStatusResponse,
    ArtifactKeysResponse,
    CGStatusResponse,
    CGSummary,
    ComparisonGroupDetailResponse,
    EntryAgentStatus,
    EvaluationResultResponse,
    HarnessAgentStatus,
    ListArtifactsParams,
    ListCGsParams,
)
from smai_api_spec.entries import (
    EntryDetailResponse,
    EntrySummary,
    EntryWithRuns,
)
from smai_api_spec.errors import (
    APIError,
    ErrorCode,
    ErrorEnvelope,
    ValidationIssue,
)
from smai_api_spec.events import (
    StateChangeEvent,
    WorkerHeartbeatEvent,
)
from smai_api_spec.experiments import (
    CompiledArtifacts,
    CompileExperimentRequest,
    CompileExperimentResponse,
    CreatedCGRef,
    SubmitExperimentRequest,
    SubmitExperimentResponse,
)
from smai_api_spec.pagination import (
    CursorPage,
    PaginationParams,
)
from smai_api_spec.papers import (
    ListPapersParams,
    PaperDetailResponse,
    PaperSubmissionResponse,
    PaperSummary,
    PromotePartialResponse,
    SubmitPaperRequest,
    TechniqueRefSummary,
)
from smai_api_spec.proposals import (
    ApproveProposalResponse,
    ListProposalsParams,
    ProposalDetailResponse,
    ProposalSubmissionResponse,
    ProposalSummary,
    RejectProposalRequest,
    RejectProposalResponse,
    SubmitProposalRequest,
)
from smai_api_spec.runs import (
    ListRunsParams,
    RunDetailResponse,
    RunSummary,
)
from smai_api_spec.system import (
    PluginInfo,
    PluginVerifyResult,
    RecentActivityItem,
    SummaryCounts,
    SystemConfigResponse,
    SystemDashboardResponse,
    SystemHealthResponse,
    SystemMigrateStatusResponse,
    SystemPluginsResponse,
    SystemVerifyRequest,
    SystemVerifyResponse,
    SystemVersionResponse,
)

__all__ = [
    # === Configured base + Literals (smai_api_spec._common) ===
    "APIBaseModel",
    "BaseAuditedResponse",
    "CGState",
    "EntityKind",
    "EntryState",
    "PaperState",
    "ProposalState",
    "RunState",
    "ScreenDecision",
    "SubmissionKind",
    "UserDecision",
    # === Pagination ===
    "CursorPage",
    "PaginationParams",
    # === Errors ===
    "APIError",
    "ErrorCode",
    "ErrorEnvelope",
    "ValidationIssue",
    # === Events (SSE) ===
    "StateChangeEvent",
    "WorkerHeartbeatEvent",
    # === Proposals ===
    "ApproveProposalResponse",
    "ListProposalsParams",
    "ProposalDetailResponse",
    "ProposalSubmissionResponse",
    "ProposalSummary",
    "RejectProposalRequest",
    "RejectProposalResponse",
    "SubmitProposalRequest",
    # === Papers ===
    "ListPapersParams",
    "PaperDetailResponse",
    "PaperSubmissionResponse",
    "PaperSummary",
    "PromotePartialResponse",
    "SubmitPaperRequest",
    "TechniqueRefSummary",
    # === Experiments ===
    "CompiledArtifacts",
    "CompileExperimentRequest",
    "CompileExperimentResponse",
    "CreatedCGRef",
    "SubmitExperimentRequest",
    "SubmitExperimentResponse",
    # === Comparison Groups ===
    "AgentStatusResponse",
    "ArtifactKeysResponse",
    "CGStatusResponse",
    "CGSummary",
    "ComparisonGroupDetailResponse",
    "EntryAgentStatus",
    "EvaluationResultResponse",
    "HarnessAgentStatus",
    "ListArtifactsParams",
    "ListCGsParams",
    # === Entries ===
    "EntryDetailResponse",
    "EntrySummary",
    "EntryWithRuns",
    # === Runs ===
    "ListRunsParams",
    "RunDetailResponse",
    "RunSummary",
    # === System ===
    "PluginInfo",
    "PluginVerifyResult",
    "RecentActivityItem",
    "SummaryCounts",
    "SystemConfigResponse",
    "SystemDashboardResponse",
    "SystemHealthResponse",
    "SystemMigrateStatusResponse",
    "SystemPluginsResponse",
    "SystemVerifyRequest",
    "SystemVerifyResponse",
    "SystemVersionResponse",
    # === Submodules ===
    "comparison_groups",
    "entries",
    "errors",
    "events",
    "experiments",
    "pagination",
    "papers",
    "paths",
    "proposals",
    "runs",
    "system",
]
