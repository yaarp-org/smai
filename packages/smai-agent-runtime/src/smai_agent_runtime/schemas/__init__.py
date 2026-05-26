"""Structured input + output Pydantic schemas for the sandbox-side
agent-reasoning steps.

Per architectural_decisions.md §12 #1: each per-step input bundle is the
complete informational world the agent reasons over. No ``read_file`` at
the body-generation steps; bundle-completeness is enforced in the schema.
The diagnose step (D7c) is the single exception with a controlled
``read_file`` escape hatch (5 calls / attempt, workspace-rooted).

Sub-PR C1 landed the two body-generation schemas (D7a + D7b). Sub-PR C2
adds D7c's diagnose schema alongside the real diagnose-step handler.
"""

from __future__ import annotations

from smai_agent_runtime.schemas.body_generation import (
    ABIFunctionName,
    ExtensionPointSpec,
    FunctionSignature,
    GroundingContext,
    HarnessBuilderBodyGenerationInput,
    HarnessBuilderBodyGenerationOutput,
    IntegrationPattern,
    LintFailure,
    NoOpBaselineGrounding,
    PaperExtractGrounding,
    PriorFailedAttempt,
    PriorFailureDetail,
    PriorTechniqueAttempt,
    ProposalGrounding,
    ReviewerAttestedGrounding,
    StandardLibraryGrounding,
    TechniqueBodyGenerationBundle,
    TechniqueBodyOutput,
    ValidationFailure,
)
from smai_agent_runtime.schemas.diagnose import (
    READ_FILE_BUDGET_PER_ATTEMPT,
    READ_FILE_MAX_BYTES,
    STDERR_TAIL_BYTES,
    STDERR_TAIL_LINES,
    DiagnoseBundle,
    Diagnosis,
    FixTarget,
    HarnessOriginalBundleRef,
    OriginalInputBundleRef,
    PriorDiagnosis,
    TechniqueOriginalBundleRef,
)

__all__ = [
    "ABIFunctionName",
    "DiagnoseBundle",
    "Diagnosis",
    "ExtensionPointSpec",
    "FixTarget",
    "FunctionSignature",
    "GroundingContext",
    "HarnessBuilderBodyGenerationInput",
    "HarnessBuilderBodyGenerationOutput",
    "HarnessOriginalBundleRef",
    "IntegrationPattern",
    "LintFailure",
    "NoOpBaselineGrounding",
    "OriginalInputBundleRef",
    "PaperExtractGrounding",
    "PriorDiagnosis",
    "PriorFailedAttempt",
    "PriorFailureDetail",
    "PriorTechniqueAttempt",
    "ProposalGrounding",
    "READ_FILE_BUDGET_PER_ATTEMPT",
    "READ_FILE_MAX_BYTES",
    "ReviewerAttestedGrounding",
    "STDERR_TAIL_BYTES",
    "STDERR_TAIL_LINES",
    "StandardLibraryGrounding",
    "TechniqueBodyGenerationBundle",
    "TechniqueBodyOutput",
    "TechniqueOriginalBundleRef",
    "ValidationFailure",
]
