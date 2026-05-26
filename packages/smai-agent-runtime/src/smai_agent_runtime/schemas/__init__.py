"""Structured input + output Pydantic schemas for the sandbox-side
agent-reasoning steps.

Per architectural_decisions.md §12 #1: each per-step input bundle is the
complete informational world the agent reasons over. No ``read_file`` at
the body-generation steps; bundle-completeness is enforced in the schema.

Sub-PR C1 lands the two body-generation schemas (D7a + D7b). D7c's
diagnose schema lands in sub-PR C2 alongside the real diagnose-step
handler.
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

__all__ = [
    "ABIFunctionName",
    "ExtensionPointSpec",
    "FunctionSignature",
    "GroundingContext",
    "HarnessBuilderBodyGenerationInput",
    "HarnessBuilderBodyGenerationOutput",
    "IntegrationPattern",
    "LintFailure",
    "NoOpBaselineGrounding",
    "PaperExtractGrounding",
    "PriorFailedAttempt",
    "PriorFailureDetail",
    "PriorTechniqueAttempt",
    "ProposalGrounding",
    "ReviewerAttestedGrounding",
    "StandardLibraryGrounding",
    "TechniqueBodyGenerationBundle",
    "TechniqueBodyOutput",
    "ValidationFailure",
]
