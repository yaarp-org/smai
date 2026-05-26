"""Workflow generator + ``WorkflowStep`` type hierarchy (sub-PR A).

Per ``designs/smai/agent_refactor/architectural_decisions.md`` §12 items
3 and 4 and the D9 design note. Lands the typed step hierarchy plus the
pure-Python ``generate_workflow`` mapping that the rest of Step 4 of the
agent-layer refactor builds on. Mini-orchestrator ``main()`` (sub-PR B)
and per-step agent-reasoning bodies (sub-PR C) consume what this module
produces.
"""

from __future__ import annotations

from smai_agent_runtime.workflow.generator import TaskRole, generate_workflow
from smai_agent_runtime.workflow.step_types import (
    STEP_REGISTRY,
    ApplyReviewFeedbackStep,
    BaselineGenerationStep,
    BodyGenerationStep,
    DiagnoseOnFailureStep,
    HarnessBuilderBodyGenerationStep,
    ManifestEmitStep,
    StepKind,
    TechniqueImplementerBodyGenerationStep,
    ValidationStep,
    WorkflowStep,
    WorkflowStepAdapter,
    WorkflowStepBase,
    register_step,
)

__all__ = [
    "STEP_REGISTRY",
    "ApplyReviewFeedbackStep",
    "BaselineGenerationStep",
    "BodyGenerationStep",
    "DiagnoseOnFailureStep",
    "HarnessBuilderBodyGenerationStep",
    "ManifestEmitStep",
    "StepKind",
    "TaskRole",
    "TechniqueImplementerBodyGenerationStep",
    "ValidationStep",
    "WorkflowStep",
    "WorkflowStepAdapter",
    "WorkflowStepBase",
    "generate_workflow",
    "register_step",
]
