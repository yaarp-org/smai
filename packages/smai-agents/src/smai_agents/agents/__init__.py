"""Role-shaped wrappers around the agent substrate.

Per ``04-agents.md`` §2.2 / §2.3 / §2.4 / §2.5: the multi-turn agents
(harness builder, technique implementer) compose the loop substrate
plus prompt config plus tool registry; the single-call agents (code
reviewer, contextual evaluator, supervisor) bypass the loop and call
:func:`smai_agents.structured_call` directly.

Four of five agents ship across 2.B3 + 2.B4:

* :func:`smai_agents.agents.harness_builder.run_harness_builder_session` — §2.2
* :func:`smai_agents.agents.technique_implementer.run_technique_implementer_session` — §2.3
* :func:`smai_agents.agents.code_reviewer.run_code_review` — §2.4
* :func:`smai_agents.agents.contextual_evaluator.run_contextual_evaluation` — §2.5

Supervisor (§2.6) lands in Task 3.G4. Planner (§2.1) lands in Task 3.E1.
"""

from smai_agents.agents.code_reviewer import (
    CodeReviewerInput,
    EntryUnderReview,
    run_code_review,
)
from smai_agents.agents.contextual_evaluator import (
    CGMetadata,
    ContextualEvaluatorEntry,
    ContextualEvaluatorInput,
    run_contextual_evaluation,
)
from smai_agents.agents.harness_builder import (
    make_dispatch_harness_build,
    run_harness_builder_session,
)
from smai_agents.agents.manifest_tool import (
    EMIT_HARNESS_MANIFEST_TOOL_NAME,
    make_emit_harness_manifest_tool,
)
from smai_agents.agents.planner import (
    DEFAULT_DESIGN_PLAN_KEY_TEMPLATE,
    DEFAULT_PAPER_PLAN_KEY_TEMPLATE,
    DraftComparisonGroup,
    DraftEntry,
    DraftLevel,
    DraftTechnique,
    DraftValidationCriteria,
    PlannerBuffer,
    PlannerInput,
    PlannerSessionResult,
    PlannerVariant,
    make_dispatch_planner,
    run_planner_session,
    variant_for_submission_kind,
)
from smai_agents.agents.supervisor import (
    SupervisorInput,
    run_supervisor_check,
)
from smai_agents.agents.technique_implementer import (
    ContextKind,
    make_dispatch_technique_implementation,
    run_technique_implementer_session,
)

__all__ = [
    "DEFAULT_DESIGN_PLAN_KEY_TEMPLATE",
    "DEFAULT_PAPER_PLAN_KEY_TEMPLATE",
    "EMIT_HARNESS_MANIFEST_TOOL_NAME",
    "CGMetadata",
    "CodeReviewerInput",
    "ContextKind",
    "ContextualEvaluatorEntry",
    "ContextualEvaluatorInput",
    "DraftComparisonGroup",
    "DraftEntry",
    "DraftLevel",
    "DraftTechnique",
    "DraftValidationCriteria",
    "EntryUnderReview",
    "PlannerBuffer",
    "PlannerInput",
    "PlannerSessionResult",
    "PlannerVariant",
    "SupervisorInput",
    "make_dispatch_harness_build",
    "make_dispatch_planner",
    "make_dispatch_technique_implementation",
    "make_emit_harness_manifest_tool",
    "run_code_review",
    "run_contextual_evaluation",
    "run_harness_builder_session",
    "run_planner_session",
    "run_supervisor_check",
    "run_technique_implementer_session",
    "variant_for_submission_kind",
]
