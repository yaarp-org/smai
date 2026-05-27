"""Role-shaped wrappers around the agent substrate.

Per ``04-agents.md`` §2.1 / §2.4 / §2.5 / §2.6: the inline single-call
agents (code reviewer, contextual evaluator, supervisor) bypass the
loop and call :func:`smai_inline_agents.structured_call` directly; the
planner runs the multi-turn loop substrate in-process.

The sandboxed roles (harness builder §2.2, technique implementer §2.3)
live in :mod:`smai_agent_runtime`; their host-side dispatcher factories
are NOT re-exported here (created at Step 8 of the agent-layer
refactor; the host-side surface moved to
:mod:`smai_orchestrator.sandboxed_dispatch` in Step 8 Wave 2).

Inline roles exported here:

* :func:`smai_inline_agents.agents.planner.run_planner_session` /
  :func:`smai_inline_agents.agents.planner.make_dispatch_planner` (§2.1)
* :func:`smai_inline_agents.agents.code_reviewer.run_code_review` (§2.4)
* :func:`smai_inline_agents.agents.contextual_evaluator.run_contextual_evaluation` (§2.5)
* :func:`smai_inline_agents.agents.supervisor.run_supervisor_check` (§2.6)
* :func:`smai_inline_agents.agents.screener.*` (paper-ingestion screener)
* :func:`smai_inline_agents.agents.enricher.*` (paper-ingestion enricher)
"""

from smai_inline_agents.agents.code_reviewer import (
    CodeReviewerInput,
    EntryUnderReview,
    run_code_review,
)
from smai_inline_agents.agents.contextual_evaluator import (
    CGMetadata,
    ContextualEvaluatorEntry,
    ContextualEvaluatorInput,
    run_contextual_evaluation,
)
from smai_inline_agents.agents.planner import (
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
from smai_inline_agents.agents.supervisor import (
    SupervisorInput,
    run_supervisor_check,
)

__all__ = [
    "DEFAULT_DESIGN_PLAN_KEY_TEMPLATE",
    "DEFAULT_PAPER_PLAN_KEY_TEMPLATE",
    "CGMetadata",
    "CodeReviewerInput",
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
    "make_dispatch_planner",
    "run_code_review",
    "run_contextual_evaluation",
    "run_planner_session",
    "run_supervisor_check",
    "variant_for_submission_kind",
]
