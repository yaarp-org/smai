"""Role-shaped wrappers around the structured-output substrate.

Per ``04-agents.md`` §2.4 / §2.5: the single-call agents (code reviewer,
contextual evaluator, supervisor) bypass the multi-turn loop and call
:func:`smai_agents.structured_call` directly. Each role here owns the
user-message assembly + prompt-config defaults +
``output_schema``/``tool_name`` selection; the structured-call substrate
itself is role-agnostic per §6.

Two of three single-call agents ship in 2.B4:

* :func:`smai_agents.agents.code_reviewer.run_code_review` — §2.4
* :func:`smai_agents.agents.contextual_evaluator.run_contextual_evaluation` — §2.5

The third (supervisor, §2.6) lands in Task 3.G4.
"""

from smai_agents.agents._prompt_config import (
    DEFAULT_CODE_REVIEW_PROMPT_CONFIG,
    DEFAULT_CONTEXTUAL_EVALUATOR_PROMPT_CONFIGS,
    PromptConfig,
)
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

__all__ = [
    "DEFAULT_CODE_REVIEW_PROMPT_CONFIG",
    "DEFAULT_CONTEXTUAL_EVALUATOR_PROMPT_CONFIGS",
    "CGMetadata",
    "CodeReviewerInput",
    "ContextualEvaluatorEntry",
    "ContextualEvaluatorInput",
    "EntryUnderReview",
    "PromptConfig",
    "run_code_review",
    "run_contextual_evaluation",
]
