"""Custom Bedrock-Converse-style agent loop, prompt-config surface, fleet roles.

Public API surface for Task 2.B1 — the engine + utilities. Per-role
agents and the prompt-config YAML loader land in 2.B2 / 2.B3 / 2.B4.

The shape exposed here matches the brief's deliverable list:

* Multi-turn loop driver (:func:`run_loop`) and its session shape
  (:class:`AgentSession`, :class:`AgentOutcome`, :class:`AgentLoopConfig`).
* Tool primitives (:class:`Tool`, :class:`ToolContext`,
  :class:`ToolRegistry`, :func:`make_finish_tool`,
  :func:`truncate_tool_output`).
* Single-call structured output (:func:`structured_call`,
  :class:`StructuredCallFailed`).
* Truncation policy (:class:`TruncationPolicy`).
* Cache config (:class:`CacheConfig`, :data:`DEFAULT_CACHE_CONFIG`,
  :data:`SINGLE_CALL_CACHE_CONFIG`).
* Per-task model selection (:data:`TaskRole`, :data:`TASK_DEFAULTS`,
  :func:`get_model_for_task`, :class:`ModelSelectionError`).
* DEC-023 retry-context helper (:func:`load_retry_context`,
  :data:`PREV_CONVERSATION_TRACE_FILENAME`).
* Between-turn helpers (:func:`wait_for_compute_job`,
  :func:`lint_after_python_write`,
  :data:`SUPERVISOR_NUDGE_PREFIX`).
"""

from smai_agents.agents import (
    DEFAULT_CODE_REVIEW_PROMPT_CONFIG,
    DEFAULT_CONTEXTUAL_EVALUATOR_PROMPT_CONFIGS,
    CGMetadata,
    CodeReviewerInput,
    ContextualEvaluatorEntry,
    ContextualEvaluatorInput,
    EntryUnderReview,
    PromptConfig,
    run_code_review,
    run_contextual_evaluation,
)
from smai_agents.between_turn import (
    SUPERVISOR_NUDGE_PREFIX,
    lint_after_python_write,
    wait_for_compute_job,
)
from smai_agents.cache import (
    DEFAULT_CACHE_CONFIG,
    SINGLE_CALL_CACHE_CONFIG,
    CacheConfig,
)
from smai_agents.loop import (
    AgentLoopConfig,
    AgentOutcome,
    AgentSession,
    run_loop,
)
from smai_agents.model_selection import (
    TASK_DEFAULTS,
    ModelSelectionError,
    TaskRole,
    get_model_for_task,
)
from smai_agents.retry_context import (
    PREV_CONVERSATION_TRACE_FILENAME,
    load_retry_context,
)
from smai_agents.schemas import (
    CodeReviewResult,
    ContextualVerdict,
    EntryRanking,
    Finding,
)
from smai_agents.structured_call import StructuredCallFailed, structured_call
from smai_agents.tools import (
    EXECUTE_MAX_LINES,
    EXECUTE_MAX_TOKENS,
    FINISH_TOOL_NAME,
    FinishInput,
    Tool,
    ToolContext,
    ToolHandler,
    ToolRegistry,
    ToolResultHook,
    make_finish_tool,
    truncate_tool_output,
)
from smai_agents.truncation import TruncationPolicy

__all__ = [
    "DEFAULT_CACHE_CONFIG",
    "DEFAULT_CODE_REVIEW_PROMPT_CONFIG",
    "DEFAULT_CONTEXTUAL_EVALUATOR_PROMPT_CONFIGS",
    "EXECUTE_MAX_LINES",
    "EXECUTE_MAX_TOKENS",
    "FINISH_TOOL_NAME",
    "PREV_CONVERSATION_TRACE_FILENAME",
    "SINGLE_CALL_CACHE_CONFIG",
    "SUPERVISOR_NUDGE_PREFIX",
    "TASK_DEFAULTS",
    "AgentLoopConfig",
    "AgentOutcome",
    "AgentSession",
    "CGMetadata",
    "CacheConfig",
    "CodeReviewResult",
    "CodeReviewerInput",
    "ContextualEvaluatorEntry",
    "ContextualEvaluatorInput",
    "ContextualVerdict",
    "EntryRanking",
    "EntryUnderReview",
    "Finding",
    "FinishInput",
    "ModelSelectionError",
    "PromptConfig",
    "StructuredCallFailed",
    "TaskRole",
    "Tool",
    "ToolContext",
    "ToolHandler",
    "ToolRegistry",
    "ToolResultHook",
    "TruncationPolicy",
    "get_model_for_task",
    "lint_after_python_write",
    "load_retry_context",
    "make_finish_tool",
    "run_code_review",
    "run_contextual_evaluation",
    "run_loop",
    "structured_call",
    "truncate_tool_output",
    "wait_for_compute_job",
]
