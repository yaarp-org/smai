"""Custom Bedrock-Converse-style agent loop, prompt-config surface, fleet roles.

Public API surface across Tasks 2.B1 + 2.B2 + 2.B4. Per-role multi-turn
agent dispatch handlers (harness builder / technique implementer /
planner / supervisor) land in 2.B3 / 3.E1 / 3.G4.

The shape exposed here:

* Multi-turn loop driver (:func:`run_loop`) and its session shape
  (:class:`AgentSession`, :class:`AgentOutcome`, :class:`AgentLoopConfig`).
* Tool primitives (:class:`Tool`, :class:`ToolContext`,
  :class:`ToolRegistry`, :func:`make_finish_tool`,
  :func:`truncate_tool_output`).
* Standard tool inventory — file ops, search, list_files, execute,
  run_experiment — under :mod:`smai_agents.std_tools`. Convenience
  helper :func:`build_standard_tool_registry`.
* Prompt-config surface (2.B2) — YAML-on-disk + Pydantic-in-process —
  under :mod:`smai_agents.prompts`. Public entry points
  :func:`load_prompt_config`, :func:`render_initial_user_message`,
  :class:`PromptLayer`, :class:`ToolBinding`,
  :class:`StructuredOutputTool`. The full layered ``PromptConfig`` is
  reachable as :class:`smai_agents.prompts.PromptConfig` (the
  top-level :class:`PromptConfig` symbol is the narrower 2.B4 inline
  shape pending reconciliation).
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
from smai_agents.prompts import (
    PromptConfigError,
    PromptConfigNotFound,
    PromptConfigValidationError,
    PromptLayer,
    PromptLayerKind,
    StructuredOutputTool,
    ToolBinding,
    clear_prompt_config_cache,
    load_prompt_config,
    render_initial_user_message,
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
from smai_agents.std_tools import (
    EDIT_FILE_TOOL_NAME,
    EXECUTE_DEFAULT_TIMEOUT_SECONDS,
    EXECUTE_TOOL_NAME,
    LIST_FILES_TOOL_NAME,
    MAX_LIST_ENTRIES,
    MAX_SEARCH_MATCHES,
    MAX_VALIDATION_EPOCHS,
    MAX_VALIDATION_SUBSET,
    READ_FILE_TOOL_NAME,
    RUN_EXPERIMENT_TOOL_NAME,
    SEARCH_TOOL_NAME,
    WRITE_FILE_TOOL_NAME,
    EditFileInput,
    ExecuteInput,
    ListFilesInput,
    ReadFileInput,
    RunExperimentInput,
    RunExperimentResult,
    SearchInput,
    StandardToolName,
    WriteFileInput,
    build_standard_tool_registry,
    make_edit_file_tool,
    make_execute_tool,
    make_list_files_tool,
    make_read_file_tool,
    make_run_experiment_tool,
    make_search_tool,
    make_write_file_tool,
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
    "EDIT_FILE_TOOL_NAME",
    "EXECUTE_DEFAULT_TIMEOUT_SECONDS",
    "EXECUTE_MAX_LINES",
    "EXECUTE_MAX_TOKENS",
    "EXECUTE_TOOL_NAME",
    "FINISH_TOOL_NAME",
    "LIST_FILES_TOOL_NAME",
    "MAX_LIST_ENTRIES",
    "MAX_SEARCH_MATCHES",
    "MAX_VALIDATION_EPOCHS",
    "MAX_VALIDATION_SUBSET",
    "PREV_CONVERSATION_TRACE_FILENAME",
    "READ_FILE_TOOL_NAME",
    "RUN_EXPERIMENT_TOOL_NAME",
    "SEARCH_TOOL_NAME",
    "SINGLE_CALL_CACHE_CONFIG",
    "SUPERVISOR_NUDGE_PREFIX",
    "TASK_DEFAULTS",
    "WRITE_FILE_TOOL_NAME",
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
    "EditFileInput",
    "EntryRanking",
    "EntryUnderReview",
    "ExecuteInput",
    "Finding",
    "FinishInput",
    "ListFilesInput",
    "ModelSelectionError",
    "PromptConfig",
    "PromptConfigError",
    "PromptConfigNotFound",
    "PromptConfigValidationError",
    "PromptLayer",
    "PromptLayerKind",
    "ReadFileInput",
    "RunExperimentInput",
    "RunExperimentResult",
    "SearchInput",
    "StandardToolName",
    "StructuredCallFailed",
    "StructuredOutputTool",
    "TaskRole",
    "Tool",
    "ToolBinding",
    "ToolContext",
    "ToolHandler",
    "ToolRegistry",
    "ToolResultHook",
    "TruncationPolicy",
    "WriteFileInput",
    "build_standard_tool_registry",
    "clear_prompt_config_cache",
    "get_model_for_task",
    "lint_after_python_write",
    "load_prompt_config",
    "load_retry_context",
    "make_edit_file_tool",
    "make_execute_tool",
    "make_finish_tool",
    "make_list_files_tool",
    "make_read_file_tool",
    "make_run_experiment_tool",
    "make_search_tool",
    "make_write_file_tool",
    "render_initial_user_message",
    "run_code_review",
    "run_contextual_evaluation",
    "run_loop",
    "structured_call",
    "truncate_tool_output",
    "wait_for_compute_job",
]
