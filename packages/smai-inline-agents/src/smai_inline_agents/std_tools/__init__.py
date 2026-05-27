"""Standard tool inventory for multi-turn agents.

Per ``04-agents.md`` §12. Seven tools, registered against the
:class:`smai_inline_agents.tools.Tool` primitive shipped in 2.B1:

* :func:`make_read_file_tool` — read a file from the workspace.
* :func:`make_write_file_tool` — write/overwrite a file.
* :func:`make_edit_file_tool` — search-and-replace within a file.
* :func:`make_search_tool` — regex search across workspace files.
* :func:`make_list_files_tool` — list files (optionally glob-filtered).
* :func:`make_execute_tool` — run a shell command in the workspace.
* :func:`make_run_experiment_tool` — DEC-021 structured GPU validation.

The ``finish`` tool ships from :mod:`smai_inline_agents.tools`
(:func:`smai_inline_agents.tools.make_finish_tool`); it is part of the loop
substrate, not the standard inventory.

The functions in this package are **factories** rather than singletons:
deployments parameterize per-tool defaults (image / GPU type / timeout
for ``run_experiment``, the lint-on-write hook binding for
``write_file`` / ``edit_file``, etc.) and the factory shape keeps the
configuration explicit at registration time.

:func:`build_standard_tool_registry` is the convenience helper —
constructs a :class:`smai_inline_agents.tools.ToolRegistry` populated with the
standard inventory minus tools the caller excludes (e.g., the planner
role excludes ``run_experiment`` per §12.3 last paragraph).
"""

from __future__ import annotations

from smai_inline_agents.std_tools.execute import (
    EXECUTE_DEFAULT_TIMEOUT_SECONDS,
    EXECUTE_TOOL_NAME,
    ExecuteInput,
    make_execute_tool,
)
from smai_inline_agents.std_tools.files import (
    EDIT_FILE_TOOL_NAME,
    LIST_FILES_TOOL_NAME,
    MAX_LIST_ENTRIES,
    MAX_SEARCH_MATCHES,
    READ_FILE_TOOL_NAME,
    SEARCH_TOOL_NAME,
    WRITE_FILE_TOOL_NAME,
    EditFileInput,
    ListFilesInput,
    ReadFileInput,
    SearchInput,
    WriteFileInput,
    make_edit_file_tool,
    make_list_files_tool,
    make_read_file_tool,
    make_search_tool,
    make_write_file_tool,
)
from smai_inline_agents.std_tools.registry import (
    DEFAULT_RUN_EXPERIMENT_CPU_IMAGE,
    DEFAULT_RUN_EXPERIMENT_IMAGE,
    StandardToolName,
    build_standard_tool_registry,
)
from smai_inline_agents.std_tools.run_experiment import (
    MAX_VALIDATION_EPOCHS,
    MAX_VALIDATION_SUBSET,
    RUN_EXPERIMENT_DEFAULT_TIMEOUT_SECONDS,
    RUN_EXPERIMENT_TOOL_NAME,
    RunExperimentInput,
    RunExperimentResult,
    make_run_experiment_tool,
)

__all__ = [
    "DEFAULT_RUN_EXPERIMENT_CPU_IMAGE",
    "DEFAULT_RUN_EXPERIMENT_IMAGE",
    "EDIT_FILE_TOOL_NAME",
    "EXECUTE_DEFAULT_TIMEOUT_SECONDS",
    "EXECUTE_TOOL_NAME",
    "LIST_FILES_TOOL_NAME",
    "MAX_LIST_ENTRIES",
    "MAX_SEARCH_MATCHES",
    "MAX_VALIDATION_EPOCHS",
    "MAX_VALIDATION_SUBSET",
    "READ_FILE_TOOL_NAME",
    "RUN_EXPERIMENT_DEFAULT_TIMEOUT_SECONDS",
    "RUN_EXPERIMENT_TOOL_NAME",
    "SEARCH_TOOL_NAME",
    "WRITE_FILE_TOOL_NAME",
    "EditFileInput",
    "ExecuteInput",
    "ListFilesInput",
    "ReadFileInput",
    "RunExperimentInput",
    "RunExperimentResult",
    "SearchInput",
    "StandardToolName",
    "WriteFileInput",
    "build_standard_tool_registry",
    "make_edit_file_tool",
    "make_execute_tool",
    "make_list_files_tool",
    "make_read_file_tool",
    "make_run_experiment_tool",
    "make_search_tool",
    "make_write_file_tool",
]
