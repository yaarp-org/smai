"""Helper to assemble a :class:`ToolRegistry` populated with the
standard inventory.

Per ``04-agents.md`` §12. Different roles need different subsets of the
standard inventory:

* **Harness builder + technique implementer** — full inventory plus
  the loop-exit ``finish``.
* **Planner** — file ops + search + list_files + execute + finish.
  Excludes ``run_experiment`` per §12.3 last paragraph (planner's loop
  is in-memory reasoning, not GPU validation).
* **Single-call agents** (code reviewer / contextual evaluator /
  supervisor / screener / enricher) — none of these tools, only their
  structured-output tool via :func:`smai_agents.structured_call`.

The :func:`build_standard_tool_registry` helper consolidates the
construction call sites: 2.B3 / 2.B4's dispatch handlers compose the
registry by passing the role-specific exclusion set, so the standard
inventory's exact shape lives in one place.
"""

from __future__ import annotations

from typing import Literal

from smai_agents.std_tools.execute import EXECUTE_TOOL_NAME, make_execute_tool
from smai_agents.std_tools.files import (
    EDIT_FILE_TOOL_NAME,
    LIST_FILES_TOOL_NAME,
    READ_FILE_TOOL_NAME,
    SEARCH_TOOL_NAME,
    WRITE_FILE_TOOL_NAME,
    make_edit_file_tool,
    make_list_files_tool,
    make_read_file_tool,
    make_search_tool,
    make_write_file_tool,
)
from smai_agents.std_tools.run_experiment import (
    RUN_EXPERIMENT_TOOL_NAME,
    make_run_experiment_tool,
)
from smai_agents.tools import ToolRegistry, make_finish_tool

# Default placeholder image for ``run_experiment``. v1's convention was
# ``smai-runtime:dev`` (DEC-021 / Task 2.A4); deployments override at
# registration time.
DEFAULT_RUN_EXPERIMENT_IMAGE = "smai-runtime:dev"


# All tools in the standard inventory (excluding ``finish`` — that's a
# loop substrate primitive, not part of the inventory). The
# :class:`Literal` type lets exclusion sets be type-checked at the call
# site.
StandardToolName = Literal[
    "read_file",
    "write_file",
    "edit_file",
    "search",
    "list_files",
    "execute",
    "run_experiment",
]


_STANDARD_TOOL_NAMES: frozenset[str] = frozenset(
    {
        READ_FILE_TOOL_NAME,
        WRITE_FILE_TOOL_NAME,
        EDIT_FILE_TOOL_NAME,
        SEARCH_TOOL_NAME,
        LIST_FILES_TOOL_NAME,
        EXECUTE_TOOL_NAME,
        RUN_EXPERIMENT_TOOL_NAME,
    }
)


def build_standard_tool_registry(
    *,
    exclude: frozenset[StandardToolName] | set[StandardToolName] | None = None,
    include_finish: bool = True,
    run_experiment_image: str = DEFAULT_RUN_EXPERIMENT_IMAGE,
    run_experiment_gpu: bool = True,
    run_experiment_timeout_seconds: int | None = None,
    run_experiment_extra_env: dict[str, str] | None = None,
) -> ToolRegistry:
    """Build a :class:`ToolRegistry` populated with the standard inventory.

    Args:
        exclude: Tool names to omit. The planner role excludes
            ``"run_experiment"`` per §12.3 last paragraph; other roles
            typically pass ``None``.
        include_finish: Whether to register the loop-exit ``finish``
            tool. ``True`` for multi-turn agents; ``False`` for
            single-call agents that bypass the loop entirely.
        run_experiment_image: Container image for ``run_experiment``
            jobs. Threaded through to :func:`make_run_experiment_tool`.
        run_experiment_gpu: GPU flag for ``run_experiment`` jobs.
        run_experiment_timeout_seconds: Optional per-job timeout. ``None``
            uses :data:`run_experiment.RUN_EXPERIMENT_DEFAULT_TIMEOUT_SECONDS`.
        run_experiment_extra_env: Optional environment variables for
            ``run_experiment`` jobs.

    Returns:
        Populated :class:`ToolRegistry` ready to assign to
        :attr:`AgentSession.tools`.
    """
    excluded = frozenset(exclude or ())

    registry = ToolRegistry()

    if READ_FILE_TOOL_NAME not in excluded:
        registry.register(make_read_file_tool())
    if WRITE_FILE_TOOL_NAME not in excluded:
        registry.register(make_write_file_tool())
    if EDIT_FILE_TOOL_NAME not in excluded:
        registry.register(make_edit_file_tool())
    if SEARCH_TOOL_NAME not in excluded:
        registry.register(make_search_tool())
    if LIST_FILES_TOOL_NAME not in excluded:
        registry.register(make_list_files_tool())
    if EXECUTE_TOOL_NAME not in excluded:
        registry.register(make_execute_tool())

    if RUN_EXPERIMENT_TOOL_NAME not in excluded:
        from smai_agents.std_tools.run_experiment import (  # noqa: PLC0415
            RUN_EXPERIMENT_DEFAULT_TIMEOUT_SECONDS,
        )

        registry.register(
            make_run_experiment_tool(
                image=run_experiment_image,
                gpu=run_experiment_gpu,
                timeout_seconds=(
                    run_experiment_timeout_seconds
                    if run_experiment_timeout_seconds is not None
                    else RUN_EXPERIMENT_DEFAULT_TIMEOUT_SECONDS
                ),
                extra_env=run_experiment_extra_env,
            )
        )

    # ``finish`` is not in StandardToolName, so it is never in ``exclude``;
    # the only knob is ``include_finish``.
    if include_finish:
        registry.register(make_finish_tool())

    return registry


def standard_tool_names() -> frozenset[str]:
    """The set of standard-inventory tool names (excluding ``finish``)."""
    return _STANDARD_TOOL_NAMES


__all__ = [
    "DEFAULT_RUN_EXPERIMENT_IMAGE",
    "StandardToolName",
    "build_standard_tool_registry",
    "standard_tool_names",
]
