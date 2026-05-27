"""File-operation tools: ``read_file`` / ``write_file`` / ``edit_file`` /
``search`` / ``list_files``.

Per ``04-agents.md`` §12.1. Inherited verbatim from v1
(``packages/agents/src/loop/tools.ts``) with v2 names (snake_case
Python) and Pydantic-typed inputs.

Path validation: every tool resolves the requested path against the
workspace root and rejects anything that resolves outside. The
"do-not-modify-harness" no-go-zone constraint
(``10-runtime-and-templates.md`` §7) is enforced at run time by the
runtime's hash check, not at tool dispatch — a write into ``harness/``
or ``experiment.py`` succeeds at the tool layer; the run fails before
any compute is consumed.

Lint-on-write integration: tools that write Python files
(``write_file`` / ``edit_file``) bind
:func:`smai_inline_agents.between_turn.lint_after_python_write` via
``Tool.post_result_hooks``. The hook keys off the parsed input's
``path`` attribute, so the input schemas below MUST expose ``path``
as an attribute name. v1's ``edit_file`` used ``oldString`` /
``newString`` keys (camelCase); v2 uses snake_case for both shape and
attribute name.
"""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from smai_core.plugins import ToolResultContent

from smai_inline_agents.between_turn import lint_after_python_write
from smai_inline_agents.tools import Tool, ToolContext

# Per §12.1 — soft caps on tool output to keep tool_result blocks small.
# The values match v1 verbatim (``packages/agents/src/loop/tools.ts``).
MAX_SEARCH_MATCHES = 100
MAX_LIST_ENTRIES = 200

# Read-file output cap — long files routinely blow context windows; v1 caps
# at 10,000 lines and shows the agent that it was truncated.
MAX_READ_LINES = 10_000

# Directories that are always hidden from ``list_files`` by name. v1
# settled on this set in ``packages/agents/src/loop/tools.ts``; the
# rationale is that these are infrastructure noise the agent should not
# spend turns on.
_LIST_HIDDEN_DIRS: frozenset[str] = frozenset({"__pycache__", "node_modules"})


READ_FILE_TOOL_NAME = "read_file"
WRITE_FILE_TOOL_NAME = "write_file"
EDIT_FILE_TOOL_NAME = "edit_file"
SEARCH_TOOL_NAME = "search"
LIST_FILES_TOOL_NAME = "list_files"


# =============================================================================
# Path resolution (workspace-escape safety)
# =============================================================================


class _PathEscapeError(ValueError):
    """Raised when a requested path resolves outside the workspace.

    Local to this module — the caller's job is to translate to
    :class:`ToolResultContent` with ``is_error=True``.
    """


def _resolve_within_workspace(workspace: Path, raw: str) -> Path:
    """Resolve ``raw`` against ``workspace`` and reject escapes.

    Mirrors v1's ``resolveSafePath``. Both relative and absolute paths
    are admitted as long as the result resolves inside the workspace —
    so that an agent that writes ``/workspace/foo.py`` (the absolute
    form on a typical Compute layout) works the same way as
    ``foo.py``.
    """
    raw_path = Path(raw)
    if raw_path.is_absolute():
        candidate = raw_path.resolve()
    else:
        candidate = (workspace / raw_path).resolve()
    workspace_resolved = workspace.resolve()
    try:
        candidate.relative_to(workspace_resolved)
    except ValueError as exc:
        raise _PathEscapeError(
            f"path {raw!r} resolves outside workspace; access denied",
        ) from exc
    return candidate


def _error_result(message: str) -> ToolResultContent:
    """Format a tool-result with ``is_error=True``.

    The loop backfills ``tool_use_id`` from the assistant's block, so
    the empty placeholder here is intentional — see
    :func:`smai_inline_agents.loop._execute_tool_uses`.
    """
    return ToolResultContent(tool_use_id="", content=message, is_error=True)


def _ok_result(message: str) -> ToolResultContent:
    return ToolResultContent(tool_use_id="", content=message, is_error=False)


# =============================================================================
# read_file (§12.1)
# =============================================================================


class ReadFileInput(BaseModel):
    """Schema for :func:`read_file` (§12.1).

    Fields:

    * ``path`` — workspace-relative path (or absolute under workspace).
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        ...,
        description="Path to the file, relative to the workspace root.",
    )


async def _read_file_handler(
    parsed_input: BaseModel,
    context: ToolContext,
) -> ToolResultContent:
    """Handler for :func:`read_file`.

    Returns the file contents with line numbers prepended (matching
    v1's ergonomic — agents get tab-separated ``<n>\\t<content>``
    lines so they can reference specific line numbers in subsequent
    edits).
    """
    if not isinstance(parsed_input, ReadFileInput):
        raise TypeError(
            f"read_file handler expected ReadFileInput, got {type(parsed_input).__name__}"
        )

    try:
        target = _resolve_within_workspace(context.workspace_path, parsed_input.path)
    except _PathEscapeError as exc:
        return _error_result(str(exc))

    if not target.exists():
        return _error_result(f"file not found: {parsed_input.path}")
    if target.is_dir():
        return _error_result(
            f"{parsed_input.path!r} is a directory; use list_files instead",
        )

    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return _error_result(
            f"file {parsed_input.path!r} is not valid UTF-8: {exc}",
        )
    except OSError as exc:
        return _error_result(
            f"failed to read {parsed_input.path!r}: {exc}",
        )

    lines = content.split("\n")
    if len(lines) > MAX_READ_LINES:
        kept = lines[:MAX_READ_LINES]
        numbered = "\n".join(f"{i + 1}\t{line}" for i, line in enumerate(kept))
        suffix = f"\n\n[truncated: showing first {MAX_READ_LINES} of {len(lines)} lines]"
        return _ok_result(f"{numbered}{suffix}")

    numbered = "\n".join(f"{i + 1}\t{line}" for i, line in enumerate(lines))
    return _ok_result(numbered)


def make_read_file_tool() -> Tool:
    """Build the :func:`read_file` tool (§12.1)."""
    return Tool(
        name=READ_FILE_TOOL_NAME,
        description=(
            "Read a file from the workspace. Takes one REQUIRED "
            "structured-input field: `path` (workspace-relative, or "
            "absolute as long as it resolves inside the workspace — "
            "omitting `path` surfaces as a Pydantic validation error). "
            "Output includes line numbers for use in subsequent "
            "edit_file calls. "
            f"Files longer than {MAX_READ_LINES:,} lines are truncated."
        ),
        input_schema=ReadFileInput,
        handler=_read_file_handler,
    )


# =============================================================================
# write_file (§12.1) — lint-on-write hook attaches in factory
# =============================================================================


class WriteFileInput(BaseModel):
    """Schema for :func:`write_file` (§12.1).

    Fields:

    * ``path`` — workspace-relative target path.
    * ``content`` — file content to write (overwrites if file exists).
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        ...,
        description="Path to write, relative to the workspace root.",
    )
    content: str = Field(
        ...,
        description="File content to write (overwrites any existing file).",
    )


async def _write_file_handler(
    parsed_input: BaseModel,
    context: ToolContext,
) -> ToolResultContent:
    """Handler for :func:`write_file`.

    Creates parent directories. ``write_file`` always overwrites; agents
    invoking it on an existing file see the bytes-written count in the
    success message, which is enough signal that the prior content is
    gone.
    """
    if not isinstance(parsed_input, WriteFileInput):
        raise TypeError(
            f"write_file handler expected WriteFileInput, got {type(parsed_input).__name__}"
        )

    try:
        target = _resolve_within_workspace(context.workspace_path, parsed_input.path)
    except _PathEscapeError as exc:
        return _error_result(str(exc))

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(parsed_input.content, encoding="utf-8")
    except OSError as exc:
        return _error_result(
            f"failed to write {parsed_input.path!r}: {exc}",
        )

    bytes_written = len(parsed_input.content.encode("utf-8"))
    return _ok_result(f"wrote {bytes_written} bytes to {parsed_input.path}")


def make_write_file_tool() -> Tool:
    """Build the :func:`write_file` tool (§12.1).

    Binds :func:`smai_inline_agents.between_turn.lint_after_python_write` as a
    post-result hook, per §3.2 / the 2.B1 carry-forward — ``.py``
    writes get ``ruff check`` output appended to the tool result.
    """
    return Tool(
        name=WRITE_FILE_TOOL_NAME,
        description=(
            "Write a file to the workspace. Takes two REQUIRED "
            "structured-input fields: `path` (workspace-relative target) "
            "and `content` (the file body, a string). Omitting either "
            "surfaces as a Pydantic validation error. Creates parent "
            "directories; overwrites any existing file at the path. "
            "For Python files, ruff check output is appended to the "
            "result."
        ),
        input_schema=WriteFileInput,
        handler=_write_file_handler,
        post_result_hooks=[lint_after_python_write],
    )


# =============================================================================
# edit_file (§12.1) — lint-on-write hook attaches in factory
# =============================================================================


class EditFileInput(BaseModel):
    """Schema for :func:`edit_file` (§12.1).

    Fields:

    * ``path`` — workspace-relative target path.
    * ``old_string`` — exact text to find. MUST occur exactly once.
    * ``new_string`` — replacement text.
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        ...,
        description="Path to edit, relative to the workspace root.",
    )
    old_string: str = Field(
        ...,
        description=(
            "Exact text to find and replace. Must occur exactly once "
            "in the file (whitespace and indentation count). Provide "
            "more surrounding context to disambiguate if needed."
        ),
    )
    new_string: str = Field(
        ...,
        description="Replacement text.",
    )


async def _edit_file_handler(
    parsed_input: BaseModel,
    context: ToolContext,
) -> ToolResultContent:
    """Handler for :func:`edit_file`.

    Per §12.1 / v1 behavior: ``old_string`` must match exactly once.
    Zero matches and multiple matches are both surfaced as tool errors
    so the agent can disambiguate by adding context.
    """
    if not isinstance(parsed_input, EditFileInput):
        raise TypeError(
            f"edit_file handler expected EditFileInput, got {type(parsed_input).__name__}"
        )

    try:
        target = _resolve_within_workspace(context.workspace_path, parsed_input.path)
    except _PathEscapeError as exc:
        return _error_result(str(exc))

    if not target.exists():
        return _error_result(f"file not found: {parsed_input.path}")
    if target.is_dir():
        return _error_result(
            f"{parsed_input.path!r} is a directory; cannot edit",
        )

    try:
        content = target.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        return _error_result(
            f"failed to read {parsed_input.path!r} for editing: {exc}",
        )

    occurrences = content.count(parsed_input.old_string)
    if occurrences == 0:
        return _error_result(
            f"old_string not found in {parsed_input.path!r}; check whitespace and indentation",
        )
    if occurrences > 1:
        return _error_result(
            f"old_string matches {occurrences} times in "
            f"{parsed_input.path!r}; provide more surrounding context "
            f"to make the match unique",
        )

    new_content = content.replace(parsed_input.old_string, parsed_input.new_string, 1)
    try:
        target.write_text(new_content, encoding="utf-8")
    except OSError as exc:
        return _error_result(
            f"failed to write {parsed_input.path!r}: {exc}",
        )

    return _ok_result(f"replaced 1 occurrence in {parsed_input.path}")


def make_edit_file_tool() -> Tool:
    """Build the :func:`edit_file` tool (§12.1)."""
    return Tool(
        name=EDIT_FILE_TOOL_NAME,
        description=(
            "Search-and-replace within a workspace file. Takes three "
            "REQUIRED structured-input fields: `path` (workspace-relative "
            "target file, REQUIRED — omitting it surfaces as a Pydantic "
            "validation error, not a tool error the loop will retry "
            "automatically), `old_string` (exact text to find — must "
            "occur exactly once; whitespace and indentation count, add "
            "surrounding context to disambiguate), and `new_string` "
            "(the replacement). The tool fails on zero or multiple "
            "matches. For Python files, ruff check output is appended "
            "to the result."
        ),
        input_schema=EditFileInput,
        handler=_edit_file_handler,
        post_result_hooks=[lint_after_python_write],
    )


# =============================================================================
# search (§12.1)
# =============================================================================


class SearchInput(BaseModel):
    """Schema for :func:`search` (§12.1).

    Fields:

    * ``pattern`` — Python ``re`` regex pattern.
    * ``path`` — optional workspace-relative subdirectory to search.
    * ``include`` — optional glob filter on file names (e.g., ``"*.py"``).
    """

    model_config = ConfigDict(extra="forbid")

    pattern: str = Field(
        ...,
        description="Python regex pattern to search for.",
    )
    path: str | None = Field(
        default=None,
        description=(
            "Subdirectory to search, relative to the workspace root. "
            "Omit to search the entire workspace."
        ),
    )
    include: str | None = Field(
        default=None,
        description=(
            "Glob pattern filtering which files to search (e.g., '*.py'). Omit to search all files."
        ),
    )


async def _search_handler(
    parsed_input: BaseModel,
    context: ToolContext,
) -> ToolResultContent:
    """Handler for :func:`search`.

    Pure-Python implementation — no ``ripgrep`` / ``grep`` dep. v1
    shelled to grep for performance; for v2's smaller workspace
    footprint and conformance-test reproducibility, a Python regex pass
    is fast enough and keeps the tool runnable in any container.
    """
    if not isinstance(parsed_input, SearchInput):
        raise TypeError(f"search handler expected SearchInput, got {type(parsed_input).__name__}")

    try:
        compiled = re.compile(parsed_input.pattern)
    except re.error as exc:
        return _error_result(f"invalid regex {parsed_input.pattern!r}: {exc}")

    base = context.workspace_path
    if parsed_input.path is not None:
        try:
            base = _resolve_within_workspace(context.workspace_path, parsed_input.path)
        except _PathEscapeError as exc:
            return _error_result(str(exc))
        if not base.exists():
            return _error_result(f"path not found: {parsed_input.path}")
        if not base.is_dir():
            return _error_result(
                f"{parsed_input.path!r} is a file, not a directory; search expects a directory",
            )

    workspace_resolved = context.workspace_path.resolve()
    matches: list[str] = []
    truncated = False

    for file_path in _walk_files(base, include=parsed_input.include):
        if len(matches) >= MAX_SEARCH_MATCHES:
            truncated = True
            break
        try:
            text = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            # Skip binary or unreadable files quietly.
            continue
        rel = file_path.resolve().relative_to(workspace_resolved)
        for line_idx, line in enumerate(text.split("\n"), start=1):
            if compiled.search(line):
                matches.append(f"{rel}:{line_idx}:{line}")
                if len(matches) >= MAX_SEARCH_MATCHES:
                    truncated = True
                    break

    if not matches:
        return _ok_result("no matches found")

    body = "\n".join(matches)
    if truncated:
        body += (
            f"\n\n[truncated: showing first {MAX_SEARCH_MATCHES} matches; "
            "narrow the pattern for fewer results]"
        )
    return _ok_result(f"{len(matches)} match(es):\n{body}")


def make_search_tool() -> Tool:
    """Build the :func:`search` tool (§12.1)."""
    return Tool(
        name=SEARCH_TOOL_NAME,
        description=(
            "Regex search across files in the workspace. Takes one "
            "REQUIRED structured-input field: `pattern` (a Python "
            "regex — omitting it surfaces as a Pydantic validation "
            "error). Two optional fields: `path` must be a DIRECTORY "
            "(the tool walks it recursively, not a single file; to "
            "grep a single file, use read_file and filter the result "
            "yourself); `include` is a filename glob like '*.py'. "
            "Returns '<path>:<line_number>:<content>' for each match. "
            f"Capped at {MAX_SEARCH_MATCHES} matches."
        ),
        input_schema=SearchInput,
        handler=_search_handler,
    )


# =============================================================================
# list_files (§12.1)
# =============================================================================


class ListFilesInput(BaseModel):
    """Schema for :func:`list_files` (§12.1).

    Fields:

    * ``path`` — optional workspace-relative subdirectory.
    * ``glob`` — optional glob filter on file names (e.g., ``"*.py"``).
    * ``recursive`` — whether to descend into subdirectories. Default ``True``.
    """

    model_config = ConfigDict(extra="forbid")

    path: str | None = Field(
        default=None,
        description=(
            "Directory to list, relative to the workspace root. Omit to list the workspace root."
        ),
    )
    glob: str | None = Field(
        default=None,
        description=(
            "Glob pattern filtering listed file names (e.g., '*.py'). Omit to list all files."
        ),
    )
    recursive: bool = Field(
        default=True,
        description="Whether to recurse into subdirectories.",
    )


async def _list_files_handler(
    parsed_input: BaseModel,
    context: ToolContext,
) -> ToolResultContent:
    """Handler for :func:`list_files`."""
    if not isinstance(parsed_input, ListFilesInput):
        raise TypeError(
            f"list_files handler expected ListFilesInput, got {type(parsed_input).__name__}"
        )

    base = context.workspace_path
    if parsed_input.path is not None:
        try:
            base = _resolve_within_workspace(context.workspace_path, parsed_input.path)
        except _PathEscapeError as exc:
            return _error_result(str(exc))
        if not base.exists():
            return _error_result(f"path not found: {parsed_input.path}")
        if not base.is_dir():
            return _error_result(
                f"{parsed_input.path!r} is a file, not a directory",
            )

    workspace_resolved = context.workspace_path.resolve()
    found: list[str] = []
    truncated = False

    for file_path in _walk_files(
        base,
        include=parsed_input.glob,
        recursive=parsed_input.recursive,
    ):
        if len(found) >= MAX_LIST_ENTRIES:
            truncated = True
            break
        rel = file_path.resolve().relative_to(workspace_resolved)
        found.append(str(rel))

    if not found:
        return _ok_result("no files found")

    body = "\n".join(found)
    if truncated:
        body += (
            f"\n\n[truncated: showing first {MAX_LIST_ENTRIES} entries; "
            "narrow the path or glob for fewer results]"
        )
    return _ok_result(body)


def make_list_files_tool() -> Tool:
    """Build the :func:`list_files` tool (§12.1)."""
    return Tool(
        name=LIST_FILES_TOOL_NAME,
        description=(
            "List files in the workspace. Takes `path` (optional "
            "subdirectory), `glob` (optional filename pattern like "
            "'*.py'), and `recursive` (bool, default True) as structured "
            "inputs; it does NOT accept a shell `command` argument, do "
            "not pass shell argv like 'find ...'. Returns paths relative "
            "to the workspace root, one per line. "
            f"Capped at {MAX_LIST_ENTRIES} entries."
        ),
        input_schema=ListFilesInput,
        handler=_list_files_handler,
    )


# =============================================================================
# Shared file-walk helper
# =============================================================================


def _walk_files(
    base: Path,
    *,
    include: str | None = None,
    recursive: bool = True,
) -> list[Path]:
    """Yield the files under ``base`` matching ``include``.

    Returns a list rather than an iterator so callers can compute the
    truncation state from ``len(...)``. Skips hidden directories
    (``.*``) and infrastructure noise (``__pycache__``, ``node_modules``)
    per v1's ``executeListFiles`` heuristic.
    """
    if not base.exists() or not base.is_dir():
        return []

    found: list[Path] = []
    if recursive:
        for root, dirnames, filenames in os.walk(base):
            # Prune hidden / noise dirs in place so os.walk doesn't
            # descend into them.
            dirnames[:] = [
                d for d in dirnames if not d.startswith(".") and d not in _LIST_HIDDEN_DIRS
            ]
            root_path = Path(root)
            for name in sorted(filenames):
                if include is not None and not fnmatch.fnmatch(name, include):
                    continue
                found.append(root_path / name)
    else:
        for entry in sorted(base.iterdir()):
            if not entry.is_file():
                continue
            if include is not None and not fnmatch.fnmatch(entry.name, include):
                continue
            found.append(entry)
    return found


__all__ = [
    "EDIT_FILE_TOOL_NAME",
    "LIST_FILES_TOOL_NAME",
    "MAX_LIST_ENTRIES",
    "MAX_READ_LINES",
    "MAX_SEARCH_MATCHES",
    "READ_FILE_TOOL_NAME",
    "SEARCH_TOOL_NAME",
    "WRITE_FILE_TOOL_NAME",
    "EditFileInput",
    "ListFilesInput",
    "ReadFileInput",
    "SearchInput",
    "WriteFileInput",
    "make_edit_file_tool",
    "make_list_files_tool",
    "make_read_file_tool",
    "make_search_tool",
    "make_write_file_tool",
]
