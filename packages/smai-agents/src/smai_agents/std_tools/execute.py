"""``execute`` tool — run a shell command in the workspace.

Per ``04-agents.md`` §12.2. Aggressive output truncation per §3.3 /
§12.2 — the last 200 lines, capped at 5000 tokens — applied via
:func:`smai_agents.tools.truncate_tool_output`.

Per DEC-021 / §12.2: a regex guard rejects commands that invoke
``experiment.py`` directly (``python experiment.py``,
``python -m experiment``, ``./experiment.py``). The guard's purpose is
**discoverability and prompt-reinforcement**, not security: the agent's
prompt is the primary mechanism keeping ``experiment.py`` out of
``execute``; the regex backstop surfaces violations as a tool error so
the agent self-corrects to ``run_experiment`` rather than silently
running unbounded training jobs.

Sandboxing is the :class:`smai_core.plugins.Compute` plugin's
responsibility (per ``07-plugin-interfaces.md`` §7), not this tool's —
this is the v2 carry-forward of v1's "execute is a thin shell-out, the
boundary is the container" stance.
"""

from __future__ import annotations

import asyncio
import re

from pydantic import BaseModel, ConfigDict, Field
from smai_core.plugins import ToolResultContent

from smai_agents.tools import Tool, ToolContext, truncate_tool_output

# Default per-command timeout. v1 used 120 seconds; v2 inherits.
EXECUTE_DEFAULT_TIMEOUT_SECONDS = 120

# Hard ceiling on per-command timeout — shielded against an agent that
# tries to set a multi-hour timeout for what should be a quick
# diagnostic command. Long-running compute belongs in
# :func:`run_experiment` / Compute.submit, not here.
EXECUTE_MAX_TIMEOUT_SECONDS = 600

# Combined-output buffer cap. Mirrors v1's 10MB ceiling — the
# truncation step trims to ~5000 tokens before returning, but we want
# a cap on raw bytes too so a runaway process can't flood memory.
EXECUTE_MAX_OUTPUT_BYTES = 10 * 1024 * 1024

EXECUTE_TOOL_NAME = "execute"

# Per DEC-021 / §12.2: reject commands that invoke ``experiment.py``
# directly. The regex covers the three common shapes v1 catalogued in
# ``packages/agents/src/loop/tools.ts``:
#
#   1. ``python experiment.py …`` / ``python3 experiment.py …``
#   2. ``python -m experiment`` / ``python3 -m experiment`` (module form)
#   3. ``./experiment.py …`` (executable form)
#
# v1's regex also caught a few partial paths; v2's set is broader to
# also catch invocations from a subdirectory (``./harness/experiment.py``)
# and ``uv run experiment.py``. This is a discoverability hint, not a
# security boundary; the cost of a few false positives is low.
_EXPERIMENT_PY_GUARD = re.compile(
    r"(?:^|[\s/])(?:python(?:3)?(?:\s+-m\s+experiment(?:\s|$)|\s+\S*experiment\.py)|"
    r"\S*experiment\.py(?:\s|$)|"
    r"uv\s+run\s+\S*experiment\.py)",
    re.IGNORECASE,
)


class ExecuteInput(BaseModel):
    """Schema for :func:`execute` (§12.2).

    Fields:

    * ``command`` — shell command (passed to ``/bin/sh -c``).
    * ``timeout_seconds`` — optional per-call timeout, default 120s.
    """

    model_config = ConfigDict(extra="forbid")

    command: str = Field(
        ...,
        description=(
            "Shell command to execute in the workspace. The command "
            "runs via /bin/sh -c with the workspace as cwd. "
            "Output is truncated to the last 200 lines / ~5000 tokens."
        ),
    )
    timeout_seconds: int | None = Field(
        default=None,
        description=(
            "Per-call timeout in seconds (default 120, max 600). "
            "Long-running training runs belong in run_experiment, not here."
        ),
        ge=1,
        le=EXECUTE_MAX_TIMEOUT_SECONDS,
    )


def _error_result(message: str) -> ToolResultContent:
    return ToolResultContent(tool_use_id="", content=message, is_error=True)


def _ok_result(message: str) -> ToolResultContent:
    return ToolResultContent(tool_use_id="", content=message, is_error=False)


async def _execute_handler(
    parsed_input: BaseModel,
    context: ToolContext,
) -> ToolResultContent:
    """Handler for :func:`execute`.

    Runs the command via ``asyncio.create_subprocess_shell`` so the loop's
    event loop stays cooperative — long-running commands don't block
    other agent sessions sharing the worker.
    """
    if not isinstance(parsed_input, ExecuteInput):
        raise TypeError(
            f"execute handler expected ExecuteInput, "
            f"got {type(parsed_input).__name__}"
        )

    if _EXPERIMENT_PY_GUARD.search(parsed_input.command):
        # Per §12.2 — surface as a tool error so the agent self-corrects
        # to run_experiment. NOT a security boundary; the regex is
        # narrow on purpose.
        return _error_result(
            "execute does not run experiment.py — use the run_experiment "
            "tool instead. run_experiment enforces hard caps on epochs / "
            "subset and submits the run through the Compute plugin where "
            "the sandbox boundary applies."
        )

    timeout = parsed_input.timeout_seconds or EXECUTE_DEFAULT_TIMEOUT_SECONDS

    try:
        process = await asyncio.create_subprocess_shell(
            parsed_input.command,
            cwd=str(context.workspace_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=None,  # inherit
        )
    except OSError as exc:
        return _error_result(f"failed to spawn shell: {exc}")

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout,
        )
    except TimeoutError:
        # Kill the runaway process; harvest whatever it wrote so far.
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            pass
        return _error_result(
            f"command exceeded {timeout}s timeout and was killed",
        )

    exit_code = process.returncode if process.returncode is not None else -1

    stdout_text = _decode_output(stdout_bytes)
    stderr_text = _decode_output(stderr_bytes)

    parts: list[str] = [f"exit code: {exit_code}"]
    combined: list[str] = []
    if stdout_text:
        combined.append(stdout_text)
    if stderr_text:
        combined.append(f"---stderr---\n{stderr_text}")
    body = "\n".join(combined) if combined else ""
    truncated = truncate_tool_output(body)

    if truncated:
        parts.extend(["", truncated])
    payload = "\n".join(parts)

    # Non-zero exit isn't a tool failure (the agent often runs commands
    # expecting failures and reads the output). Surface as is_error=False;
    # the agent reads exit_code from the body.
    return _ok_result(payload)


def _decode_output(raw: bytes) -> str:
    if len(raw) > EXECUTE_MAX_OUTPUT_BYTES:
        # Keep the tail — that's where the failure typically surfaces.
        raw = raw[-EXECUTE_MAX_OUTPUT_BYTES:]
    return raw.decode("utf-8", errors="replace")


def make_execute_tool() -> Tool:
    """Build the :func:`execute` tool (§12.2)."""
    return Tool(
        name=EXECUTE_TOOL_NAME,
        description=(
            "Run a shell command in the workspace. Returns exit code, "
            "stdout, and stderr. Output is truncated to the last 200 "
            "lines / ~5000 tokens. Commands that invoke experiment.py "
            "are rejected — use the run_experiment tool for GPU "
            "validation runs."
        ),
        input_schema=ExecuteInput,
        handler=_execute_handler,
    )


__all__ = [
    "EXECUTE_DEFAULT_TIMEOUT_SECONDS",
    "EXECUTE_MAX_OUTPUT_BYTES",
    "EXECUTE_MAX_TIMEOUT_SECONDS",
    "EXECUTE_TOOL_NAME",
    "ExecuteInput",
    "make_execute_tool",
]
