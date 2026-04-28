"""Between-turn deterministic logic for the agent loop.

Per ``04-agents.md`` §3.2. The behaviors that v1 (``agent_design.md``
§4) settled as belonging *between* turns — not inside the prompt, not
inside the tool, not inside the LLM client:

* :func:`maybe_truncate_context` — middle-truncate when conversation
  crosses §7's threshold.
* :func:`maybe_check_supervisor_nudge` — read a nudge file from
  :class:`ArtifactStore` and inject it as a user-role message with the
  literal ``[supervisor nudge]`` parsing convention prefix.
* :func:`write_status` — write the per-turn status JSON.
* :func:`lint_after_python_write` — :class:`ToolResultHook` registered
  by Task 2.B2 on ``write_file`` / ``edit_file`` that runs ``ruff``
  on freshly-written ``.py`` files and appends the output.
* :func:`wait_for_compute_job` — helper for tool handlers that submit a
  Compute job and need to block until terminal.

These are all between-turn because they are deterministic side effects
that should not depend on the agent remembering to do them and should
not consume turn capacity. v1's design rationale carries over verbatim.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from smai_core.plugins import (
    ArtifactNotFound,
    Compute,
    JobHandle,
    JobStatus,
    NormalizedMessage,
    TextContent,
    ToolResultContent,
)

from smai_agents.tools import Tool, ToolContext
from smai_agents.truncation import should_truncate, truncate_messages

if TYPE_CHECKING:
    from smai_agents.loop import AgentSession

logger = logging.getLogger(__name__)


# Parsing convention per §3.2 — the multi-turn agent prompts (B2 / B3)
# learn to recognize this prefix as a supervisor signal rather than
# genuine user input. NOT a security boundary; per §3.2 paragraph 5
# the convention is editorial, not adversarial.
SUPERVISOR_NUDGE_PREFIX = "[supervisor nudge]"


# Default poll interval for :func:`wait_for_compute_job`. Tool handlers
# pass through their own value when they want a tighter or looser cadence.
DEFAULT_COMPUTE_POLL_INTERVAL = 5.0


# === Truncation (§3.2 / §7) =================================================


async def maybe_truncate_context(session: AgentSession) -> None:
    """Middle-truncate the session's messages if §7's threshold fires.

    No-op when ``should_truncate(...)`` returns ``False``. When it does
    fire, drops the middle messages in place and increments
    :attr:`AgentSession.truncations_fired` for §3.4's debug logging.
    """
    capabilities = session.llm.capabilities
    if not should_truncate(
        messages=session.messages,
        capabilities=capabilities,
        policy=session.truncation_policy,
        system=session.system_prompt,
        tools=session.tools.to_provider_definitions(),
    ):
        return

    before = len(session.messages)
    session.messages = truncate_messages(session.messages, session.truncation_policy)
    after = len(session.messages)
    if before != after:
        session.truncations_fired += 1
        logger.info(
            "agent loop: truncated context (turn=%d, role=%s, before=%d msgs, "
            "after=%d msgs)",
            session.turn_count,
            session.current_role,
            before,
            after,
        )


# === Supervisor nudge polling (§3.2) ========================================


async def maybe_check_supervisor_nudge(session: AgentSession) -> None:
    """Read the nudge file from ArtifactStore and inject it on hit.

    Per §3.2: the orchestrator-driven supervisor agent (§2.6) writes a
    nudge file to a deployment-controlled path; the loop polls and, on
    hit, injects the nudge as
    ``role="user", content=[Text("[supervisor nudge] ...")]`` and
    deletes the file. The literal ``[supervisor nudge]`` prefix is
    a parsing convention the agent's prompt is taught to recognize.

    No-op when the session has no
    :attr:`AgentSession.nudge_artifact_path` configured or no
    :attr:`AgentSession.artifact_store`.
    """
    store = session.artifact_store
    path = session.nudge_artifact_path
    if store is None or path is None:
        return

    try:
        raw = await store.get(path)
    except ArtifactNotFound:
        return

    text = raw.decode("utf-8", errors="replace")
    nudge_message = NormalizedMessage(
        role="user",
        content=[TextContent(text=f"{SUPERVISOR_NUDGE_PREFIX} {text}")],
    )
    session.messages.append(nudge_message)
    session.nudges_consumed += 1
    logger.info(
        "agent loop: consumed supervisor nudge (turn=%d, role=%s)",
        session.turn_count,
        session.current_role,
    )

    # Best-effort delete — the nudge has been consumed so the next poll
    # shouldn't see it. ArtifactStore.delete is idempotent per
    # 07-plugin-interfaces.md §6.2.
    try:
        await store.delete(path)
    except Exception:  # noqa: BLE001 — delete failure is non-fatal
        logger.warning(
            "agent loop: failed to delete consumed supervisor nudge at %r",
            path,
            exc_info=True,
        )


# === Status writes (§3.2) ===================================================


async def write_status(session: AgentSession) -> None:
    """Write the per-turn status JSON to ArtifactStore.

    Per §3.2: "v1 wrote on every turn; v2 inherits this default." The
    status JSON lets the orchestrator's polling cycle and the supervisor
    agent (§2.6) observe progress. Schema is intentionally narrow —
    enough to power v1's stuck-detection signals (status unchanged 25+
    min, same error 3+ times, many turns without writes).

    No-op when no :attr:`AgentSession.artifact_store` /
    :attr:`AgentSession.status_artifact_path`. The loop's
    ``status_write_every_turns`` config controls cadence; this function
    runs unconditionally when called.
    """
    store = session.artifact_store
    path = session.status_artifact_path
    if store is None or path is None:
        return

    payload: dict[str, Any] = {
        "role": session.current_role,
        "turn_count": session.turn_count,
        "monotonic_timestamp": session.time_provider(),
        "usage_total": session.usage_total.model_dump(),
        "nudges_consumed": session.nudges_consumed,
        "truncations_fired": session.truncations_fired,
    }
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    await store.put(path, body, content_type="application/json")


# === Lint-on-write hook (§3.2 / §12.1) ======================================


async def lint_after_python_write(
    tool: Tool,
    parsed_input: BaseModel,
    result: ToolResultContent,
    context: ToolContext,
) -> ToolResultContent:
    """Append ``ruff`` output to the tool result for ``.py`` writes.

    Per §3.2: "When the agent writes or edits a ``.py`` file via
    ``write_file`` / ``edit_file``, run a linter (``ruff`` or
    ``pyflakes``; substrate-deferred) and append the output to the
    tool result. This catches syntax errors before the agent moves on.
    Inherited verbatim from v1." This hook is shipped here so 2.B2 can
    register it on its file-write tools without B1 needing to know
    about the specific tools.

    The hook is keyed by the parsed input — it expects the input model
    to expose a ``path`` attribute pointing at the file just written.
    Tools whose input shape is different just don't register this hook.

    On lint error, the hook does **not** mark ``is_error=True`` — a
    lint warning isn't a tool failure, just diagnostic output the
    agent should see. The agent decides whether to act on it.
    """
    del tool  # tool identity is not load-bearing for this hook
    path_field = getattr(parsed_input, "path", None)
    if not isinstance(path_field, str | Path) or not _is_python_file(path_field):
        return result

    abs_path = _resolve_workspace_relative(context.workspace_path, str(path_field))
    if abs_path is None or not abs_path.exists():
        return result

    ruff = shutil.which("ruff")
    if ruff is None:
        return result

    try:
        completed = await asyncio.to_thread(
            subprocess.run,
            [ruff, "check", "--output-format=concise", str(abs_path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except Exception:  # noqa: BLE001 — best-effort hook
        logger.warning("lint_after_python_write: ruff invocation failed", exc_info=True)
        return result

    output = (completed.stdout or "") + (completed.stderr or "")
    output = output.strip()
    if not output:
        return result

    appended = f"{result.content}\n\n[ruff check {abs_path.name}]\n{output}"
    return result.model_copy(update={"content": appended})


def _is_python_file(path: str | Path) -> bool:
    return str(path).endswith(".py")


def _resolve_workspace_relative(workspace_path: Path, raw: str) -> Path | None:
    """Resolve ``raw`` relative to the workspace, refusing escapes."""
    raw_path = Path(raw)
    if raw_path.is_absolute():
        candidate = raw_path.resolve()
    else:
        candidate = (workspace_path / raw_path).resolve()
    workspace_resolved = workspace_path.resolve()
    try:
        candidate.relative_to(workspace_resolved)
    except ValueError:
        return None
    return candidate


# === Compute job wait (§3.2 validation polling) =============================


async def wait_for_compute_job(
    *,
    handle: JobHandle,
    compute: Compute,
    time_provider: Callable[[], float],
    poll_interval: float = DEFAULT_COMPUTE_POLL_INTERVAL,
    max_wait_seconds: float | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> JobStatus:
    """Block until ``handle`` reaches a terminal :class:`JobState`.

    Used by the ``run_experiment`` tool (B2's territory) to surface
    validation results synchronously to the agent — per §3.2,
    "the agent doesn't manage the wait; the loop does."

    Polls :meth:`Compute.status` every ``poll_interval`` seconds; raises
    :class:`asyncio.TimeoutError` when ``max_wait_seconds`` elapses. The
    ``time_provider`` and ``sleep`` seams let tests fast-forward the
    poll loop deterministically.
    """
    sleep_fn = sleep if sleep is not None else asyncio.sleep
    start = time_provider()
    terminal_states = {"succeeded", "failed", "cancelled", "timeout"}
    while True:
        status = await compute.status(handle)
        if status.state in terminal_states:
            return status
        if max_wait_seconds is not None and (time_provider() - start) >= max_wait_seconds:
            raise TimeoutError(
                f"wait_for_compute_job: handle {handle.handle!r} did not reach a "
                f"terminal state within {max_wait_seconds}s "
                f"(last_state={status.state!r})"
            )
        await sleep_fn(poll_interval)


__all__ = [
    "DEFAULT_COMPUTE_POLL_INTERVAL",
    "SUPERVISOR_NUDGE_PREFIX",
    "lint_after_python_write",
    "maybe_check_supervisor_nudge",
    "maybe_truncate_context",
    "wait_for_compute_job",
    "write_status",
]
