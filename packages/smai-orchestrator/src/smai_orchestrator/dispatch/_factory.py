"""Unified ``make_compute_dispatcher`` factory.

Implements the factory sketch from
``designs/smai/agent_refactor/compute_dispatch_decisions.md`` §3:

.. code-block:: python

    make_compute_dispatcher(
        role,             # TaskRole literal, for retry-policy + role-model lookup
        image_resolver,   # agent-runtime vs experiment-runtime image
        command_builder,  # python -m smai_agent_runtime.X vs python -m smai_runtime.runner
        inputs,           # files to push in (WorkspaceInputs)
        outputs,          # files to harvest on exit (WorkspaceOutputs)
        retry_policy,     # round-10's declarative shape on DispatchAction
    )

The handler internals are deliberately small — per the lineage statement
(``architectural_decisions.md`` §5) the engine already owns retry counter
bumps, retry-exhausted terminal synthesis, and the write-first dispatch
ordering. The factory is the seam that lets every dispatcher (seed-run,
harness-builder, technique-implementer) share substrate without
re-implementing those mechanics.

Round-20 generalization: container stderr surfaces uniformly into
``last_error`` on terminal-failure transitions. The factory's failure
path on submit-time errors (``Compute.submit`` raising or returning no
handle) is already covered by the engine's
:func:`_handle_dispatch_failure`; the post-submit terminal-failure
surface lives in :func:`smai_orchestrator.engine.phase1_step` (also
extended in this PR).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict
from smai_core.plugins import JobHandle, WorkspaceHandle

from smai_orchestrator.engine.types import (
    DispatchContext,
    DispatchHandler,
    DispatchOutcome,
    PostTerminalContext,
    PostTerminalHandler,
    RetryPolicy,
)

# Tail length for stderr surfaces (round-20 generalization). 100 lines
# per ``implementation_plan.md`` Step 2 (suggest 100). Lives at module
# scope so callers can reach it from tests.
STDERR_TAIL_LINES: int = 100


# === Workspace I/O shape ====================================================


class WorkspaceInputs(BaseModel):
    """Description of the workspace contents pushed into the sandbox.

    The factory consults :attr:`resolver` (an async callable) at dispatch
    time. ``None`` means "no workspace staging" — the runtime image is
    self-contained and the substrate does not need a host directory
    bind-mounted in. Seed-run dispatch uses this shape (the runtime
    image carries everything; per-run inputs are passed via env vars).

    When non-``None``, :attr:`resolver` returns a host :class:`Path`; the
    factory then calls :meth:`Compute.stage_workspace` and threads the
    returned :class:`WorkspaceHandle` through :meth:`Compute.submit`'s
    ``workspace=`` plugin_option.

    The callable returns ``None`` to opt out per-dispatch (e.g., a
    harness-builder dispatch that has no contract on disk yet — the
    factory then submits without ``workspace=``).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    resolver: Callable[[DispatchContext], Awaitable[Path | None]] | None = None

    @classmethod
    def empty(cls) -> WorkspaceInputs:
        """Empty inputs: skip ``stage_workspace`` entirely.

        Used by the seed-run dispatcher migration in Step 2 — seed runs
        carry everything they need in the runtime image plus env vars.
        """
        return cls(resolver=None)


class WorkspaceOutputs(BaseModel):
    """Description of the workspace contents harvested from the sandbox.

    The factory does NOT call :meth:`Compute.harvest_workspace` at
    submit time — the harvest happens on terminal-state observation
    (phase-1) when the job has actually produced output. This type
    carries the *configuration* that the spec author wires up; the
    factory's contribution at Step 2 is the type's shape (and the
    submit-time validation that ``workspace_distribution`` is
    compatible).

    :attr:`destination` returns the host path to harvest INTO. When
    ``None``, no harvest is performed. Seed-run dispatch uses this
    shape — the runtime publishes to ``ArtifactStore`` via host-side
    keys (``RUN_METRICS_KEY_TEMPLATE``); no per-run workspace pull-out
    is needed today.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    destination: Callable[[DispatchContext], Awaitable[Path | None]] | None = None

    @classmethod
    def empty(cls) -> WorkspaceOutputs:
        """Empty outputs: skip ``harvest_workspace`` entirely."""
        return cls(destination=None)


class CommandSpec(BaseModel):
    """Container-command spec the dispatcher passes to :meth:`Compute.submit`.

    The :data:`command_builder` callable returns one of these per
    dispatch. The factory threads the fields into the ``submit`` call.

    :attr:`env` is plugin-substrate environment variables (HF tokens,
    LLM provider keys for agent dispatch, seed-run identity vars for
    the seed-run case). :attr:`gpu` and :attr:`timeout_seconds` mirror
    :meth:`Compute.submit`'s signature.

    :attr:`plugin_options` is a free-form dict the factory passes
    through as kwargs. Pre-populated by the factory with
    ``workspace=<WorkspaceHandle>`` when staging fired; callers should
    NOT pre-populate ``workspace`` here (the factory's value wins, per
    the round-18 / round-20 same-PR discipline).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    command: list[str]
    env: dict[str, str]
    gpu: bool = False
    timeout_seconds: int = 3600
    plugin_options: dict[str, Any] = {}  # noqa: RUF012 — pydantic deep-copies field defaults; safe.


# === Helpers =================================================================


def format_stderr_tail(logs: str, *, max_lines: int = STDERR_TAIL_LINES) -> str:
    """Truncate ``logs`` to the last ``max_lines`` newline-separated lines.

    Returns the tail with a ``"... (truncated, N lines hidden) ..."``
    prefix when truncation occurred. Empty input returns the empty
    string. Used by the round-20 generalization in
    :func:`smai_orchestrator.engine.phase1_step` to attach container
    stderr to ``last_error`` on terminal-failure transitions, and by
    the unit tests that pin the consistency-of-use contract.
    """
    if not logs:
        return ""
    lines = logs.splitlines()
    if len(lines) <= max_lines:
        return logs
    hidden = len(lines) - max_lines
    return f"... (truncated, {hidden} lines hidden) ...\n" + "\n".join(lines[-max_lines:])


# === Factory =================================================================


@dataclass(frozen=True)
class DispatcherBundle:
    """A dispatch handler plus its optional post-terminal-success hook.

    Returned by :func:`make_compute_dispatcher`. The two callables are
    a pair: the handler submits the Compute job; the post-terminal
    handler (if any) runs once on phase-1's terminal-success
    observation, after the engine has CAS'd the entity into the
    success-target state. The pairing exists so the factory can wire a
    workspace-harvest invocation alongside the submit shape without
    requiring callers to thread the :class:`WorkspaceOutputs`
    declaration through the engine themselves.

    Sub-PR B (agent-refactor Step 4): introduced so the
    harness_builder dispatch can declare ``outputs`` (harness/,
    techniques/baseline.py, etc.) and have phase-1 invoke
    :meth:`Compute.harvest_workspace` on success without leaking the
    harvest concern into the spec author's wiring beyond a single
    field on :class:`DispatchAction`.

    :attr:`post_terminal_handler` is ``None`` when
    :attr:`WorkspaceOutputs.destination` is ``None`` (the seed-run
    shape — no harvest configured).
    """

    handler: DispatchHandler
    post_terminal_handler: PostTerminalHandler | None = None


def make_compute_dispatcher(
    *,
    role: str,
    image_resolver: Callable[[DispatchContext], Awaitable[str]],
    command_builder: Callable[[DispatchContext], Awaitable[CommandSpec]],
    inputs: WorkspaceInputs,
    outputs: WorkspaceOutputs,
    retry_policy: RetryPolicy | None = None,
) -> DispatcherBundle:
    """Produce a :data:`DispatchHandler` that submits a :class:`Compute`
    job per the unified shape (agent_refactor §3 / §5).

    The returned handler is wired into a :class:`DispatchAction` by the
    caller — alongside the ``handle_field`` (which entity column holds
    the :class:`JobHandle`) and the ``pool`` (which concurrency pool
    accounts the in-flight slot). The factory does not set those; they
    are spec-author concerns.

    Args:
        role: :data:`smai_agents.TaskRole` literal (typed as ``str`` here
            to keep the orchestrator's dependency direction one-way per
            DEC-026 / ``tools/check_deps.py``). The role is reserved for
            future per-role policy lookups (per-step model selection in
            Step 4; agent-image selection); the seed-run migration
            in Step 2 does not consume it at runtime.
        image_resolver: Async callable returning the container image
            tag for this dispatch (e.g., GPU vs CPU runtime image per
            the seed-run :class:`HarnessContract` body; agent-runtime
            image for agent dispatches). The dispatch context carries
            :attr:`DispatchContext.artifact_store` so resolvers can read
            artifacts (seed-run's GPU flag).
        command_builder: Async callable returning the :class:`CommandSpec`
            for this dispatch.
        inputs: :class:`WorkspaceInputs` description. ``empty()`` means
            no host workspace is staged.
        outputs: :class:`WorkspaceOutputs` description. ``empty()`` means
            no harvest is performed on terminal. Step 2's seed-run
            migration uses both empty; agent dispatches in later steps
            populate both.
        retry_policy: Round-10's declarative shape. Reserved on the
            factory signature so callers carry the policy alongside the
            handler; the engine reads it directly off the
            :class:`DispatchAction`, not from a handler attribute.
            Pass-through here keeps the factory's signature complete per
            ``compute_dispatch_decisions.md`` §3.

    Returns:
        :class:`DispatcherBundle` carrying the
        :data:`DispatchHandler` and the optional
        :data:`PostTerminalHandler`. Callers wire the handler onto
        :attr:`DispatchAction.handler` and the post-terminal handler
        onto :attr:`DispatchAction.post_terminal_handler`. When
        :attr:`WorkspaceOutputs.destination` is ``None`` the bundle's
        :attr:`post_terminal_handler` is ``None`` (seed-run shape).
    """
    # ``retry_policy`` is part of the documented factory signature
    # (compute_dispatch §3) but is not consumed by the handler body —
    # the engine reads it off ``DispatchAction.retry_policy``. ``role``
    # is reserved for future per-role policy lookups (per-step model
    # selection in Step 4; agent-image selection). Both are accepted
    # to match the design sketch verbatim.
    _ = retry_policy
    _ = role

    async def _dispatch(ctx: DispatchContext) -> DispatchOutcome:
        image = await image_resolver(ctx)
        spec = await command_builder(ctx)

        plugin_options: dict[str, Any] = dict(spec.plugin_options)

        staged: WorkspaceHandle | None = None
        if inputs.resolver is not None:
            local_path = await inputs.resolver(ctx)
            if local_path is not None:
                staged = await ctx.compute.stage_workspace(local_path)
                # The factory's value wins over any caller-supplied
                # ``workspace`` in ``plugin_options`` — round-18 / round-20
                # same-PR discipline (the dispatcher owns workspace
                # threading; agent-supplied options must not override
                # session-internal correctness).
                plugin_options["workspace"] = staged

        handle: JobHandle = await ctx.compute.submit(
            image=image,
            command=spec.command,
            env=spec.env,
            gpu=spec.gpu,
            timeout_seconds=spec.timeout_seconds,
            **plugin_options,
        )

        # ``outputs`` declares what gets harvested on terminal. The
        # harvest itself fires in phase-1 (post-terminal observation)
        # via the bundle's ``post_terminal_handler`` below; the
        # factory's submit-time half only stages the inputs side.

        return DispatchOutcome(submitted_handles=[handle])

    post_terminal: PostTerminalHandler | None = (
        _make_workspace_harvest_handler(outputs) if outputs.destination is not None else None
    )

    return DispatcherBundle(handler=_dispatch, post_terminal_handler=post_terminal)


def _make_workspace_harvest_handler(outputs: WorkspaceOutputs) -> PostTerminalHandler:
    """Build the :data:`PostTerminalHandler` that calls
    :meth:`Compute.harvest_workspace` on terminal-success observation.

    Sub-PR B's harvest path is the bind-mount-substrate happy path:
    LocalGpu's :meth:`Compute.stage_workspace` is identity on its host
    path and :meth:`Compute.harvest_workspace` is a no-op (the host
    already sees what the container wrote via the mount). The handler
    re-derives the :class:`WorkspaceHandle` from the destination path
    via a fresh ``stage_workspace`` call rather than persisting the
    original handle across dispatch ↔ phase-1; this works under
    bind-mount semantics by construction.

    Upload-download substrates (Modal) require the original
    :class:`WorkspaceHandle` (with the substrate's volume identifier)
    to harvest. Sub-PR B does not persist the workspace handle across
    the dispatch ↔ phase-1 boundary — the persisted entity carries
    only the :class:`JobHandle`. A future sub-PR that exercises Modal
    against this hook needs handle persistence (D2's note on Modal
    ``Sandbox.from_id`` covers the equivalent for the JobHandle side).
    TODO at the architectural_decisions §7 "host attests-and-persists"
    boundary.

    Best-effort: any exception is allowed to escape so phase-1 can
    catch and log without blocking the state-machine transition.
    Phase-1 wraps the call in a try/except for that reason.
    """

    async def _post_terminal(ctx: PostTerminalContext) -> None:
        if outputs.destination is None:  # defensive — bundle skips this case
            return
        dest = await outputs.destination(ctx.dispatch_context)
        if dest is None:
            return
        dest.mkdir(parents=True, exist_ok=True)
        handle = await ctx.compute.stage_workspace(dest)
        await ctx.compute.harvest_workspace(handle, dest)

    return _post_terminal
