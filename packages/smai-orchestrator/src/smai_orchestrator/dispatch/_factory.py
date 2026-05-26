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
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict
from smai_core.plugins import JobHandle, WorkspaceHandle

from smai_orchestrator.engine.types import (
    DispatchContext,
    DispatchHandler,
    DispatchOutcome,
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


def make_compute_dispatcher(
    *,
    role: str,
    image_resolver: Callable[[DispatchContext], Awaitable[str]],
    command_builder: Callable[[DispatchContext], Awaitable[CommandSpec]],
    inputs: WorkspaceInputs,
    outputs: WorkspaceOutputs,
    retry_policy: RetryPolicy | None = None,
) -> DispatchHandler:
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
        :data:`DispatchHandler` — an async callable of
        ``DispatchContext -> DispatchOutcome``.
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

        # ``outputs`` is the spec author's declaration of what gets
        # harvested on terminal. The harvest itself happens in phase-1
        # (post-terminal observation); the factory's submit-time half
        # only validates the shape exists. ``outputs`` is reserved on
        # the signature for symmetry with ``inputs`` and so the wire-up
        # point exists when later steps add the harvest hook.

        return DispatchOutcome(submitted_handles=[handle])

    return _dispatch
