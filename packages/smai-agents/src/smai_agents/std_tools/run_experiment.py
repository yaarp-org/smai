"""``run_experiment`` tool — DEC-021 structured GPU validation.

Per ``04-agents.md`` §12.3 and DEC-021. The structured tool agents use
to run ``experiment.py`` validation jobs, *not* raw
``execute("python experiment.py …")`` — because:

* **Hard caps are enforced in tool code, not by prompt.** ``MAX_VALIDATION_EPOCHS``
  / ``MAX_VALIDATION_SUBSET`` are clamped at the tool layer; an agent
  asking for ``epochs=100`` gets ``epochs=5`` silently (with a
  ``[capped]`` note in the result).
* **Sandbox boundary is enforced by the Compute plugin** (per
  ``07-plugin-interfaces.md`` §7). The tool calls
  :meth:`smai_core.plugins.Compute.submit` and
  :func:`smai_agents.between_turn.wait_for_compute_job`; the agent
  never sees the substrate.
* **Result shape is structured** — :class:`RunExperimentResult` carries
  the parsed ``metrics.json`` and a synthesized ``run_status`` block.
  Failed runs return ``completed=False`` plus a ``failure_reason``
  string the agent can reason about deterministically.

The tool is registered for the harness builder and the technique
implementer (§12.3 last paragraph). The planner does not register it —
its loop is in-memory reasoning, not GPU validation.

Image / GPU / timeout selection are deployment-specific knobs baked in
at registration time via :func:`make_run_experiment_tool`. The factory
shape lets 2.B3's dispatch handler register the tool with the per-CG
configured image without 2.B2 having to know about specific images.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from smai_core.plugins import (
    ComputeError,
    JobStatus,
    ToolResultContent,
)

from smai_agents.between_turn import wait_for_compute_job
from smai_agents.tools import Tool, ToolContext

# Per DEC-021 / §12.3 — hard caps enforced in tool code. The values
# match v1's settled defaults (``packages/agents/src/loop/tools.ts``):
# 5 epochs and 0.2 subset. These are absolute ceilings — a deployment
# that wants tighter caps configures them at the harness/runtime layer
# (validation runs ALSO honor harness ``--epochs`` /  ``--subset``
# clamping per ``10-runtime-and-templates.md`` §10.1), not by relaxing
# these.
MAX_VALIDATION_EPOCHS = 5
MAX_VALIDATION_SUBSET = 0.2

# Default substrate timeout for a validation run. v1 settled on 600s
# for Modal (per DEC-021 / ``modal_migration.md``); v2 inherits.
RUN_EXPERIMENT_DEFAULT_TIMEOUT_SECONDS = 600

# Default poll interval the wait helper uses while the job runs.
# Tighter than :data:`smai_agents.between_turn.DEFAULT_COMPUTE_POLL_INTERVAL`
# because validation runs are short and the agent is blocked waiting
# for the result.
RUN_EXPERIMENT_POLL_INTERVAL = 5.0

RUN_EXPERIMENT_TOOL_NAME = "run_experiment"

# File names the runtime writes after a validation run, per
# ``10-runtime-and-templates.md`` §2 / §10.1. ``metrics.json`` is the
# canonical output; ``validation_results.json`` is the validation-mode
# structured result (loss decreased / metrics in range / NaN-OOM
# absent). The tool harvests both when present.
_METRICS_FILENAME = "metrics.json"
_VALIDATION_RESULTS_FILENAME = "validation_results.json"


# =============================================================================
# Input + result schemas (§12.3 verbatim)
# =============================================================================


class RunExperimentInput(BaseModel):
    """Schema for :func:`run_experiment` (§12.3 verbatim).

    Fields:

    * ``technique`` — technique module name (e.g., ``"baseline"``,
      ``"cutout"``). ``None`` means the additive baseline (per
      DEC-013 / ``CLAUDE.md`` "Nullable techniqueId").
    * ``seed`` — RNG seed for reproducibility.
    * ``epochs`` — capped at :data:`MAX_VALIDATION_EPOCHS`.
    * ``subset_fraction`` — capped at :data:`MAX_VALIDATION_SUBSET`.
    """

    model_config = ConfigDict(extra="forbid")

    technique: str | None = Field(
        default=None,
        description=(
            "Technique module name (must match a file in techniques/). "
            "Pass null for the additive baseline (no technique applied)."
        ),
    )
    seed: int = Field(
        ...,
        description="RNG seed for reproducibility.",
    )
    epochs: int = Field(
        default=MAX_VALIDATION_EPOCHS,
        description=(f"Number of training epochs (capped at {MAX_VALIDATION_EPOCHS})."),
        ge=1,
    )
    subset_fraction: float = Field(
        default=MAX_VALIDATION_SUBSET,
        description=(f"Fraction of dataset to use (capped at {MAX_VALIDATION_SUBSET})."),
        gt=0.0,
        le=1.0,
    )


class RunExperimentResult(BaseModel):
    """Result schema for :func:`run_experiment` (§12.3 verbatim).

    Per §12.3:

    * ``completed`` — ``True`` iff the substrate reported state
      ``succeeded`` AND the runtime wrote ``metrics.json``.
    * ``metrics`` — parsed ``metrics.json`` (or ``None`` if missing).
    * ``run_status`` — synthesized from :class:`JobStatus`
      (state / exit_code / started_at / finished_at). Per §12.3's
      Pydantic typing, values are ``str | int | None``.
    * ``failure_reason`` — populated on the failed-run path; carries
      the substrate's ``failure_reason`` plus an excerpt of the
      validation_results.json error context if present.
    """

    model_config = ConfigDict(extra="forbid")

    completed: bool
    metrics: dict[str, float | int] | None
    run_status: dict[str, str | int | None]
    failure_reason: str | None


# =============================================================================
# Handler factory (deployments parameterize image / gpu / timeout)
# =============================================================================


# Factory-time hook that lets tests inject a custom wait helper. Default
# is :func:`smai_agents.between_turn.wait_for_compute_job`; tests pass
# a fast-clock variant.
_DefaultWaitHelper = Callable[..., Awaitable[JobStatus]]


def make_run_experiment_tool(
    *,
    image: str,
    gpu: bool,
    timeout_seconds: int = RUN_EXPERIMENT_DEFAULT_TIMEOUT_SECONDS,
    poll_interval: float = RUN_EXPERIMENT_POLL_INTERVAL,
    extra_env: dict[str, str] | None = None,
    plugin_options: dict[str, Any] | None = None,
    wait_helper: _DefaultWaitHelper | None = None,
) -> Tool:
    """Build the :func:`run_experiment` tool (§12.3).

    Args:
        image: Container image to dispatch (e.g., ``"smai-runtime:dev"``
            for ``LocalGpuCompute``). The Compute plugin's eager image
            validation surfaces a malformed image as
            :class:`smai_core.plugins.JobImageInvalid`.
        gpu: Whether the validation run requires a GPU. Required (no
            default) because picking the wrong value silently —
            ``gpu=True`` on a CPU-only experiment, ``gpu=False`` on a
            GPU-bound one — wedges the agent on its first
            ``run_experiment`` call. The session runners
            (:func:`run_harness_builder_session` /
            :func:`run_technique_implementer_session`) derive this from
            :attr:`HarnessContractBody.compute.gpu` so the agent never
            needs to reason about it.
        timeout_seconds: Substrate-enforced job timeout. The tool also
            wraps the wait helper with the same value as the wall-clock
            budget so a substrate that fails to enforce its timeout
            doesn't hang the agent loop.
        poll_interval: Seconds between :meth:`Compute.status` polls.
        extra_env: Optional environment variables passed to the
            substrate. The substrate-set "validation mode" flag goes
            here when applicable per
            ``10-runtime-and-templates.md`` §10.1's mode-signal note.
        plugin_options: Optional ``**plugin_options`` forwarded to
            :meth:`Compute.submit`. Substrate-specific (Modal volumes,
            S3 mount config, etc.).
        wait_helper: Test-only seam. Defaults to
            :func:`smai_agents.between_turn.wait_for_compute_job`.

    Returns:
        A :class:`Tool` whose handler dispatches a validation run
        through :class:`Compute`, harvests ``metrics.json`` from the
        workspace, and returns a :class:`RunExperimentResult` formatted
        as JSON in the tool result.
    """
    base_env: dict[str, str] = dict(extra_env or {})
    options: dict[str, Any] = dict(plugin_options or {})
    actual_wait = wait_helper or wait_for_compute_job

    async def _handler(
        parsed_input: BaseModel,
        context: ToolContext,
    ) -> ToolResultContent:
        if not isinstance(parsed_input, RunExperimentInput):
            raise TypeError(
                f"run_experiment handler expected RunExperimentInput, "
                f"got {type(parsed_input).__name__}"
            )

        compute = context.compute
        if compute is None:
            return _error_result(
                "run_experiment requires a Compute plugin on the agent "
                "session; the dispatch handler did not provide one"
            )

        # Apply hard caps per §12.3. Track which knobs got clamped so
        # the agent sees the cap in the result body.
        epochs = min(parsed_input.epochs, MAX_VALIDATION_EPOCHS)
        subset = min(parsed_input.subset_fraction, MAX_VALIDATION_SUBSET)
        capped: list[str] = []
        if epochs != parsed_input.epochs:
            capped.append(f"epochs capped from {parsed_input.epochs} to {epochs}")
        if subset != parsed_input.subset_fraction:
            capped.append(f"subset_fraction capped from {parsed_input.subset_fraction} to {subset}")

        command = _build_command(
            technique=parsed_input.technique,
            seed=parsed_input.seed,
            epochs=epochs,
            subset_fraction=subset,
        )

        # The workspace mount is a session-internal concern, not an
        # agent-tunable option: the container must see the SAME workspace
        # the agent is writing experiment.py / harness/ / techniques/ into,
        # so ``_harvest_result`` can read back metrics.json /
        # validation_results.json from the same host path. Without this,
        # LocalGpuCompute's container has no bind-mount and the run fails
        # with "can't open file '/workspace/experiment.py'". The
        # tool-controlled value must win over anything in ``options`` so a
        # caller / agent cannot redirect the mount to a different
        # directory.
        #
        # ModalCompute.submit silently ignores unknown ``plugin_options``
        # keys (it reads only gpu_type / cpu / memory_mb; confirmed at
        # plugins/smai-compute-modal/src/smai_compute_modal/_compute.py
        # ~line 171-173), so passing ``workspace=...`` is a no-op on
        # Modal. TODO: Modal-substrate ``run_experiment`` needs a separate
        # workspace-distribution mechanism (Modal volumes, or runtime
        # fetch from ArtifactStore) before validation runs there can see
        # the agent's workspace.
        merged_options = {**options, "workspace": str(context.workspace_path)}
        try:
            handle = await compute.submit(
                image=image,
                command=command,
                env=base_env,
                gpu=gpu,
                timeout_seconds=timeout_seconds,
                **merged_options,
            )
        except ComputeError as exc:
            return _failed_dispatch_result(
                reason=f"{type(exc).__name__}: {exc}",
                capped=capped,
            )

        try:
            status = await actual_wait(
                handle=handle,
                compute=compute,
                time_provider=context.session.time_provider,
                poll_interval=poll_interval,
                # ``timeout_seconds`` is the substrate's budget; we wait
                # a bit longer wall-clock-wise so a slow-to-report
                # substrate doesn't trip our ceiling before its own.
                max_wait_seconds=timeout_seconds + poll_interval * 2,
            )
        except TimeoutError as exc:
            return _failed_dispatch_result(
                reason=f"wait_for_compute_job timed out: {exc}",
                capped=capped,
            )
        except ComputeError as exc:
            return _failed_dispatch_result(
                reason=f"{type(exc).__name__}: {exc}",
                capped=capped,
            )

        result = await _harvest_result(
            workspace=context.workspace_path,
            status=status,
            capped=capped,
        )
        return _result_to_tool_result(result, capped=capped)

    return Tool(
        name=RUN_EXPERIMENT_TOOL_NAME,
        description=(
            "Run experiment.py for validation. Takes one REQUIRED "
            "structured-input field: `seed` (an int — omitting it "
            "surfaces as a Pydantic validation error). Optional fields: "
            "`technique` (module name; null for the additive baseline), "
            "`epochs`, `subset_fraction`. Submits the job through the "
            "configured Compute plugin (image + GPU configured at "
            f"registration time). Caps epochs at {MAX_VALIDATION_EPOCHS} "
            f"and subset_fraction at {MAX_VALIDATION_SUBSET}. Returns "
            "structured metrics + run_status; failures carry a "
            "failure_reason instead of free-text logs."
        ),
        input_schema=RunExperimentInput,
        handler=_handler,
    )


# =============================================================================
# Helpers
# =============================================================================


def _build_command(
    *,
    technique: str | None,
    seed: int,
    epochs: int,
    subset_fraction: float,
) -> list[str]:
    """Construct the ``experiment.py`` argv for the substrate.

    Per ``10-runtime-and-templates.md`` §3.1: ``experiment.py`` parses
    ``--technique``, ``--seed``, optional ``--epochs`` / ``--subset``.
    The additive-baseline case (``technique is None``) omits
    ``--technique`` so the runtime branches into the no-op default.

    ``--mode validation`` is always passed: this tool is the agents'
    in-loop check, never the orchestrator's seed-run dispatcher (that
    path goes through :mod:`smai_orchestrator.specs.run_record`). The
    runtime's full-mode contract requires the
    :class:`HarnessAPIManifest` to be present at
    ``contracts/harness_api_manifest.json`` — but the harness builder
    runs validation BEFORE emitting the manifest (manifest is the
    outcome of a passing validation per ``04-agents.md`` §9 step 5),
    so without ``--mode validation`` the run aborts with
    ``contract artifact missing``. Validation mode also tells the runtime
    to write ``validation_results.json`` on success, which
    :func:`smai_agents.agents.manifest_tool` reads to decide whether
    the manifest may be emitted (§9 step 3).
    """
    argv: list[str] = [
        "python",
        "experiment.py",
        "--mode",
        "validation",
        "--seed",
        str(seed),
    ]
    if technique is not None:
        argv.extend(["--technique", technique])
    argv.extend(["--epochs", str(epochs)])
    argv.extend(["--subset", str(subset_fraction)])
    return argv


def _status_to_run_status(status: JobStatus) -> dict[str, str | int | None]:
    """Render a :class:`JobStatus` as the run_status dict §12.3 specs."""
    return {
        "state": status.state,
        "exit_code": status.exit_code,
        "started_at": status.started_at,
        "finished_at": status.finished_at,
        "failure_reason": status.failure_reason,
    }


async def _harvest_result(
    *,
    workspace: Path,
    status: JobStatus,
    capped: list[str],
) -> RunExperimentResult:
    """Harvest ``metrics.json`` / ``validation_results.json`` post-run."""
    del capped  # caller threads cap notes through ``_result_to_tool_result``

    run_status = _status_to_run_status(status)
    metrics: dict[str, float | int] | None = None
    failure_reason: str | None = status.failure_reason

    metrics_path = workspace / _METRICS_FILENAME
    if metrics_path.exists():
        metrics = _load_metrics(metrics_path)
        if metrics is None:
            failure_reason = (
                f"metrics.json present but unreadable; substrate state={status.state!r}"
            )

    validation_path = workspace / _VALIDATION_RESULTS_FILENAME
    validation_payload: str | None = None
    if validation_path.exists():
        try:
            validation_payload = validation_path.read_text(encoding="utf-8")[:2000]
        except OSError:
            validation_payload = None

    succeeded = status.state == "succeeded" and metrics is not None
    if not succeeded and failure_reason is None:
        failure_reason = f"run did not produce metrics.json (state={status.state!r})"
    if validation_payload is not None and not succeeded:
        # Append a snippet of validation_results.json on failure for
        # debugging — the agent may use it to decide whether to retry
        # with a different config or escalate.
        failure_reason = f"{failure_reason}\n--- validation_results.json ---\n{validation_payload}"

    return RunExperimentResult(
        completed=succeeded,
        metrics=metrics,
        run_status=run_status,
        failure_reason=failure_reason,
    )


def _load_metrics(path: Path) -> dict[str, float | int] | None:
    """Read ``metrics.json`` and coerce to ``dict[str, float | int]``.

    Uses a Pydantic adapter to do the heavy lifting — the type-narrowing
    dance pyright would otherwise demand for a hand-rolled
    ``dict[Unknown, Unknown]`` walk is bypassed by letting Pydantic
    validate the canonical flat-numeric-dict shape per
    ``10-runtime-and-templates.md`` §10.1.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        return _METRICS_ADAPTER.validate_json(raw)
    except ValidationError:
        return None


# Pydantic v2 TypeAdapter for the canonical metrics.json shape. Booleans
# are JSON-distinct from numbers but ``isinstance(True, int)`` is True
# in Python, so the adapter's ``strict=True`` mode rejects them.
_METRICS_ADAPTER: TypeAdapter[dict[str, float | int]] = TypeAdapter(
    dict[str, float | int],
    config=ConfigDict(strict=True),
)


def _result_to_tool_result(
    result: RunExperimentResult,
    *,
    capped: list[str],
) -> ToolResultContent:
    """Encode :class:`RunExperimentResult` as a :class:`ToolResultContent`.

    The body is the JSON-serialized result so an agent reading the
    tool_result can parse it deterministically. Cap notes (when knobs
    were clamped) prepend the body so the agent can see them up front.
    """
    payload = result.model_dump_json()
    if capped:
        cap_lines = "\n".join(f"[capped] {entry}" for entry in capped)
        payload = f"{cap_lines}\n{payload}"
    is_error = not result.completed
    return ToolResultContent(tool_use_id="", content=payload, is_error=is_error)


def _failed_dispatch_result(
    *,
    reason: str,
    capped: list[str],
) -> ToolResultContent:
    """Produce a tool result for the no-substrate path (submit/wait failure)."""
    result = RunExperimentResult(
        completed=False,
        metrics=None,
        run_status={
            "state": "failed",
            "exit_code": None,
            "started_at": None,
            "finished_at": None,
            "failure_reason": reason,
        },
        failure_reason=reason,
    )
    return _result_to_tool_result(result, capped=capped)


def _error_result(message: str) -> ToolResultContent:
    return ToolResultContent(tool_use_id="", content=message, is_error=True)


__all__ = [
    "MAX_VALIDATION_EPOCHS",
    "MAX_VALIDATION_SUBSET",
    "RUN_EXPERIMENT_DEFAULT_TIMEOUT_SECONDS",
    "RUN_EXPERIMENT_POLL_INTERVAL",
    "RUN_EXPERIMENT_TOOL_NAME",
    "RunExperimentInput",
    "RunExperimentResult",
    "make_run_experiment_tool",
]
