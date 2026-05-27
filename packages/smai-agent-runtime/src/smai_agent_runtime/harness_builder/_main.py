"""Harness-builder sandbox-side mini-orchestrator.

Agent-refactor Step 4 sub-PR C2 replaces sub-PR B's remaining three
fake handlers (ValidationStep, DiagnoseOnFailureStep, ManifestEmitStep)
with real implementations:

* ``ValidationStep`` runs ``python experiment.py --mode validation``
  via real ``subprocess.run`` and captures the stderr tail.
* ``DiagnoseOnFailureStep`` builds a D7c :class:`DiagnoseBundle`, calls
  a PydanticAI Agent bound to :class:`Diagnosis`, and applies the
  ``fix_payload`` to the named ``fix_target``. ``read_file`` is the
  only tool exposed and is workspace-scoped per architectural_decisions
  §12 #1.
* ``ManifestEmitStep`` recomputes ``harness_version_hash`` from the
  current on-disk ``harness/`` contents (round-15 discipline; never
  reuse a stale hash from a prior turn), builds the
  :class:`HarnessAPIManifest`, freezes it, and writes ``manifest.json``
  to the workspace.

D10's stdout-JSON status emit lands on the workflow's step iteration:
``session.start`` / ``step.start`` / ``step.end`` / ``session.cost`` /
``session.end`` events stream to stdout and to
``status/events.jsonl`` per architectural_decisions §7.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, cast

from jinja2 import UndefinedError
from pydantic import BaseModel, ValidationError
from smai_core.artifacts.harness_contract import HarnessContract
from smai_core.artifacts.technique_contract import TechniqueContract
from smai_runtime.components import ADMISSIBLE_PATTERNS_FOR_KEY, COMPONENT_FIELD_FOR_KEY
from smai_runtime.manifest import (
    MANIFEST_SCHEMA_VERSION,
    HarnessAPIManifest,
    HarnessExtensionPoint,
    compute_harness_version_hash,
    freeze_manifest,
)
from smai_runtime.no_go_zone import RUNTIME_TEMPLATE_VERSION

from smai_agent_runtime.agent_reasoning import build_agent, get_model_for_step
from smai_agent_runtime.agent_reasoning.model_selection import (
    OverrideMap,
    SandboxedRole,
)
from smai_agent_runtime.prompts import load_step_prompt
from smai_agent_runtime.prompts._loader import render_user_message
from smai_agent_runtime.schemas import (
    READ_FILE_BUDGET_PER_ATTEMPT,
    READ_FILE_MAX_BYTES,
    STDERR_TAIL_BYTES,
    STDERR_TAIL_LINES,
    DiagnoseBundle,
    Diagnosis,
    ExtensionPointSpec,
    FunctionSignature,
    GroundingContext,
    HarnessBuilderBodyGenerationInput,
    HarnessBuilderBodyGenerationOutput,
    HarnessOriginalBundleRef,
    LintFailure,
    NoOpBaselineGrounding,
    OriginalInputBundleRef,
    PaperExtractGrounding,
    PriorDiagnosis,
    PriorFailedAttempt,
    PriorTechniqueAttempt,
    ProposalGrounding,
    ReviewerAttestedGrounding,
    StandardLibraryGrounding,
    TechniqueBodyGenerationBundle,
    TechniqueBodyOutput,
)
from smai_agent_runtime.status import (
    StatusEmitter,
    WorkflowPlanItem,
    WorkflowStepKind,
    _step_label_for_kind,
)
from smai_agent_runtime.workflow.generator import TaskRole, generate_workflow
from smai_agent_runtime.workflow.step_types import (
    BaselineGenerationStep,
    DiagnoseOnFailureStep,
    HarnessBuilderBodyGenerationStep,
    ManifestEmitStep,
    TechniqueImplementerBodyGenerationStep,
    ValidationStep,
    WorkflowStep,
)

# Sandbox-side exit codes. The host's ``Compute.status(handle).exit_code``
# reads these.
EXIT_OK = 0
EXIT_STEP_FAILED = 1
EXIT_RESUME_STUB = 66
EXIT_BAD_WORKSPACE = 65

# Backwards-compatible alias for sub-PR B's exit-code name. Sub-PR D
# renames the constant to ``EXIT_RESUME_STUB`` (the architectural hedge
# is "resume entry signature exists; workflow logic is a stub"); the
# old name remains a re-export so importers staged between sub-PRs
# don't break in lockstep.
EXIT_RESUME_NOT_IMPLEMENTED = EXIT_RESUME_STUB

# Hardcoded role for this module; the per-step model resolution uses it
# to look up the right ``SMAI_MODEL_HARNESS_BUILDER__<STEP>`` env var.
_ROLE: SandboxedRole = "harness_builder"

# Bounded body-generation retries on lint failure per architectural_decisions
# §12 #1 + research_report §8.5 (retries pinned at 3 across all roles).
_MAX_LINT_RETRIES: int = 3

# Cap the lint output the agent sees so retry bundles stay below the
# 50KB-token budget. ~3KB head+tail matches the D7a schema's docstring
# convention.
_LINT_OUTPUT_CAP_CHARS: int = 3000

# Validation subprocess timeout. CPU-smoke validation in the sandbox is
# expected to land in seconds; the cap is for safety, not throughput.
_VALIDATION_TIMEOUT_SECONDS: int = 300

# Default seed if the contract carries no seeds (defensive — the generator
# already defaults to 0 in that case).
_DEFAULT_SEED: int = 0

# Validation runner CLI: the in-container ``python experiment.py`` driver
# with ``--mode validation`` per round-20.
_VALIDATION_TECHNIQUE_NAME: str = "baseline"

# Dependency-injected fake-LLM runner (sub-PR D thread 5). The
# ``--fake-llm`` argparse flag swaps this in at main()-entry boundary so
# cross-process subprocess tests can exercise the mini-orchestrator
# without real LLM credentials. The pre-sub-PR-D
# ``SMAI_AGENT_RUNTIME_FAKE_LLM`` env-var read was deleted because
# env-var-as-mode-switch is the kind of smell that hides in production
# code paths.

# Type alias for the per-step agent runner. Tests can substitute a fake;
# production wires the real PydanticAI run via :func:`_real_agent_run`.
AgentRunner = Callable[[Any, str, type[Any]], Any]


@dataclass
class _StepOutcome:
    """One step's result. Mirrors D9's sketch shape."""

    step_index: int
    step_type: str
    succeeded: bool
    error: str | None = None
    # ValidationStep populates this on failure so the diagnose step's
    # bundle assembler has the canonical stderr-tail without re-reading
    # the workspace.
    captured_stderr: str | None = None


@dataclass
class _DispatchContext:
    """Per-session immutable context the dispatcher threads through every step."""

    cg_id: str
    workspace: Path
    contract: HarnessContract
    technique_contract: TechniqueContract | None
    overrides: OverrideMap | None
    # Populated lazily in :func:`main` once the workspace is resolved.
    # Held on context so step handlers can emit lifecycle events without
    # passing the emitter through every helper signature.
    status: StatusEmitter | None = None
    # Records the workflow step shape so D7c bundle assembly can
    # back-reference the body-generation step that produced the failing
    # code. Filled by :func:`_dispatch_step` before each step runs.
    body_step_kinds: dict[int, str] = field(default_factory=dict[int, str])
    # Dependency-injected agent runner (sub-PR D thread 5). ``None``
    # means "use the real PydanticAI run path"; tests substitute a
    # fake via the ``--fake-llm`` flag at the entry-point boundary.
    # In-process tests still monkeypatch :func:`_run_agent_sync`
    # directly for finer-grained per-call assertions.
    agent_runner: AgentRunner | None = None


def main(args: argparse.Namespace) -> int:
    """Entry point invoked by :mod:`smai_agent_runtime.__main__`."""
    if args.cg_id is None:
        _emit_status_line("error", reason="harness_builder requires --cg-id")
        return EXIT_BAD_WORKSPACE

    if args.resume is not None:
        return _resume_main(
            cg_id=args.cg_id,
            workspace_arg=args.workspace,
            resume_from=args.resume,
        )

    workspace = _resolve_workspace(args.workspace)
    if workspace is None:
        _emit_status_line(
            "error",
            cg_id=args.cg_id,
            reason=f"--workspace path {args.workspace!r} does not exist or is not a directory",
        )
        return EXIT_BAD_WORKSPACE

    contract = _load_contract(workspace)
    if contract is None:
        _emit_status_line(
            "error",
            cg_id=args.cg_id,
            reason=(
                "no harness_contract.json under contracts/ in the staged workspace; "
                "host-side materialization must run before sandbox dispatch"
            ),
        )
        return EXIT_BAD_WORKSPACE

    workflow = generate_workflow(contract, TaskRole.HARNESS_BUILDER)

    # Construct the status emitter once the workspace is resolved so
    # subsequent emits land in ``status/events.jsonl`` next to the agent
    # workspace's outputs (host harvests both per Step 1's
    # ``harvest_workspace``).
    session_id = f"agent-session-{args.cg_id}-{uuid.uuid4().hex[:8]}"
    emitter = StatusEmitter(session_id=session_id, workspace=workspace)
    workflow_plan = [
        WorkflowPlanItem(
            step_index=idx,
            step_kind=step.step_type,
            step_label=_label_for_step(step),
        )
        for idx, step in enumerate(workflow)
    ]
    emitter.emit_session_start(
        agent_role="harness_builder",
        parent_kind="cg",
        parent_id=args.cg_id,
        workflow_plan=workflow_plan,
    )

    context = _DispatchContext(
        cg_id=args.cg_id,
        workspace=workspace,
        contract=contract,
        technique_contract=_load_technique_contract(workspace),
        overrides=None,  # sub-PR D wires the host-side override map.
        status=emitter,
        agent_runner=_fake_agent_runner if getattr(args, "fake_llm", False) else None,
    )

    outcomes: list[_StepOutcome] = []
    last_succeeded_index: int | None = None
    for index, step in enumerate(workflow):
        step_kind: WorkflowStepKind = step.step_type
        # Capture per-index step kind for downstream D7c bundle assembly
        # so DiagnoseOnFailureStep can resolve the body-generation step
        # that produced the failing code (used by
        # ``_resolve_original_bundle_ref``).
        context.body_step_kinds[index] = step.step_type

        emitter.emit_step_start(
            step_index=index,
            step_kind=step_kind,
            step_label=_label_for_step(step),
            input_summary=_input_summary_for(step),
        )
        start_t = time.monotonic()
        outcome = _dispatch_step(step, index, context, outcomes)
        elapsed = time.monotonic() - start_t
        outcomes.append(outcome)

        emitter.emit_step_end(
            step_index=index,
            step_kind=step_kind,
            outcome="success" if outcome.succeeded else "failure",
            duration_seconds=elapsed,
            failure_reason=outcome.error if not outcome.succeeded else None,
            captured_stderr=outcome.captured_stderr,
        )

        if outcome.succeeded:
            last_succeeded_index = index
        else:
            # Round-21 finding (2026-05-26): the outer loop used to
            # fail-fast on any failure, which prevented the diagnose
            # step (architectural_decisions §6 step 7) from ever firing
            # after a validation failure. Diagnose's anchor-step-index
            # pattern already handles "anchor succeeded → pass through"
            # internally; the outer loop only needs to keep iterating
            # past a ValidationStep failure so the next-in-line
            # DiagnoseOnFailureStep gets its chance. Body-gen / baseline
            # / manifest-emit failures stay fail-fast since no
            # downstream step is designed to handle them.
            if not isinstance(step, ValidationStep):
                emitter.emit_session_cost(
                    input_tokens=0,
                    output_tokens=0,
                    cache_read_tokens=0,
                    cache_write_tokens=0,
                    turn_count=0,
                    tool_errors_fired=0,
                )
                emitter.emit_session_end(
                    outcome="failure",
                    last_completed_step_index=last_succeeded_index,
                    failure_reason=outcome.error or "unknown",
                )
                _write_status_summary(workspace, args.cg_id, outcomes, succeeded=False)
                return EXIT_STEP_FAILED

    emitter.emit_session_cost(
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        turn_count=0,
        tool_errors_fired=0,
    )
    emitter.emit_session_end(
        outcome="success",
        last_completed_step_index=last_succeeded_index,
    )
    _write_status_summary(workspace, args.cg_id, outcomes, succeeded=True)
    return EXIT_OK


# === Resume-prep hedge (architectural_decisions §12 #4) =====================


def _resume_main(
    *,
    cg_id: str,
    workspace_arg: object,
    resume_from: str,
) -> int:
    """Stub resume-mode entry that loads prior state and exits.

    The architectural hedge per architectural_decisions §12 item 4: the
    refactor does not ship the multi-cycle review-feedback loop, but the
    mini-orchestrator entry signature accepts ``--resume <prior_session_id>``
    so a future PR can wire the workflow logic without retrofitting the
    argparse + workspace + state-load surface. Sub-PR D fills in the
    prior-state load; the actual "do something with the loaded
    message_history" lands later.

    Loads:

    * ``conversation_traces/*.json`` files from the staged workspace
      (written by :func:`_persist_message_history` during prior runs).
    * Workspace contents (harness/, techniques/, validation_results.json,
      manifest.json) are already on disk via the host-side workspace
      stager; the resume path reads them in place.

    Emits a structured ``resume_loaded`` event with the loaded step list,
    followed by a ``resume_stub`` event indicating no workflow logic
    ran, then exits :data:`EXIT_RESUME_STUB`.
    """
    workspace = _resolve_workspace(workspace_arg)
    if workspace is None:
        _emit_status_line(
            "error",
            cg_id=cg_id,
            resume_from=resume_from,
            reason=(
                f"--workspace path {workspace_arg!r} does not exist or is not "
                "a directory; resume-mode requires a staged workspace"
            ),
        )
        return EXIT_BAD_WORKSPACE

    traces_dir = workspace / "conversation_traces"
    loaded_steps: list[str] = []
    if traces_dir.is_dir():
        for path in sorted(traces_dir.glob("*.json")):
            loaded_steps.append(path.stem)
    _emit_status_line(
        "resume_loaded",
        cg_id=cg_id,
        resume_from=resume_from,
        workspace=str(workspace),
        loaded_steps=loaded_steps,
    )
    _emit_status_line(
        "resume_stub",
        cg_id=cg_id,
        resume_from=resume_from,
        reason=(
            "loaded prior state; resume-mode workflow logic is not yet "
            "implemented (architectural hedge per architectural_decisions "
            "§12 item 4; future PR adds the multi-cycle loop)"
        ),
    )
    return EXIT_RESUME_STUB


def _persist_message_history(
    workspace: Path,
    step_name: str,
    result: Any,
) -> None:
    """Dump a PydanticAI :class:`AgentRunResult`'s message history.

    Writes ``conversation_traces/<step_name>.json`` per
    architectural_decisions §12 item 4. The file uses PydanticAI's
    native :meth:`all_messages_json` serialization so a future resume
    path can rehydrate via ``ModelMessagesTypeAdapter.validate_json``.

    Best-effort: serialization failures or workspace-write failures
    do not abort the workflow step. The traces are diagnostic / hedge
    state, not load-bearing for the current run.
    """
    traces_dir = workspace / "conversation_traces"
    try:
        traces_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    try:
        history_bytes = result.all_messages_json()
    except (AttributeError, RuntimeError, ValueError):
        return
    try:
        (traces_dir / f"{step_name}.json").write_bytes(history_bytes)
    except OSError:
        return


# === Argparse / workspace helpers ============================================


def _resolve_workspace(workspace_arg: object) -> Path | None:
    """Resolve and validate the ``--workspace`` argv."""
    if workspace_arg is None:
        return None
    path = Path(str(workspace_arg))
    if not path.exists() or not path.is_dir():
        return None
    return path.resolve()


def _load_contract(workspace: Path) -> HarnessContract | None:
    """Read and validate the staged :class:`HarnessContract`."""
    contract_path = workspace / "contracts" / "harness_contract.json"
    if not contract_path.exists():
        return None
    try:
        return HarnessContract.model_validate_json(contract_path.read_text())
    except (ValueError, OSError):
        return None


def _load_technique_contract(workspace: Path) -> TechniqueContract | None:
    """Read the staged :class:`TechniqueContract` if the host materialized one."""
    contract_path = workspace / "contracts" / "technique_contract.json"
    if not contract_path.exists():
        return None
    try:
        return TechniqueContract.model_validate_json(contract_path.read_text())
    except (ValueError, OSError):
        return None


# === Status emission ========================================================


def _emit_status_line(event_type: str, **fields: Any) -> None:
    """Write one JSON line to stdout for pre-emitter / error-path events.

    Used for the narrow set of pre-session-construction errors (bad
    workspace, missing cg-id, resume stub) that fire before the
    :class:`StatusEmitter` is wired. After session.start lands, all
    transitions go through :class:`StatusEmitter`.
    """
    line = json.dumps({"event_type": event_type, **fields}, separators=(",", ":"))
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _write_status_summary(
    workspace: Path,
    cg_id: str,
    outcomes: list[_StepOutcome],
    *,
    succeeded: bool,
) -> None:
    """Drop a ``status.json`` under ``outputs/`` summarizing the workflow run.

    Companion summary file to ``status/events.jsonl`` per D10's design
    (the events.jsonl is the per-emit stream; status.json is the single
    "what happened" summary the host's post-terminal harvest is most
    likely to surface in ``smai status <cg_id>``).
    """
    outputs_dir = workspace / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "cg_id": cg_id,
        "succeeded": succeeded,
        "step_outcomes": [
            {
                "step_index": o.step_index,
                "step_type": o.step_type,
                "succeeded": o.succeeded,
                "error": o.error,
            }
            for o in outcomes
        ],
    }
    (outputs_dir / "status.json").write_text(json.dumps(payload, indent=2))


def _label_for_step(step: WorkflowStep) -> str:
    """Human-readable step label for status emit + operator surface."""
    if isinstance(step, HarnessBuilderBodyGenerationStep):
        return _step_label_for_kind(step.step_type, function_name=step.function_name)
    return _step_label_for_kind(step.step_type)


def _input_summary_for(step: WorkflowStep) -> dict[str, str]:
    """Build the ``step.start`` ``input_summary`` payload.

    Capped at the load-bearing identification fields per the D10 design
    note (~10 keys total across step types).
    """
    if isinstance(step, HarnessBuilderBodyGenerationStep):
        return {
            "function_name": step.function_name,
            "function_index": str(step.function_index),
            "write_to_path": step.write_to_path,
        }
    # Step 7 of the agent-layer refactor: technique_implementer reuses
    # this helper via cross-module import; an entry for the technique-
    # body-generation step type lands here so the operator's
    # ``smai status`` rendering doesn't fall through to ``{}``.
    if isinstance(step, TechniqueImplementerBodyGenerationStep):
        return {
            "technique_id": step.technique_id,
            "write_to_path": step.write_to_path,
        }
    if isinstance(step, BaselineGenerationStep):
        return {
            "factor_type": step.factor_type,
            "baseline_technique_id": step.baseline_technique_id,
            "write_to_path": step.write_to_path,
        }
    if isinstance(step, ValidationStep):
        return {
            "technique_id": step.technique_id,
            "seed": str(step.seed),
            "dispatch_target": step.dispatch_target,
        }
    if isinstance(step, DiagnoseOnFailureStep):
        return {
            "anchor_step_index": str(step.anchor_step_index),
            "max_retries": str(step.max_retries),
        }
    if isinstance(step, ManifestEmitStep):
        return {
            "runtime_template_version": step.runtime_template_version,
            "parent_harness_contract_hash": step.parent_harness_contract_hash,
        }
    return {}


# === Per-step dispatcher =====================================================


def _dispatch_step(
    step: WorkflowStep,
    index: int,
    context: _DispatchContext,
    prior_outcomes: list[_StepOutcome],
) -> _StepOutcome:
    """Per-step dispatcher: every handler is a real implementation."""
    if isinstance(step, HarnessBuilderBodyGenerationStep):
        return _run_body_generation_step(step, index, context)
    if isinstance(step, BaselineGenerationStep):
        return _run_baseline_step(step, index, context)
    if isinstance(step, ValidationStep):
        return _run_validation_step(step, index, context)
    if isinstance(step, DiagnoseOnFailureStep):
        return _run_diagnose_step(step, index, context, prior_outcomes)
    if isinstance(step, ManifestEmitStep):
        return _run_manifest_emit_step(step, index, context, prior_outcomes)
    return _StepOutcome(
        step_index=index,
        step_type=step.step_type,
        succeeded=False,
        error=f"mini-orchestrator has no handler for {step.step_type!r}",
    )


# === Body-generation step (D7a) =============================================


def _run_body_generation_step(
    step: HarnessBuilderBodyGenerationStep,
    index: int,
    context: _DispatchContext,
) -> _StepOutcome:
    """Real handler for :class:`HarnessBuilderBodyGenerationStep`."""
    try:
        prompt = load_step_prompt(_ROLE, "step_2_fill_init_py.yaml")
    except (FileNotFoundError, ValueError) as exc:
        return _StepOutcome(
            step_index=index,
            step_type=step.step_type,
            succeeded=False,
            error=f"failed to load harness_builder/step_2_fill_init_py.yaml: {exc}",
        )

    target_path = context.workspace / step.write_to_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    current_module_source = target_path.read_text() if target_path.exists() else ""

    extension_points = _resolve_extension_points(context.contract)
    # Round-21 finding (2026-05-26): the host stages the reference at
    # ``contracts/harness_api_reference.md`` (see
    # ``smai_agents/agents/harness_api_reference.py``
    # ``WORKSPACE_HARNESS_API_REFERENCE_PATH``). The original sandbox-
    # side path was the workspace root, so the reference was always
    # empty — and the agent hallucinated module symbols like
    # ``TrainingConfig`` / ``model_registry`` that don't exist in
    # ``smai_runtime``, blowing the validation subprocess.
    harness_api_reference = _read_or_empty(
        context.workspace / "contracts" / "harness_api_reference.md"
    )

    provider, model_id = get_model_for_step(
        _ROLE,
        step.step_type,
        overrides=context.overrides,
    )

    prior_failed_attempts: list[PriorFailedAttempt] = []
    for attempt_index in range(_MAX_LINT_RETRIES + 1):
        bundle = HarnessBuilderBodyGenerationInput(
            target_file_path=step.write_to_path,
            function_signature=FunctionSignature(
                name=_coerce_abi_name(step.function_name),
                signature=step.function_signature,
                purpose=_abi_purpose(step.function_name),
            ),
            extension_points=extension_points,
            factor=context.contract.body.factor,
            locked_config=list(context.contract.body.fixed_variables),
            harness_api_reference=harness_api_reference,
            current_module_source=current_module_source,
            prior_failed_attempts=list(prior_failed_attempts),
        )

        try:
            user_message = render_user_message(
                prompt.initial_user_message_template,
                bundle.model_dump(mode="json"),
            )
        except UndefinedError as exc:
            return _StepOutcome(
                step_index=index,
                step_type=step.step_type,
                succeeded=False,
                error=f"prompt template missing variable: {exc}",
            )

        agent = build_agent(
            provider=provider,
            model_id=model_id,
            output_type=HarnessBuilderBodyGenerationOutput,
            system_prompt=prompt.system_prompt,
        )

        try:
            output = _run_agent_sync(
                agent,
                user_message,
                HarnessBuilderBodyGenerationOutput,
                workspace=context.workspace,
                trace_step_name=(f"{index:02d}_body_{step.function_name}_attempt_{attempt_index}"),
                agent_runner=context.agent_runner,
            )
        except _AgentRunError as exc:
            return _StepOutcome(
                step_index=index,
                step_type=step.step_type,
                succeeded=False,
                error=f"agent call failed: {exc}",
            )

        if output.function_name != _coerce_abi_name(step.function_name):
            return _StepOutcome(
                step_index=index,
                step_type=step.step_type,
                succeeded=False,
                error=(
                    f"agent echoed function_name={output.function_name!r} but step "
                    f"targets {step.function_name!r}"
                ),
            )

        target_path.write_text(output.module_source)
        lint_outcome = _run_ruff_check(target_path)
        if lint_outcome is None:
            current_module_source = output.module_source
            return _StepOutcome(
                step_index=index,
                step_type=step.step_type,
                succeeded=True,
            )

        prior_failed_attempts.append(
            PriorFailedAttempt(
                attempt_index=attempt_index,
                prior_module_source=output.module_source,
                failure=LintFailure(linter="ruff", output=lint_outcome),
            )
        )

    return _StepOutcome(
        step_index=index,
        step_type=step.step_type,
        succeeded=False,
        error=f"lint-retry budget exhausted ({_MAX_LINT_RETRIES} retries)",
    )


# === Baseline-generation step (D7b) =========================================


def _run_baseline_step(
    step: BaselineGenerationStep,
    index: int,
    context: _DispatchContext,
) -> _StepOutcome:
    """Real handler for :class:`BaselineGenerationStep`."""
    try:
        prompt = load_step_prompt(_ROLE, "step_4_fill_baseline.yaml")
    except (FileNotFoundError, ValueError) as exc:
        return _StepOutcome(
            step_index=index,
            step_type=step.step_type,
            succeeded=False,
            error=f"failed to load harness_builder/step_4_fill_baseline.yaml: {exc}",
        )

    target_path = context.workspace / step.write_to_path
    target_path.parent.mkdir(parents=True, exist_ok=True)

    grounding = _resolve_baseline_grounding(step, context)
    harness_source = _read_harness_source(context.workspace)
    manifest_json = _read_or_empty(context.workspace / "harness_api_manifest.json")

    technique_id = step.baseline_technique_id
    entry_id = (
        context.technique_contract.body.entry_id
        if context.technique_contract is not None
        else f"{context.cg_id}-baseline"
    )
    technique_params = (
        dict(context.technique_contract.body.technique_params)
        if context.technique_contract is not None
        and context.technique_contract.body.technique_params is not None
        else None
    )

    provider, model_id = get_model_for_step(
        _ROLE,
        step.step_type,
        overrides=context.overrides,
    )

    prior_failed_attempts: list[PriorTechniqueAttempt] = []
    for attempt_index in range(_MAX_LINT_RETRIES + 1):
        bundle = TechniqueBodyGenerationBundle(
            step_kind="baseline",
            cg_id=context.cg_id,
            entry_id=entry_id,
            technique_name="baseline",
            is_baseline=True,
            factor_dimension=context.contract.body.factor.name,
            factor_type=step.factor_type,
            technique_id=technique_id,
            technique_params=technique_params,
            grounding=grounding,
            harness_api_manifest_json=manifest_json,
            harness_source=harness_source,
            baseline_source=None,
            prior_failed_attempts=list(prior_failed_attempts),
        )

        try:
            user_message = render_user_message(
                prompt.initial_user_message_template,
                bundle.model_dump(mode="json"),
            )
        except UndefinedError as exc:
            return _StepOutcome(
                step_index=index,
                step_type=step.step_type,
                succeeded=False,
                error=f"prompt template missing variable: {exc}",
            )

        agent = build_agent(
            provider=provider,
            model_id=model_id,
            output_type=TechniqueBodyOutput,
            system_prompt=prompt.system_prompt,
        )

        try:
            output = _run_agent_sync(
                agent,
                user_message,
                TechniqueBodyOutput,
                workspace=context.workspace,
                trace_step_name=f"{index:02d}_baseline_attempt_{attempt_index}",
                agent_runner=context.agent_runner,
            )
        except _AgentRunError as exc:
            return _StepOutcome(
                step_index=index,
                step_type=step.step_type,
                succeeded=False,
                error=f"agent call failed: {exc}",
            )

        target_path.write_text(output.technique_py_source)
        lint_outcome = _run_ruff_check(target_path)
        if lint_outcome is None:
            return _StepOutcome(
                step_index=index,
                step_type=step.step_type,
                succeeded=True,
            )

        prior_failed_attempts.append(
            PriorTechniqueAttempt(
                attempt_index=attempt_index,
                prior_source=output.technique_py_source,
                failure_kind="lint",
                failure_excerpt=lint_outcome,
            )
        )

    return _StepOutcome(
        step_index=index,
        step_type=step.step_type,
        succeeded=False,
        error=f"lint-retry budget exhausted ({_MAX_LINT_RETRIES} retries)",
    )


# === Validation step =========================================================


def _run_validation_step(
    step: ValidationStep,
    index: int,
    context: _DispatchContext,
) -> _StepOutcome:
    """Real handler for :class:`ValidationStep`.

    Runs ``python experiment.py --mode validation`` via real
    ``subprocess.run`` inside the sandbox. On success (exit 0 +
    ``validation_results.json`` written by the runner), the step
    succeeds and the workflow advances to the diagnose step (which
    passes through as a no-op when the anchor succeeded) and then to
    the manifest emit step.

    On failure (non-zero exit OR missing/malformed
    ``validation_results.json``), the step records the captured stderr
    tail onto :attr:`_StepOutcome.captured_stderr` so the diagnose step
    can build a D7c :class:`DiagnoseBundle` without re-running
    validation.

    ``dispatch_target == "compute_submit"`` is the architectural §1
    escape hatch (GPU validation via callback to the host Compute
    Protocol). Not implemented in sub-PR C2; surfaces as an error so
    the workflow doesn't silently skip validation.
    """
    if step.dispatch_target == "compute_submit":
        return _StepOutcome(
            step_index=index,
            step_type=step.step_type,
            succeeded=False,
            error=(
                "dispatch_target='compute_submit' (GPU-validation escape hatch per "
                "architectural_decisions §1) is not implemented in sub-PR C2; "
                "the workflow generator currently emits dispatch_target='subprocess' "
                "and the substrate escape lands in a follow-up."
            ),
        )

    experiment_path = context.workspace / "experiment.py"
    if not experiment_path.exists():
        return _StepOutcome(
            step_index=index,
            step_type=step.step_type,
            succeeded=False,
            error=(
                f"validation requires experiment.py at the workspace root, but "
                f"{experiment_path} does not exist; the host-side workspace stager "
                "must materialize the runtime template before sandbox dispatch"
            ),
        )

    cmd = [
        sys.executable,
        str(experiment_path),
        "--technique",
        step.technique_id or _VALIDATION_TECHNIQUE_NAME,
        "--seed",
        str(step.seed),
        "--mode",
        "validation",
        "--workspace",
        str(context.workspace),
    ]
    try:
        proc = subprocess.run(  # noqa: S603
            cmd,
            cwd=str(context.workspace),
            capture_output=True,
            text=True,
            check=False,
            timeout=_VALIDATION_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        # text=True above means exc.stderr is str when populated; pyright
        # doesn't narrow the Optional[str | bytes] union, so coerce here.
        raw_stderr = exc.stderr
        if isinstance(raw_stderr, bytes):
            raw_stderr = raw_stderr.decode("utf-8", errors="replace")
        tail = _tail_text(raw_stderr or "", STDERR_TAIL_LINES, STDERR_TAIL_BYTES)
        return _StepOutcome(
            step_index=index,
            step_type=step.step_type,
            succeeded=False,
            error=(
                f"validation subprocess timed out after {_VALIDATION_TIMEOUT_SECONDS}s; "
                "the experiment-runner did not return within the configured ceiling"
            ),
            captured_stderr=tail,
        )
    except FileNotFoundError as exc:
        return _StepOutcome(
            step_index=index,
            step_type=step.step_type,
            succeeded=False,
            error=f"validation subprocess could not start: {exc}",
        )

    stderr_tail = _tail_text(proc.stderr, STDERR_TAIL_LINES, STDERR_TAIL_BYTES)

    if proc.returncode != 0:
        return _StepOutcome(
            step_index=index,
            step_type=step.step_type,
            succeeded=False,
            error=(
                f"validation subprocess exited with code {proc.returncode}; see "
                "captured stderr for the runner diagnostic"
            ),
            captured_stderr=stderr_tail,
        )

    # Runner exit code is 0 but the runner must have written
    # ``validation_results.json`` on the success path (round-20 contract).
    # Missing or malformed JSON => failure, hand off to diagnose.
    results_path = context.workspace / "validation_results.json"
    if not results_path.exists():
        return _StepOutcome(
            step_index=index,
            step_type=step.step_type,
            succeeded=False,
            error=(
                "validation exited 0 but did not produce validation_results.json; "
                "the runner's success-path writer (write_validation_results) did "
                "not fire — likely a mode-routing or workspace-path bug"
            ),
            captured_stderr=stderr_tail,
        )
    try:
        results = json.loads(results_path.read_text())
    except (ValueError, OSError) as exc:
        return _StepOutcome(
            step_index=index,
            step_type=step.step_type,
            succeeded=False,
            error=f"validation_results.json is malformed: {exc}",
            captured_stderr=stderr_tail,
        )
    if not isinstance(results, dict) or cast(dict[str, object], results).get("passed") is not True:
        return _StepOutcome(
            step_index=index,
            step_type=step.step_type,
            succeeded=False,
            error=(
                "validation_results.json does not report passed=True; the runner "
                "wrote the file but the structured outcome reflects a soft fail"
            ),
            captured_stderr=stderr_tail,
        )

    return _StepOutcome(
        step_index=index,
        step_type=step.step_type,
        succeeded=True,
    )


# === Diagnose-on-failure step (D7c) =========================================


def _run_diagnose_step(
    step: DiagnoseOnFailureStep,
    index: int,
    context: _DispatchContext,
    prior_outcomes: list[_StepOutcome],
) -> _StepOutcome:
    """Real handler for :class:`DiagnoseOnFailureStep`.

    Passes through as a no-op when the anchor step succeeded. On anchor
    failure, builds a D7c :class:`DiagnoseBundle`, calls a PydanticAI
    Agent bound to :class:`Diagnosis`, applies the ``fix_payload`` to
    the ``fix_target``, and loops back to revalidate. The agent has
    one tool — ``read_file`` — workspace-scoped per architectural_decisions
    §12 #1.

    Bounded retries per :attr:`DiagnoseOnFailureStep.max_retries`. On
    budget exhaustion, the step returns a clear failure error so the
    outer orchestrator surfaces "we tried N diagnoses and validation
    still doesn't pass".
    """
    if step.anchor_step_index >= len(prior_outcomes):
        return _StepOutcome(
            step_index=index,
            step_type=step.step_type,
            succeeded=False,
            error=(
                f"anchor_step_index {step.anchor_step_index} out of range "
                f"(only {len(prior_outcomes)} prior steps)"
            ),
        )
    anchor = prior_outcomes[step.anchor_step_index]
    if anchor.succeeded:
        # Pass-through: validation succeeded, no diagnose needed.
        return _StepOutcome(step_index=index, step_type=step.step_type, succeeded=True)

    try:
        prompt = load_step_prompt(_ROLE, "step_7_diagnose.yaml")
    except (FileNotFoundError, ValueError) as exc:
        return _StepOutcome(
            step_index=index,
            step_type=step.step_type,
            succeeded=False,
            error=f"failed to load harness_builder/step_7_diagnose.yaml: {exc}",
        )

    provider, model_id = get_model_for_step(
        _ROLE,
        step.step_type,
        overrides=context.overrides,
    )

    prior_diagnoses: list[PriorDiagnosis] = []
    last_anchor = anchor
    for attempt_index in range(step.max_retries):
        bundle = _build_diagnose_bundle(
            anchor=last_anchor,
            context=context,
            prior_diagnoses=prior_diagnoses,
        )

        try:
            user_message = render_user_message(
                prompt.initial_user_message_template,
                bundle.model_dump(mode="json"),
            )
        except UndefinedError as exc:
            return _StepOutcome(
                step_index=index,
                step_type=step.step_type,
                succeeded=False,
                error=f"prompt template missing variable: {exc}",
            )

        agent = build_agent(
            provider=provider,
            model_id=model_id,
            output_type=Diagnosis,
            system_prompt=prompt.system_prompt,
        )
        _register_read_file_tool(agent, context.workspace)

        try:
            diagnosis = _run_agent_sync(
                agent,
                user_message,
                Diagnosis,
                workspace=context.workspace,
                trace_step_name=f"{index:02d}_diagnose_attempt_{attempt_index}",
                agent_runner=context.agent_runner,
            )
        except _AgentRunError as exc:
            return _StepOutcome(
                step_index=index,
                step_type=step.step_type,
                succeeded=False,
                error=f"diagnose agent call failed: {exc}",
            )

        apply_outcome = _apply_diagnosis(diagnosis, context)
        if not apply_outcome.applied:
            # fix_target / kind mismatch or target write failed; record
            # as did_not_apply and loop (gives the agent another shot
            # with the corrected context).
            prior_diagnoses.append(
                PriorDiagnosis(
                    attempt_index=attempt_index,
                    diagnosis_summary=_first_line(diagnosis.diagnosis),
                    fix_target=diagnosis.fix_target,
                    fix_payload_preview=diagnosis.fix_payload[:200],
                    outcome="did_not_apply",
                )
            )
            continue

        # Re-validate by running a fresh ValidationStep against the
        # anchor's substrate config. If it now passes, the diagnose step
        # succeeds and the workflow advances to manifest emit.
        revalidation = _run_validation_step(
            _validation_step_from_anchor(step.anchor_step_index, context),
            step.anchor_step_index,  # re-use anchor index for symmetry; outcome is not appended
            context,
        )
        if revalidation.succeeded:
            # Patch the original anchor outcome so manifest emit's
            # validation-precondition check passes; this is a deliberate
            # mutation that reflects the "we fixed it and revalidated"
            # reality the workflow needs.
            prior_outcomes[step.anchor_step_index] = _StepOutcome(
                step_index=step.anchor_step_index,
                step_type=anchor.step_type,
                succeeded=True,
            )
            return _StepOutcome(
                step_index=index,
                step_type=step.step_type,
                succeeded=True,
            )

        # Revalidation still failed; record the outcome and loop.
        last_anchor = revalidation
        prior_diagnoses.append(
            PriorDiagnosis(
                attempt_index=attempt_index,
                diagnosis_summary=_first_line(diagnosis.diagnosis),
                fix_target=diagnosis.fix_target,
                fix_payload_preview=diagnosis.fix_payload[:200],
                outcome="applied_but_validation_failed_again",
            )
        )

    return _StepOutcome(
        step_index=index,
        step_type=step.step_type,
        succeeded=False,
        error=(
            f"diagnose-retry budget exhausted ({step.max_retries} attempts); "
            "validation did not pass after applying the agent's diagnoses"
        ),
    )


# === Manifest-emit step ======================================================


def _run_manifest_emit_step(
    step: ManifestEmitStep,
    index: int,
    context: _DispatchContext,
    prior_outcomes: list[_StepOutcome],
) -> _StepOutcome:
    """Real handler for :class:`ManifestEmitStep`.

    Validation-precondition: a passing :class:`ValidationStep` must
    exist in the prior outcomes (round-15 / round-20 discipline). On
    success, recomputes ``harness_version_hash`` from the current
    on-disk ``harness/`` contents (never reuses a stale hash from a
    prior turn), builds the :class:`HarnessAPIManifest`, freezes it,
    and writes ``manifest.json`` to the workspace root.

    Per architectural_decisions.md §12 sub-step 4.8 "all deterministic":
    the agent does not call a tool to emit the manifest; the workflow
    runs the emit deterministically.
    """
    # Validation precondition: at least one prior step is a successful
    # ValidationStep. The workflow generator always emits the validation
    # step before manifest emit, so this also surfaces an out-of-order
    # workflow as a hard failure rather than silently emitting a
    # manifest against unvalidated harness code.
    validation_succeeded = any(
        outcome.step_type == "validation" and outcome.succeeded for outcome in prior_outcomes
    )
    if not validation_succeeded:
        return _StepOutcome(
            step_index=index,
            step_type=step.step_type,
            succeeded=False,
            error=(
                "manifest emit refuses to run: no prior successful ValidationStep "
                "in the workflow outcomes (round-15 / round-20 discipline; the "
                "manifest documents a validated harness, never a hopefully-correct one)"
            ),
        )

    harness_files = _read_harness_files_bytes(context.workspace)
    if not harness_files:
        return _StepOutcome(
            step_index=index,
            step_type=step.step_type,
            succeeded=False,
            error=(
                "manifest emit refuses to run: workspace harness/ directory contains "
                "no files; the harness body-generation steps must have produced at "
                "least harness/__init__.py before manifest emit"
            ),
        )
    # Round-15: recompute the harness_version_hash from the on-disk
    # files, never reuse a stale value from an earlier turn. This is
    # the canonical input the manifest carries.
    harness_version_hash = compute_harness_version_hash(harness_files)

    # Manifest payload. Extension points come from the harness contract's
    # factor type (closed v1 set per round-15 introspection); the
    # integration_pattern_summary is a one-line operator-surface aid.
    extension_points = _build_manifest_extension_points(context.contract)
    integration_pattern_summary = _summarize_integration_patterns(extension_points)
    manifest = HarnessAPIManifest(
        extension_points=extension_points,
        integration_pattern_summary=integration_pattern_summary,
        harness_version_hash=harness_version_hash,
        parent_harness_contract_hash=step.parent_harness_contract_hash,
        manifest_schema_version=MANIFEST_SCHEMA_VERSION,
        runtime_template_version=step.runtime_template_version,
    )
    if step.runtime_template_version != RUNTIME_TEMPLATE_VERSION:
        return _StepOutcome(
            step_index=index,
            step_type=step.step_type,
            succeeded=False,
            error=(
                f"step.runtime_template_version {step.runtime_template_version!r} "
                f"does not match live RUNTIME_TEMPLATE_VERSION "
                f"{RUNTIME_TEMPLATE_VERSION!r}; the runtime that consumes this "
                "manifest would reject it at run start"
            ),
        )

    frozen = freeze_manifest(manifest)
    try:
        (context.workspace / "manifest.json").write_text(
            frozen.model_dump_json(indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        return _StepOutcome(
            step_index=index,
            step_type=step.step_type,
            succeeded=False,
            error=f"failed to write manifest.json to workspace: {exc}",
        )

    return _StepOutcome(
        step_index=index,
        step_type=step.step_type,
        succeeded=True,
    )


# === Bundle-construction helpers =============================================


def _resolve_extension_points(contract: HarnessContract) -> list[ExtensionPointSpec]:
    """Construct the D7a ``extension_points`` list from the contract."""
    factor_role = (
        "additive_baseline_default"
        if contract.body.factor.type == "additive"
        else "substitutive_slot"
    )
    specs: list[ExtensionPointSpec] = []
    for key, component_field in COMPONENT_FIELD_FOR_KEY.items():
        admissible = ADMISSIBLE_PATTERNS_FOR_KEY.get(key, frozenset())
        specs.append(
            ExtensionPointSpec(
                key=key,
                component_field=component_field,
                admissible_patterns=list(admissible),
                factor_role=factor_role,
            )
        )
    return specs


def _build_manifest_extension_points(
    contract: HarnessContract,
) -> list[HarnessExtensionPoint]:
    """Build the manifest's :class:`HarnessExtensionPoint` list.

    Sub-PR C2 emits an empty extension-point list because the workflow's
    validation-mode runner already runs against the empty-extension-
    points stub manifest (round-20 contract). The real per-contract
    derivation lands in sub-PR D when the host-side dispatcher passes
    the planner-derived extension-point set through.
    """
    _ = contract
    return []


def _summarize_integration_patterns(
    extension_points: list[HarnessExtensionPoint],
) -> str:
    """One-line operator-surface summary of the manifest's patterns."""
    if not extension_points:
        return "no extension points (validation-mode stub manifest)"
    patterns = sorted({ep.integration_pattern for ep in extension_points})
    return f"integration patterns: {', '.join(patterns)}"


def _resolve_baseline_grounding(
    step: BaselineGenerationStep,
    context: _DispatchContext,
) -> GroundingContext:
    """Resolve the D7b :data:`GroundingContext` for the baseline step."""
    staged = context.workspace / "grounding" / "baseline_grounding.json"
    if staged.exists():
        try:
            raw = json.loads(staged.read_text())
            return _adapter_validate_grounding(raw)
        except (ValueError, OSError):
            # Fall through to contract-driven resolution; the planner
            # may not have staged grounding for this baseline yet.
            pass

    if context.technique_contract is None:
        if step.factor_type == "additive":
            return NoOpBaselineGrounding()
        return StandardLibraryGrounding(
            technique_description=(
                f"Baseline technique {step.baseline_technique_id!r}: standard library "
                "default for the factor. No grounding artifact was staged; the "
                "implementation should rely on canonical library APIs for the "
                "factor's reference arm."
            )
        )

    anchor = context.technique_contract.body.fidelity_anchor
    if anchor is None:
        if step.factor_type == "additive":
            return NoOpBaselineGrounding()
        return StandardLibraryGrounding(
            technique_description=(
                f"Standard baseline for technique {step.baseline_technique_id!r}; "
                "no fidelity anchor staged on the technique contract."
            )
        )
    if anchor.kind == "paper":
        return PaperExtractGrounding(
            arxiv_id=anchor.arxiv_id or "unknown",
            technique_id=step.baseline_technique_id,
            method_extraction=(
                "No method extraction was staged for this baseline; the host-side "
                "dispatcher (sub-PR D) materializes the EnrichmentResult payload "
                "into grounding/baseline_grounding.json."
            ),
            implementability="medium",
        )
    if anchor.kind == "proposal":
        return ProposalGrounding(
            proposal_id=anchor.proposal_id,
            technique_description=(
                f"Baseline technique {step.baseline_technique_id!r}: planner-authored "
                "description not staged inline. Use the proposal id to anchor the "
                "reference arm and implement per the technique-id naming convention."
            ),
        )
    if anchor.kind == "reviewer_attested":
        return ReviewerAttestedGrounding(
            spec_text=anchor.spec_text,
            attested_by=anchor.attested_by,
        )
    # Exhaustive over FidelityAnchor.kind; defensive fall-through.
    return StandardLibraryGrounding(
        technique_description=f"Baseline technique {step.baseline_technique_id!r}."
    )


def _adapter_validate_grounding(raw: Any) -> GroundingContext:
    from pydantic import TypeAdapter

    adapter: TypeAdapter[GroundingContext] = TypeAdapter(GroundingContext)
    return adapter.validate_python(raw)


def _read_harness_source(workspace: Path) -> dict[str, str]:
    """Collect every ``.py`` file under ``harness/`` into a relative-path map."""
    harness_dir = workspace / "harness"
    if not harness_dir.is_dir():
        return {}
    sources: dict[str, str] = {}
    for path in sorted(harness_dir.rglob("*.py")):
        relative = path.relative_to(workspace)
        try:
            sources[str(relative)] = path.read_text()
        except OSError:
            continue
    return sources


def _read_harness_files_bytes(workspace: Path) -> dict[str, bytes]:
    """Collect ``{relative_path: bytes}`` for every regular file under
    ``<workspace>/harness/``.

    Mirrors the host-side ``emit_harness_manifest`` tool's
    ``_read_harness_files`` shape so the hash is byte-stable across the
    sandbox-side and host-side surfaces. Paths are relative to the
    harness directory so the digest is workspace-position-independent
    (``compute_harness_version_hash`` sorts internally).
    """
    base = workspace / "harness"
    if not base.is_dir():
        return {}
    files: dict[str, bytes] = {}
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(base).as_posix()
        try:
            files[rel] = path.read_bytes()
        except OSError:
            continue
    return files


def _read_or_empty(path: Path) -> str:
    """Read a text file or return an empty string."""
    if not path.exists():
        return ""
    try:
        return path.read_text()
    except OSError:
        return ""


def _coerce_abi_name(name: str) -> Any:
    """Cast a raw ABI function name to the schema's ``ABIFunctionName`` Literal."""
    return name


def _abi_purpose(name: str) -> str:
    """One-sentence purpose for the ABI function ``name``."""
    purposes = {
        "build_harness": (
            "Construct the experiment harness from `config` and return a "
            "HarnessComponents instance; called once per run by "
            "smai_runtime.runner step 5."
        ),
        "run_training_loop": (
            "Drive the training loop using the harness and technique; return "
            "a TrainingResult capturing metrics and artifacts."
        ),
        "evaluate": (
            "Compute the contract's required metrics from the TrainingResult "
            "and return them as a Metrics mapping."
        ),
    }
    return purposes.get(name, f"ABI function `{name}`")


# === Diagnose-bundle assembly + fix application ==============================


def _build_diagnose_bundle(
    *,
    anchor: _StepOutcome,
    context: _DispatchContext,
    prior_diagnoses: list[PriorDiagnosis],
) -> DiagnoseBundle:
    """Assemble the D7c :class:`DiagnoseBundle` from workspace context."""
    stderr_tail = anchor.captured_stderr or ""
    failure_reason = anchor.error or "validation failed without a structured reason"

    # Parse validation_results.json if present (the runtime writes it
    # only on success, so this is typically None on failure; the field
    # exists for the "soft fail with structured results" future-path).
    validation_results: dict[str, object] | None = None
    results_path = context.workspace / "validation_results.json"
    if results_path.exists():
        try:
            raw = json.loads(results_path.read_text())
            if isinstance(raw, dict):
                validation_results = cast(dict[str, object], raw)
        except (ValueError, OSError):
            validation_results = None

    current_code = _collect_current_code(context.workspace)
    original_input_bundle = _resolve_original_bundle_ref(context)

    return DiagnoseBundle(
        failure_reason=failure_reason,
        container_stderr=stderr_tail,
        validation_results=validation_results,
        current_code=current_code,
        prior_diagnoses=prior_diagnoses,
        original_input_bundle=original_input_bundle,
    )


def _collect_current_code(workspace: Path) -> dict[str, str]:
    """Gather the most-recently-written code files for the diagnose bundle.

    Includes every ``.py`` under ``harness/`` and ``techniques/``;
    these are the surfaces the body-generation steps wrote into. The
    map is workspace-relative-path -> file content; the agent sees what
    the prior steps produced without dereferencing any path.
    """
    files: dict[str, str] = {}
    for sub in ("harness", "techniques"):
        sub_dir = workspace / sub
        if not sub_dir.is_dir():
            continue
        for path in sorted(sub_dir.rglob("*.py")):
            relative = path.relative_to(workspace)
            try:
                files[str(relative)] = path.read_text()
            except OSError:
                continue
    return files


def _resolve_original_bundle_ref(context: _DispatchContext) -> OriginalInputBundleRef:
    """Build the discriminated bundle reference for the harness flow.

    Harness-flow diagnose steps always reference the harness body-
    generation bundle that produced the failing code. Sub-PR C2's
    discriminator-walk returns a :class:`HarnessOriginalBundleRef`
    populated with the contract's factor type plus a one-line summary
    of the validation-mode stub manifest the runner synthesizes (round-
    23 R1: closed-v1 keys allowlisted with safe-default integration
    patterns; previously empty).
    """
    return HarnessOriginalBundleRef(
        function_signature="def build_harness(config: dict) -> HarnessComponents:",
        factor_type=context.contract.body.factor.type,
        extension_points_summary=(
            "(validation-mode stub manifest: closed-v1 keys "
            "{train_transforms, model_wrapper, loss_fn, training_overrides, callbacks} "
            "declared optional with safe-default patterns)"
        ),
    )


def _validation_step_from_anchor(
    anchor_index: int,
    context: _DispatchContext,
) -> ValidationStep:
    """Re-construct a :class:`ValidationStep` for re-running validation
    after the diagnose step applied a fix.

    Mirrors the shape the generator originally emitted (round-15
    drift-guard discipline; consult ``contract.body.seeds[0]`` rather
    than hardcoding).
    """
    _ = anchor_index
    seed = context.contract.body.seeds[0] if context.contract.body.seeds else _DEFAULT_SEED
    return ValidationStep(
        technique_id=_VALIDATION_TECHNIQUE_NAME,
        seed=seed,
        dispatch_target="subprocess",
    )


@dataclass
class _FixApplication:
    """Result of applying a :class:`Diagnosis`'s ``fix_payload`` to disk."""

    applied: bool
    target_path: Path | None = None
    error: str | None = None


def _apply_diagnosis(
    diagnosis: Diagnosis,
    context: _DispatchContext,
) -> _FixApplication:
    """Write ``diagnosis.fix_payload`` to the surface named by ``fix_target``.

    Per architectural_decisions.md §12 sub-step 4.8 "all deterministic":
    the mini-orchestrator applies the fix, the agent does not. The
    surface table is:

    * ``harness`` -> ``harness/__init__.py``
    * ``baseline`` -> ``techniques/baseline.py``
    * ``config`` -> ``config.yaml``
    * ``technique`` -> not applicable to harness_builder flows; returns
      ``applied=False`` so the diagnose step records ``did_not_apply``
      and the agent picks a different target.
    """
    surface_table = {
        "harness": "harness/__init__.py",
        "baseline": "techniques/baseline.py",
        "config": "config.yaml",
    }
    if diagnosis.fix_target == "technique":
        return _FixApplication(
            applied=False,
            error=(
                "fix_target='technique' is not applicable to a harness_builder "
                "diagnose step; the workflow has no per-technique writable surface "
                "here (technique_implementer flows accept this target)"
            ),
        )
    relative = surface_table.get(diagnosis.fix_target)
    if relative is None:
        return _FixApplication(
            applied=False,
            error=f"unknown fix_target {diagnosis.fix_target!r}",
        )
    target = context.workspace / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.write_text(diagnosis.fix_payload)
    except OSError as exc:
        return _FixApplication(
            applied=False,
            error=f"failed to write {relative}: {exc}",
        )
    return _FixApplication(applied=True, target_path=target)


def _register_read_file_tool(agent: Any, workspace: Path) -> None:
    """Register the workspace-scoped ``read_file`` PydanticAI tool.

    Per architectural_decisions.md §12 #1 "Design discipline":
    ``read_file`` is the diagnose-step-only escape hatch for
    genuinely unforeseen lookups. Workspace-rooted paths only (no
    escape via ``..``); responses are size-capped per
    :data:`READ_FILE_MAX_BYTES`. The per-attempt call budget
    (:data:`READ_FILE_BUDGET_PER_ATTEMPT`) is enforced via a
    closure-scoped counter.

    Implemented as ``@agent.tool_plain`` so the tool signature does not
    need a ``RunContext`` argument; the workspace path is captured in
    the closure at registration time.
    """
    state = {"call_count": 0}

    @agent.tool_plain
    def read_file(path: str) -> str:
        """Read a UTF-8 text file from the agent's workspace.

        ``path`` is workspace-relative (``..`` escapes rejected). The
        response is truncated at the configured size cap with a
        structured flag if exceeded. Per-attempt call budget is enforced
        in-process and surfaces as a structured error.

        Use only when the bundle is missing context you genuinely need
        for the diagnosis; the bundle is designed to be complete.
        """
        if state["call_count"] >= READ_FILE_BUDGET_PER_ATTEMPT:
            return (
                f"read_file budget exhausted ({READ_FILE_BUDGET_PER_ATTEMPT} calls "
                "per diagnose attempt); the bundle is structurally complete by "
                "design and further reads are a signal that the bundle needs "
                "extension, not more tool latitude."
            )
        state["call_count"] += 1
        try:
            resolved = (workspace / path).resolve()
            resolved.relative_to(workspace.resolve())
        except ValueError:
            return f"path {path!r} resolves outside workspace; access denied"
        if not resolved.exists() or not resolved.is_file():
            return f"path {path!r} does not exist or is not a regular file"
        try:
            data = resolved.read_bytes()
        except OSError as exc:
            return f"failed to read {path!r}: {exc}"
        if len(data) > READ_FILE_MAX_BYTES:
            truncated = data[:READ_FILE_MAX_BYTES].decode("utf-8", errors="replace")
            return (
                truncated
                + f"\n\n[file truncated at {READ_FILE_MAX_BYTES} bytes "
                + f"(total {len(data)} bytes)]"
            )
        return data.decode("utf-8", errors="replace")

    _ = read_file  # silence unused-binding lint; the decorator registers


def _first_line(text: str) -> str:
    """Return the first non-empty line of ``text`` (capped at 200 chars)."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:200]
    return ""


def _tail_text(text: str, max_lines: int, max_bytes: int) -> str:
    """Tail ``text`` to the last ``max_lines`` lines, then cap at
    ``max_bytes`` from the end so the most recent content survives.

    Mirrors the D7c design-note §1 cap-from-end rule.
    """
    if not text:
        return ""
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    tail = "\n".join(lines)
    encoded = tail.encode("utf-8")
    if len(encoded) > max_bytes:
        encoded = encoded[-max_bytes:]
        tail = encoded.decode("utf-8", errors="replace")
    return tail


# === Agent / lint helpers ====================================================


_AgentOutputT = TypeVar("_AgentOutputT", bound=BaseModel)


class _AgentRunError(RuntimeError):
    """Raised when a PydanticAI agent call surfaces an unrecoverable error."""


def _run_agent_sync(
    agent: Any,
    user_message: str,
    output_type: type[_AgentOutputT],
    *,
    workspace: Path | None = None,
    trace_step_name: str | None = None,
    agent_runner: AgentRunner | None = None,
) -> _AgentOutputT:
    """Invoke an :class:`AgentRunner` synchronously and unwrap the output.

    When ``agent_runner`` is ``None``, drives the real PydanticAI Agent
    via :meth:`Agent.run_sync` and extracts ``.output``. Tests may pass
    a callable that returns a canned output (the ``--fake-llm`` flag at
    the entry-point boundary substitutes :func:`_fake_agent_runner`).

    When ``workspace`` and ``trace_step_name`` are both supplied, the
    PydanticAI :class:`AgentRunResult`'s message history is dumped to
    ``<workspace>/conversation_traces/<trace_step_name>.json`` per the
    resume-prep architectural hedge (architectural_decisions §12 #4).
    The fake-runner path skips trace persistence — fakes have no
    PydanticAI :class:`AgentRunResult` to serialize.
    """
    if agent_runner is not None:
        # DI: tests / --fake-llm path substitutes a runner that returns
        # a canned output directly. No trace persistence (no
        # AgentRunResult to project).
        canned = agent_runner(agent, user_message, output_type)
        if not isinstance(canned, output_type):
            raise _AgentRunError(
                f"injected agent_runner returned output of type "
                f"{type(canned).__name__!r}, expected {output_type.__name__!r}"
            )
        return canned

    try:
        result = agent.run_sync(user_message)
    except (RuntimeError, ValueError, ValidationError) as exc:
        raise _AgentRunError(str(exc)) from exc
    output = getattr(result, "output", None)
    if not isinstance(output, output_type):
        raise _AgentRunError(
            f"agent returned output of type {type(output).__name__!r}, "
            f"expected {output_type.__name__!r}"
        )
    if workspace is not None and trace_step_name is not None:
        _persist_message_history(workspace, trace_step_name, result)
    return output


def _fake_agent_runner(
    agent: Any,
    user_message: str,
    output_type: type[Any],
) -> Any:
    """Deterministic test runner — yields canned outputs per
    ``output_type`` so cross-process subprocess tests can drive the
    full mini-orchestrator workflow without real LLM credentials.

    Wired in by the ``--fake-llm`` flag at main()-entry. Production
    code never invokes this directly; production assigns
    ``context.agent_runner = None`` and drives the real PydanticAI
    run path.
    """
    del agent  # the fake doesn't introspect the PydanticAI Agent
    return _fake_llm_output(output_type, user_message)


def _fake_llm_output(
    output_type: type[_AgentOutputT],
    user_message: str,
) -> _AgentOutputT:
    """Deterministic conforming output for the requested schema."""
    if output_type is HarnessBuilderBodyGenerationOutput:
        fn_name: Any = "build_harness"
        for candidate in ("build_harness", "run_training_loop", "evaluate"):
            if f"body for `{candidate}`" in user_message:
                fn_name = candidate
                break
        module_source = (
            '"""Fake-LLM stub module for sub-PR C tests."""\n'
            "\n"
            "def build_harness(config):\n"
            "    return {}\n"
            "\n"
            "def run_training_loop(harness, technique):\n"
            "    return {}\n"
            "\n"
            "def evaluate(harness, result):\n"
            "    return {}\n"
        )
        return HarnessBuilderBodyGenerationOutput(
            function_name=fn_name,
            module_source=module_source,
            reasoning="fake-llm stub: deterministic conforming output for tests",
        )  # type: ignore[return-value]
    if output_type is TechniqueBodyOutput:
        return TechniqueBodyOutput(
            technique_py_source=(
                '"""Fake-LLM baseline stub for sub-PR C tests."""\n'
                "\n"
                "def baseline(*args, **kwargs):\n"
                "    return None\n"
            ),
            reasoning="fake-llm stub: deterministic conforming output for tests",
        )  # type: ignore[return-value]
    if output_type is Diagnosis:
        return Diagnosis(
            diagnosis=(
                "Fake-LLM stub diagnosis for tests. The diagnose step's "
                "real output requires real LLM access."
            ),
            fix_target="harness",
            fix_payload=(
                '"""Fake-LLM stub harness body for tests."""\n'
                "\n"
                "def build_harness(config):\n"
                "    return {}\n"
                "\n"
                "def run_training_loop(harness, technique):\n"
                "    return {}\n"
                "\n"
                "def evaluate(harness, result):\n"
                "    return {}\n"
            ),
        )  # type: ignore[return-value]
    raise _AgentRunError(
        f"_FAKE_LLM_ENV_VAR active but no canned output registered for {output_type.__name__!r}"
    )


def _run_ruff_check(target_path: Path) -> str | None:
    """Run ``ruff check`` against ``target_path``.

    Returns ``None`` on clean lint, or the captured output (head + tail
    capped) on failure.
    """
    try:
        proc = subprocess.run(  # noqa: S603
            ["ruff", "check", "--no-cache", str(target_path)],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return "ruff invocation error: ruff binary not on PATH"
    except OSError as exc:
        return f"ruff invocation error: {exc}"
    if proc.returncode == 0:
        return None
    output = (proc.stdout + proc.stderr).strip()
    if len(output) > _LINT_OUTPUT_CAP_CHARS:
        head_cap = _LINT_OUTPUT_CAP_CHARS // 2
        tail_cap = _LINT_OUTPUT_CAP_CHARS - head_cap
        return (
            output[:head_cap]
            + f"\n...[truncated {len(output) - _LINT_OUTPUT_CAP_CHARS} chars]...\n"
            + output[-tail_cap:]
        )
    return output


__all__ = [
    "EXIT_BAD_WORKSPACE",
    "EXIT_OK",
    "EXIT_RESUME_NOT_IMPLEMENTED",
    "EXIT_RESUME_STUB",
    "EXIT_STEP_FAILED",
    "main",
]
