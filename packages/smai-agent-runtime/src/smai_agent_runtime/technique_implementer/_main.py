"""Technique-implementer sandbox-side mini-orchestrator.

Agent-refactor Step 7 ports the technique_implementer role onto the
mini-orchestrator pattern that Step 4 + Steps C1/C2/D landed for the
harness_builder. The workflow generator already produces the 3-step
shape per D9 (one body-generation + one validation + one diagnose);
this module fills in the per-step handlers + the outer iteration loop.

Per ``architectural_decisions.md`` §6 and ``implementation_plan.md`` §7
("Replicate Step 4's structure for technique_implementer ... Reuses
sandbox-side and dispatch infrastructure"): role-agnostic helpers
(``_run_validation_step``, ``_run_diagnose_step``, ``_run_ruff_check``,
``_run_agent_sync``, status emitter, etc.) are imported from
:mod:`smai_agent_runtime.harness_builder._main` rather than duplicated.
The role-specific code here is:

* :func:`main` — argparse + workspace + technique_contract load +
  workflow iteration.
* :func:`_run_technique_body_generation_step` — analog of harness
  builder's ``_run_baseline_step`` but bound to
  :class:`TechniqueImplementerBodyGenerationStep` and writing
  ``techniques/<technique_name>.py`` for a non-baseline entry.
* :func:`_load_technique_contract` and
  :func:`_load_parent_harness_contract` — the primary + grounding
  contracts the mini-orchestrator needs.
"""

from __future__ import annotations

import argparse
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from jinja2 import UndefinedError
from smai_core.artifacts.harness_contract import HarnessContract
from smai_core.artifacts.technique_contract import TechniqueContract

from smai_agent_runtime.agent_reasoning import build_agent, get_model_for_step
from smai_agent_runtime.agent_reasoning.model_selection import OverrideMap, SandboxedRole

# Sub-PR Step 7: role-agnostic helpers imported from the harness_builder
# sibling module per implementation_plan.md §7 ("Reuses sandbox-side and
# dispatch infrastructure"). These names start with ``_`` because they
# were private inside that module before the second role landed; the
# import here is documented as a deliberate cross-module reach within
# the same package. A future cleanup may extract them to a shared
# ``_shared.py`` once a third sandboxed role appears or the cross-role
# surface becomes painful — three concrete consumers is when the
# pattern stops being premature abstraction.
from smai_agent_runtime.harness_builder._main import (  # noqa: PLC2701
    EXIT_BAD_WORKSPACE,
    EXIT_OK,
    EXIT_STEP_FAILED,
    AgentRunner,
    _AgentRunError,  # pyright: ignore[reportPrivateUsage]
    _emit_status_line,  # pyright: ignore[reportPrivateUsage]
    _fake_agent_runner,  # pyright: ignore[reportPrivateUsage]
    _input_summary_for,  # pyright: ignore[reportPrivateUsage]
    _label_for_step,  # pyright: ignore[reportPrivateUsage]
    _read_or_empty,  # pyright: ignore[reportPrivateUsage]
    _resolve_workspace,  # pyright: ignore[reportPrivateUsage]
    _run_agent_sync,  # pyright: ignore[reportPrivateUsage]
    _run_diagnose_step,  # pyright: ignore[reportPrivateUsage]
    _run_ruff_check,  # pyright: ignore[reportPrivateUsage]
    _run_validation_step,  # pyright: ignore[reportPrivateUsage]
    _StepOutcome,  # pyright: ignore[reportPrivateUsage]
    _write_status_summary,  # pyright: ignore[reportPrivateUsage]
)
from smai_agent_runtime.prompts import load_step_prompt
from smai_agent_runtime.prompts._loader import render_user_message
from smai_agent_runtime.schemas import (
    GroundingContext,
    NoOpBaselineGrounding,
    PriorTechniqueAttempt,
    StandardLibraryGrounding,
    TechniqueBodyGenerationBundle,
    TechniqueBodyOutput,
)
from smai_agent_runtime.status import (
    StatusEmitter,
    WorkflowPlanItem,
    WorkflowStepKind,
)
from smai_agent_runtime.workflow.generator import TaskRole, generate_workflow
from smai_agent_runtime.workflow.step_types import (
    DiagnoseOnFailureStep,
    TechniqueImplementerBodyGenerationStep,
    ValidationStep,
    WorkflowStep,
)

# Hardcoded role for this module; the per-step model resolution uses it
# to look up the right ``SMAI_MODEL_TECHNIQUE_IMPLEMENTER__<STEP>`` env
# var.
_ROLE: SandboxedRole = "technique_implementer"

# Bounded body-generation retries on lint failure per architectural_decisions
# §12 #1 + research_report §8.5 (retries pinned at 3 across all roles).
_MAX_LINT_RETRIES: int = 3


@dataclass
class _DispatchContext:
    """Per-session immutable context the dispatcher threads through every step.

    Mirrors the harness_builder's ``_DispatchContext`` shape so the
    shared step handlers (``_run_validation_step``,
    ``_run_diagnose_step``) accept it via duck typing. Both contracts
    are non-None for technique_implementer (the primary is
    ``technique_contract``; the parent ``contract`` carries the factor
    + extension-point grounding the agent needs).
    """

    cg_id: str
    entry_id: str
    workspace: Path
    contract: HarnessContract
    technique_contract: TechniqueContract | None
    overrides: OverrideMap | None
    status: StatusEmitter | None = None
    body_step_kinds: dict[int, str] | None = None
    agent_runner: AgentRunner | None = None

    def __post_init__(self) -> None:
        if self.body_step_kinds is None:
            self.body_step_kinds = {}


def main(args: argparse.Namespace) -> int:
    """Entry point invoked by :mod:`smai_agent_runtime.__main__`.

    Required args: ``--entry-id``, ``--workspace``. Optional:
    ``--fake-llm`` (test seam matching harness_builder's), ``--resume``
    (architectural hedge per architectural_decisions §12 item 4; not
    yet wired for technique_implementer — surfaces as an error so the
    workflow doesn't silently swallow a resume request).
    """
    if args.entry_id is None:
        _emit_status_line(
            "error",
            reason="technique_implementer requires --entry-id",
        )
        return EXIT_BAD_WORKSPACE

    if args.resume is not None:
        # Architectural hedge: the harness_builder branch ships the
        # resume-mode loader; technique_implementer doesn't have any
        # outer-orchestrator caller emitting --resume today and the
        # multi-cycle review-feedback loop is the cross-role item.
        # Surface explicitly rather than silently fall through.
        _emit_status_line(
            "error",
            entry_id=args.entry_id,
            reason=(
                "--resume is not yet wired for technique_implementer "
                "(architectural hedge per architectural_decisions §12 item 4 "
                "lands the multi-cycle loop at a future PR)"
            ),
        )
        return EXIT_BAD_WORKSPACE

    workspace = _resolve_workspace(args.workspace)
    if workspace is None:
        _emit_status_line(
            "error",
            entry_id=args.entry_id,
            reason=f"--workspace path {args.workspace!r} does not exist or is not a directory",
        )
        return EXIT_BAD_WORKSPACE

    technique_contract = _load_technique_contract(workspace)
    if technique_contract is None:
        _emit_status_line(
            "error",
            entry_id=args.entry_id,
            reason=(
                "no technique_contract.json under contracts/ in the staged workspace; "
                "host-side materialization must run before sandbox dispatch"
            ),
        )
        return EXIT_BAD_WORKSPACE

    harness_contract = _load_parent_harness_contract(workspace)
    if harness_contract is None:
        _emit_status_line(
            "error",
            entry_id=args.entry_id,
            reason=(
                "no harness_contract.json under contracts/ in the staged workspace; "
                "the parent harness contract is required to resolve the factor + "
                "extension-point grounding for the technique body"
            ),
        )
        return EXIT_BAD_WORKSPACE

    cg_id = technique_contract.body.parent_experiment_id

    workflow = generate_workflow(harness_contract, TaskRole.TECHNIQUE_IMPLEMENTER)

    session_id = f"agent-session-{args.entry_id}-{uuid.uuid4().hex[:8]}"
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
        agent_role="technique_implementer",
        parent_kind="entry",
        parent_id=args.entry_id,
        workflow_plan=workflow_plan,
    )

    context = _DispatchContext(
        cg_id=cg_id,
        entry_id=args.entry_id,
        workspace=workspace,
        contract=harness_contract,
        technique_contract=technique_contract,
        overrides=None,
        status=emitter,
        agent_runner=_fake_agent_runner if getattr(args, "fake_llm", False) else None,
    )

    outcomes: list[_StepOutcome] = []
    last_succeeded_index: int | None = None
    for index, step in enumerate(workflow):
        step_kind: WorkflowStepKind = step.step_type
        assert context.body_step_kinds is not None  # noqa: S101 — pyright narrowing
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
            # Same fail-fast discipline as harness_builder (round-21
            # finding): ValidationStep failures fall through to
            # diagnose; everything else is terminal.
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
                _write_status_summary(workspace, args.entry_id, outcomes, succeeded=False)
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
    _write_status_summary(workspace, args.entry_id, outcomes, succeeded=True)
    return EXIT_OK


# === Workspace loaders ======================================================


def _load_technique_contract(workspace: Path) -> TechniqueContract | None:
    """Read ``contracts/technique_contract.json`` from the staged workspace."""
    path = workspace / "contracts" / "technique_contract.json"
    if not path.exists():
        return None
    try:
        return TechniqueContract.model_validate_json(path.read_text())
    except (ValueError, OSError):
        return None


def _load_parent_harness_contract(workspace: Path) -> HarnessContract | None:
    """Read ``contracts/harness_contract.json`` from the staged workspace.

    The host-side dispatcher stages the parent harness contract
    alongside the per-entry technique contract so the mini-orchestrator
    has the factor + extension-point grounding the agent needs for
    body generation.
    """
    path = workspace / "contracts" / "harness_contract.json"
    if not path.exists():
        return None
    try:
        return HarnessContract.model_validate_json(path.read_text())
    except (ValueError, OSError):
        return None


# === Per-step dispatcher ====================================================


def _dispatch_step(
    step: WorkflowStep,
    index: int,
    context: _DispatchContext,
    prior_outcomes: list[_StepOutcome],
) -> _StepOutcome:
    """Per-step dispatcher: body-generation, validation, diagnose."""
    if isinstance(step, TechniqueImplementerBodyGenerationStep):
        return _run_technique_body_generation_step(step, index, context)
    if isinstance(step, ValidationStep):
        # The shared handler accepts any object exposing
        # ``workspace`` + ``cg_id``; technique_implementer's context
        # exposes both. Cast for pyright; runtime is duck-typed.
        return _run_validation_step(step, index, cast(Any, context))
    if isinstance(step, DiagnoseOnFailureStep):
        return _run_diagnose_step(step, index, cast(Any, context), prior_outcomes)
    return _StepOutcome(
        step_index=index,
        step_type=step.step_type,
        succeeded=False,
        error=(
            f"technique_implementer mini-orchestrator has no handler for "
            f"{step.step_type!r}; the workflow generator emitted an "
            "unexpected step (harness_builder-only steps don't belong here)"
        ),
    )


# === Body-generation step (D7b, step_kind="technique") =======================


def _run_technique_body_generation_step(
    step: TechniqueImplementerBodyGenerationStep,
    index: int,
    context: _DispatchContext,
) -> _StepOutcome:
    """Real handler for :class:`TechniqueImplementerBodyGenerationStep`.

    Mirror of the harness_builder's baseline-step shape but bound to
    ``step_kind="technique"`` in the D7b bundle and writing
    ``techniques/<technique_name>.py`` for the non-baseline entry.
    """
    try:
        prompt = load_step_prompt(_ROLE, "step_2_fill_technique_body.yaml")
    except (FileNotFoundError, ValueError) as exc:
        return _StepOutcome(
            step_index=index,
            step_type=step.step_type,
            succeeded=False,
            error=f"failed to load technique_implementer/step_2_fill_technique_body.yaml: {exc}",
        )

    target_path = context.workspace / step.write_to_path
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if context.technique_contract is None:
        return _StepOutcome(
            step_index=index,
            step_type=step.step_type,
            succeeded=False,
            error=(
                "technique_implementer body-generation requires a loaded "
                "technique_contract on the dispatch context; this is a "
                "wiring bug in main()'s workspace materialization"
            ),
        )

    technique_contract = context.technique_contract

    grounding = _resolve_technique_grounding(step, context)
    harness_source = _read_harness_source(context.workspace)
    manifest_json = _read_or_empty(context.workspace / "harness_api_manifest.json")
    baseline_source = _read_or_empty(context.workspace / "techniques" / "baseline.py") or None

    technique_id = technique_contract.body.technique_id
    technique_params = (
        dict(technique_contract.body.technique_params)
        if technique_contract.body.technique_params is not None
        else None
    )
    technique_name = _technique_name_from_write_path(step.write_to_path)

    provider, model_id = get_model_for_step(
        _ROLE,
        step.step_type,
        overrides=context.overrides,
    )

    prior_failed_attempts: list[PriorTechniqueAttempt] = []
    for attempt_index in range(_MAX_LINT_RETRIES + 1):
        bundle = TechniqueBodyGenerationBundle(
            step_kind="technique",
            cg_id=context.cg_id,
            entry_id=context.entry_id,
            technique_name=technique_name,
            is_baseline=False,
            factor_dimension=context.contract.body.factor.name,
            factor_type=context.contract.body.factor.type,
            technique_id=technique_id,
            technique_params=technique_params,
            grounding=grounding,
            harness_api_manifest_json=manifest_json,
            harness_source=harness_source,
            baseline_source=baseline_source,
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
                trace_step_name=f"{index:02d}_technique_attempt_{attempt_index}",
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


# === Grounding + helpers ====================================================


def _resolve_technique_grounding(
    step: TechniqueImplementerBodyGenerationStep,
    context: _DispatchContext,
) -> GroundingContext:
    """Resolve the per-step grounding for a technique body-generation.

    Reads the staged ``grounding/technique_grounding.json`` if present
    (host-side dispatcher writes it from the TechniqueRef + grounding
    context the orchestrator carries). Falls back to a standard-library
    grounding when the file is missing and the technique_id is
    populated (the orchestrator-default path for non-paper-grounded
    techniques). For the architectural edge case where the orchestrator
    emits a technique step against a null technique_id (which the
    generator shouldn't normally do but the type system allows),
    returns a ``no_op_baseline`` grounding so the agent flags the gap
    rather than silently inventing semantics.
    """
    grounding_path = context.workspace / "grounding" / "technique_grounding.json"
    if grounding_path.exists():
        try:
            return _adapter_validate_grounding(grounding_path.read_text())
        except (ValueError, OSError):
            # Fall through to default below
            pass

    if context.technique_contract is None or context.technique_contract.body.technique_id is None:
        # Schema carries kind only; the bundle's other fields (technique_id,
        # technique_params) already surface the missing-grounding shape to
        # the agent. The prompt's no_op_baseline branch explicitly tells
        # the agent to flag this gap in reasoning.
        return NoOpBaselineGrounding()

    return StandardLibraryGrounding(
        technique_description=(
            f"Standard-library technique '{context.technique_contract.body.technique_id}'. "
            "No paper extraction or proposal grounding was staged into the workspace; "
            "implement from your library knowledge of this technique name."
        ),
    )


def _adapter_validate_grounding(raw: str) -> GroundingContext:
    """Validate a grounding-context JSON against the discriminated union.

    Lazy import keeps the pydantic TypeAdapter construction out of the
    module-import hot path.
    """
    from pydantic import TypeAdapter  # noqa: PLC0415

    adapter: TypeAdapter[GroundingContext] = TypeAdapter(GroundingContext)
    return adapter.validate_json(raw)


def _read_harness_source(workspace: Path) -> dict[str, str]:
    """Read all ``harness/*.py`` files into a {relative-path: content} map.

    Mirrors the harness_builder helper but is duplicated here to keep
    the cross-module import surface tight (the original is private to
    harness_builder._main and adding a fourth public name there for
    one consumer in this module is more friction than the copy).
    Round-21's harness body-generation step expected ~50KB total; the
    agent's harness output for round 21 was ~14KB so there's room.
    """
    harness_dir = workspace / "harness"
    if not harness_dir.is_dir():
        return {}
    out: dict[str, str] = {}
    for path in sorted(harness_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(workspace).as_posix()
        try:
            out[rel] = path.read_text(encoding="utf-8")
        except OSError:
            out[rel] = ""
    return out


def _technique_name_from_write_path(write_to_path: str) -> str:
    """Extract the technique file stem from ``techniques/<name>.py``.

    The workflow generator constructs ``write_to_path`` as
    ``f"techniques/{technique_id}.py"`` per
    :func:`_generate_technique_implementer_workflow`. This helper
    re-extracts the stem for the bundle's ``technique_name`` field
    (the agent uses the name as the filename it writes).
    """
    name = Path(write_to_path).stem
    if not name:
        return "unknown_technique"
    return name


__all__ = [
    "main",
]
