"""Harness-builder sandbox-side mini-orchestrator.

Agent-refactor Step 4 sub-PR C1 replaces sub-PR B's fake handlers for
the two body-generation step types with real PydanticAI
``Agent(output_type=...)`` calls. The remaining three fake handlers
(ValidationStep, DiagnoseOnFailureStep, ManifestEmitStep) stay until
sub-PR C2.

Workflow shape:

1. Parse ``--cg-id`` / ``--workspace`` / ``--resume`` argv.
2. Read :class:`HarnessContract` from ``contracts/harness_contract.json``.
3. Call :func:`smai_agent_runtime.workflow.generate_workflow` (sub-PR A).
4. Iterate the resulting :class:`WorkflowStep` list. Body-generation
   steps invoke PydanticAI Agents bound to the D7a / D7b output schemas;
   scripted-and-fake steps run the sub-PR B canned-output path.
5. Write a placeholder ``status.json`` to ``outputs/`` on exit.

The ``--resume <prior_session_id>`` flag is accepted at argparse and
routed to a no-op stub that exits with "resume not yet implemented in
workflow shape" — sub-PR D's resume-mode wiring lands on top.

Workspace layout assumed:

    /workspace/contracts/harness_contract.json
    /workspace/contracts/technique_contract.json    (optional — used by baseline step)
    /workspace/harness_api_reference.md             (optional — used by body-gen steps)
    /workspace/harness_api_manifest.json            (optional — used by baseline step)
    /workspace/grounding/baseline_grounding.json    (optional — used by baseline step)
    /workspace/harness/ (placeholder from create_workspace_skeleton)
    /workspace/techniques/ (placeholder from create_workspace_skeleton)
    /workspace/outputs/ (created here; status.json + validation_results.json land here)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from jinja2 import UndefinedError
from pydantic import BaseModel, ValidationError
from smai_core.artifacts.harness_contract import HarnessContract
from smai_core.artifacts.technique_contract import TechniqueContract
from smai_runtime.components import ADMISSIBLE_PATTERNS_FOR_KEY, COMPONENT_FIELD_FOR_KEY

from smai_agent_runtime.agent_reasoning import build_agent, get_model_for_step
from smai_agent_runtime.agent_reasoning.model_selection import (
    OverrideMap,
    SandboxedRole,
)
from smai_agent_runtime.prompts import load_step_prompt
from smai_agent_runtime.prompts._loader import render_user_message
from smai_agent_runtime.schemas import (
    ExtensionPointSpec,
    FunctionSignature,
    GroundingContext,
    HarnessBuilderBodyGenerationInput,
    HarnessBuilderBodyGenerationOutput,
    LintFailure,
    NoOpBaselineGrounding,
    PaperExtractGrounding,
    PriorFailedAttempt,
    PriorTechniqueAttempt,
    ProposalGrounding,
    ReviewerAttestedGrounding,
    StandardLibraryGrounding,
    TechniqueBodyGenerationBundle,
    TechniqueBodyOutput,
)
from smai_agent_runtime.workflow.generator import TaskRole, generate_workflow
from smai_agent_runtime.workflow.step_types import (
    BaselineGenerationStep,
    DiagnoseOnFailureStep,
    HarnessBuilderBodyGenerationStep,
    ManifestEmitStep,
    ValidationStep,
    WorkflowStep,
)

# Sandbox-side exit codes. The host's
# ``Compute.status(handle).exit_code`` reads these.
EXIT_OK = 0
EXIT_STEP_FAILED = 1
EXIT_RESUME_NOT_IMPLEMENTED = 64
EXIT_BAD_WORKSPACE = 65

# Hardcoded role for this module; the per-step model resolution uses it
# to look up the right ``SMAI_MODEL_HARNESS_BUILDER__<STEP>`` env var.
_ROLE: SandboxedRole = "harness_builder"

# Bounded body-generation retries on lint failure per architectural_decisions
# §12 #1 + research_report §8.5 (retries pinned at 3 across all roles).
_MAX_LINT_RETRIES: int = 3

# Cap the lint output the agent sees so retry bundles stay below the
# 50KB-token budget. ~3KB head+tail matches the D7a schema's docstring
# convention; we hand back the tail (which carries the actionable
# diagnostic) plus the count of dropped lines if the linter spewed
# heavily.
_LINT_OUTPUT_CAP_CHARS: int = 3000

# Test-substitution env var. When set to a truthy value, _run_agent_sync
# returns a deterministic conforming output for the requested
# output_type rather than calling the configured LLM. This is the only
# way the cross-process subprocess test (test_python_m_subprocess_runs_workflow
# in sub-PR B's test_entry.py) can drive the workflow end-to-end without
# real Bedrock credentials. In-process tests prefer the per-step
# monkeypatch-on-_run_agent_sync pattern because it captures the bundle
# the dispatcher built (the assertion surface the brief calls for).
_FAKE_LLM_ENV_VAR: str = "SMAI_AGENT_RUNTIME_FAKE_LLM"


@dataclass
class _StepOutcome:
    """One step's result. Mirrors D9's sketch shape."""

    step_index: int
    step_type: str
    succeeded: bool
    error: str | None = None


@dataclass
class _DispatchContext:
    """Per-session immutable context the dispatcher threads through every step."""

    cg_id: str
    workspace: Path
    contract: HarnessContract
    technique_contract: TechniqueContract | None
    overrides: OverrideMap | None


def main(args: argparse.Namespace) -> int:
    """Entry point invoked by :mod:`smai_agent_runtime.__main__`.

    Sub-PR B (Step 4) replaced the Step 3 "not yet implemented" stub
    with the mini-orchestrator skeleton; sub-PR C1 wires real PydanticAI
    Agent calls in place of the body-generation fake handlers below.
    """
    if args.cg_id is None:
        _emit_status_line("error", reason="harness_builder requires --cg-id")
        return EXIT_BAD_WORKSPACE

    if args.resume is not None:
        # Sub-PR D wires resume-mode workflow logic. Sub-PR B's
        # contract: the flag is accepted at argparse and routes to a
        # no-op stub so the entry signature is pinned (per arch §12
        # item 4 — the resume-prep architectural hedge).
        _emit_status_line(
            "resume_not_implemented",
            cg_id=args.cg_id,
            resume_from=args.resume,
            reason=(
                "resume not yet implemented in workflow shape "
                "(architectural hedge per architectural_decisions §12 item 4; "
                "sub-PR D wires the resume-mode workflow logic)"
            ),
        )
        return EXIT_RESUME_NOT_IMPLEMENTED

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
    _emit_status_line(
        "session_start",
        cg_id=args.cg_id,
        step_count=len(workflow),
        step_types=[s.step_type for s in workflow],
    )

    context = _DispatchContext(
        cg_id=args.cg_id,
        workspace=workspace,
        contract=contract,
        technique_contract=_load_technique_contract(workspace),
        overrides=None,  # sub-PR D wires the host-side override map; for now env-only.
    )

    outcomes: list[_StepOutcome] = []
    for index, step in enumerate(workflow):
        _emit_status_line("step_start", step_index=index, step_type=step.step_type)
        outcome = _dispatch_step(step, index, context, outcomes)
        outcomes.append(outcome)
        if outcome.succeeded:
            _emit_status_line("step_success", step_index=index, step_type=step.step_type)
        else:
            _emit_status_line(
                "step_failure",
                step_index=index,
                step_type=step.step_type,
                reason=outcome.error or "unknown",
            )
            _write_status_summary(workspace, args.cg_id, outcomes, succeeded=False)
            return EXIT_STEP_FAILED

    _write_status_summary(workspace, args.cg_id, outcomes, succeeded=True)
    _emit_status_line("session_end_success", step_count=len(workflow))
    return EXIT_OK


# === Argparse / workspace helpers ============================================


def _resolve_workspace(workspace_arg: object) -> Path | None:
    """Resolve and validate the ``--workspace`` argv.

    Returns ``None`` if the path does not exist or is not a directory.
    The :mod:`__main__` entry point's argparse may pass a ``str`` or a
    pre-parsed ``Path`` depending on how ``type=`` was wired; this
    helper handles both.
    """
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
    """Read the staged :class:`TechniqueContract` if the host materialized one.

    Optional: sub-PR B's tests stage only the harness contract. Sub-PR C1's
    baseline-generation handler uses this when present to drive the
    grounding lookup; absent, it falls back to a no-op-baseline shape
    plus a workflow-level warning.
    """
    contract_path = workspace / "contracts" / "technique_contract.json"
    if not contract_path.exists():
        return None
    try:
        return TechniqueContract.model_validate_json(contract_path.read_text())
    except (ValueError, OSError):
        return None


# === Status emission ========================================================


def _emit_status_line(event_type: str, **fields: Any) -> None:
    """Write one JSON line to stdout per D10's status-emit schema (placeholder).

    Sub-PR B's status-emit is the placeholder shape — the full schema
    (D10 envelope) lands in sub-PR C2 alongside the real status emit.
    The fields here are intentionally minimal so the host's
    ``Compute.logs(handle)`` tail surfaces enough to read what
    happened, without committing to the full schema.

    Flush eagerly so a host worker tailing logs sees progress
    incrementally (matters for the long-running real-agent path).
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

    Placeholder shape; the real D10 stdout-JSON status emit lands in
    sub-PR C2. The host's post-terminal harvest pulls this back so the
    operator can read what ran without scraping stdout.
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
        "note": "sub-PR B/C1 placeholder; real D10 schema lands in sub-PR C2",
    }
    (outputs_dir / "status.json").write_text(json.dumps(payload, indent=2))


# === Per-step dispatcher =====================================================


def _dispatch_step(
    step: WorkflowStep,
    index: int,
    context: _DispatchContext,
    prior_outcomes: list[_StepOutcome],
) -> _StepOutcome:
    """Sub-PR C1 dispatcher: real handlers for body-generation, fakes elsewhere.

    Sub-PR C2 replaces the ValidationStep / DiagnoseOnFailureStep /
    ManifestEmitStep fakes with real implementations.
    """
    if isinstance(step, HarnessBuilderBodyGenerationStep):
        return _run_body_generation_step(step, index, context)
    if isinstance(step, BaselineGenerationStep):
        return _run_baseline_step(step, index, context)
    if isinstance(step, ValidationStep):
        return _fake_validation_step(step, index, context.workspace)
    if isinstance(step, DiagnoseOnFailureStep):
        return _fake_diagnose_step(step, index, prior_outcomes)
    if isinstance(step, ManifestEmitStep):
        return _fake_manifest_emit_step(step, index, context.workspace)
    # Unknown / not-yet-handled step types: explicit failure so the
    # workflow surfaces them rather than silently advancing.
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
    """Real handler for :class:`HarnessBuilderBodyGenerationStep`.

    Builds the D7a input bundle from workspace artifacts, calls the
    PydanticAI Agent bound to :class:`HarnessBuilderBodyGenerationOutput`,
    writes the agent's ``module_source`` to ``write_to_path``, runs
    ``ruff check`` for syntax, and re-prompts on lint failure with
    ``prior_failed_attempts`` populated (bounded retries, 3).

    Body-generation steps register no tools per architectural_decisions
    §12 #1 "Design discipline." If the agent appears to need more
    information, the fix is to extend the D7a bundle, not to add a
    ``read_file`` tool.
    """
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
    harness_api_reference = _read_or_empty(context.workspace / "harness_api_reference.md")

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
            output = _run_agent_sync(agent, user_message, HarnessBuilderBodyGenerationOutput)
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
        _emit_status_line(
            "lint_retry",
            step_index=index,
            step_type=step.step_type,
            attempt_index=attempt_index,
            output_tail=lint_outcome[-200:],
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
    """Real handler for :class:`BaselineGenerationStep`.

    Builds the D7b input bundle with ``step_kind="baseline"``, resolves
    the :class:`GroundingContext` from the staged technique-contract +
    optional grounding artifact, calls the PydanticAI Agent bound to
    :class:`TechniqueBodyOutput`, writes ``techniques/baseline.py``, and
    runs the lint-retry loop (bounded retries, 3).
    """
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
            output = _run_agent_sync(agent, user_message, TechniqueBodyOutput)
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
        _emit_status_line(
            "lint_retry",
            step_index=index,
            step_type=step.step_type,
            attempt_index=attempt_index,
            output_tail=lint_outcome[-200:],
        )

    return _StepOutcome(
        step_index=index,
        step_type=step.step_type,
        succeeded=False,
        error=f"lint-retry budget exhausted ({_MAX_LINT_RETRIES} retries)",
    )


# === Bundle-construction helpers =============================================


def _resolve_extension_points(contract: HarnessContract) -> list[ExtensionPointSpec]:
    """Construct the D7a ``extension_points`` list from the contract.

    Iterates :data:`COMPONENT_FIELD_FOR_KEY` (the round-15 introspection
    pattern). The sub-PR C1 placeholder surfaces every closed-v1 key
    with the contract's factor type setting the ``factor_role``; sub-PR
    D narrows this to only the keys the contract's factor type
    implicates once the host-side dispatcher passes a filtered subset
    through to the sandbox.
    """
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


def _resolve_baseline_grounding(
    step: BaselineGenerationStep,
    context: _DispatchContext,
) -> GroundingContext:
    """Resolve the D7b :data:`GroundingContext` for the baseline step.

    Sub-PR C1 reads from one of two sources, in priority order:

    1. ``grounding/baseline_grounding.json`` in the workspace (host
       staged this from upstream ArtifactStore / MetadataStore lookups
       via :class:`FidelityAnchor.kind`; sub-PR D wires the host side).
    2. The staged :class:`TechniqueContract`'s ``fidelity_anchor``
       dispatched per its ``kind``. For non-baseline-friendly variants
       (paper / proposal anchors with no staged extract / description),
       falls back to :class:`StandardLibraryGrounding` with the
       technique-id as the description (visibly broken; sub-PR D
       finishes the host-side staging so this branch becomes
       unreachable in practice).

    Additive factors with no anchor default to
    :class:`NoOpBaselineGrounding` per DEC-013.
    """
    staged = context.workspace / "grounding" / "baseline_grounding.json"
    if staged.exists():
        try:
            raw = json.loads(staged.read_text())
            return _adapter_validate_grounding(raw)
        except (ValueError, OSError) as exc:
            _emit_status_line(
                "grounding_staged_invalid",
                staged_path=str(staged),
                reason=str(exc),
            )

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
    # Exhaustive over FidelityAnchor.kind; the type checker treats this
    # branch as unreachable. Defensive fall-through preserves a clean
    # diagnostic if the union grows in a future revision.
    return StandardLibraryGrounding(
        technique_description=f"Baseline technique {step.baseline_technique_id!r}."
    )


def _adapter_validate_grounding(raw: Any) -> GroundingContext:
    """Validate a raw dict into the :data:`GroundingContext` union.

    Pydantic v2's :class:`pydantic.TypeAdapter` is the canonical entry
    point for validating against a discriminated-union alias; we
    instantiate inline because the alias is the only consumer and the
    overhead is negligible.
    """
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


def _read_or_empty(path: Path) -> str:
    """Read a text file or return an empty string. Used for optional staged inputs."""
    if not path.exists():
        return ""
    try:
        return path.read_text()
    except OSError:
        return ""


def _coerce_abi_name(name: str) -> Any:
    """Cast a raw ABI function name to the schema's ``ABIFunctionName`` Literal.

    Pydantic enforces the Literal at validation time; this helper exists
    so the type checker is satisfied with the cast at the call site. The
    workflow generator produces names from the v1 ABI table that match
    the Literal by construction, so the cast is never lossy at runtime.
    """
    return name


def _abi_purpose(name: str) -> str:
    """One-sentence purpose for the ABI function ``name``.

    Hardcoded for the v1 ABI; mirrors the spec text in
    ``10-runtime-and-templates.md`` §8.2. Future ABIs extend the lookup
    or carry purpose directly on the contract.
    """
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


# === Agent / lint helpers ====================================================


_AgentOutputT = TypeVar("_AgentOutputT", bound=BaseModel)


class _AgentRunError(RuntimeError):
    """Raised when a PydanticAI agent call surfaces an unrecoverable error.

    Sub-PR C1 wraps the run with a thin diagnostic so the caller can
    drop a clean step-failure message without leaking PydanticAI's
    internal exception types into the status emit.
    """


def _run_agent_sync(
    agent: Any,
    user_message: str,
    output_type: type[_AgentOutputT],
) -> _AgentOutputT:
    """Invoke a PydanticAI :class:`Agent` synchronously and unwrap the output.

    Wraps :meth:`Agent.run_sync` so the call site does not import
    ``pydantic_ai`` directly (keeps the harness_builder module
    PydanticAI-agnostic at the type level; the dependency lives in
    :mod:`smai_agent_runtime.agent_reasoning`).

    Test-substitution: when the :data:`_FAKE_LLM_ENV_VAR` env var is
    truthy, returns :func:`_fake_llm_output` instead of calling the
    real agent. Required for cross-process subprocess tests; in-process
    tests prefer monkeypatching this callable directly so they can
    capture the bundle the dispatcher built.
    """
    if os.environ.get(_FAKE_LLM_ENV_VAR):
        return _fake_llm_output(output_type, user_message)

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
    return output


def _fake_llm_output(
    output_type: type[_AgentOutputT],
    user_message: str,
) -> _AgentOutputT:
    """Deterministic conforming output for the requested schema.

    Active only when :data:`_FAKE_LLM_ENV_VAR` is truthy. Produces a
    minimum-valid Python source body so the downstream lint + write
    path runs end-to-end; the bodies are intentionally trivial.

    Inspects ``user_message`` for the target function name (body steps
    re-render with the right discriminator) so the
    ``output.function_name == step.function_name`` assertion stays
    intact across the three v1 ABI functions.
    """
    if output_type is HarnessBuilderBodyGenerationOutput:
        # The body-generation prompt renders the target ABI function as
        # ``fill harness/__init__.py body for `<name>``` near the top of
        # the user message (the H1 in step_2_fill_init_py.yaml's
        # initial_user_message_template). Look for that exact marker so
        # the three ABI names disambiguate cleanly: substring matches
        # like "evaluate" lose to "run_training_loop" because the latter
        # also contains snippets that overlap with other names'
        # ``purpose`` prose.
        fn_name: Any = "build_harness"
        for candidate in ("build_harness", "run_training_loop", "evaluate"):
            if f"body for `{candidate}`" in user_message:
                fn_name = candidate
                break
        module_source = (
            '"""Fake-LLM stub module for sub-PR C1 cross-process tests."""\n'
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
                '"""Fake-LLM baseline stub for sub-PR C1 cross-process tests."""\n'
                "\n"
                "def baseline(*args, **kwargs):\n"
                "    return None\n"
            ),
            reasoning="fake-llm stub: deterministic conforming output for tests",
        )  # type: ignore[return-value]
    raise _AgentRunError(
        f"_FAKE_LLM_ENV_VAR active but no canned output registered for {output_type.__name__!r}"
    )


def _run_ruff_check(target_path: Path) -> str | None:
    """Run ``ruff check`` against ``target_path``.

    Returns ``None`` on clean lint, or the captured output (head + tail
    capped) on failure. Errors invoking ruff itself surface as failure
    output prefixed with ``"ruff invocation error: "`` so the agent
    sees a structured signal rather than silent success.
    """
    try:
        proc = subprocess.run(
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


# === Sub-PR B fake handlers (kept until sub-PR C2) ===========================


def _fake_validation_step(
    step: ValidationStep,
    index: int,
    workspace: Path,
) -> _StepOutcome:
    """Canned passing validation_results.json. Sub-PR C2 replaces with
    a real ``python experiment.py --mode validation`` subprocess (or
    Compute submit per the dispatch_target field)."""
    outputs_dir = workspace / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "succeeded": True,
        "technique_id": step.technique_id,
        "seed": step.seed,
        "dispatch_target": step.dispatch_target,
        "note": "sub-PR B fake handler; real subprocess validation lands in sub-PR C2",
    }
    (outputs_dir / "validation_results.json").write_text(json.dumps(payload, indent=2))
    (workspace / "validation_results.json").write_text(json.dumps(payload, indent=2))
    print(
        f"[sub-PR B fake] validation step {index} wrote canned passing payload",
        file=sys.stderr,
    )
    return _StepOutcome(step_index=index, step_type=step.step_type, succeeded=True)


def _fake_diagnose_step(
    step: DiagnoseOnFailureStep,
    index: int,
    prior_outcomes: list[_StepOutcome],
) -> _StepOutcome:
    """Conditional: pass-through when the anchor succeeded.

    In the sub-PR B fake-handler shape every :class:`ValidationStep`
    succeeds, so this branch is always the pass-through path. Sub-PR C2
    threads the real D7c diagnose bundle through when the anchor
    actually fails."""
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
        return _StepOutcome(step_index=index, step_type=step.step_type, succeeded=True)
    print(
        f"[sub-PR B fake] diagnose step {index} would diagnose failure of step "
        f"{step.anchor_step_index} ({anchor.step_type}); real loop lands in sub-PR C2",
        file=sys.stderr,
    )
    return _StepOutcome(step_index=index, step_type=step.step_type, succeeded=True)


def _fake_manifest_emit_step(
    step: ManifestEmitStep,
    index: int,
    workspace: Path,
) -> _StepOutcome:
    """Canned manifest.json. Sub-PR C2 wires the real round-15
    harness_version_hash recompute + ``HarnessAPIManifest`` freeze."""
    payload: dict[str, Any] = {
        "runtime_template_version": step.runtime_template_version,
        "parent_harness_contract_hash": step.parent_harness_contract_hash,
        "extension_points": [],
        "harness_version_hash": "sub-pr-b-placeholder-hash",
        "note": "sub-PR B fake; real round-15 hash + freeze lands in sub-PR C2",
    }
    (workspace / "manifest.json").write_text(json.dumps(payload, indent=2))
    print(
        f"[sub-PR B fake] manifest_emit step {index} wrote canned manifest.json",
        file=sys.stderr,
    )
    return _StepOutcome(step_index=index, step_type=step.step_type, succeeded=True)


__all__ = [
    "EXIT_BAD_WORKSPACE",
    "EXIT_OK",
    "EXIT_RESUME_NOT_IMPLEMENTED",
    "EXIT_STEP_FAILED",
    "main",
]
