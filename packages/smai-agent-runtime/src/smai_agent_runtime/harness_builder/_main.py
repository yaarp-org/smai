"""Harness-builder sandbox-side mini-orchestrator skeleton.

Agent-refactor Step 4 sub-PR B. Replaces the Step 3 stub body with a
mini-orchestrator that:

1. Parses ``--cg-id`` / ``--workspace`` / ``--resume`` argv.
2. Reads the :class:`HarnessContract` from the bind-mounted workspace.
3. Calls :func:`smai_agent_runtime.workflow.generate_workflow` (sub-PR A).
4. Iterates the resulting :class:`WorkflowStep` list, dispatching each
   to a **fake handler** that logs to stdout and writes a canned output
   to the workspace.
5. Writes a placeholder ``status.json`` to ``outputs/`` on exit.

The fake-handler shape (no real agent reasoning) is deliberate per the
sub-PR B brief: this sub-PR validates the dispatch round-trip + the
host ↔ sandbox plumbing without committing to PydanticAI integration.
Sub-PR C replaces the fakes with real :class:`Agent(output_type=...)`
calls.

The ``--resume <prior_session_id>`` flag is accepted at argparse and
routed to a no-op stub that exits with "resume not yet implemented in
workflow shape" — sub-PR D's resume-mode wiring lands on top.

Workspace layout assumed (per
:func:`smai_agents.agents.harness_builder._materialize_harness_builder_workspace`):

    /workspace/contracts/harness_contract.json
    /workspace/contracts/technique_contract.json
    /workspace/harness_api_reference.md
    /workspace/harness/ (placeholder from create_workspace_skeleton)
    /workspace/techniques/ (placeholder from create_workspace_skeleton)
    /workspace/outputs/ (created here; status.json + validation_results.json land here)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smai_core.artifacts.harness_contract import HarnessContract

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


@dataclass
class _StepOutcome:
    """One step's result. Mirrors D9's sketch shape."""

    step_index: int
    step_type: str
    succeeded: bool
    error: str | None = None


def main(args: argparse.Namespace) -> int:
    """Entry point invoked by :mod:`smai_agent_runtime.__main__`.

    Sub-PR B (Step 4) replaces the Step 3 "not yet implemented" stub
    with the mini-orchestrator skeleton. Sub-PR C wires real
    PydanticAI Agent calls in place of the fake handlers below.
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

    outcomes: list[_StepOutcome] = []
    for index, step in enumerate(workflow):
        _emit_status_line("step_start", step_index=index, step_type=step.step_type)
        outcome = _dispatch_fake(step, index, workspace, outcomes)
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


# === Status emission ========================================================


def _emit_status_line(event_type: str, **fields: Any) -> None:
    """Write one JSON line to stdout per D10's status-emit schema (placeholder).

    Sub-PR B's status-emit is the placeholder shape — the full schema
    (D10 envelope) lands in sub-PR C alongside the real PydanticAI
    integration. The fields here are intentionally minimal so the host's
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
    sub-PR C. The host's post-terminal harvest pulls this back so the
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
        "note": "sub-PR B placeholder; real D10 schema lands in sub-PR C",
    }
    (outputs_dir / "status.json").write_text(json.dumps(payload, indent=2))


# === Per-step fake handlers =================================================


def _dispatch_fake(
    step: WorkflowStep,
    index: int,
    workspace: Path,
    prior_outcomes: list[_StepOutcome],
) -> _StepOutcome:
    """Sub-PR B fake-handler dispatcher.

    Each branch writes a canned output to the workspace so the post-
    terminal harvest has files to pull back. Sub-PR C replaces these
    with real PydanticAI :class:`Agent(output_type=...)` calls per
    D7a / D7b / D7c bundle shapes.

    :class:`DiagnoseOnFailureStep` is the conditional path: its anchor
    step (a :class:`ValidationStep`) succeeded in the fake-handler shape,
    so the diagnose step is a no-op pass-through (mirrors D9's sketch
    semantics).
    """
    if isinstance(step, HarnessBuilderBodyGenerationStep):
        return _fake_body_generation_step(step, index, workspace)
    if isinstance(step, BaselineGenerationStep):
        return _fake_baseline_step(step, index, workspace)
    if isinstance(step, ValidationStep):
        return _fake_validation_step(step, index, workspace)
    if isinstance(step, DiagnoseOnFailureStep):
        return _fake_diagnose_step(step, index, prior_outcomes)
    if isinstance(step, ManifestEmitStep):
        return _fake_manifest_emit_step(step, index, workspace)
    # Unknown / not-yet-handled step types: explicit failure so the
    # workflow surfaces them rather than silently advancing.
    return _StepOutcome(
        step_index=index,
        step_type=step.step_type,
        succeeded=False,
        error=f"sub-PR B mini-orchestrator has no fake handler for {step.step_type!r}",
    )


def _fake_body_generation_step(
    step: HarnessBuilderBodyGenerationStep,
    index: int,
    workspace: Path,
) -> _StepOutcome:
    """Canned ABI-function body. Writes a stub file at the step's
    ``write_to_path`` annotated with the function name + signature so
    the post-harvest output has something operators can eyeball."""
    target = workspace / step.write_to_path
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = target.read_text() if target.exists() else ""
    body = (
        f"# [sub-PR B fake] {step.function_name} - index {step.function_index}\n"
        f"# Signature: {step.function_signature}\n"
        "# Real implementation lands in sub-PR C via "
        "PydanticAI Agent(output_type=FilledHarnessBody).\n\n"
        f"def {step.function_name}(*args, **kwargs):\n"
        f"    raise NotImplementedError({step.function_name!r})\n\n"
    )
    target.write_text(existing + body)
    print(
        f"[sub-PR B fake] body_generation step {index} "
        f"({step.function_name}) wrote stub to {target}",
        file=sys.stderr,
    )
    return _StepOutcome(step_index=index, step_type=step.step_type, succeeded=True)


def _fake_baseline_step(
    step: BaselineGenerationStep,
    index: int,
    workspace: Path,
) -> _StepOutcome:
    """Canned baseline-technique body. Writes a stub at
    ``techniques/baseline.py`` annotated with the factor type."""
    target = workspace / step.write_to_path
    target.parent.mkdir(parents=True, exist_ok=True)
    body = (
        f"# [sub-PR B fake] baseline ({step.factor_type} factor, "
        f"technique_id={step.baseline_technique_id!r})\n"
        "# Real implementation lands in sub-PR C via "
        "PydanticAI Agent(output_type=FilledTechniqueBody).\n\n"
        "def baseline(*args, **kwargs):\n"
        "    raise NotImplementedError('baseline')\n"
    )
    target.write_text(body)
    print(
        f"[sub-PR B fake] baseline_generation step {index} wrote stub to {target}",
        file=sys.stderr,
    )
    return _StepOutcome(step_index=index, step_type=step.step_type, succeeded=True)


def _fake_validation_step(
    step: ValidationStep,
    index: int,
    workspace: Path,
) -> _StepOutcome:
    """Canned passing validation_results.json. Sub-PR C replaces with
    a real ``python experiment.py --mode validation`` subprocess (or
    Compute submit per the dispatch_target field)."""
    outputs_dir = workspace / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "succeeded": True,
        "technique_id": step.technique_id,
        "seed": step.seed,
        "dispatch_target": step.dispatch_target,
        "note": "sub-PR B fake handler; real subprocess validation lands in sub-PR C",
    }
    (outputs_dir / "validation_results.json").write_text(json.dumps(payload, indent=2))
    # Also drop it at the workspace root for the publish-prefix's "validation_results.json"
    # root entry — the host harvest publishes from there, mirroring the
    # legacy harness-builder workspace layout.
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
    succeeds, so this branch is always the pass-through path. Sub-PR C
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
    # Anchor failed — sub-PR B fake just records the would-fix; sub-PR C
    # runs the real diagnose loop.
    print(
        f"[sub-PR B fake] diagnose step {index} would diagnose failure of step "
        f"{step.anchor_step_index} ({anchor.step_type}); real loop lands in sub-PR C",
        file=sys.stderr,
    )
    return _StepOutcome(step_index=index, step_type=step.step_type, succeeded=True)


def _fake_manifest_emit_step(
    step: ManifestEmitStep,
    index: int,
    workspace: Path,
) -> _StepOutcome:
    """Canned manifest.json. Sub-PR C wires the real round-15
    harness_version_hash recompute + ``HarnessAPIManifest`` freeze."""
    payload: dict[str, Any] = {
        "runtime_template_version": step.runtime_template_version,
        "parent_harness_contract_hash": step.parent_harness_contract_hash,
        "extension_points": [],
        "harness_version_hash": "sub-pr-b-placeholder-hash",
        "note": "sub-PR B fake; real round-15 hash + freeze lands in sub-PR C",
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
