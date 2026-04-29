"""SMAI :class:`PipelineSpec` instances — Phase 2 / Task 2.C4.

Per ``designs/smai/03-state-machine.md`` §3 and DEC-032, the SMAI v2
engine binary registers multiple :class:`PipelineSpec` instances at
worker startup. Phase 2 ships the CG-execution + entry-side specs;
Phase 3 (Tasks 3.E1 / 3.E2 / 3.E3) lifts the proposal-pipeline,
paper-ingestion, and run sub-state-machine specs into peers.

Public surface:

* :func:`build_cg_execution_spec` — entity_kind ``"cg"``; the CG-level
  state machine (``draft → implementing → implemented → running →
  evaluating → complete`` plus the three failure terminals).
* :func:`build_cg_entries_spec` — entity_kind ``"entry"``; per-entry
  technique-implementer dispatch.
* :func:`register_smai_specs` — convenience helper that constructs and
  registers both specs in one call. The CLI bootstrap (Task 2.D2)
  invokes this after :class:`RuntimeConfig` is assembled.

Spec authors who want finer control (e.g., overriding the seeds set
or wiring a non-default container image) call the factory functions
directly and register manually via
:func:`smai_orchestrator.runtime.register_pipeline_spec`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from smai_core.plugins import LlmProvider

from smai_orchestrator.runtime.registry import register_pipeline_spec
from smai_orchestrator.runtime.spec import PipelineSpec
from smai_orchestrator.specs.cg_entries import build_cg_entries_spec
from smai_orchestrator.specs.cg_execution import (
    CG_ENTRIES_SPEC_NAME,
    CG_EXECUTION_SPEC_NAME,
    CODE_REVIEW_KEY_TEMPLATE,
    CONTEXTUAL_VERDICT_KEY_TEMPLATE,
    EVALUATION_ERROR_KEY_TEMPLATE,
    EVALUATION_RESULT_KEY_TEMPLATE,
    HARNESS_CONTRACT_KEY_TEMPLATE,
    HARNESS_MANIFEST_KEY_TEMPLATE,
    HARNESS_VALIDATION_KEY_TEMPLATE,
    POOL_AGENTS,
    POOL_INLINE,
    POOL_RUNS,
    RUN_IN_PROGRESS_STATES,
    RUN_METRICS_KEY_TEMPLATE,
    RUN_TERMINAL_STATES,
    TECHNIQUE_CONTRACT_KEY_TEMPLATE,
    TECHNIQUE_VALIDATION_KEY_TEMPLATE,
    VALIDATION_CONFIG_KEY_TEMPLATE,
    build_cg_execution_spec,
)


def register_smai_specs(
    *,
    workspace_root: Path,
    llm_for_code_reviewer: LlmProvider,
    llm_for_contextual_evaluator: LlmProvider,
    seeds: Sequence[int] = (0,),
    runtime_image: str = "smai-runtime:dev",
    max_review_attempts: int = 1,
    require_human_approval: bool = False,
    evaluation_dispatch_trace: list[str] | None = None,
) -> tuple[PipelineSpec, PipelineSpec]:
    """Construct and register both Phase-2 SMAI specs into the process registry.

    Convenience helper for the CLI bootstrap (Task 2.D2). Constructs the
    CG-execution and CG-entries specs, registers each via
    :func:`smai_orchestrator.runtime.register_pipeline_spec`, and
    returns both for callers that want to reference them directly
    (e.g., to project to :class:`EngineSpec` for the worker loop).

    Args mirror :func:`build_cg_execution_spec`; the entry spec
    inherits ``workspace_root`` and ``runtime_image`` only (it doesn't
    use the LlmProviders since the implementer agent runs inside the
    dispatched Compute container).

    Raises :class:`smai_orchestrator.runtime.DuplicateSpecError` if
    either spec name is already registered. Tests call
    :func:`smai_orchestrator.runtime.reset_registry` between runs.
    """
    cg_spec = build_cg_execution_spec(
        workspace_root=workspace_root,
        llm_for_code_reviewer=llm_for_code_reviewer,
        llm_for_contextual_evaluator=llm_for_contextual_evaluator,
        seeds=seeds,
        runtime_image=runtime_image,
        max_review_attempts=max_review_attempts,
        require_human_approval=require_human_approval,
        evaluation_dispatch_trace=evaluation_dispatch_trace,
    )
    entry_spec = build_cg_entries_spec(
        workspace_root=workspace_root,
        runtime_image=runtime_image,
    )
    register_pipeline_spec(cg_spec)
    register_pipeline_spec(entry_spec)
    return cg_spec, entry_spec


__all__ = [
    "CG_ENTRIES_SPEC_NAME",
    "CG_EXECUTION_SPEC_NAME",
    "CODE_REVIEW_KEY_TEMPLATE",
    "CONTEXTUAL_VERDICT_KEY_TEMPLATE",
    "EVALUATION_ERROR_KEY_TEMPLATE",
    "EVALUATION_RESULT_KEY_TEMPLATE",
    "HARNESS_CONTRACT_KEY_TEMPLATE",
    "HARNESS_MANIFEST_KEY_TEMPLATE",
    "HARNESS_VALIDATION_KEY_TEMPLATE",
    "POOL_AGENTS",
    "POOL_INLINE",
    "POOL_RUNS",
    "RUN_IN_PROGRESS_STATES",
    "RUN_METRICS_KEY_TEMPLATE",
    "RUN_TERMINAL_STATES",
    "TECHNIQUE_CONTRACT_KEY_TEMPLATE",
    "TECHNIQUE_VALIDATION_KEY_TEMPLATE",
    "VALIDATION_CONFIG_KEY_TEMPLATE",
    "build_cg_entries_spec",
    "build_cg_execution_spec",
    "register_smai_specs",
]
