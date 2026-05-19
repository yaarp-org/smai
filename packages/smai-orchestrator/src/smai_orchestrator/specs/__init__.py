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
from typing import Any

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
from smai_orchestrator.specs.paper_ingestion import (
    METHOD_EXTRACTION_KEY_TEMPLATE,
    PAPER_INGESTION_SPEC_NAME,
    POOL_PAPER_INGESTION,
    POOL_PAPER_INGESTION_LIMIT,
    POOL_PAPER_INGESTION_PRIORITY,
    SCREEN_RESULT_KEY_TEMPLATE,
    TECHNIQUE_BUFFER_KEY_TEMPLATE,
    ArxivLatexFetcher,
    FetchedPaper,
    PaperFetcher,
    PaperFetchError,
    build_paper_ingestion_spec,
    register_paper_ingestion_pipeline,
)
from smai_orchestrator.specs.paper_ingestion import (
    PAPER_TEXT_KEY_TEMPLATE as PAPER_INGESTION_PAPER_TEXT_KEY_TEMPLATE,
)
from smai_orchestrator.specs.proposal import (
    APPROVAL_KEY_TEMPLATE as PROPOSAL_APPROVAL_KEY_TEMPLATE,
)
from smai_orchestrator.specs.proposal import (
    DESIGN_PLAN_KEY_TEMPLATE,
    POOL_PROPOSAL_PIPELINE,
    POOL_PROPOSAL_PIPELINE_LIMIT,
    POOL_PROPOSAL_PIPELINE_PRIORITY,
    PROPOSAL_PIPELINE_SPEC_NAME,
    build_proposal_pipeline_spec,
    register_proposal_pipeline,
)
from smai_orchestrator.specs.proposal import (
    REJECTION_KEY_TEMPLATE as PROPOSAL_REJECTION_KEY_TEMPLATE,
)
from smai_orchestrator.specs.proposal import (
    TECHNIQUE_DESCRIPTION_KEY_TEMPLATE as PROPOSAL_TECHNIQUE_DESCRIPTION_KEY_TEMPLATE,
)
from smai_orchestrator.specs.run_record import (
    RUN_RECORD_SPEC_NAME,
    build_run_record_spec,
    register_run_record_spec,
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
    harness_builder_inline_runner: Any = None,
    technique_implementer_inline_runner: Any = None,
) -> tuple[PipelineSpec, PipelineSpec]:
    """Construct and register both CG-side SMAI specs into the process registry.

    Convenience helper for the CLI bootstrap (Task 2.D2). Constructs the
    CG-execution and CG-entries specs, registers each via
    :func:`smai_orchestrator.runtime.register_pipeline_spec`, and
    returns both for callers that want to reference them directly
    (e.g., to project to :class:`EngineSpec` for the worker loop).

    Args mirror :func:`build_cg_execution_spec`; the entry spec
    inherits ``workspace_root`` and ``runtime_image`` only (it doesn't
    use the LlmProviders — the worker resolves the per-role provider
    when it drives the in-process technique-implementer dispatch). The
    ``*_inline_runner`` kwargs are test-only runner overrides threaded
    into the two agent dispatch factories.

    The proposal-pipeline, paper-ingestion, and run-record specs are
    registered separately by their own helpers
    (:func:`register_proposal_pipeline`,
    :func:`register_paper_ingestion_pipeline`,
    :func:`register_run_record_spec`) — each takes per-spec arguments
    that don't fit naturally on this CG-side surface (per-role
    LlmProvider distinct from code-review / contextual-evaluator, the
    paper fetcher seam, etc.). The CLI bootstrap in
    :class:`smai_cli.runtime.Runtime.start_in_band` calls all four
    registration helpers in sequence.

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
        harness_builder_inline_runner=harness_builder_inline_runner,
    )
    entry_spec = build_cg_entries_spec(
        workspace_root=workspace_root,
        runtime_image=runtime_image,
        technique_implementer_inline_runner=technique_implementer_inline_runner,
    )
    register_pipeline_spec(cg_spec)
    register_pipeline_spec(entry_spec)
    return cg_spec, entry_spec


__all__ = [
    "ArxivLatexFetcher",
    "CG_ENTRIES_SPEC_NAME",
    "CG_EXECUTION_SPEC_NAME",
    "CODE_REVIEW_KEY_TEMPLATE",
    "CONTEXTUAL_VERDICT_KEY_TEMPLATE",
    "DESIGN_PLAN_KEY_TEMPLATE",
    "EVALUATION_ERROR_KEY_TEMPLATE",
    "EVALUATION_RESULT_KEY_TEMPLATE",
    "FetchedPaper",
    "HARNESS_CONTRACT_KEY_TEMPLATE",
    "HARNESS_MANIFEST_KEY_TEMPLATE",
    "HARNESS_VALIDATION_KEY_TEMPLATE",
    "METHOD_EXTRACTION_KEY_TEMPLATE",
    "PAPER_INGESTION_PAPER_TEXT_KEY_TEMPLATE",
    "PAPER_INGESTION_SPEC_NAME",
    "POOL_AGENTS",
    "POOL_INLINE",
    "POOL_PAPER_INGESTION",
    "POOL_PAPER_INGESTION_LIMIT",
    "POOL_PAPER_INGESTION_PRIORITY",
    "POOL_PROPOSAL_PIPELINE",
    "POOL_PROPOSAL_PIPELINE_LIMIT",
    "POOL_PROPOSAL_PIPELINE_PRIORITY",
    "POOL_RUNS",
    "PROPOSAL_APPROVAL_KEY_TEMPLATE",
    "PROPOSAL_PIPELINE_SPEC_NAME",
    "PROPOSAL_REJECTION_KEY_TEMPLATE",
    "PROPOSAL_TECHNIQUE_DESCRIPTION_KEY_TEMPLATE",
    "PaperFetchError",
    "PaperFetcher",
    "RUN_IN_PROGRESS_STATES",
    "RUN_METRICS_KEY_TEMPLATE",
    "RUN_RECORD_SPEC_NAME",
    "RUN_TERMINAL_STATES",
    "SCREEN_RESULT_KEY_TEMPLATE",
    "TECHNIQUE_BUFFER_KEY_TEMPLATE",
    "TECHNIQUE_CONTRACT_KEY_TEMPLATE",
    "TECHNIQUE_VALIDATION_KEY_TEMPLATE",
    "VALIDATION_CONFIG_KEY_TEMPLATE",
    "build_cg_entries_spec",
    "build_cg_execution_spec",
    "build_paper_ingestion_spec",
    "build_proposal_pipeline_spec",
    "build_run_record_spec",
    "register_paper_ingestion_pipeline",
    "register_proposal_pipeline",
    "register_run_record_spec",
    "register_smai_specs",
]
