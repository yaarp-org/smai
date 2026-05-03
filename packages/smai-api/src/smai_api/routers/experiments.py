"""``/api/v1/experiments`` router per ``designs/smai/11-api.md`` §4.3.

Two endpoints:

* ``POST /experiments/compile`` — pure compile (methodology-only); no
  ``MetadataStore`` or ``ArtifactStore`` touch.
* ``POST /experiments`` — compile + persist + create CGs in ``draft``
  (the ``smai run`` HTTP analog).
"""

from __future__ import annotations

from fastapi import APIRouter
from smai_api_spec import (
    CompiledArtifacts,
    CompileExperimentRequest,
    CompileExperimentResponse,
    CreatedCGRef,
    SubmitExperimentRequest,
    SubmitExperimentResponse,
)
from smai_api_spec.paths import EXPERIMENTS, EXPERIMENTS_COMPILE

from smai_api._deps import RuntimeDep

router = APIRouter()


# === POST /api/v1/experiments/compile ======================================


@router.post(EXPERIMENTS_COMPILE, response_model=CompileExperimentResponse)
async def compile_experiment(
    body: CompileExperimentRequest,
    runtime: RuntimeDep,
) -> CompileExperimentResponse:
    """Pure compile per ``11`` §4.3 — runs the methodology compiler over
    the YAML body and returns the four contract artifacts per resulting
    CG. Never touches ``MetadataStore`` / ``Compute``."""
    artifact_sets = await runtime.experiments.compile_text(body.definition_text)
    compilations: list[CompiledArtifacts] = []
    for cg_id, artifact_set in artifact_sets.items():
        compilations.append(
            CompiledArtifacts(
                cg_id=cg_id,
                experiment_plan=artifact_set.experiment_plan.model_dump(mode="json"),
                harness_contract=artifact_set.harness_contract.model_dump(mode="json"),
                technique_contracts=[
                    contract.model_dump(mode="json")
                    for contract in artifact_set.technique_contracts
                ],
                validation_config=artifact_set.validation_config.model_dump(mode="json"),
            )
        )
    return CompileExperimentResponse(compilations=compilations)


# === POST /api/v1/experiments ==============================================


@router.post(EXPERIMENTS, status_code=202, response_model=SubmitExperimentResponse)
async def submit_experiment(
    body: SubmitExperimentRequest,
    runtime: RuntimeDep,
) -> SubmitExperimentResponse:
    """The ``smai run`` HTTP analog — compile, persist artifacts to
    ``ArtifactStore``, and create the ``ComparisonGroupRecord``(s) in
    ``draft``. Worker picks them up via the CG-execution pipeline-spec."""
    cg_ids = await runtime.experiments.submit_text(body.definition_text)
    cgs: list[CreatedCGRef] = []
    for cg_id in cg_ids:
        record = await runtime.status.get_cg_record(cg_id)
        cgs.append(CreatedCGRef(cg_id=record.id, state=record.state))
    return SubmitExperimentResponse(cgs=cgs)


__all__ = ["router"]
