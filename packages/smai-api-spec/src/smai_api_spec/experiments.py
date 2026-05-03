"""Experiment request / response shapes per ``designs/smai/11-api.md`` §4.3.

The ``smai run`` adapter — bypasses the proposal lifecycle since the
technique is already fully specified in the YAML. Two endpoints:

* ``POST /api/v1/experiments/compile`` — pure compile, no plugin touch.
* ``POST /api/v1/experiments``         — compile + persist artifacts +
  create 1+ CGs in ``draft``.

Per ``11`` §13 OQ4 (still open as of 2026-05-03), the compile-result
shape on the wire is a flat list of compilations rather than the CLI's
dict-keyed-by-cg-id form. The CLI's ``--stdout`` mode is expected to
align in a follow-up reconciliation pass.
"""

from __future__ import annotations

from smai_api_spec._common import APIBaseModel, CGState

# === POST /api/v1/experiments/compile =======================================


class CompileExperimentRequest(APIBaseModel):
    """Pure-compile request. Methodology-only — no ``MetadataStore`` or
    ``Compute`` touch.

    ``definition_text`` is the raw YAML body (a multi-line string). The
    server runs the methodology compiler over it and returns the four
    contract artifacts per resulting CG.
    """

    definition_text: str


class CompiledArtifacts(APIBaseModel):
    """The four contract artifacts emitted per compiled CG.

    Per ``02-dsl-and-contracts.md`` §1: the methodology compiler emits
    ``ExperimentPlan``, ``HarnessContract``, ``TechniqueContract[]`` and
    ``ValidationConfig``. The wire form here is a JSON projection
    suitable for downstream tooling (UI rendering, contract-diff in a
    review tool, codegen). The exact field shape inside each artifact
    is the methodology compiler's output and is not re-asserted here —
    it is opaque to this contract package.
    """

    cg_id: str
    experiment_plan: dict[str, object]
    harness_contract: dict[str, object]
    technique_contracts: list[dict[str, object]]
    validation_config: dict[str, object]


class CompileExperimentResponse(APIBaseModel):
    """``200 OK`` body for ``POST /api/v1/experiments/compile``.

    Flat list per ``11`` §13 OQ4 — easier to type than the CLI's
    dict-keyed form. ``compilations`` is non-empty: the smallest valid
    DSL document yields a single ``ExperimentDefinition`` and therefore
    one compilation; factor models (``02`` §3) yield N.
    """

    compilations: list[CompiledArtifacts]


# === POST /api/v1/experiments ===============================================


class SubmitExperimentRequest(APIBaseModel):
    """``smai run``'s HTTP analog: compile + persist + create CG records."""

    definition_text: str


class CreatedCGRef(APIBaseModel):
    """One CG created by ``POST /api/v1/experiments``."""

    cg_id: str
    state: CGState


class SubmitExperimentResponse(APIBaseModel):
    """``202 Accepted`` body for ``POST /api/v1/experiments``.

    The CGs land in ``draft``; the worker's CG-execution pipeline-spec
    drives them to ``implementing`` on its own cadence.
    """

    cgs: list[CreatedCGRef]


__all__ = [
    "CompileExperimentRequest",
    "CompileExperimentResponse",
    "CompiledArtifacts",
    "CreatedCGRef",
    "SubmitExperimentRequest",
    "SubmitExperimentResponse",
]
