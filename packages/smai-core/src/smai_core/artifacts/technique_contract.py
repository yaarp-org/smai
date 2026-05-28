""":class:`TechniqueContract` — one per :class:`Entry`.

Per ``02-dsl-and-contracts.md`` §7.5 (post-Decision-1 narrowing): identity,
configured params, parent-contract hashes, ``fidelity_anchor`` (DEC-032).
Per-key output-key declarations are gone; those live on the pipeline-layer
``HarnessAPIManifest``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from smai_core.artifacts._envelope import ArtifactEnvelope
from smai_core.entities.numeric import NumericValue
from smai_core.entities.technique import ContextKind, FidelityAnchor, TechniqueParams


class TechniqueContractBody(BaseModel):
    """Body fields of :class:`TechniqueContract` (§7.5).

    ``context_kind`` (``agent_refactor/upstream_requirements §1``) declares
    the grounding flavor explicitly so downstream consumers (the
    implementer agent's bundle-construction step, the contextual
    evaluator) read it from the contract rather than re-inferring from
    ``standard`` + ``fidelity_anchor.kind``. The four ref-backed variants
    (``paper_extract`` / ``proposal`` / ``reviewer_attested`` /
    ``standard``) mirror :attr:`TechniqueRef.context_kind`; additive
    baselines (``technique_id is None`` + ``is_baseline=True``, per
    DEC-013) get ``no_op_baseline`` synthesized by the compiler so no
    consumer ever branches on ``context_kind is None``.
    """

    model_config = ConfigDict(extra="forbid")

    entry_id: str
    parent_experiment_id: str
    parent_experiment_hash: str
    parent_harness_contract_hash: str
    technique_id: str | None
    technique_params: TechniqueParams | None
    level_value: NumericValue | None
    is_baseline: bool
    fidelity_anchor: FidelityAnchor | None
    standard: bool
    context_kind: ContextKind


class TechniqueContract(BaseModel):
    """One technique's compiled contract (§7.5)."""

    model_config = ConfigDict(extra="forbid")

    envelope: ArtifactEnvelope
    body: TechniqueContractBody
