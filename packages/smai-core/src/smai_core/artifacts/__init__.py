"""Contract artifact emission surface.

Per ``02-dsl-and-contracts.md`` §7 + §8. The four contract artifacts
(``ExperimentPlan``, ``HarnessContract``, ``TechniqueContract``,
``ValidationConfig``) plus their packaging container
(``ContractArtifactSet``) and the public :func:`emit_artifacts` /
:func:`emit_factor_model_artifacts` entry points.

Tier B integrators import from ``smai_core`` directly. This subpackage's
module split is an implementation detail.
"""

from smai_core.artifacts._canonical import (
    artifact_hash,
    canonical_json,
    freeze_with_hash,
    hash_registry_dict,
    hash_string_set,
)
from smai_core.artifacts._emit import (
    ARTIFACT_SCHEMA_VERSION,
    COMPILER_VERSION,
    emit_artifacts,
    emit_factor_model_artifacts,
)
from smai_core.artifacts._envelope import (
    ArtifactEnvelope,
    ArtifactKind,
    SurfaceKind,
)
from smai_core.artifacts._set import ContractArtifactSet
from smai_core.artifacts._surface_maps import (
    EXPERIMENT_PLAN_SURFACE_MAP,
    HARNESS_CONTRACT_SURFACE_MAP,
    TECHNIQUE_CONTRACT_SURFACE_MAP,
    VALIDATION_CONFIG_SURFACE_MAP,
)
from smai_core.artifacts.experiment_plan import ExperimentPlan, ExperimentPlanBody
from smai_core.artifacts.harness_contract import (
    FixedVariable,
    HarnessContract,
    HarnessContractBody,
)
from smai_core.artifacts.technique_contract import (
    TechniqueContract,
    TechniqueContractBody,
)
from smai_core.artifacts.validation_config import (
    ValidationConfig,
    ValidationConfigBody,
)

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "COMPILER_VERSION",
    "ArtifactEnvelope",
    "ArtifactKind",
    "ContractArtifactSet",
    "EXPERIMENT_PLAN_SURFACE_MAP",
    "ExperimentPlan",
    "ExperimentPlanBody",
    "FixedVariable",
    "HARNESS_CONTRACT_SURFACE_MAP",
    "HarnessContract",
    "HarnessContractBody",
    "SurfaceKind",
    "TECHNIQUE_CONTRACT_SURFACE_MAP",
    "TechniqueContract",
    "TechniqueContractBody",
    "VALIDATION_CONFIG_SURFACE_MAP",
    "ValidationConfig",
    "ValidationConfigBody",
    "artifact_hash",
    "canonical_json",
    "emit_artifacts",
    "emit_factor_model_artifacts",
    "freeze_with_hash",
    "hash_registry_dict",
    "hash_string_set",
]
