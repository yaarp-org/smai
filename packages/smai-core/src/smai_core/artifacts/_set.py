""":class:`ContractArtifactSet` — runtime container for one CG's emitted artifacts.

Per ``02-dsl-and-contracts.md`` §8.1 / §8.6. Bundles the four (3 + N)
artifacts a single CG's emission produces. Each child carries its own
envelope and ``content_hash``; pre-distribution packaging (zip / tar /
directory) is a pipeline-layer concern.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from smai_core.artifacts.experiment_plan import ExperimentPlan
from smai_core.artifacts.harness_contract import HarnessContract
from smai_core.artifacts.technique_contract import TechniqueContract
from smai_core.artifacts.validation_config import ValidationConfig


class ContractArtifactSet(BaseModel):
    """Runtime container for the four contract artifacts of one CG (§8.1)."""

    model_config = ConfigDict(extra="forbid")

    experiment_plan: ExperimentPlan
    harness_contract: HarnessContract
    technique_contracts: list[TechniqueContract]
    validation_config: ValidationConfig
