"""Per-task fixtures for sub-PR C1 baseline-generation step tests.

Per the SMAI test convention (``tests/_<task>_<purpose>.py``).

Provides workspace builders that stage the optional inputs the
baseline-generation step consumes: a manifest JSON, a harness source
tree, and a grounding artifact. The body-generation tests don't need
any of these; this module narrows to the baseline-specific inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

from smai_core.artifacts._envelope import ArtifactEnvelope
from smai_core.artifacts.technique_contract import TechniqueContract, TechniqueContractBody
from smai_core.entities.technique import (
    PaperFidelityAnchor,
    ProposalFidelityAnchor,
    ReviewerAttestedFidelityAnchor,
)


def stage_grounding_file(workspace: Path, payload: dict[str, str]) -> None:
    """Drop a pre-resolved grounding JSON under ``grounding/`` in the
    workspace. Mirrors what the host-side dispatcher (sub-PR D) will
    materialize."""
    grounding_dir = workspace / "grounding"
    grounding_dir.mkdir(parents=True, exist_ok=True)
    (grounding_dir / "baseline_grounding.json").write_text(json.dumps(payload))


def stage_harness_api_manifest(workspace: Path, manifest: dict[str, object]) -> None:
    """Drop a manifest JSON at the workspace root for the baseline bundle."""
    (workspace / "harness_api_manifest.json").write_text(json.dumps(manifest))


def stage_harness_source_module(workspace: Path, module_path: str, source: str) -> None:
    """Drop a ``.py`` module under ``harness/`` so the bundle's
    ``harness_source`` map gets populated."""
    target = workspace / module_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source)


def stage_technique_contract(
    workspace: Path,
    *,
    is_baseline: bool = True,
    technique_id: str | None = "tech-baseline",
    standard: bool = True,
    fidelity_anchor: object | None = None,
) -> TechniqueContract:
    """Stage a minimal :class:`TechniqueContract` under ``contracts/``."""
    envelope = ArtifactEnvelope(
        artifact_kind="technique_contract",
        artifact_id="tc-fixture",
        schema_version=1,
        compiler_version="0.0.0-test",
        content_hash="cafebabe" * 8,
        parent_experiment_id="exp-fixture",
    )
    body = TechniqueContractBody(
        entry_id="entry-test",
        parent_experiment_id="exp-fixture",
        parent_experiment_hash="deadbeef" * 8,
        parent_harness_contract_hash="cafebabe" * 8,
        technique_id=technique_id,
        technique_params=None,
        level_value=None,
        is_baseline=is_baseline,
        fidelity_anchor=fidelity_anchor,  # type: ignore[arg-type]
        standard=standard,
    )
    contract = TechniqueContract(envelope=envelope, body=body)
    contracts_dir = workspace / "contracts"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    (contracts_dir / "technique_contract.json").write_text(contract.model_dump_json(indent=2))
    return contract


__all__ = [
    "PaperFidelityAnchor",
    "ProposalFidelityAnchor",
    "ReviewerAttestedFidelityAnchor",
    "stage_grounding_file",
    "stage_harness_api_manifest",
    "stage_harness_source_module",
    "stage_technique_contract",
]
