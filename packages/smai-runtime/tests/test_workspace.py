"""Workspace materialization tests (§2)."""

from __future__ import annotations

from pathlib import Path

from smai_core import HarnessContract, TechniqueContract
from smai_runtime import (
    HARNESS_API_MANIFEST_FILENAME,
    HARNESS_CONTRACT_FILENAME,
    TECHNIQUE_CONTRACT_FILENAME,
    HarnessAPIManifest,
    build_runtime_config,
    create_workspace_skeleton,
    load_contracts,
    materialize_workspace,
)


def test_skeleton_creates_expected_subdirs(tmp_path: Path) -> None:
    workspace = create_workspace_skeleton(tmp_path / "ws")
    assert (workspace / "harness").is_dir()
    assert (workspace / "techniques").is_dir()
    assert (workspace / "contracts").is_dir()


def test_materialize_drops_templates_and_contracts(
    tmp_path: Path,
    additive_harness_contract: HarnessContract,
    additive_technique_contract: TechniqueContract,
    additive_manifest: HarnessAPIManifest,
) -> None:
    workspace = materialize_workspace(
        tmp_path / "ws",
        harness_contract=additive_harness_contract,
        technique_contract=additive_technique_contract,
        manifest=additive_manifest,
    )

    assert (workspace / "experiment.py").is_file()
    assert (workspace / "techniques" / "__init__.py").is_file()
    assert (workspace / "contracts" / HARNESS_CONTRACT_FILENAME).is_file()
    assert (workspace / "contracts" / TECHNIQUE_CONTRACT_FILENAME).is_file()
    assert (workspace / "contracts" / HARNESS_API_MANIFEST_FILENAME).is_file()


def test_load_contracts_round_trips(
    tmp_path: Path,
    additive_harness_contract: HarnessContract,
    additive_technique_contract: TechniqueContract,
    additive_manifest: HarnessAPIManifest,
) -> None:
    workspace = materialize_workspace(
        tmp_path / "ws",
        harness_contract=additive_harness_contract,
        technique_contract=additive_technique_contract,
        manifest=additive_manifest,
    )
    hc, tc, m = load_contracts(workspace)
    assert hc == additive_harness_contract
    assert tc == additive_technique_contract
    assert m == additive_manifest


def test_build_runtime_config_flattens_fixed_variables(
    additive_harness_contract: HarnessContract,
    additive_technique_contract: TechniqueContract,
) -> None:
    config = build_runtime_config(
        harness_contract=additive_harness_contract,
        technique_contract=additive_technique_contract,
        seed=2,
    )
    assert config["seed"] == 2
    assert config["params"] == {"alpha": 0.1}
    # Fixed variables are flattened by JSONPath under top-level config keys.
    assert config["dataset.name"] == "cifar10"
    assert config["optimization.lr"] == 0.001
