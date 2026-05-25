"""End-to-end runner test.

A hand-written workspace + technique runs ``experiment.py``'s logic to
completion and emits ``metrics.json`` that the methodology evaluator
accepts. This test covers §3.4's full startup sequence and the
acceptance criterion for Task 2.D1.
"""

from __future__ import annotations

import json
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest
from smai_core import (
    EntryMetrics,
    HarnessContract,
    RawMetrics,
    TechniqueContract,
)
from smai_runtime import (
    EXIT_OK,
    EXIT_RUNTIME_FAILURE,
    EXIT_USAGE_ERROR,
    HARNESS_API_MANIFEST_FILENAME,
    HARNESS_CONTRACT_FILENAME,
    METRICS_FILENAME,
    TECHNIQUE_CONTRACT_FILENAME,
    VALIDATION_RESULTS_FILENAME,
    HarnessAPIManifest,
    build_seed_run_outcome,
    create_workspace_skeleton,
    materialize_workspace,
    run,
    write_template_files,
)


def _write_harness(workspace: Path) -> None:
    """Hand-written harness exposing the §8.2 ABI."""
    harness_dir = workspace / "harness"
    harness_dir.mkdir(parents=True, exist_ok=True)
    (harness_dir / "__init__.py").write_text(
        textwrap.dedent(
            '''
            """Hand-written test harness."""
            from typing import Any

            from smai_runtime import HarnessComponents


            def build_harness(config: dict[str, Any]) -> HarnessComponents:
                return HarnessComponents(
                    train_transforms=[lambda x: x],
                    callbacks=[lambda: None],
                    training_config={"epochs": int(config.get("epochs", 5))},
                )


            def run_training_loop(components, config, seed):  # type: ignore[no-untyped-def]
                # Deterministic "training": use the seed + technique-supplied
                # transform count to drive the metric.
                base = 0.80
                bonus = 0.01 * len(components.train_transforms)
                return {"accuracy": base + bonus, "seed": seed}


            def evaluate(model, components, config):  # type: ignore[no-untyped-def]
                acc = float(model["accuracy"])
                return {"accuracy": acc, "loss": 1.0 - acc}
            '''
        ).strip()
        + "\n"
    )


def _write_technique(workspace: Path, name: str, body: str) -> None:
    techniques_dir = workspace / "techniques"
    techniques_dir.mkdir(parents=True, exist_ok=True)
    (techniques_dir / f"{name}.py").write_text(body)


@pytest.fixture(autouse=True)
def _clean_imports() -> Iterator[None]:
    """Drop any harness/techniques modules between tests so each materialized
    workspace gets imported fresh.
    """
    yield
    for mod in list(sys.modules):
        if mod == "harness" or mod.startswith("harness.") or mod.startswith("techniques"):
            del sys.modules[mod]


def test_runner_completes_and_writes_metrics(
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
    _write_harness(workspace)
    _write_technique(
        workspace,
        "tech_abc",
        textwrap.dedent(
            '''
            """Test technique that contributes one transform."""
            from typing import Any


            def apply(config: dict[str, Any]) -> dict[str, Any]:
                return {"train_transforms": [lambda x: x + 1]}
            '''
        ).strip()
        + "\n",
    )

    code = run(["--technique", "tech_abc", "--seed", "1", "--workspace", str(workspace)])
    assert code == EXIT_OK
    metrics_path = workspace / METRICS_FILENAME
    assert metrics_path.is_file()
    metrics = json.loads(metrics_path.read_text())
    assert "accuracy" in metrics
    assert metrics["accuracy"] == pytest.approx(0.82, rel=1e-9)

    # And the metrics round-trip through the methodology evaluator's input shape.
    outcome = build_seed_run_outcome(metrics, additive_harness_contract)
    rm = RawMetrics(by_entry={"t": EntryMetrics(entry_id="t", seed_outcomes={1: outcome})})
    assert rm.by_entry["t"].seed_outcomes[1].required == {"accuracy": pytest.approx(0.82)}


def test_runner_fails_on_no_go_zone_modification(
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
    _write_harness(workspace)
    _write_technique(
        workspace,
        "tech_abc",
        "def apply(config): return {}\n",
    )

    # Tamper with experiment.py.
    target = workspace / "experiment.py"
    target.write_text(target.read_text() + "\n# tampered\n")

    code = run(["--technique", "tech_abc", "--seed", "1", "--workspace", str(workspace)])
    assert code == EXIT_RUNTIME_FAILURE
    err_path = workspace / "no_go_zone_error.json"
    assert err_path.is_file()
    err_payload = json.loads(err_path.read_text())
    assert err_payload["code"] == "hash_mismatch"
    assert err_payload["path"] == "experiment.py"


def test_runner_fails_on_wrong_shaped_apply_output(
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
    _write_harness(workspace)
    _write_technique(
        workspace,
        "tech_abc",
        textwrap.dedent(
            '''
            """Bad technique: returns a key not declared in the manifest."""
            from typing import Any


            def apply(config: dict[str, Any]) -> dict[str, Any]:
                return {"completely_unknown": "junk"}
            '''
        ).strip()
        + "\n",
    )

    code = run(["--technique", "tech_abc", "--seed", "1", "--workspace", str(workspace)])
    assert code == EXIT_RUNTIME_FAILURE
    err_path = workspace / "contract_error.json"
    assert err_path.is_file()
    err = json.loads(err_path.read_text())
    assert err["code"] == "unknown_key"
    assert err["entry_id"] == additive_technique_contract.body.entry_id
    assert err["technique_id"] == additive_technique_contract.body.technique_id


def test_runner_fails_when_apply_raises(
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
    _write_harness(workspace)
    _write_technique(
        workspace,
        "tech_abc",
        textwrap.dedent(
            """
            from typing import Any


            def apply(config: dict[str, Any]) -> dict[str, Any]:
                raise RuntimeError("boom")
            """
        ).strip()
        + "\n",
    )

    code = run(["--technique", "tech_abc", "--seed", "1", "--workspace", str(workspace)])
    assert code == EXIT_RUNTIME_FAILURE
    err = json.loads((workspace / "contract_error.json").read_text())
    assert err["code"] == "apply_raised"
    assert "boom" in err["message"]


def test_runner_rejects_full_run_with_epochs(
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
    _write_harness(workspace)
    _write_technique(workspace, "tech_abc", "def apply(config): return {}\n")

    code = run(
        [
            "--technique",
            "tech_abc",
            "--seed",
            "1",
            "--workspace",
            str(workspace),
            "--epochs",
            "1",
        ]
    )
    assert code == EXIT_USAGE_ERROR


def test_runner_validation_mode_accepts_epochs(
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
    _write_harness(workspace)
    _write_technique(workspace, "tech_abc", "def apply(config): return {}\n")

    code = run(
        [
            "--technique",
            "tech_abc",
            "--seed",
            "1",
            "--mode",
            "validation",
            "--epochs",
            "1",
            "--workspace",
            str(workspace),
        ]
    )
    assert code == EXIT_OK


def test_runner_factor_aware_additive_baseline_runs(
    tmp_path: Path,
    additive_harness_contract: HarnessContract,
    additive_baseline_technique_contract: TechniqueContract,
    additive_manifest: HarnessAPIManifest,
) -> None:
    """§9.1 — additive baseline runs the harness with no contribution."""
    workspace = materialize_workspace(
        tmp_path / "ws",
        harness_contract=additive_harness_contract,
        technique_contract=additive_baseline_technique_contract,
        manifest=additive_manifest,
    )
    _write_harness(workspace)
    _write_technique(
        workspace,
        "baseline",
        textwrap.dedent(
            '''
            """Additive baseline returns empty dict — harness defaults stand."""
            from typing import Any


            def apply(config: dict[str, Any]) -> dict[str, Any]:
                return {}
            '''
        ).strip()
        + "\n",
    )

    code = run(["--technique", "baseline", "--seed", "1", "--workspace", str(workspace)])
    assert code == EXIT_OK
    metrics = json.loads((workspace / METRICS_FILENAME).read_text())
    # No technique contribution: only the harness's default single transform.
    assert metrics["accuracy"] == pytest.approx(0.81, rel=1e-9)


def _materialize_partial_workspace(
    workspace_root: Path,
    *,
    harness_contract: HarnessContract,
    technique_contract: TechniqueContract,
) -> Path:
    """Lay down workspace skeleton + templates + the two contracts the
    harness-builder validation flow stages (harness + baseline technique),
    WITHOUT the manifest (it is the OUTPUT of a passing validation per
    04-agents.md §9 step 5 — the catch-22 round 20 fixes).
    """
    workspace = create_workspace_skeleton(workspace_root)
    write_template_files(workspace)
    contracts_dir = workspace / "contracts"
    (contracts_dir / HARNESS_CONTRACT_FILENAME).write_text(harness_contract.model_dump_json())
    (contracts_dir / TECHNIQUE_CONTRACT_FILENAME).write_text(technique_contract.model_dump_json())
    return workspace


def test_validation_mode_runs_with_no_manifest_and_writes_validation_results(
    tmp_path: Path,
    additive_harness_contract: HarnessContract,
    additive_baseline_technique_contract: TechniqueContract,
) -> None:
    """Round 20 — validation mode tolerates a missing manifest (the agent
    runs validation before emitting it) and writes
    ``validation_results.json`` on the success path so the manifest tool's
    ``passed: true`` check sees it.
    """
    workspace = _materialize_partial_workspace(
        tmp_path / "ws",
        harness_contract=additive_harness_contract,
        technique_contract=additive_baseline_technique_contract,
    )
    assert not (workspace / "contracts" / HARNESS_API_MANIFEST_FILENAME).exists()
    _write_harness(workspace)
    _write_technique(workspace, "baseline", "def apply(config): return {}\n")

    code = run(
        [
            "--technique",
            "baseline",
            "--seed",
            "1",
            "--mode",
            "validation",
            "--workspace",
            str(workspace),
        ]
    )
    assert code == EXIT_OK

    validation_path = workspace / VALIDATION_RESULTS_FILENAME
    assert validation_path.is_file(), "validation mode must write validation_results.json"
    payload = json.loads(validation_path.read_text())
    assert payload["passed"] is True
    assert payload["mode"] == "validation"
    assert payload["technique"] == "baseline"
    assert payload["seed"] == 1
    assert "metrics" in payload


def test_full_mode_still_rejects_missing_manifest(
    tmp_path: Path,
    additive_harness_contract: HarnessContract,
    additive_baseline_technique_contract: TechniqueContract,
) -> None:
    """Full mode is the orchestrator's seed-run path; the dispatcher
    materializes all three contracts before invoking the runtime, so a
    missing manifest there IS genuine corruption — unchanged by round 20.
    """
    workspace = _materialize_partial_workspace(
        tmp_path / "ws",
        harness_contract=additive_harness_contract,
        technique_contract=additive_baseline_technique_contract,
    )
    _write_harness(workspace)
    _write_technique(workspace, "baseline", "def apply(config): return {}\n")

    code = run(["--technique", "baseline", "--seed", "1", "--workspace", str(workspace)])
    assert code == EXIT_USAGE_ERROR


def test_validation_mode_clear_error_when_technique_contract_absent(
    tmp_path: Path,
    additive_harness_contract: HarnessContract,
    additive_baseline_technique_contract: TechniqueContract,
    additive_manifest: HarnessAPIManifest,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The technique contract stays REQUIRED in validation mode —
    ``build_runtime_config`` reads its params + level_value. The error
    message points at the staging gap so the agent (or the dispatch
    handler) sees what's missing.
    """
    workspace = materialize_workspace(
        tmp_path / "ws",
        harness_contract=additive_harness_contract,
        technique_contract=additive_baseline_technique_contract,
        manifest=additive_manifest,
    )
    # Delete the technique contract that materialize_workspace just wrote.
    (workspace / "contracts" / TECHNIQUE_CONTRACT_FILENAME).unlink()
    _write_harness(workspace)
    _write_technique(workspace, "baseline", "def apply(config): return {}\n")

    code = run(
        [
            "--technique",
            "baseline",
            "--seed",
            "1",
            "--mode",
            "validation",
            "--workspace",
            str(workspace),
        ]
    )
    assert code == EXIT_USAGE_ERROR
    err_text = capsys.readouterr().err
    assert TECHNIQUE_CONTRACT_FILENAME in err_text
    assert "baseline entry" in err_text or "stage" in err_text
