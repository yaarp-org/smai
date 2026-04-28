"""Experiment driver — the substance behind the fixed ``experiment.py`` template.

Per ``10-runtime-and-templates.md`` §3.4. The fixed template is intentionally
short — its job is to call into this module, which carries the heavy logic.
That keeps the byte-stable no-go-zone surface narrow (any future change to
the dispatch sequence lands here, not in the hashed template).

Startup sequence (§3.4 steps 1–11):

  1. Parse CLI args.
  2. Load three contract artifacts from ``contracts/``.
  3. No-go-zone hash check.
  4. Mode-and-seed gating.
  5. Load harness modules + named technique.
  6. Call ``apply(config)`` (catching exceptions per §6.2 class (a)).
  7. Manifest-driven type check on output (§6.1, class (b)).
  8. Splice via :func:`integrator.integrate_technique_output`.
  9. Run training procedure.
 10. Run evaluator; write ``metrics.json``.
 11. Exit with the training procedure's return code.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from smai_runtime.errors import (
    NoGoZoneHashError,
    TechniqueOutputContractError,
    write_contract_error,
    write_no_go_zone_error,
)
from smai_runtime.integrator import integrate_technique_output
from smai_runtime.metrics import write_metrics
from smai_runtime.no_go_zone import check_no_go_zones
from smai_runtime.seed import seed_everything
from smai_runtime.type_check import check_technique_output
from smai_runtime.workspace import build_runtime_config, load_contracts

RunMode = Literal["validation", "full"]

# Exit-code taxonomy, consumed by the orchestrator's run-completion logic
# (§6.2 / §7.4).
EXIT_OK: int = 0
EXIT_RUNTIME_FAILURE: int = 1  # technique-output contract or no-go-zone failure
EXIT_USAGE_ERROR: int = 2  # mismatched CLI / contract pair (full-run --epochs, etc.)


@dataclass(frozen=True)
class RunnerArgs:
    """Parsed CLI args (§3.4 step 1)."""

    technique: str
    seed: int
    mode: RunMode
    epochs: int | None
    subset: str | None
    workspace: Path


def parse_args(argv: list[str] | None = None) -> RunnerArgs:
    parser = argparse.ArgumentParser(
        prog="smai-experiment",
        description="smai-runtime experiment driver (§3.4 of 10-runtime-and-templates.md)",
    )
    parser.add_argument("--technique", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--mode",
        choices=("validation", "full"),
        default="full",
        help=("validation runs accept --epochs / --subset; full runs reject them per §10.4."),
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--subset", default=None)
    parser.add_argument(
        "--workspace",
        default=None,
        help="defaults to the current working directory (canonically /workspace).",
    )
    ns = parser.parse_args(argv)

    workspace = Path(ns.workspace) if ns.workspace is not None else Path.cwd()
    return RunnerArgs(
        technique=str(ns.technique),
        seed=int(ns.seed),
        mode=cast(RunMode, ns.mode),
        epochs=ns.epochs,
        subset=ns.subset,
        workspace=workspace,
    )


def run(argv: list[str] | None = None) -> int:
    """Entry point invoked by the fixed ``experiment.py`` template (§3.4)."""
    args = parse_args(argv)
    workspace = args.workspace.resolve()

    try:
        harness_contract, technique_contract, manifest = load_contracts(workspace)
    except FileNotFoundError as exc:
        print(f"contract artifact missing: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    # Step 3 — no-go-zone hash check before any harness import / training.
    try:
        check_no_go_zones(
            workspace,
            no_go_zones=harness_contract.body.no_go_zones,
            runtime_template_version=manifest.runtime_template_version,
        )
    except NoGoZoneHashError as err:
        write_no_go_zone_error(workspace, err)
        print(f"no-go-zone failure: {err}", file=sys.stderr)
        return EXIT_RUNTIME_FAILURE

    # Step 4 — mode-and-seed gating.
    if args.mode == "full" and (args.epochs is not None or args.subset is not None):
        print(
            "full runs reject --epochs / --subset; pass --mode validation if "
            "you intend to override (§10.4).",
            file=sys.stderr,
        )
        return EXIT_USAGE_ERROR
    if args.seed not in harness_contract.body.seeds:
        print(
            f"seed {args.seed} not in HarnessContract.seeds={harness_contract.body.seeds}",
            file=sys.stderr,
        )
        return EXIT_USAGE_ERROR
    if (
        technique_contract.body.parent_harness_contract_hash
        != harness_contract.envelope.content_hash
    ):
        print(
            "TechniqueContract.parent_harness_contract_hash does not match "
            "loaded HarnessContract.envelope.content_hash — stale contract pair.",
            file=sys.stderr,
        )
        return EXIT_USAGE_ERROR
    if manifest.parent_harness_contract_hash != harness_contract.envelope.content_hash:
        print(
            "Manifest.parent_harness_contract_hash does not match loaded "
            "HarnessContract.envelope.content_hash — stale manifest.",
            file=sys.stderr,
        )
        return EXIT_USAGE_ERROR

    # Make workspace-resident packages (`harness`, `techniques`) importable.
    workspace_str = str(workspace)
    if workspace_str not in sys.path:
        sys.path.insert(0, workspace_str)

    seed_everything(args.seed)

    # Step 5 — load harness + technique by string (§8.2 ABI).
    harness_module = importlib.import_module("harness")
    techniques_module = importlib.import_module("techniques")

    build_harness = cast(
        "Callable[[dict[str, Any]], Any]",
        harness_module.build_harness,
    )
    run_training_loop = cast(
        "Callable[..., Any]",
        harness_module.run_training_loop,
    )
    evaluate_fn = cast(
        "Callable[..., dict[str, Any]]",
        harness_module.evaluate,
    )
    load_technique = cast(
        "Callable[[str], Callable[[dict[str, Any]], dict[str, Any]]]",
        techniques_module.load_technique,
    )

    config = build_runtime_config(
        harness_contract=harness_contract,
        technique_contract=technique_contract,
        seed=args.seed,
    )
    if args.mode == "validation":
        if args.epochs is not None:
            config["epochs"] = args.epochs
        if args.subset is not None:
            config["subset"] = args.subset

    components = build_harness(config)
    technique_apply = load_technique(args.technique)

    # Step 6 — call apply, catching anything (§6.2 class (a)).
    try:
        technique_output: dict[str, Any] = technique_apply(config)
    except Exception as exc:
        contract_err = TechniqueOutputContractError(
            code="apply_raised",
            message=f"apply() raised {type(exc).__name__}: {exc}",
            extension_point_key=None,
            expected_signature=None,
            actual_value_repr=None,
            manifest_hash=manifest.content_hash,
            technique_id=technique_contract.body.technique_id,
            entry_id=technique_contract.body.entry_id,
        )
        write_contract_error(workspace, contract_err)
        print(f"apply() raised: {exc}", file=sys.stderr)
        return EXIT_RUNTIME_FAILURE

    # Step 7 — manifest-driven type check (§6.1, class (b)).
    try:
        check_technique_output(technique_output, manifest)
    except TechniqueOutputContractError as err:
        err = err.with_identity(
            manifest_hash=manifest.content_hash,
            technique_id=technique_contract.body.technique_id,
            entry_id=technique_contract.body.entry_id,
        )
        write_contract_error(workspace, err)
        print(f"contract violation: {err.message}", file=sys.stderr)
        return EXIT_RUNTIME_FAILURE

    # Step 8 — splice technique output into harness components.
    spliced = integrate_technique_output(components, technique_output, manifest)

    # Step 9 — run training.
    trained_model = run_training_loop(spliced, config, args.seed)

    # Step 10 — evaluator + metrics.json.
    metrics = evaluate_fn(trained_model, spliced, config)
    write_metrics(workspace, metrics, harness_contract)

    return EXIT_OK
