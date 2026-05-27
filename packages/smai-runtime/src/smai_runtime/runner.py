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

from smai_core import HarnessContract, TechniqueContract

from smai_runtime.components import (
    COMPONENT_FIELD_FOR_KEY,
    DEFAULT_VALIDATION_STUB_PATTERN_FOR_KEY,
    DEFAULT_VALIDATION_STUB_TYPE_SIGNATURE_FOR_KEY,
)
from smai_runtime.errors import (
    NoGoZoneHashError,
    TechniqueOutputContractError,
    write_contract_error,
    write_no_go_zone_error,
)
from smai_runtime.integrator import integrate_technique_output
from smai_runtime.manifest import (
    MANIFEST_SCHEMA_VERSION,
    HarnessAPIManifest,
    HarnessExtensionPoint,
)
from smai_runtime.metrics import write_metrics
from smai_runtime.no_go_zone import RUNTIME_TEMPLATE_VERSION, check_no_go_zones
from smai_runtime.seed import seed_everything
from smai_runtime.type_check import check_technique_output
from smai_runtime.workspace import (
    HARNESS_API_MANIFEST_FILENAME,
    HARNESS_CONTRACT_FILENAME,
    TECHNIQUE_CONTRACT_FILENAME,
    build_runtime_config,
    write_validation_results,
)

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

    # Mode-specific contract loading. ``full`` mode requires the full
    # contract triple (the orchestrator's seed-run dispatcher materializes
    # all three before invoking the runtime). ``validation`` mode is the
    # harness-builder / technique-implementer in-loop check that runs
    # BEFORE the manifest exists — the agent is validating their harness
    # against the locked contract pair, then emits the manifest as the
    # outcome of a passing validation. So in validation mode the manifest
    # is optional; if absent the runtime synthesizes a stub that declares
    # the closed-v1 well-known extension-point keys (per round-23 R1) so
    # the agent's baseline can return real component overrides and have
    # them spliced through. The technique contract stays required either
    # way — ``build_runtime_config`` reads its ``technique_params`` /
    # ``level_value`` to produce the ``config`` dict the harness consumes.
    try:
        harness_contract, technique_contract, manifest = _load_contracts_by_mode(
            workspace, mode=args.mode
        )
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

    # Step 11 — validation-mode structured output (§10.4). Written ONLY on
    # the success path so the manifest tool's ``passed: true`` check (per
    # 04-agents.md §9 step 3) cannot mistake an aborted run for a passing
    # validation. Full-mode runs skip this — they're real seed runs, the
    # verdict path consumes metrics.json directly.
    if args.mode == "validation":
        write_validation_results(
            workspace,
            {
                "passed": True,
                "mode": "validation",
                "technique": args.technique,
                "seed": args.seed,
                "epochs": args.epochs,
                "subset": args.subset,
                "metrics": metrics,
            },
        )

    return EXIT_OK


# === Mode-dispatched contract loading =======================================


def _load_contracts_by_mode(
    workspace: Path,
    *,
    mode: RunMode,
) -> tuple[HarnessContract, TechniqueContract, HarnessAPIManifest]:
    """Load contract artifacts with mode-specific manifest handling.

    Full mode delegates to :func:`load_contracts` — both per-entry seed-run
    dispatchers materialize all three artifacts before invoking the runtime,
    so a missing one is genuine corruption.

    Validation mode is the agents' in-loop check. The harness builder
    validates BEFORE emitting the manifest (manifest is the OUTPUT of a
    passing validation per 04-agents.md §9 step 5 — running validation
    before manifest emission is the only way to get a passing validation
    in the first place). So validation mode tolerates an absent manifest
    and synthesizes one.

    Round-23 sub-PR R1 (see ``project_round22_real_llm_dogfood.md``
    Wall #2 + ``smai_runtime.components.DEFAULT_VALIDATION_STUB_PATTERN_FOR_KEY``):
    the synthesized stub now declares all five closed-v1 well-known
    extension-point keys (from
    :data:`smai_runtime.components.COMPONENT_FIELD_FOR_KEY`) as
    ``optional=True`` with their per-key safe-default integration
    patterns. The previous behavior emitted ``extension_points=[]``,
    which round-20 documented as "safe for additive baselines (return
    {}) but a substitutive baseline returning a non-empty dict will
    surface as type_check 'unknown_key'". Round-22 confirmed the agent's
    diagnose loop frequently can't navigate that blind spot in 3
    attempts; the richer stub eliminates the false-negative entirely
    while preserving the "validation actually validates" semantic
    (out-of-set keys still get rejected — that catches genuinely-wrong
    hallucinations like an LLM-invented ``"my_custom_hook"``).

    The technique contract stays required —
    :func:`build_runtime_config` needs ``technique_params`` /
    ``level_value`` to assemble the config dict; the harness builder's
    workspace setup is responsible for staging the baseline's contract
    (see :func:`smai_agents.agents.harness_builder.run_harness_builder_session`).
    """
    contracts_dir = workspace / "contracts"
    harness_path = contracts_dir / HARNESS_CONTRACT_FILENAME
    technique_path = contracts_dir / TECHNIQUE_CONTRACT_FILENAME
    manifest_path = contracts_dir / HARNESS_API_MANIFEST_FILENAME

    if not harness_path.is_file():
        raise FileNotFoundError(str(harness_path))
    harness_contract = HarnessContract.model_validate_json(harness_path.read_text())

    if not technique_path.is_file():
        raise FileNotFoundError(
            f"{technique_path} (stage the per-entry technique_contract.json before "
            "invoking the runtime; in validation mode the harness builder must stage "
            "the baseline entry's contract from ArtifactStore)"
        )
    technique_contract = TechniqueContract.model_validate_json(technique_path.read_text())

    if manifest_path.is_file():
        manifest = HarnessAPIManifest.model_validate_json(manifest_path.read_text())
    elif mode == "validation":
        # Round-23 R1: declare all closed-v1 keys as ``optional=True`` with
        # their per-key safe-default integration patterns + types
        # (see ``smai_runtime.components.DEFAULT_VALIDATION_STUB_*``).
        # A baseline that returns any subset of the v1 keys passes
        # ``check_technique_output``; the splice in
        # ``integrate_technique_output`` actually fires, so validation
        # exercises the real harness + baseline mechanics rather than
        # short-circuiting on an empty manifest. Keys outside the v1 set
        # (e.g. an LLM-hallucinated ``"my_custom_hook"``) still get
        # ``unknown_key`` — the validation surface keeps its real signal.
        manifest = HarnessAPIManifest(
            extension_points=[
                HarnessExtensionPoint(
                    key=key,
                    type_signature=DEFAULT_VALIDATION_STUB_TYPE_SIGNATURE_FOR_KEY[key],
                    purpose=(
                        f"validation-mode stub allowlist for closed-v1 key {key!r}; "
                        "the real per-harness manifest is emitted by the harness "
                        "builder after a passing validation"
                    ),
                    optional=True,
                    integration_pattern=DEFAULT_VALIDATION_STUB_PATTERN_FOR_KEY[key],
                )
                for key in COMPONENT_FIELD_FOR_KEY
            ],
            integration_pattern_summary=(
                "validation-mode stub (closed-v1 keys allowlisted, safe-default patterns)"
            ),
            harness_version_hash="",
            parent_harness_contract_hash=harness_contract.envelope.content_hash,
            manifest_schema_version=MANIFEST_SCHEMA_VERSION,
            runtime_template_version=RUNTIME_TEMPLATE_VERSION,
        )
    else:
        raise FileNotFoundError(str(manifest_path))

    return harness_contract, technique_contract, manifest
