"""``smai`` CLI entry point — Phase-2 + Phase-3 (Task 3.E1) verbs.

Per ``designs/smai/09-cli.md`` §1: Phase-2 ships ``dev``, ``run``,
``status``, ``compile``, ``init``, ``plugins``, ``version``. Phase-3
Task 3.E1 adds ``submit-proposal``, ``approve-proposal``, and
``reject-proposal`` (per DEC-032 — proposal submission is the v2
primary input verb). Other Phase-3 verbs (``start``, ``serve``,
``ingest``, ``migrate``) ship in their own Phase-3 tasks.

The CLI is a thin adapter over :mod:`smai_cli.runtime` (per `09` §9 /
§1.2 — the verb-to-service mapping). Each verb does just enough to
parse args + load config + invoke the underlying service call.

Implementation choice: Typer.

* Pydantic-friendly typing surface, matches the typed style of the
  rest of the workspace.
* First-class auto-help generation surfaces only the verbs the CLI
  actually defines (`09` §1's stated test-of-record is "smai --help
  shows only the Phase-2 verbs").
* Single dependency, zero shell-completion ceremony.

Argparse / Click / Cleo were considered. Argparse needs more
boilerplate; Click's decorator + context machinery is heavier; Cleo
isn't used elsewhere in the workspace.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer

from smai_cli.config import (
    PHASE_2_DEFAULT_PIPELINES,
    ConfigFileError,
    ConfigValidationError,
    base_defaults,
    dev_defaults,
    load_runtime_config,
)
from smai_cli.runtime import (
    CGNotFoundError,
    PaperNotFoundError,
    PaperStateError,
    ProposalNotFoundError,
    ProposalStateError,
    Runtime,
    WaitTimeoutError,
)

app = typer.Typer(
    name="smai",
    help="SMAI v2 — methodology-grounded experiment platform.",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode=None,
)


# === Helpers =================================================================


def _err(message: str, *, exit_code: int = 1) -> NoReturn:
    """Print a user-facing error and exit non-zero."""
    typer.echo(f"smai: error: {message}", err=True)
    raise typer.Exit(code=exit_code)


def _smai_home() -> Path:
    """Resolve the per-user ``~/.smai`` directory.

    Honors ``SMAI_HOME`` if set; otherwise falls back to ``~/.smai``.
    """
    override = os.environ.get("SMAI_HOME")
    if override:
        return Path(override)
    return Path.home() / ".smai"


def _dev_artifact_paths() -> tuple[Path, Path, Path]:
    """``smai dev``'s default working directories.

    Returns ``(workspace_root, artifact_root, sqlite_path)``. Created
    on demand by :func:`smai_dev`.
    """
    home = _smai_home()
    return (
        home / "workspaces",
        home / "artifacts",
        home / "state.db",
    )


def _apply_dev_filesystem_defaults(layered_overrides: dict[str, Any]) -> dict[str, Any]:
    """Materialize the ``smai dev`` filesystem defaults.

    Per `09` §5.1: SQLite at ``~/.smai/state.db``, LocalFs at
    ``~/.smai/artifacts``, workspaces at ``~/.smai/workspaces``. The
    config-layering pipeline doesn't know these defaults statically
    (they depend on ``$HOME`` resolution); the CLI fills them in just
    before invoking :func:`load_runtime_config`.
    """
    workspace_root, artifact_root, sqlite_path = _dev_artifact_paths()
    workspace_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    plugins = layered_overrides.setdefault("plugins", {})
    metadata_cfg = plugins.setdefault("metadata_store_config", {})
    metadata_cfg.setdefault("uri", f"sqlite+aiosqlite:///{sqlite_path}")
    artifact_cfg = plugins.setdefault("artifact_store_config", {})
    artifact_cfg.setdefault("root", str(artifact_root))
    return layered_overrides


# === Verb 1: dev =============================================================


@app.command("dev")
def smai_dev(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to a smai.yaml; overrides search order."),
    ] = None,
    no_dashboard: Annotated[
        bool,
        typer.Option("--no-dashboard", help="(Phase 3 placeholder.) Reserved."),
    ] = False,
) -> None:
    """Boot the laptop demo (`09` §5).

    Defaults: ``sqlite`` MetadataStore (``~/.smai/state.db``),
    ``localfs`` ArtifactStore (``~/.smai/artifacts``), ``localgpu``
    Compute, ``bedrock`` LlmProvider. ``poll_interval_seconds=10`` for
    interactive feel. Worker runs in-process; Ctrl+C drains gracefully.
    """
    del no_dashboard  # Phase 3 wires the read-only dashboard (`09` §5.3).
    overrides: dict[str, Any] = _apply_dev_filesystem_defaults({})
    try:
        runtime_config = load_runtime_config(
            config_path=config,
            defaults=dev_defaults(),
            flag_overrides=overrides,
        )
    except (ConfigFileError, ConfigValidationError) as exc:
        _err(str(exc))

    workspace_root, _artifacts, _sqlite = _dev_artifact_paths()

    async def _run() -> None:
        async with Runtime.start_in_band(
            runtime_config,
            workspace_root=workspace_root,
        ) as runtime:
            stop_event = asyncio.Event()

            def _signal_handler(*_: Any) -> None:
                stop_event.set()

            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, _signal_handler)

            typer.echo(
                "smai dev: worker running. "
                f"workspace_root={runtime.workspace_root} "
                f"poll_interval={runtime.config.engine.poll_interval_seconds}s. "
                "Press Ctrl+C to stop."
            )
            await stop_event.wait()
            typer.echo("smai dev: shutdown signal received; draining...")

    asyncio.run(_run())


# === Verb 2: run =============================================================


@app.command("run")
def smai_run(
    experiment_yaml: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Experiment YAML path."),
    ],
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to a smai.yaml."),
    ] = None,
    watch: Annotated[
        bool,
        typer.Option("--watch", help="Tail state transitions until terminal."),
    ] = False,
) -> None:
    """Submit ``experiment.yaml`` to a running deployment (`09` §5.2).

    Compiles + persists artifacts + creates a CG row in ``draft``.
    Returns the CG ID on stdout. ``--watch`` polls until terminal.
    """
    overrides: dict[str, Any] = _apply_dev_filesystem_defaults({})
    try:
        runtime_config = load_runtime_config(
            config_path=config,
            defaults=dev_defaults(),
            flag_overrides=overrides,
        )
    except (ConfigFileError, ConfigValidationError) as exc:
        _err(str(exc))

    yaml_text = experiment_yaml.read_text(encoding="utf-8")

    async def _run() -> None:
        # Don't spin up the worker — `smai run` is a one-shot submit.
        async with Runtime.start_in_band(runtime_config, run_worker=False) as runtime:
            cg_ids = await runtime.experiments.submit_text(yaml_text)
            for cg_id in cg_ids:
                typer.echo(cg_id)
            if watch:
                for cg_id in cg_ids:
                    snap = await runtime.status.wait_for_terminal(cg_id, timeout=None)
                    typer.echo(f"{cg_id}: {snap.state}")

    asyncio.run(_run())


# === Verb 3: status ==========================================================


@app.command("status")
def smai_status(
    cg_id: Annotated[str, typer.Argument(help="CG identifier.")],
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to a smai.yaml."),
    ] = None,
    watch: Annotated[
        bool,
        typer.Option("--watch", help="Poll until the CG reaches a terminal state."),
    ] = False,
    poll_interval: Annotated[
        float,
        typer.Option(
            "--poll-interval",
            help="Watch poll interval (seconds).",
            min=0.1,
        ),
    ] = 5.0,
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            help="Output format: 'text' or 'json'.",
        ),
    ] = "text",
) -> None:
    """Read CG state from MetadataStore (`09` §7).

    Read-only. ``--watch`` polls every ``--poll-interval`` seconds
    until terminal. JSON format for machine consumption.
    """
    overrides: dict[str, Any] = _apply_dev_filesystem_defaults({})
    try:
        runtime_config = load_runtime_config(
            config_path=config,
            defaults=dev_defaults(),
            flag_overrides=overrides,
        )
    except (ConfigFileError, ConfigValidationError) as exc:
        _err(str(exc))

    async def _run() -> None:
        async with Runtime.start_in_band(runtime_config, run_worker=False) as runtime:
            try:
                snap = await runtime.status.get(cg_id)
            except CGNotFoundError as exc:
                _err(str(exc), exit_code=2)
                return
            _emit_status(snap, output_format)
            if watch and not snap.is_terminal:
                try:
                    final = await runtime.status.wait_for_terminal(
                        cg_id,
                        timeout=None,
                        poll_interval_seconds=poll_interval,
                    )
                except WaitTimeoutError as exc:
                    _err(str(exc), exit_code=3)
                    return
                _emit_status(final, output_format)

    asyncio.run(_run())


def _emit_status(snap: Any, output_format: str) -> None:
    if output_format == "json":
        typer.echo(
            json.dumps(
                {
                    "cg_id": snap.cg_id,
                    "state": snap.state,
                    "updated_at": snap.updated_at.isoformat(),
                    "is_terminal": snap.is_terminal,
                }
            )
        )
    else:
        terminal_marker = " (terminal)" if snap.is_terminal else ""
        typer.echo(f"{snap.cg_id}: {snap.state}{terminal_marker}")


# === Verb 4: compile =========================================================


@app.command("compile")
def smai_compile(
    experiment_yaml: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Experiment YAML path."),
    ],
    out_dir: Annotated[
        Path | None,
        typer.Option(
            "--out",
            "-o",
            help="Directory to write the four artifact JSON files. "
            "If absent, prints a JSON bundle to stdout.",
            file_okay=False,
        ),
    ] = None,
) -> None:
    """Compile a YAML to the four contract artifacts (`09` §1 / `02` §8).

    Pure methodology call — does not touch MetadataStore / Compute. No
    plugin instantiation. Useful for offline validation before
    submission.
    """
    yaml_text = experiment_yaml.read_text(encoding="utf-8")
    # Compile in-process; no plugins needed.
    import yaml as _yaml  # noqa: PLC0415
    from smai_core import (  # noqa: PLC0415 — lazy-import to keep `--help` fast
        DslDocumentAdapter,
        FactorModelDocument,
        compile_experiment,
        load_default_registries,
    )

    payload: Any = _yaml.safe_load(yaml_text)
    if not isinstance(payload, dict):
        _err("experiment YAML must be a mapping at the top level.")
    document = DslDocumentAdapter.validate_python(payload, context={"smai_mode": "dsl"})
    registries = load_default_registries()

    if isinstance(document, FactorModelDocument):
        artifact_sets = compile_experiment(document, registries)
    else:
        single = compile_experiment(document, registries)
        artifact_sets = {document.experiment.id: single}

    if out_dir is None:
        # Bundle to stdout.
        bundle: dict[str, Any] = {
            cg_id: {
                "experiment_plan": json.loads(s.experiment_plan.model_dump_json()),
                "harness_contract": json.loads(s.harness_contract.model_dump_json()),
                "technique_contracts": [
                    json.loads(t.model_dump_json()) for t in s.technique_contracts
                ],
                "validation_config": json.loads(s.validation_config.model_dump_json()),
            }
            for cg_id, s in artifact_sets.items()
        }
        typer.echo(json.dumps(bundle, indent=2))
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for cg_id, s in artifact_sets.items():
        cg_dir = out_dir / cg_id
        cg_dir.mkdir(parents=True, exist_ok=True)
        (cg_dir / "experiment_plan.json").write_text(
            s.experiment_plan.model_dump_json(indent=2), encoding="utf-8"
        )
        (cg_dir / "harness_contract.json").write_text(
            s.harness_contract.model_dump_json(indent=2), encoding="utf-8"
        )
        for tc in s.technique_contracts:
            (cg_dir / f"technique_contract_{tc.body.entry_id}.json").write_text(
                tc.model_dump_json(indent=2), encoding="utf-8"
            )
        (cg_dir / "validation_config.json").write_text(
            s.validation_config.model_dump_json(indent=2), encoding="utf-8"
        )
    typer.echo(f"compiled {len(artifact_sets)} CG(s) → {out_dir}")


# === Verb 5: init ============================================================


_STARTER_SMAI_YAML = """\
# smai.yaml — generated by `smai init`. Edit values to customize.
# See designs/smai/09-cli.md §2 for the canonical schema.

engine:
  poll_interval_seconds: 10
  worker_count: 1
  fair_scheduling: "off"

plugins:
  llm_provider: bedrock
  metadata_store: sqlite
  artifact_store: localfs
  compute: localgpu
  llm_provider_config:
    region: us-east-1
    model_id: us.anthropic.claude-opus-4-7-v1
  metadata_store_config: {}
  artifact_store_config: {}
  compute_config: {}

pipelines:
  - smai_cg_execution
  - smai_cg_entries
"""


_STARTER_EXPERIMENT_YAML = """\
# experiment.yaml — illustrative worked example.
#
# Submit via: smai run experiment.yaml
#
# Note: this example references a `tech_cutout` technique that is NOT
# in the default technique registry. To compile/run this experiment as
# written, populate the technique registry programmatically (Tier B
# integrators construct their own `Registries` per `02-dsl-and-contracts.md`
# §4.1). In Phase 3 the canonical input path is `smai submit-proposal`,
# which drives the planner agent to register required techniques first.
kind: experiment
experiment:
  id: cg_example
  hypothesis: "Cutout improves accuracy on CIFAR-10."
  factors:
    - name: augmentation
      type: additive
      description: "cutout on/off"
  controlled_conditions:
    dataset:
      name: cifar10
      split: train
      version: v1
    optimization:
      optimizer: sgd
      lr: 0.1
    seeds: [1, 2, 3]
  entries:
    - id: entry_baseline
      is_baseline: true
      level:
        factor: augmentation
        name: absent
    - id: entry_cutout
      is_baseline: false
      level:
        factor: augmentation
        name: cutout
        technique_id: tech_cutout
        technique_params:
          patch_size: 16
  validation:
    metric: { kind: atomic, ref: accuracy }
    direction: higher_is_better
    aggregation: { method: mean }
    comparison:
      rule: compare_to_baseline
      threshold: 0.01
      # baseline_entry_id is filled in by the compiler from
      # `entries[].is_baseline=true` (DSL gate per `02` §2.5).
    seed_count_required: 3
"""


@app.command("init")
def smai_init(
    directory: Annotated[
        Path,
        typer.Argument(help="Where to write the starter files. Defaults to '.'"),
    ] = Path("."),
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite existing files."),
    ] = False,
) -> None:
    """Bootstrap a starter project (`09` §1 / Task 2.D2).

    Writes ``smai.yaml`` + a sample ``experiment.yaml``. Refuses to
    overwrite existing files unless ``--force`` is passed.
    """
    directory.mkdir(parents=True, exist_ok=True)
    smai_yaml = directory / "smai.yaml"
    experiment_yaml = directory / "experiment.yaml"

    if not force:
        existing = [p for p in (smai_yaml, experiment_yaml) if p.exists()]
        if existing:
            names = ", ".join(p.name for p in existing)
            _err(f"refusing to overwrite existing file(s): {names}. Pass --force to overwrite.")

    smai_yaml.write_text(_STARTER_SMAI_YAML, encoding="utf-8")
    experiment_yaml.write_text(_STARTER_EXPERIMENT_YAML, encoding="utf-8")
    typer.echo(f"wrote {smai_yaml}")
    typer.echo(f"wrote {experiment_yaml}")


# === Verb 6: plugins =========================================================


@app.command("plugins")
def smai_plugins(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to a smai.yaml."),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format: 'text' or 'json'."),
    ] = "text",
) -> None:
    """Enumerate discovered plugins + the SMAI Phase-2 pipeline-specs (`09` §4.1).

    Walks the four entry-point namespaces and lists what pip has
    installed; flags which one is currently selected (if a config is
    loadable). Plugin instantiation is deliberately NOT performed
    here — discovery is a read-only surface (no AWS / Docker / etc.).
    """
    from smai_orchestrator import list_discovered_plugins  # noqa: PLC0415

    discovered = list_discovered_plugins()

    # Pipeline-specs are registered at boot time by `smai dev`; the
    # `plugins` verb prints the canonical Phase-2 names without
    # registering them (registration would require constructing live
    # plugins, which we explicitly want to avoid here).
    specs = list(PHASE_2_DEFAULT_PIPELINES)

    selected: dict[str, str] = {}
    try:
        runtime_config = load_runtime_config(config_path=config, defaults=base_defaults())
    except (ConfigFileError, ConfigValidationError):
        runtime_config = None
    if runtime_config is not None:
        selected = {
            "smai.llm_providers": runtime_config.plugins.llm_provider,
            "smai.metadata_stores": runtime_config.plugins.metadata_store,
            "smai.artifact_stores": runtime_config.plugins.artifact_store,
            "smai.computes": runtime_config.plugins.compute,
        }

    if output_format == "json":
        typer.echo(
            json.dumps(
                {
                    "discovered": discovered,
                    "selected": selected,
                    "registered_pipeline_specs": specs,
                },
                indent=2,
            )
        )
        return

    typer.echo("Discovered plugins:")
    for group, names in sorted(discovered.items()):
        typer.echo(f"  {group}:")
        sel = selected.get(group)
        for name in names:
            marker = "  (selected)" if sel == name else ""
            typer.echo(f"    - {name}{marker}")
    typer.echo("")
    typer.echo("Registered pipeline-specs:")
    for spec_name in specs:
        typer.echo(f"  - {spec_name}")


# === Verb 8: submit-proposal =================================================


def _generate_proposal_id() -> str:
    """Generate a deterministic-shaped proposal id.

    v1 uses ``proposal-<unix_ms>-<rand4>``; the format-validator
    accepts ULID-shaped strings or any non-whitespace ASCII ≤ 64
    chars per ``01-data-model.md`` §5.2.2. Tests pass an explicit
    ``--id`` override to make assertions deterministic.
    """
    import secrets  # noqa: PLC0415
    import time  # noqa: PLC0415

    return f"proposal-{int(time.time() * 1000)}-{secrets.token_hex(4)}"


@app.command("submit-proposal")
def smai_submit_proposal(
    description: Annotated[
        str | None,
        typer.Argument(
            help=(
                "Inline JSON technique description. Pass `-` to read from "
                "stdin. Mutually exclusive with --description-file and "
                "--reproduce-paper."
            ),
        ),
    ] = None,
    description_file: Annotated[
        Path | None,
        typer.Option(
            "--description-file",
            "-f",
            help="Path to a JSON file with the technique description.",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    reproduce_paper: Annotated[
        str | None,
        typer.Option(
            "--reproduce-paper",
            help=(
                "ArXiv id to reproduce. The paper must already be ingested "
                "via `smai ingest <arxiv-id>` per DEC-032 OQ1."
            ),
        ),
    ] = None,
    proposal_id: Annotated[
        str | None,
        typer.Option(
            "--id",
            help="Override the auto-generated proposal id (useful for tests).",
        ),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to a smai.yaml."),
    ] = None,
) -> None:
    """Submit a novel-technique or reproduce-paper proposal (`09` §1).

    The PRIMARY input verb per DEC-032. Synchronously creates the
    :class:`ProposalRecord` in ``proposal_submitted`` and persists the
    submission artifact. The next worker cycle picks it up and fires
    the planner.
    """
    if reproduce_paper is None and description is None and description_file is None:
        _err(
            "submit-proposal requires one of: a description argument, "
            "--description-file, or --reproduce-paper."
        )
    if reproduce_paper is not None and (description is not None or description_file is not None):
        _err(
            "submit-proposal: --reproduce-paper is mutually exclusive with "
            "an inline description argument or --description-file."
        )

    submission_kind = "reproduce_paper" if reproduce_paper is not None else "novel_technique"
    technique_payload: Any = None
    if description_file is not None:
        technique_payload = description_file.read_text(encoding="utf-8")
    elif description == "-":
        technique_payload = sys.stdin.read()
    elif description is not None:
        technique_payload = description

    final_proposal_id = proposal_id or _generate_proposal_id()

    overrides: dict[str, Any] = _apply_dev_filesystem_defaults({})
    try:
        runtime_config = load_runtime_config(
            config_path=config,
            defaults=dev_defaults(),
            flag_overrides=overrides,
        )
    except (ConfigFileError, ConfigValidationError) as exc:
        _err(str(exc))

    async def _run() -> None:
        async with Runtime.start_in_band(runtime_config, run_worker=False) as runtime:
            submission = await runtime.proposals.submit(
                proposal_id=final_proposal_id,
                submission_kind=submission_kind,
                technique_description=technique_payload,
                reproduce_paper_arxiv_id=reproduce_paper,
            )
            typer.echo(submission.proposal_id)

    asyncio.run(_run())


# === Verb 9: approve-proposal ================================================


@app.command("approve-proposal")
def smai_approve_proposal(
    proposal_id: Annotated[str, typer.Argument(help="Proposal identifier.")],
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to a smai.yaml."),
    ] = None,
) -> None:
    """Approve a proposal in the ``designed`` resting state (`09` §1).

    Per ``03-state-machine.md`` §4.2 (edge 5,
    ``proposal.user_approved``). The next worker cycle fires the
    ``designed → registered`` edge — the registration handler atomically
    writes the methodology + tracking entities and the CGs land in
    ``draft`` ready for the CG-execution spec.
    """
    overrides: dict[str, Any] = _apply_dev_filesystem_defaults({})
    try:
        runtime_config = load_runtime_config(
            config_path=config,
            defaults=dev_defaults(),
            flag_overrides=overrides,
        )
    except (ConfigFileError, ConfigValidationError) as exc:
        _err(str(exc))

    async def _run() -> None:
        async with Runtime.start_in_band(runtime_config, run_worker=False) as runtime:
            try:
                record = await runtime.proposals.approve(proposal_id)
            except ProposalNotFoundError as exc:
                _err(str(exc), exit_code=2)
                return
            except ProposalStateError as exc:
                _err(str(exc), exit_code=4)
                return
            typer.echo(f"approved: {record.id} (state={record.state})")

    asyncio.run(_run())


# === Verb 10: reject-proposal ================================================


@app.command("reject-proposal")
def smai_reject_proposal(
    proposal_id: Annotated[str, typer.Argument(help="Proposal identifier.")],
    reason: Annotated[
        str | None,
        typer.Option("--reason", "-r", help="Free-text rejection reason."),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to a smai.yaml."),
    ] = None,
) -> None:
    """Reject a proposal in the ``designed`` resting state (`09` §1).

    Per ``03-state-machine.md`` §4.2 (edge 6,
    ``proposal.user_rejected``). The proposal transitions to
    ``rejected`` (terminal). The draft buffer artifact is preserved.
    """
    overrides: dict[str, Any] = _apply_dev_filesystem_defaults({})
    try:
        runtime_config = load_runtime_config(
            config_path=config,
            defaults=dev_defaults(),
            flag_overrides=overrides,
        )
    except (ConfigFileError, ConfigValidationError) as exc:
        _err(str(exc))

    async def _run() -> None:
        async with Runtime.start_in_band(runtime_config, run_worker=False) as runtime:
            try:
                record = await runtime.proposals.reject(proposal_id, reason=reason)
            except ProposalNotFoundError as exc:
                _err(str(exc), exit_code=2)
                return
            except ProposalStateError as exc:
                _err(str(exc), exit_code=4)
                return
            typer.echo(f"rejected: {record.id} (state={record.state})")

    asyncio.run(_run())


# === Verb 11: ingest =========================================================


@app.command("ingest")
def smai_ingest(
    arxiv_id: Annotated[str, typer.Argument(help="arXiv id (e.g., 1804.07612 or cs.LG/9701001).")],
    promote_partial: Annotated[
        bool,
        typer.Option(
            "--promote-partial",
            help=(
                "Promote an existing partial paper to ``submitted`` "
                "(per `08-novel-technique-pipeline.md` §5.7). The paper "
                "must already exist in `partial` state — populated by "
                "another paper's enrichment step. Mutually exclusive "
                "with the default ingestion flow."
            ),
        ),
    ] = False,
    title: Annotated[
        str | None,
        typer.Option(
            "--title",
            help=(
                "Optional bibliographic title to record on the new "
                "PaperRecord. The fetcher overwrites this with the "
                "arXiv-reported title once content lands."
            ),
        ),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to a smai.yaml."),
    ] = None,
) -> None:
    """Submit a paper for ingestion (`09` §1 / DEC-032 OQ1).

    The paper-ingestion verb. Per `08` §5 / DEC-032: paper ingestion is
    a *supporting utility* that produces ``TechniqueRef``s + paper-
    fidelity-anchor metadata only — it does NOT produce CGs. To create
    CGs from a paper, ``smai ingest`` first, then ``smai submit-proposal
    --reproduce-paper <arxiv-id>``.

    ``smai ingest <arxiv-id>``: create a :class:`PaperRecord` in
    ``submitted``; the worker's next cycle picks it up via
    ``get_ready_for_paper_fetch`` and drives the ingestion pipeline.

    ``smai ingest --promote-partial <arxiv-id>``: synchronous transition
    of a paper from ``partial`` to ``submitted`` (per `08` §5.7 — partial
    papers are link-target / dedup parking spots created by another
    paper's enrichment step; promotion fully ingests them).
    """
    overrides: dict[str, Any] = _apply_dev_filesystem_defaults({})
    try:
        runtime_config = load_runtime_config(
            config_path=config,
            defaults=dev_defaults(),
            flag_overrides=overrides,
        )
    except (ConfigFileError, ConfigValidationError) as exc:
        _err(str(exc))

    async def _run() -> None:
        async with Runtime.start_in_band(runtime_config, run_worker=False) as runtime:
            try:
                if promote_partial:
                    record = await runtime.papers.promote_partial(arxiv_id)
                    typer.echo(f"promoted: {record.arxiv_id} (state={record.state})")
                else:
                    submission = await runtime.papers.submit(arxiv_id=arxiv_id, title=title)
                    suffix = " (promoted from partial)" if submission.promoted else ""
                    typer.echo(f"{submission.arxiv_id} (state={submission.state}){suffix}")
            except PaperNotFoundError as exc:
                _err(str(exc), exit_code=2)
                return
            except PaperStateError as exc:
                _err(str(exc), exit_code=4)
                return

    asyncio.run(_run())


# === Verb 12: serve ==========================================================


@app.command("serve")
def smai_serve(
    host: Annotated[
        str,
        typer.Option(
            "--host",
            help=(
                "Bind address. Defaults to 127.0.0.1 (loopback only) per "
                "the single-user local dashboard contract — the OSS "
                "dashboard has no authentication (DEC-027). Production "
                "deployments that want broader binding pass --host "
                "explicitly; this is a deliberate choice each deployment "
                "makes for itself."
            ),
        ),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", help="Port to listen on.", min=1, max=65535),
    ] = 8000,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to a smai.yaml."),
    ] = None,
) -> None:
    """Boot the read-only dashboard (`09` §5.4).

    Renders pages over the configured ``MetadataStore`` + ``ArtifactStore``
    via the existing :class:`Runtime` service surface. Pages cover
    proposals, comparison groups, runs, and papers (lists + per-entity
    detail). The surface is read-only by spec; mutations go through the
    other CLI verbs (``smai run`` / ``smai submit-proposal`` /
    ``smai approve-proposal`` / ``smai reject-proposal`` /
    ``smai ingest``).

    No authentication. The default ``--host=127.0.0.1`` keeps the
    surface loopback-only; broaden the bind explicitly if you mean to.
    """
    import uvicorn  # noqa: PLC0415 — lazy-import to keep `--help` fast

    from smai_cli.dashboard import build_app  # noqa: PLC0415

    overrides: dict[str, Any] = _apply_dev_filesystem_defaults({})
    try:
        runtime_config = load_runtime_config(
            config_path=config,
            defaults=dev_defaults(),
            flag_overrides=overrides,
        )
    except (ConfigFileError, ConfigValidationError) as exc:
        _err(str(exc))

    workspace_root, _artifacts, _sqlite = _dev_artifact_paths()

    async def _run() -> None:
        # The dashboard is read-only — boot the runtime without spawning
        # a worker. Workers are managed independently by `smai dev` /
        # `smai start`; the dashboard reads through the configured
        # plugins per `09` §5.4 ("the verb that runs only the dashboard
        # ... no worker").
        async with Runtime.start_in_band(
            runtime_config,
            workspace_root=workspace_root,
            run_worker=False,
        ) as runtime:
            fastapi_app = build_app(runtime)
            uvicorn_config = uvicorn.Config(
                fastapi_app,
                host=host,
                port=port,
                log_level="info",
                access_log=True,
            )
            server = uvicorn.Server(uvicorn_config)
            typer.echo(
                f"smai serve: dashboard listening on http://{host}:{port}/ "
                "(read-only, no authentication). Press Ctrl+C to stop."
            )
            await server.serve()

    asyncio.run(_run())


# === Verb 13a: start (production worker process) =============================


def _resolve_worker_id(*, override: str | None) -> str:
    """Resolve the production worker identity per Task 3.G1's pattern.

    Resolution order: explicit ``--worker-id`` flag > ``SMAI_WORKER_ID``
    env > a ``f"{hostname}-{pid}-{uuid8}"`` host-pid-uuid fallback.
    The lease-holder field on each entity row carries this value per
    `01` §5.6 / DEC-035 #2 so operators can read it back from the DB.
    """
    if override is not None:
        return override
    env_override = os.environ.get("SMAI_WORKER_ID")
    if env_override:
        return env_override
    import socket  # noqa: PLC0415
    import uuid  # noqa: PLC0415

    hostname = socket.gethostname() or "unknown"
    return f"{hostname}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _validate_plugin_completeness(runtime_config: Any) -> None:
    """Refuse to boot if any of the four plugin selections is empty.

    Per `09-cli.md` §6.2 "smai start requires that all four plugin
    selections be set to non-default values". The CLI's
    :class:`PluginSelection` Pydantic model already requires non-empty
    string fields at parse time (every config_layering test catches
    this); this helper exists so the verb can surface a clear
    actionable error if a future config-loading path lets an empty
    string through. The downstream catches are the entry-point
    discovery (:class:`PluginNotFound`) and the plugin constructor —
    each surfaces a typed exception the verb already renders.
    """
    selection = runtime_config.plugins
    missing: list[str] = []
    if not selection.llm_provider:
        missing.append("plugins.llm_provider")
    if not selection.metadata_store:
        missing.append("plugins.metadata_store")
    if not selection.artifact_store:
        missing.append("plugins.artifact_store")
    if not selection.compute:
        missing.append("plugins.compute")
    if missing:
        _err(
            "smai start: incomplete plugin selection — "
            f"missing required field(s): {', '.join(missing)}. "
            "Set each in your smai.yaml or via SMAI_* env vars (see "
            "`smai init` for a starter template)."
        )


async def _check_schema_at_head(runtime_config: Any) -> None:
    """Refuse to boot against a stale schema (`smai start` pre-flight).

    Mirrors ``smai migrate --check``'s programmatic shape — equivalent
    in failure-mode coverage but doesn't shell out to a subprocess.
    Per Task 3.H2's status-note carry-forward, ``smai start`` runs
    this check before any plugin instantiation so a stale-schema
    deployment fails with a clear actionable error rather than dying
    mid-dispatch when a SQL feature added in a newer revision is
    missing.
    """
    from smai_orchestrator.migrations import (  # noqa: PLC0415
        get_current_revision,
        get_head_revision,
    )
    from sqlalchemy.ext.asyncio import create_async_engine  # noqa: PLC0415

    uri = _resolve_metadata_store_uri(runtime_config)
    engine = create_async_engine(uri)
    try:
        current = await get_current_revision(engine)
        head = get_head_revision()
    finally:
        await engine.dispose()
    if current != head:
        shown_current = current if current is not None else "<unstamped>"
        _err(
            "smai start: schema NOT at head — "
            f"current={shown_current}, head={head}. "
            "Run `smai migrate` to upgrade, then re-run `smai start`."
        )


def _enforce_lease_capability(runtime: Any) -> None:
    """Refuse multi-worker boot against a non-lease-capable ``MetadataStore``
    (`09-cli.md` §6.2).

    Phase-3 dispatch wraps every entity-driving step in
    ``acquire_lease`` / ``release_lease`` so concurrent workers cannot
    both fire the dispatch handler for the same entity (DEC-035 #2 /
    Task 3.G1). If the configured store reports
    ``capabilities.supports_leasing=False``, that ABA-safety contract
    is unmet and a multi-worker deployment can double-fire the dispatch
    handler — silent corruption is unacceptable, so we hard-exit 1
    rather than boot. Single-worker deployments (``worker_count == 1``)
    are allowed against non-lease-capable stores; the contention only
    materializes across workers.
    """
    worker_count = runtime.config.engine.worker_count
    capabilities = runtime.plugins.metadata_store.capabilities
    if worker_count > 1 and not capabilities.supports_leasing:
        plugin_name = runtime.config.plugins.metadata_store
        _err(
            "smai start: refusing to boot — "
            f"engine.worker_count={worker_count} but the configured "
            f"MetadataStore plugin {plugin_name!r} reports "
            "supports_leasing=False. Multi-worker deployments require "
            "a lease-capable store (per `09-cli.md` §6.2 / DEC-035 #2). "
            "Either set engine.worker_count=1, or switch to a "
            "lease-capable plugin (e.g., postgres, sqlite)."
        )


@app.command("start")
def smai_start(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to a smai.yaml; overrides search order."),
    ] = None,
    worker_id: Annotated[
        str | None,
        typer.Option(
            "--worker-id",
            help=(
                "Stable lease-holder identity for this worker process. "
                "Resolution order: --worker-id flag > SMAI_WORKER_ID env "
                "> f'{hostname}-{pid}-{uuid8}' fallback. The value lands "
                "on every leased entity row's `leased_by` column per "
                "DEC-035 #2; pin a stable id (e.g., the systemd unit "
                "name) for production deployments so operators can read "
                "it back."
            ),
        ),
    ] = None,
) -> None:
    """Boot the production-mode worker process (`09` §6 / `05` §7.2).

    The out-of-band counterpart to ``smai dev``: this CLI process IS
    the worker, and runs until killed. It does NOT serve a dashboard
    (`smai serve` is the dashboard verb), does NOT submit foreground
    experiments (`smai run` / `smai submit-proposal` / `smai ingest`
    are the submit verbs), and does NOT pretend to be a one-stop
    shell. A typical deployment runs ``smai start`` as the entrypoint
    of a long-lived container or systemd service.

    Pre-flight refuses to boot on any of:

    * Incomplete plugin selection (`09` §6.2): all four plugin slots
      MUST be set in the resolved :class:`RuntimeConfig`. The Pydantic
      model rejects empty strings at config parse time; this is a
      defense-in-depth check.
    * Stale schema: the ``MetadataStore``'s ``alembic_version`` row
      MUST match the migrations head. Equivalent to
      ``smai migrate --check`` but programmatic (no subprocess shell-
      out). Run ``smai migrate`` to upgrade.

    Operational guidance for systemd / supervisord / launchd unit
    examples + recommended connection-pool sizing + log handling lives
    in ``packages/smai-cli/OPERATIONS.md``. Production deployments
    should use a Postgres ``MetadataStore`` (multi-worker leasing
    requires lease-capability per `09` §6.2); single-worker
    SQLite-backed deployments are allowed (single-VM self-hosted) but
    MUST keep ``EngineConfig.worker_count=1``.

    Multi-worker deployments per Task 3.G1 / DEC-035 #2: each worker
    pins a stable ``worker_id``; the CAS-leased phase-3 dispatch
    wrapper round-trips it as the ``leased_by`` column. Concurrent
    workers against the same Postgres are supported.
    """
    try:
        runtime_config = load_runtime_config(
            config_path=config,
            defaults=base_defaults(),
        )
    except (ConfigFileError, ConfigValidationError) as exc:
        _err(str(exc))

    _validate_plugin_completeness(runtime_config)
    asyncio.run(_check_schema_at_head(runtime_config))

    resolved_worker_id = _resolve_worker_id(override=worker_id)

    workspace_root, _artifacts, _sqlite = _dev_artifact_paths()

    async def _run() -> None:
        async with Runtime.start_worker(
            runtime_config,
            worker_id=resolved_worker_id,
            workspace_root=workspace_root,
        ) as runtime:
            _enforce_lease_capability(runtime)

            stop_event = asyncio.Event()

            def _signal_handler(*_: Any) -> None:
                stop_event.set()

            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, _signal_handler)

            typer.echo(
                "smai start: production worker running. "
                f"worker_id={runtime.worker_id} "
                f"workspace_root={runtime.workspace_root} "
                f"poll_interval={runtime.config.engine.poll_interval_seconds}s "
                f"worker_count={runtime.config.engine.worker_count}. "
                "Send SIGTERM to drain gracefully."
            )
            await stop_event.wait()
            typer.echo("smai start: shutdown signal received; draining...")

    asyncio.run(_run())


# === Verb 13b: verify (plugin-ping pre-flight) ===============================


@app.command("verify")
def smai_verify(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to a smai.yaml; overrides search order."),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format: 'text' or 'json'."),
    ] = "text",
) -> None:
    """Ping each configured plugin (`09` §6.2 — pre-flight).

    Sibling of ``smai start``: instantiates the four plugins from the
    resolved :class:`RuntimeConfig` and calls a minimal read-only
    "ping" probe per interface. Surfaces misconfigured ``smai.yaml``
    (bad creds, unreachable bucket, network issue, wrong region) with
    a clear per-plugin diagnostic before a worker boots and starts
    dispatching jobs.

    Probe semantics (from :mod:`smai_cli.verify`):

    * **LlmProvider** — single 1-token completion. **This costs real
      tokens** (typically <10 in + 1 out, but provider-billed). The
      operator gets one round-trip's worth of token usage per
      ``smai verify`` invocation.
    * **MetadataStore** — read-only count_in_state probe; surfaces
      DB connectivity + auth + schema-at-head. Equivalent in failure
      coverage to ``smai migrate --check``.
    * **ArtifactStore** — read-only ``exists`` against a non-existent
      key; surfaces bucket reachability + creds.
    * **Compute** — read-only ``status`` against a non-existent
      handle; expects :class:`JobNotFound`. Surfaces auth +
      substrate reachability.

    Exit code: 0 iff all four plugins ping clean. Non-zero (1) if
    any plugin fails — the per-plugin reason is printed to stdout,
    and the verb exits with code 1 so CI / deployment scripts can
    gate on it.
    """
    from smai_orchestrator import instantiate_plugins  # noqa: PLC0415

    from smai_cli.verify import (  # noqa: PLC0415
        VerifyResult,
        verify_artifact_store,
        verify_compute,
        verify_llm_provider,
        verify_metadata_store,
    )

    try:
        runtime_config = load_runtime_config(
            config_path=config,
            defaults=base_defaults(),
        )
    except (ConfigFileError, ConfigValidationError) as exc:
        _err(str(exc))

    _validate_plugin_completeness(runtime_config)

    async def _run() -> dict[str, VerifyResult]:
        # ``skip_migrate=True`` keeps the verify probe strictly read-only
        # per `09-cli.md` §1: the per-plugin ping helpers below are
        # read-only, and we MUST NOT mutate the configured store's
        # schema as a side effect of probing connectivity. Operators
        # invoke ``smai migrate`` explicitly; ``smai verify`` should
        # not do it for them.
        async with instantiate_plugins(runtime_config.plugins, skip_migrate=True) as plugins:
            # Pick the first per-role provider — every role resolves
            # to the same instance unless per-role overrides apply,
            # and `smai verify` is a single-shot pre-flight, not a
            # per-role health check.
            llm_provider = next(iter(plugins.llm_providers.values()))
            results = {
                "llm_provider": await verify_llm_provider(llm_provider),
                "metadata_store": await verify_metadata_store(plugins.metadata_store),
                "artifact_store": await verify_artifact_store(plugins.artifact_store),
                "compute": await verify_compute(plugins.compute),
            }
        return results

    try:
        results = asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001 — pre-construction failure is a fail
        # Plugin instantiation failed before we reached the per-plugin
        # ping helpers. Surface as a single-line FAIL so the operator
        # sees the construction error verbatim.
        _err(f"smai verify: plugin instantiation failed: {type(exc).__name__}: {exc}")

    if output_format == "json":
        typer.echo(
            json.dumps(
                {
                    name: {
                        "ok": result.ok,
                        "reason": result.reason,
                        "latency_ms": result.latency_ms,
                    }
                    for name, result in results.items()
                },
                indent=2,
            )
        )
    else:
        for name, result in results.items():
            tag = "PASS" if result.ok else "FAIL"
            latency = f" ({result.latency_ms:.1f}ms)" if result.latency_ms is not None else ""
            typer.echo(f"{tag} {name}{latency}: {result.reason}")

    if not all(result.ok for result in results.values()):
        raise typer.Exit(code=1)


# === Verb 13: migrate ========================================================


def _resolve_metadata_store_uri(runtime_config: Any) -> str:
    """Pick the URI the migrate verbs (`smai migrate`, `--check`,
    `--dry-run`, `--prune`) operate on.

    Resolution order:

    1. ``runtime_config.plugins.metadata_store_config["uri"]`` if set
       (the layered config-pipeline result, including ``smai dev``'s
       sqlite-at-``~/.smai/state.db`` default).
    2. Plugin-specific fallbacks (``sqlite+aiosqlite:///:memory:`` for
       the sqlite plugin's empty-config default; the dockerized
       compose URL for the postgres plugin).
    3. Otherwise: error explicitly so an unknown plugin doesn't
       silently fall through.
    """
    selection = runtime_config.plugins
    cfg: dict[str, Any] = selection.metadata_store_config
    uri = cfg.get("uri")
    if isinstance(uri, str) and uri:
        return uri
    plugin_name = selection.metadata_store
    if plugin_name == "sqlite":
        return "sqlite+aiosqlite:///:memory:"
    if plugin_name == "postgres":
        return "postgresql+asyncpg://smai:smai@localhost:5433/smai"
    raise ValueError(
        f"smai migrate: cannot determine database URI for plugin "
        f"{plugin_name!r}. Set plugins.metadata_store_config.uri in your "
        "smai.yaml, or pick a plugin with a known default."
    )


@app.command("migrate")
def smai_migrate(
    check: Annotated[
        bool,
        typer.Option(
            "--check",
            help=(
                "Exit 0 if the database schema is at head, 1 otherwise. "
                "Useful as a `smai start` pre-flight per `09-cli.md` §6 — "
                "the worker refuses to boot against a stale schema."
            ),
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help=(
                "Render the SQL Alembic would emit without executing it "
                "(Alembic's `--sql` mode). Useful for review before a "
                "production migration."
            ),
        ),
    ] = False,
    prune: Annotated[
        bool,
        typer.Option(
            "--prune",
            help=(
                "Run the retention sweep (DEC-033 #1, #2). Deletes rows "
                "older than the per-table window from "
                "`engine.retention_policies` (defaults: transition_log = "
                "90d, agent_sessions = 180d, run_costs = 365d). "
                "Mutually exclusive with --check / --dry-run."
            ),
        ),
    ] = False,
    upgrade_to: Annotated[
        str | None,
        typer.Option(
            "--upgrade-to",
            help=(
                "Target a specific Alembic branch's head instead of the "
                "default canonical OSS chain (Task 3.G2). Currently the "
                "only non-default value is `tenant_aware`, which applies "
                "the opt-in tenant_id schema extension (revision "
                "0002_tenant_aware_schema). Required before flipping "
                "`PostgresStore(tenant_aware=True)` on an existing "
                "deployment if you prefer the explicit migration path; "
                "constructor-driven boot-time migrate is idempotent and "
                "works equally."
            ),
        ),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to a smai.yaml."),
    ] = None,
) -> None:
    """Run schema migrations against the configured ``MetadataStore`` (`09` §1).

    Default (no flags): equivalent to ``alembic upgrade default@head`` —
    the OSS canonical schema chain. Idempotent — safe to run repeatedly.

    Per Task 3.H2 / DEC-036: drives the shared Alembic env at
    :mod:`smai_orchestrator.migrations`. The same code path runs at
    `smai dev` boot via the plugin's ``migrate()`` method.

    Per Task 3.G2: pass ``--upgrade-to=tenant_aware`` to apply the
    opt-in tenant_id schema extension (`07` §5.5 / §5.6.8). Default
    `smai migrate` (no flag) **never** touches the tenant_aware branch.

    Rollback policy: v2 does not implement ``alembic downgrade``;
    operators recover from a bad migration by restoring from backup.
    See ``MIGRATIONS.md`` next to the migrations package for the
    operator runbook.
    """
    exclusive_count = sum([check, dry_run, prune])
    if exclusive_count > 1:
        _err("smai migrate: --check, --dry-run, and --prune are mutually exclusive.")
    if upgrade_to is not None and (check or prune):
        _err("smai migrate: --upgrade-to is incompatible with --check / --prune.")

    # Unlike ``smai dev`` / ``smai run`` we DO NOT apply
    # :func:`_apply_dev_filesystem_defaults` here — those defaults are
    # passed as ``flag_overrides`` (highest layering precedence) and
    # would shadow a user's explicit ``metadata_store_config.uri``
    # from smai.yaml. Production callers of ``smai migrate`` always
    # specify a URI in their config; the empty-override path
    # falls back to plugin defaults via :func:`_resolve_metadata_store_uri`.
    try:
        runtime_config = load_runtime_config(
            config_path=config,
            defaults=dev_defaults(),
        )
    except (ConfigFileError, ConfigValidationError) as exc:
        _err(str(exc))

    uri = _resolve_metadata_store_uri(runtime_config)

    # Lazy-import the migrations module — keeps `smai --help` fast and
    # keeps the heavy SQLAlchemy import off the cold-start path of
    # other verbs.
    from smai_orchestrator.migrations import (  # noqa: PLC0415
        get_current_revision,
        get_head_revision,
        prune_retention_tables,
        render_offline_sql,
        upgrade_to_head,
    )
    from sqlalchemy.ext.asyncio import create_async_engine  # noqa: PLC0415

    if dry_run:
        target = f"{upgrade_to}@head" if upgrade_to is not None else None
        sql = render_offline_sql(uri, target=target)
        typer.echo(sql, nl=False)
        return

    async def _run_check() -> None:
        engine = create_async_engine(uri)
        try:
            current = await get_current_revision(engine)
            head = get_head_revision()
            if current == head:
                typer.echo(f"smai migrate: schema at head ({head}).")
                return
            shown_current = current if current is not None else "<unstamped>"
            typer.echo(
                f"smai migrate: schema NOT at head — current={shown_current}, head={head}.",
                err=True,
            )
            raise typer.Exit(code=1)
        finally:
            await engine.dispose()

    async def _run_upgrade() -> None:
        engine = create_async_engine(uri)
        try:
            await upgrade_to_head(engine, branch=upgrade_to)
            head = get_head_revision(branch=upgrade_to)
            if upgrade_to is None:
                typer.echo(f"smai migrate: schema upgraded to head ({head}).")
            else:
                typer.echo(f"smai migrate: schema upgraded to {upgrade_to}@head ({head}).")
        finally:
            await engine.dispose()

    async def _run_prune() -> None:
        engine = create_async_engine(uri)
        try:
            retention = runtime_config.engine.retention_policies
            deleted = await prune_retention_tables(
                engine,
                retention_days=retention if retention else None,
            )
            for name in sorted(deleted):
                typer.echo(f"{name}: deleted {deleted[name]} row(s)")
        finally:
            await engine.dispose()

    if check:
        asyncio.run(_run_check())
        return
    if prune:
        asyncio.run(_run_prune())
        return
    asyncio.run(_run_upgrade())


# === Verb 7: version =========================================================


_VERSION_PACKAGES: tuple[str, ...] = (
    "smai-cli",
    "smai-core",
    "smai-orchestrator",
    "smai-agents",
    "smai-runtime",
    "smai-llm-bedrock",
    "smai-store-sqlite",
    "smai-artifacts-localfs",
    "smai-compute-localgpu",
)


@app.command("version")
def smai_version(
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format: 'text' or 'json'."),
    ] = "text",
) -> None:
    """Print package versions for every shipped SMAI v2 package (`09` §1)."""
    out: dict[str, str] = {}
    for pkg in _VERSION_PACKAGES:
        try:
            out[pkg] = _pkg_version(pkg)
        except PackageNotFoundError:
            out[pkg] = "<not installed>"

    if output_format == "json":
        typer.echo(json.dumps(out, indent=2))
        return
    width = max(len(p) for p in out)
    for pkg, ver in out.items():
        typer.echo(f"{pkg.ljust(width)}  {ver}")


# === Module entry-point ======================================================


# Re-export defaults for tests that want to inspect the dev-config surface.
__all__ = [
    "PHASE_2_DEFAULT_PIPELINES",
    "app",
]


if __name__ == "__main__":  # pragma: no cover
    app()


def main(argv: list[str] | None = None) -> None:
    """Programmatic entry point for tests + non-default invocation.

    Typer's :class:`typer.Typer` is callable — passing ``None``
    delegates to ``sys.argv[1:]``; passing a list runs that list.
    """
    if argv is None:
        app()
        return
    # Typer accepts ``args`` via ``standalone_mode=False`` — but we
    # want full Typer behavior including SystemExit. Just patch sys.argv
    # for the call duration.
    saved = sys.argv
    try:
        sys.argv = ["smai", *argv]
        app()
    finally:
        sys.argv = saved
