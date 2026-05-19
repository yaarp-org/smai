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
import logging
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
    EntityNotFoundError,
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


_LOG_LEVEL_NAMES = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}


def _resolve_log_level(verbosity: int) -> int:
    """Resolve the root log level for ``smai dev`` / ``smai start``.

    Precedence: an explicit ``--verbose`` (``-v`` → INFO, ``-vv`` →
    DEBUG) wins; otherwise the ``SMAI_LOG_LEVEL`` env var (a standard
    level name) is honored — this is what makes ``OPERATIONS.md`` §5's
    long-standing claim true; otherwise WARNING."""
    if verbosity >= 2:  # noqa: PLR2004 — -vv
        return logging.DEBUG
    if verbosity == 1:
        return logging.INFO
    env = (os.environ.get("SMAI_LOG_LEVEL") or "").strip().upper()
    if env in _LOG_LEVEL_NAMES:
        return getattr(logging, env)
    return logging.WARNING


def _configure_logging(verbosity: int = 0) -> None:
    """Install a stderr log handler at the resolved level.

    Without this nothing in the worker / engine ever emits — the
    ``_log.info`` / ``_log.debug`` calls in ``worker/loop.py``,
    ``engine/dispatch.py``, and ``engine/_metadata_ops.py`` go to the
    root logger, which has no handler by default in a CLI process."""
    level = _resolve_log_level(verbosity)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # basicConfig is a no-op if the root logger already has handlers
    # (e.g., a test harness installed one) — force the level regardless.
    logging.getLogger().setLevel(level)


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


def _reset_dev_state(home: Path) -> list[Path]:
    """Wipe the three ``smai dev``-managed paths under ``$SMAI_HOME``.

    Returns the list of paths that were removed (or attempted to be — a
    missing path is a no-op, not an error). Touches only the three
    canonical subpaths (``state.db``, ``artifacts/``, ``workspaces/``);
    other files the user may have placed in ``~/.smai/`` are left alone.

    Backs the ``smai dev --reset`` flag (Task: round-2 friction D-iii).
    The friction it addresses: when a record (paper, proposal, CG, run)
    gets wedged in a non-terminal state in a ``smai dev`` deployment,
    there's no API-level recovery — orphan-reclaim resumes it on
    restart and re-submitting no-ops on in-flight records, so the only
    escape was editing the SQLite file by hand.
    """
    import shutil  # noqa: PLC0415 — local import; only used by this rarely-called helper

    targets = [home / "state.db", home / "artifacts", home / "workspaces"]
    removed: list[Path] = []
    for path in targets:
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(path)
    return removed


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
    reset: Annotated[
        bool,
        typer.Option(
            "--reset",
            help=(
                "Wipe the dev-managed paths under $SMAI_HOME (state.db + "
                "artifacts/ + workspaces/) before booting. Useful for "
                "clearing test detritus or unsticking a record wedged in a "
                "non-terminal state. Other files in $SMAI_HOME are left "
                "alone. Prompts for confirmation unless --yes is passed."
            ),
        ),
    ] = False,
    assume_yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip the --reset confirmation prompt. No effect without --reset.",
        ),
    ] = False,
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Increase log verbosity: -v = INFO, -vv = DEBUG. With no -v flag, "
            "the SMAI_LOG_LEVEL env var is used (default WARNING).",
        ),
    ] = 0,
) -> None:
    """Boot the laptop demo (`09` §5).

    Defaults: ``sqlite`` MetadataStore (``~/.smai/state.db``),
    ``localfs`` ArtifactStore (``~/.smai/artifacts``), ``localgpu``
    Compute, ``bedrock`` LlmProvider. ``poll_interval_seconds=10`` for
    interactive feel. Worker runs in-process; Ctrl+C drains gracefully.

    ``--reset`` wipes the three dev-managed paths under ``$SMAI_HOME``
    (default ``~/.smai``) before booting — the recovery hatch for a
    record wedged in a non-terminal state when no cancel verb exists
    yet (post-M5 backlog).
    """
    del no_dashboard  # Phase 3 wires the read-only dashboard (`09` §5.3).
    _configure_logging(verbose)
    if reset:
        home = _smai_home()
        targets = [home / "state.db", home / "artifacts", home / "workspaces"]
        existing = [p for p in targets if p.exists() or p.is_symlink()]
        if not existing:
            typer.echo(f"smai dev: --reset: nothing to wipe under {home} (already clean).")
        else:
            if not assume_yes:
                listing = "\n  ".join(str(p) for p in existing)
                typer.echo(f"smai dev: --reset will delete:\n  {listing}")
                if not typer.confirm("Continue?", default=False):
                    _err("aborted.", exit_code=1)
            removed = _reset_dev_state(home)
            for path in removed:
                typer.echo(f"smai dev: removed {path}")
    elif assume_yes:
        # `--yes` without `--reset` is benign but probably a mistake; warn
        # so it doesn't silently get swallowed.
        typer.echo("smai dev: --yes has no effect without --reset.", err=True)
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
            _enforce_container_images_published(runtime)
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
    techniques: Annotated[
        list[Path] | None,
        typer.Option(
            "--techniques",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Technique JSON file(s) to register before submitting. Repeatable. "
            "Each file is a JSON list (or {id: ref} object) of TechniqueRef "
            "objects, byte-identical to the format `smai compile --techniques` "
            "accepts. Every TechniqueRef is upserted into the MetadataStore so "
            "an experiment whose entries reference a hand-authored technique_id "
            "clears the technique.id_registered verification rule.",
        ),
    ] = None,
) -> None:
    """Submit ``experiment.yaml`` to a running deployment (`09` §5.2).

    Compiles + persists artifacts + creates a CG row in ``draft``.
    Returns the CG ID on stdout. ``--watch`` polls until terminal.

    ``--techniques FILE`` (repeatable) registers hand-authored
    techniques into the MetadataStore before the experiment is
    compiled. The upsert is idempotent, and it happens before
    compilation: if the run fails after the upserts the techniques
    stay registered, and a re-run with the same ``--techniques``
    re-upserts them harmlessly.
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
    technique_refs = (
        list(_load_technique_refs_from_files(techniques).values()) if techniques else None
    )

    async def _run() -> None:
        # Don't spin up the worker — `smai run` is a one-shot submit.
        async with Runtime.start_in_band(runtime_config, run_worker=False) as runtime:
            cg_ids = await runtime.experiments.submit_text(yaml_text, techniques=technique_refs)
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
    entity_id: Annotated[
        str,
        typer.Argument(help="A CG, proposal, or paper identifier."),
    ],
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to a smai.yaml."),
    ] = None,
    watch: Annotated[
        bool,
        typer.Option(
            "--watch",
            help="Poll until terminal (CG ids only; ignored for proposals/papers).",
        ),
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
    """Read pipeline-tracking state from MetadataStore (`09` §7).

    Accepts a CG, proposal, or paper id (probed in that order). For an
    entity in an in-progress agent state, also surfaces the per-turn
    ``status.json`` (turn count, last tool call, last error). Read-only.
    ``--watch`` polls every ``--poll-interval`` seconds until terminal —
    CG ids only; for proposals / papers it is ignored.
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
                snap = await runtime.status.resolve_status(entity_id)
            except EntityNotFoundError as exc:
                _err(str(exc), exit_code=2)
                return
            _emit_status(snap, output_format)
            if watch and snap.kind == "cg" and not snap.is_terminal:
                try:
                    await runtime.status.wait_for_terminal(
                        entity_id,
                        timeout=None,
                        poll_interval_seconds=poll_interval,
                    )
                except WaitTimeoutError as exc:
                    _err(str(exc), exit_code=3)
                    return
                final = await runtime.status.resolve_status(entity_id)
                _emit_status(final, output_format)

    asyncio.run(_run())


def _humanize_elapsed(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s ago"
    if s < 3600:
        return f"{s // 60}m ago"
    if s < 86400:
        return f"{s // 3600}h ago"
    return f"{s // 86400}d ago"


def _emit_status(snap: Any, output_format: str) -> None:
    if output_format == "json":
        typer.echo(
            json.dumps(
                {
                    "kind": snap.kind,
                    "id": snap.entity_id,
                    "state": snap.state,
                    "is_terminal": snap.is_terminal,
                    "updated_at": snap.updated_at.isoformat(),
                    "seconds_since_updated": snap.seconds_since_updated,
                    "last_error": snap.last_error,
                    "attempts": snap.attempts,
                    "agent_status": [
                        {
                            "label": a.label,
                            "role": a.role,
                            "turn_count": a.turn_count,
                            "last_tool_call": a.last_tool_call,
                            "last_tool_error": a.last_tool_error,
                            "wall_clock_utc": a.wall_clock_utc,
                            "attempt_index": a.attempt_index,
                        }
                        for a in snap.agent_status
                    ],
                }
            )
        )
        return

    terminal_marker = " (terminal)" if snap.is_terminal else ""
    typer.echo(f"{snap.entity_id} [{snap.kind}]: {snap.state}{terminal_marker}")
    typer.echo(f"  updated {_humanize_elapsed(snap.seconds_since_updated)}")
    nonzero_attempts = {k: v for k, v in snap.attempts.items() if v}
    if nonzero_attempts:
        attempts_str = ", ".join(f"{k}={v}" for k, v in sorted(nonzero_attempts.items()))
        typer.echo(f"  attempts: {attempts_str}")
    if snap.last_error:
        typer.echo(f"  last_error: {snap.last_error}")
    for a in snap.agent_status:
        action = f"last action {a.last_tool_call}" if a.last_tool_call else "no recorded action"
        err = f", last error: {a.last_tool_error}" if a.last_tool_error else ", no error"
        turn = f"turn ~{a.turn_count}" if a.turn_count is not None else "turn ?"
        when = f", updated {a.wall_clock_utc}" if a.wall_clock_utc else ""
        role = a.role or a.label
        typer.echo(f"  {role}: {turn}, {action}{err}{when}")


# === Verb 4: compile =========================================================


def _load_technique_refs_from_files(paths: list[Path]) -> dict[str, Any]:
    """Parse ``--techniques`` JSON sidecars into a ``{id: TechniqueRef}`` map.

    Each file is JSON shaped as either a ``list[TechniqueRef]`` or a
    ``dict[id, TechniqueRef]`` (the latter's keys must agree with each
    ref's ``id``). Multiple ``--techniques`` files merge; a later file
    overriding an earlier file's id is an error (silently shadowing a
    technique definition is the kind of thing that produces confusing
    verdicts later). Returns the dict :func:`load_default_registries`
    accepts for its ``technique_registry`` argument.
    """
    import json as _json  # noqa: PLC0415
    from typing import cast  # noqa: PLC0415

    from smai_core import TechniqueRef  # noqa: PLC0415

    merged: dict[str, Any] = {}
    for path in paths:
        try:
            raw: Any = _json.loads(path.read_text(encoding="utf-8"))
        except (OSError, _json.JSONDecodeError) as exc:
            _err(f"--techniques {path}: could not read JSON ({exc})")
        entries: list[tuple[str | None, Any]]
        if isinstance(raw, dict):
            entries = [(str(k), v) for k, v in cast("dict[str, Any]", raw).items()]
        elif isinstance(raw, list):
            entries = [(None, item) for item in cast("list[Any]", raw)]
        else:
            _err(f"--techniques {path}: expected a JSON list or object, got {type(raw).__name__}")
        for key, item in entries:
            try:
                ref = TechniqueRef.model_validate(item)
            except Exception as exc:  # noqa: BLE001 — surface a friendly message
                _err(f"--techniques {path}: invalid TechniqueRef ({type(exc).__name__}: {exc})")
            if key is not None and key != ref.id:
                _err(
                    f"--techniques {path}: object key {key!r} does not match the "
                    f"ref's id {ref.id!r}"
                )
            if ref.id in merged:
                _err(f"--techniques: technique id {ref.id!r} defined more than once")
            merged[ref.id] = ref
    return merged


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
    techniques: Annotated[
        list[Path] | None,
        typer.Option(
            "--techniques",
            exists=True,
            dir_okay=False,
            readable=True,
            help="JSON file of TechniqueRef definitions (a list, or an object keyed by id) "
            "to register before compiling. Repeatable. `smai compile` is pure-methodology "
            "and does not touch the MetadataStore, so techniques referenced by the "
            "experiment must be supplied here; `smai run` reads them from the store instead.",
        ),
    ] = None,
) -> None:
    """Compile a YAML to the four contract artifacts (`09` §1 / `02` §8).

    Pure methodology call — does not touch MetadataStore / Compute. No
    plugin instantiation. Useful for offline validation before
    submission. Supply ``--techniques FILE`` for any technique the
    experiment references (the registry is otherwise empty here).
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
    technique_registry = _load_technique_refs_from_files(techniques) if techniques else None
    registries = load_default_registries(technique_registry=technique_registry)

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
# See designs/smai/09-cli.md §2 for the canonical schema. Each *_config
# block is passed verbatim as kwargs to the selected plugin's constructor
# — see the per-plugin key table in README.md "Configuration".

engine:
  poll_interval_seconds: 10
  worker_count: 1
  fair_scheduling: "off"
  # Per-role agent-model overrides. Each value is a bare model id
  # understood by the `llm_provider` selected below; roles you omit fall
  # back to the built-in per-role defaults (Opus tier for planner /
  # harness_builder / technique_implementer / code_reviewer, Sonnet tier
  # for the bounded single-call roles). Precedence: the env var
  # `SMAI_MODEL_<ROLE>` (e.g. `SMAI_MODEL_PLANNER=bedrock:<id>`, which
  # also lets one role use a different provider) > this map >
  # built-in defaults. The nested env form `SMAI_ENGINE__ROLE_MODELS__PLANNER`
  # also works. (Roles: planner, harness_builder, technique_implementer,
  # code_reviewer, contextual_evaluator, supervisor, screener, enricher.)
  role_models: {}
  #   role_models:
  #     planner: us.anthropic.claude-opus-4-6-v1
  #     code_reviewer: us.anthropic.claude-sonnet-4-6

plugins:
  llm_provider: bedrock
  metadata_store: sqlite
  artifact_store: localfs
  compute: localgpu
  llm_provider_config:
    region: us-east-1
    # A Bedrock inference-profile ID you have model access to in this
    # region (`aws bedrock list-inference-profiles --region <r>`).
    model_id: us.anthropic.claude-opus-4-6-v1
  # Empty {} ⇒ sqlite plugin default = an in-memory database (fine for
  # `smai dev` / `smai ui`, which inject ~/.smai/state.db for you, but a
  # no-op for `smai migrate` / `smai start`). For those, set a file path:
  #   metadata_store_config: { uri: "sqlite+aiosqlite:////absolute/path/state.db" }
  metadata_store_config: {}
  # Empty {} ⇒ localfs root = $SMAI_ARTIFACTS_ROOT or ~/.smai/artifacts.
  artifact_store_config: {}
  # Empty {} ⇒ localgpu uses the smai-runtime:dev / smai-agent:dev images
  # (build them locally first — SMAI does not publish them).
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

    **Deprecated in v2 per `12-ui-process.md` §7.2** — use
    ``smai ui`` (with ``--no-worker`` for the read-only-against-remote
    case) instead. ``smai serve`` ships a one-line deprecation warning
    on every invocation; behavior is otherwise unchanged. Source-tree
    removal lands in v2.1 per the ``implementation_plan.md`` §8 backlog
    item.
    """
    typer.echo(
        "smai serve: deprecated in v2; use `smai ui` instead "
        "(see designs/smai/12-ui-process.md §7.2). Source-tree removal "
        "in v2.1 — migrate to `smai ui --no-worker` for the read-only "
        "case.",
        err=True,
    )

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


# === Verb 12b: ui (API + SPA host) ===========================================
#
# Per Task 4.L1 / `12-ui-process.md`: the API + SPA process. Auto-detects
# `--with-worker` from the resolved plugin shape (`12` §4.3 / §9.3),
# runs strict pre-flights when the worker is on (mirrors `smai start`),
# soft pre-flights otherwise. Bearer-token auto-generation is delegated
# to :func:`smai_api.auth._read_or_create_token_file` (the verb only
# logs the path — never the token).


def _infer_with_worker(runtime_config: Any) -> bool:
    """Apply the ``--with-worker=auto`` rule per `12` §4.3 / §9.3.

    Returns ``True`` only for the canonical Case-A laptop shape (sqlite
    metadata store + localfs artifact store). Any drift toward
    production plugins (postgres, s3, modal, runpod) flips to
    ``False``; mixed configs are conservative-no-worker per the spec
    table. The user opts back in explicitly with ``--with-worker``.
    """
    selection = runtime_config.plugins
    return selection.metadata_store == "sqlite" and selection.artifact_store == "localfs"


def _soft_validate_plugin_completeness(runtime_config: Any) -> None:
    """`--no-worker` mode: warn about missing plugin slots, don't refuse.

    Per `12` §4.4: a misconfigured remote backend is still useful
    enough that refusing-to-boot is more hostile than per-endpoint 503
    on plugin failure. We emit a single stderr warning per missing
    slot and proceed.
    """
    selection = runtime_config.plugins
    missing: list[str] = []
    for field_name in ("llm_provider", "metadata_store", "artifact_store", "compute"):
        if not getattr(selection, field_name, None):
            missing.append(f"plugins.{field_name}")
    if missing:
        typer.echo(
            "smai ui: warning — incomplete plugin selection in --no-worker "
            f"mode (missing: {', '.join(missing)}). Endpoints reading these "
            "slots will return 503; the API still boots so existing data is "
            "readable. Set the slots in your smai.yaml or via SMAI_* env "
            "vars to enable full-write workflows.",
            err=True,
        )


@app.command("ui")
def smai_ui(  # noqa: PLR0913
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to a smai.yaml; overrides search order."),
    ] = None,
    host: Annotated[
        str | None,
        typer.Option(
            "--host",
            help=(
                "Bind address. Defaults to api.host (or 127.0.0.1 if unset) "
                "per the loopback-by-default contract in `12` §4.2 / `11` §7. "
                "Broaden explicitly only if the deployment intends remote "
                "access."
            ),
        ),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option(
            "--port",
            help="TCP port to listen on. Defaults to api.port (8000).",
            min=1,
            max=65535,
        ),
    ] = None,
    with_worker: Annotated[
        bool | None,
        typer.Option(
            "--with-worker/--no-worker",
            help=(
                "Whether to start an in-process worker `asyncio.Task` "
                "alongside the API. Default: auto-detect from plugin "
                "shape per `12` §4.3 — sqlite+localfs → on; anything "
                "else → off. The inferred decision is logged loudly on "
                "startup so the user can override on the next launch."
            ),
        ),
    ] = None,
    worker_id: Annotated[
        str | None,
        typer.Option(
            "--worker-id",
            help=(
                "When --with-worker, pins the worker's lease identity "
                "(same resolution as `smai start --worker-id`)."
            ),
        ),
    ] = None,
    reload: Annotated[
        bool,
        typer.Option(
            "--reload",
            help=(
                "Dev-only: enable uvicorn auto-reload on Python source "
                "changes. Does NOT hot-reload smai.yaml (per `09` §3 — "
                "config changes need a restart). Use only with a "
                "dev-shaped config."
            ),
        ),
    ] = False,
) -> None:
    """Boot the API + SPA process (`12-ui-process.md`).

    Composes the existing primitives — :meth:`Runtime.start_in_band` +
    :func:`smai_api.make_api_app` + uvicorn — into a single verb. The
    resolved ``--with-worker`` decision is logged loudly on startup so
    the auto-detect rule (`12` §4.3) is never hidden.

    Pre-flight regime per `12` §4.4:

    * ``--with-worker`` mode → strict (mirrors ``smai start``):
      :func:`_validate_plugin_completeness` + :func:`_check_schema_at_head`,
      plus :func:`_enforce_lease_capability` only when ``worker_count > 1``.
    * ``--no-worker`` mode → soft: warn on missing plugin slots; skip
      the schema-at-head check (remote workers should have migrated);
      skip lease-capability (no leasing in this process).

    Bearer auth is opt-in per ``api.auth.enabled``; the token file at
    ``api.auth.token_path`` is auto-generated on first launch (mode
    ``0o600``, :func:`secrets.token_urlsafe(32)`) and preserved across
    restarts (per `11` §7.3).

    The verb stays scoped to uvicorn — Vite for SPA dev runs in a
    separate terminal per `12` §8 / `13` §12.1 (the canonical two-
    terminal dev workflow). ``--reload`` toggles uvicorn auto-reload
    on Python sources only.
    """
    # Layered config: dev defaults so an empty config still boots
    # usefully on a laptop (per `12` §4.2). Apply dev filesystem paths
    # before layering so the SQLite URI / artifact root resolve
    # consistently with `smai dev`.
    overrides: dict[str, Any] = _apply_dev_filesystem_defaults({})
    # Thread the per-flag overrides through the same layering pipeline so
    # CLI flags win over file/env per the standard precedence (`09` §2).
    api_flag_overrides: dict[str, Any] = {}
    if host is not None:
        api_flag_overrides["host"] = host
    if port is not None:
        api_flag_overrides["port"] = port
    if reload:
        api_flag_overrides["reload"] = True
    if with_worker is not None:
        api_flag_overrides["with_worker"] = bool(with_worker)
    if api_flag_overrides:
        overrides["api"] = api_flag_overrides
    try:
        runtime_config = load_runtime_config(
            config_path=config,
            defaults=dev_defaults(),
            flag_overrides=overrides,
        )
    except (ConfigFileError, ConfigValidationError) as exc:
        _err(str(exc))

    api_cfg = runtime_config.api
    # Resolve `with_worker` per `12` §4.3 / §9.3. Auto-detect runs only
    # when neither the flag nor the layered ``api.with_worker`` field
    # set an explicit boolean.
    explicit_request: bool | None = None
    if with_worker is not None:
        explicit_request = bool(with_worker)
    elif isinstance(api_cfg.with_worker, bool):
        explicit_request = api_cfg.with_worker
    if explicit_request is None:
        resolved_with_worker = _infer_with_worker(runtime_config)
        worker_decision_source = "auto-detect"
    else:
        resolved_with_worker = explicit_request
        worker_decision_source = "explicit"

    # Pre-flights per `12` §4.4. Strict in --with-worker; soft otherwise.
    if resolved_with_worker:
        _validate_plugin_completeness(runtime_config)
        asyncio.run(_check_schema_at_head(runtime_config))
    else:
        _soft_validate_plugin_completeness(runtime_config)

    resolved_worker_id = _resolve_worker_id(override=worker_id) if resolved_with_worker else None
    workspace_root, _artifacts, _sqlite = _dev_artifact_paths()

    # Loud startup-log line per `12` §4.3 — the inferred decision must
    # never be hidden.
    auto_explanation = ""
    if worker_decision_source == "auto-detect":
        plugin_shape = (
            f"{runtime_config.plugins.metadata_store} + {runtime_config.plugins.artifact_store}"
        )
        auto_explanation = f" (auto-detected from plugin shape: {plugin_shape})"
    if resolved_with_worker:
        typer.echo(
            f"smai ui: starting in-process worker{auto_explanation}. Pass --no-worker to disable.",
            err=True,
        )
    else:
        typer.echo(
            f"smai ui: NOT starting in-process worker{auto_explanation}. "
            "Pass --with-worker to enable. Worker(s) should run "
            "separately as `smai start` against the same backend.",
            err=True,
        )

    if api_cfg.auth.enabled:
        typer.echo(
            f"smai ui: bearer auth enabled; token file at {api_cfg.auth.token_path} "
            "(auto-generated on first launch, mode 0o600, preserved across "
            "restarts).",
            err=True,
        )

    import uvicorn  # noqa: PLC0415 — lazy-import to keep `--help` fast
    from smai_api import AuthConfig, make_api_app  # noqa: PLC0415

    auth_config: AuthConfig | None = None
    if api_cfg.auth.enabled:
        auth_config = AuthConfig(
            enabled=True,
            token_path=api_cfg.auth.token_path,
        )

    async def _run() -> None:
        async with Runtime.start_in_band(
            runtime_config,
            workspace_root=workspace_root,
            run_worker=resolved_with_worker,
            worker_id=resolved_worker_id,
        ) as runtime:
            # Multi-worker `--with-worker` against postgres is the §12 OQ7
            # warn-don't-block case: Postgres is lease-capable so the
            # check passes, but co-locating an API server with a worker
            # couples their failure modes. The strict
            # _enforce_lease_capability call below covers the worker_count > 1
            # branch; the warn-don't-block guidance is documented in the
            # verb's docstring + OPERATIONS.md.
            if resolved_with_worker:
                _enforce_lease_capability(runtime)
                _enforce_container_images_published(runtime)
            fastapi_app = make_api_app(runtime, auth_config=auth_config)
            uvicorn_config = uvicorn.Config(
                fastapi_app,
                host=api_cfg.host,
                port=api_cfg.port,
                log_level="info",
                access_log=True,
                reload=api_cfg.reload,
            )
            server = uvicorn.Server(uvicorn_config)

            stop_event = asyncio.Event()

            def _signal_handler(*_: Any) -> None:
                # Trip uvicorn's own shutdown switch first so its
                # `serve()` loop returns cleanly; then unblock the
                # outer wait. Mirrors `smai dev`'s drain pattern.
                server.should_exit = True
                stop_event.set()

            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, _signal_handler)

            typer.echo(
                f"smai ui: listening on http://{api_cfg.host}:{api_cfg.port}/. "
                "Press Ctrl+C to stop."
            )
            await server.serve()
            # Once uvicorn returns the worker drain happens automatically
            # via the ``async with Runtime.start_in_band`` exit.
            stop_event.set()

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


def _enforce_container_images_published(runtime: Any) -> None:
    """Worker boot pre-flight: refuse a registry-pull Compute substrate
    paired with a local-only built-in default container image.

    Shared across every path that boots a worker which will dispatch
    CG-execution jobs — ``smai dev``, ``smai start``, and
    ``smai ui --with-worker``. A registry-pull substrate (Modal /
    RunPod, anything whose ``ComputeCapabilities.requires_published_image``
    is ``True``) cannot pull the local-only built-in default tags — the
    experiment-seed-run images (``smai-runtime:dev`` /
    ``smai-runtime-cpu:dev``, round 11); left unchecked the failure
    surfaces as an opaque ``RemoteError: Image build ... failed``
    mid-CG-execution. This catches it at boot with a concrete,
    actionable message.

    Round 14: the agent image is not checked — the harness-builder /
    technique-implementer agents run in-process in the worker, so that
    image is never submitted to :class:`Compute`.

    Reuses :func:`smai_cli.verify.verify_runtime_image_config` so the
    boot-path check is byte-identical to the one ``smai verify`` runs.
    """
    from smai_cli.verify import verify_runtime_image_config  # noqa: PLC0415

    engine = runtime.config.engine
    result = verify_runtime_image_config(
        runtime.plugins.compute,
        engine.runtime_image,
        engine.runtime_cpu_image,
    )
    if not result.ok:
        _err(f"worker pre-flight refused to boot — {result.reason}")


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
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Increase log verbosity: -v = INFO, -vv = DEBUG. With no -v flag, "
            "the SMAI_LOG_LEVEL env var is used (default WARNING).",
        ),
    ] = 0,
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
    * Unpublished container image: a registry-pull compute substrate
      (Modal / RunPod) paired with the local-only built-in default
      ``engine.runtime_image`` / ``runtime_cpu_image`` — the operator
      never published an image. Checked after the runtime starts via
      :func:`_enforce_container_images_published` (shared with
      ``smai dev`` / ``smai ui --with-worker``).

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
    _configure_logging(verbose)
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
            _enforce_container_images_published(runtime)

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
    probe_image: Annotated[
        bool,
        typer.Option(
            "--probe-image",
            help=(
                "Additionally submit a trivial no-op job per configured "
                "container image (the two runtime images + the agent "
                "image, deduped) to confirm it is pullable by the "
                "configured compute substrate. COSTS REAL MONEY / TIME "
                "(a real, minimal compute job per unique image) — opt-in "
                "only. The free, always-on container-image config check "
                "runs without this flag."
            ),
        ),
    ] = False,
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
    * **Container-image config** (round 11) — a free, always-on check:
      when the compute substrate is registry-pull (Modal / RunPod),
      fails if ``engine.runtime_image`` / ``runtime_cpu_image`` (the
      experiment-seed-run images) are still the local-only built-in
      defaults the substrate cannot pull. The agent image is not
      checked — round 14 moved the harness-builder / technique-
      implementer agents in-process, so that image is never submitted.
    * **Container-image probe** (``--probe-image``, opt-in) — submits a
      trivial no-op job per configured runtime image (deduped) to
      confirm pullability. **Costs real money / time**, so it is opt-in
      only.

    Exit code: 0 iff every check passes. Non-zero (1) if any check
    fails — the per-check reason is printed to stdout, and the verb
    exits with code 1 so CI / deployment scripts can gate on it.
    """
    from smai_orchestrator import instantiate_plugins  # noqa: PLC0415

    from smai_cli.verify import (  # noqa: PLC0415
        VerifyResult,
        verify_artifact_store,
        verify_compute,
        verify_llm_provider,
        verify_metadata_store,
        verify_runtime_image_config,
        verify_runtime_image_probe,
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
            engine_cfg = runtime_config.engine
            results = {
                "llm_provider": await verify_llm_provider(llm_provider),
                "metadata_store": await verify_metadata_store(plugins.metadata_store),
                "artifact_store": await verify_artifact_store(plugins.artifact_store),
                "compute": await verify_compute(plugins.compute),
                # Free, always-on: a registry-pull substrate paired with
                # the local-only default runtime image is a hard fail.
                "runtime_image": verify_runtime_image_config(
                    plugins.compute,
                    engine_cfg.runtime_image,
                    engine_cfg.runtime_cpu_image,
                ),
            }
            if probe_image:
                # Opt-in real probe — costs money/time; runs only after
                # the cheap checks above are constructed.
                results["runtime_image_probe"] = await verify_runtime_image_probe(
                    plugins.compute,
                    engine_cfg.runtime_image,
                    engine_cfg.runtime_cpu_image,
                )
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

    # Warn (don't block) when the resolved store is in-memory: a migrate
    # against ``:memory:`` reports success but the database evaporates on
    # process exit, so a later ``smai verify`` / ``smai ui`` sees
    # "no such table: cgs". This is what a bare ``metadata_store_config: {}``
    # (the `smai init` default) resolves to. `smai dev` / `smai ui` paper
    # over it by injecting ~/.smai/state.db; the other verbs use the
    # config verbatim.
    if ":memory:" in uri:
        typer.echo(
            "smai migrate: warning — the resolved metadata store is an in-memory "
            "SQLite database; migrating it has no lasting effect. Set "
            "plugins.metadata_store_config.uri to a file path (e.g. "
            "'sqlite+aiosqlite:////absolute/path/state.db') in your smai.yaml, "
            "or use 'smai dev' / 'smai ui' which manage ~/.smai/state.db for you.",
            err=True,
        )

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
