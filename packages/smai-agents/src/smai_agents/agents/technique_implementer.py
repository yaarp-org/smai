"""Technique implementer dispatch handler — ``04-agents.md`` §2.3.

Two surfaces, mirroring :mod:`smai_agents.agents.harness_builder`:

* :func:`run_technique_implementer_session` — the in-process agent
  runner. Materializes a per-entry workspace seeded with the harness
  builder's outputs (templates + harness/ + techniques/__init__.py +
  techniques/baseline.py + contracts/) plus the per-entry
  :class:`TechniqueContract` and the loaded :class:`HarnessAPIManifest`
  (read-only — the runtime's no-go-zone hash check enforces the
  "do-not-modify-harness" constraint per
  ``10-runtime-and-templates.md`` §7). The agent writes only
  ``techniques/<technique_name>.py``. On retry the previous attempt's
  ``conversation-trace.json`` is copied into the workspace as
  ``prev-conversation-trace.json`` per DEC-023 / §8.

* :func:`make_dispatch_technique_implementation` — the
  :data:`smai_orchestrator.engine.types.DispatchHandler`-shaped wrapper.
  Reads the per-entry :class:`TechniqueContract`, the parent
  :class:`HarnessContract`, and the :class:`HarnessAPIManifest` from
  :class:`ArtifactStore`; pre-filters additive baselines per DEC-013 /
  DEC-017 (those skip implementation dispatch — the orchestrator marks
  them ``implemented`` directly without calling the agent); for non-
  baseline / substitutive entries it submits a Compute job and returns
  the :class:`JobHandle`.

Per DEC-017's three grounding-context types the rendered initial user
message branches on ``context_kind``: ``method_description`` (novel-
technique pipeline), ``description_only`` (standard techniques, no
paper artifact), and ``paper_extract`` (paper-sourced techniques).

Per ``04-agents.md`` §11's retry-with-feedback flow the dispatch
handler also propagates the prior attempt's review feedback —
``review_feedback`` lands in the rendered initial user message, and the
``prev-conversation-trace.json`` workspace file gives the retry agent
the previous attempt's reasoning surface (selective reads via
``read_file`` per §8 / OQ #9).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

from smai_core import HarnessContract, TechniqueContract
from smai_core.plugins import (
    ArtifactNotFound,
    ArtifactStore,
    Compute,
    JobHandle,
    LlmProvider,
)
from smai_runtime import (
    HARNESS_API_MANIFEST_FILENAME,
    HARNESS_CONTRACT_FILENAME,
    TECHNIQUE_CONTRACT_FILENAME,
    HarnessAPIManifest,
    create_workspace_skeleton,
    is_additive_baseline,
    write_template_files,
)

from smai_agents.agent_session_telemetry import (
    close_agent_session,
    make_progress_sink,
    open_agent_session,
)
from smai_agents.agents.harness_builder import SessionRunner
from smai_agents.loop import (
    AgentLoopConfig,
    AgentOutcome,
    AgentSession,
    run_loop,
)
from smai_agents.model_selection import TaskRole
from smai_agents.prompts import (
    PromptConfig,
    load_prompt_config,
    render_initial_user_message,
)
from smai_agents.retry_context import load_retry_context
from smai_agents.std_tools import build_standard_tool_registry
from smai_agents.tools import ToolRegistry

_TECHNIQUE_IMPLEMENTER_ROLE: TaskRole = "technique_implementer"

DEFAULT_AGENT_IMAGE = "smai-agent:dev"

# Three grounding-context types per DEC-017 / §2.3 Inputs.
ContextKind = Literal[
    "method_description",
    "description_only",
    "paper_extract",
]

# Default ArtifactStore key conventions — same per-CG layout as the
# harness builder. Per-entry paths use ``entries/{entry_id}/...`` per
# v1's `comparison-groups/{cg_id}/entries/{entry_id}/...` layout.
DEFAULT_TECHNIQUE_CONTRACT_KEY_TEMPLATE = (
    "comparison-groups/{cg_id}/entries/{entry_id}/technique_contract.json"
)
DEFAULT_HARNESS_CONTRACT_KEY_TEMPLATE = "comparison-groups/{cg_id}/harness/contract.json"
DEFAULT_HARNESS_MANIFEST_KEY_TEMPLATE = "comparison-groups/{cg_id}/harness/manifest.json"
DEFAULT_PREV_TRACE_KEY_TEMPLATE = (
    "comparison-groups/{cg_id}/entries/{entry_id}/conversation-trace.json"
)
DEFAULT_STATUS_KEY_TEMPLATE = "comparison-groups/{cg_id}/entries/{entry_id}/status.json"
DEFAULT_NUDGE_KEY_TEMPLATE = "comparison-groups/{cg_id}/entries/{entry_id}/nudge.txt"


# === In-process runner ======================================================


async def run_technique_implementer_session(
    *,
    workspace_path: Path,
    cg_id: str,
    entry_id: str,
    technique_id: str | None,
    technique_name: str,
    factor_dimension: str,
    factor_type: Literal["additive", "substitutive"],
    context_kind: ContextKind,
    grounding_path: str | None,
    harness_contract: HarnessContract,
    technique_contract: TechniqueContract,
    manifest: HarnessAPIManifest,
    harness_files: dict[str, bytes],
    baseline_module: bytes | None,
    llm: LlmProvider,
    artifact_store: ArtifactStore,
    compute: Compute,
    status_artifact_path: str | None = None,
    nudge_artifact_path: str | None = None,
    run_experiment_image: str | None = None,
    run_experiment_gpu: bool = True,
    review_feedback: str | None = None,
    prev_conversation_trace_path: str | None = None,
    implementation_attempt: int = 0,
    prompt_config: PromptConfig | None = None,
    config: AgentLoopConfig | None = None,
    runner: SessionRunner | None = None,
    supervisor_llm: LlmProvider | None = None,
    progress_sink: Callable[[Any], Awaitable[None]] | None = None,
) -> AgentOutcome:
    """Drive the technique-implementer agent loop in-process.

    The dispatch handler (or a test) hands in the harness builder's
    output bytes plus the contract trio; this function lays the bytes
    down on disk byte-identically (so the runtime's no-go-zone hash
    check passes), drops the contracts under ``contracts/``, optionally
    materializes the previous-attempt conversation trace per DEC-023,
    and runs the loop.

    Args:
        workspace_path: Local-disk per-entry workspace.
        cg_id, entry_id, technique_id, technique_name: Identity threaded
            into the rendered initial user message.
        factor_dimension, factor_type: Factor framing (DEC-017).
        context_kind, grounding_path: Grounding-context flavor + the
            path the agent reads to ground its implementation. The
            three flavors are :data:`ContextKind`'s literal values.
        harness_contract, technique_contract, manifest: The three
            contract artifacts the agent reads from
            ``contracts/`` (the runtime expects this layout per
            ``10-runtime-and-templates.md`` §2).
        harness_files: ``{relative_path: bytes}`` for every file under
            the harness builder's ``harness/`` directory. Replayed
            byte-identically (the no-go-zone hash check covers this
            surface per §7).
        baseline_module: Bytes of ``techniques/baseline.py`` from the
            harness builder. ``None`` when the harness builder did not
            ship a baseline (substitutive factor without explicit
            baseline shipping — the technique implementer's own
            ``baseline`` may not be on the path).
        llm: :class:`LlmProvider` for the technique-implementer role.
        artifact_store: Storage seam.
        compute: Where ``run_experiment`` validation jobs go.
        status_artifact_path, nudge_artifact_path: Optional ArtifactStore
            keys for between-turn writes; ``None`` disables.
        run_experiment_image, run_experiment_gpu: Validation-job knobs
            threaded through to :func:`build_standard_tool_registry`.
        review_feedback: Optional code-review findings to surface in
            the initial user message on retry attempts (§11).
        prev_conversation_trace_path: Optional ArtifactStore key for the
            prior attempt's conversation trace (DEC-023). When non-
            ``None`` the function copies it into
            ``<workspace_path>/prev-conversation-trace.json`` before
            invoking the loop.
        implementation_attempt: Per-entry retry counter (`01` §5.4).
            Threaded into the rendered initial user message so the
            retry prompt-config branch can reference it.
        prompt_config: Optional pre-loaded :class:`PromptConfig`.
        config: Optional :class:`AgentLoopConfig`.
        runner: Test-only seam; defaults to :func:`run_loop`.

    Returns:
        The loop's terminal :class:`AgentOutcome`.
    """
    # === Workspace materialization ==========================================
    create_workspace_skeleton(workspace_path)
    write_template_files(workspace_path)

    # Replay the harness builder's output byte-identically. The runtime's
    # no-go-zone hash check (§7) compares ``harness/*`` and the fixed
    # templates against the harness builder's emission; any byte
    # difference fails the run loudly.
    harness_dir = workspace_path / "harness"
    harness_dir.mkdir(parents=True, exist_ok=True)
    for rel, data in harness_files.items():
        target = harness_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    if baseline_module is not None:
        techniques_dir = workspace_path / "techniques"
        techniques_dir.mkdir(parents=True, exist_ok=True)
        (techniques_dir / "baseline.py").write_bytes(baseline_module)

    # Drop contracts under contracts/.
    contracts_dir = workspace_path / "contracts"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    (contracts_dir / HARNESS_CONTRACT_FILENAME).write_text(
        harness_contract.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (contracts_dir / TECHNIQUE_CONTRACT_FILENAME).write_text(
        technique_contract.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (contracts_dir / HARNESS_API_MANIFEST_FILENAME).write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )

    # === Retry-context materialization (DEC-023 / §8) =======================
    if implementation_attempt > 0 and prev_conversation_trace_path is not None:
        await load_retry_context(
            workspace_path=workspace_path,
            artifact_store=artifact_store,
            artifact_path=prev_conversation_trace_path,
        )

    # === Prompt config + tool registry ======================================
    cfg = (
        prompt_config
        if prompt_config is not None
        else load_prompt_config(role=_TECHNIQUE_IMPLEMENTER_ROLE)
    )

    tools: ToolRegistry = build_standard_tool_registry(
        run_experiment_image=run_experiment_image
        if run_experiment_image is not None
        else "smai-runtime:dev",
        run_experiment_gpu=run_experiment_gpu,
    )
    # No emit_harness_manifest tool — that is harness-builder-only per §9.

    # === Initial user message ===============================================
    initial_user = render_initial_user_message(
        cfg,
        cg_id=cg_id,
        entry_id=entry_id,
        technique_id=technique_id,
        technique_name=technique_name,
        workspace_path=str(workspace_path),
        factor_type=factor_type,
        factor_dimension=factor_dimension,
        context_kind=context_kind,
        grounding_path=grounding_path,
        review_feedback=review_feedback,
        implementation_attempt=implementation_attempt,
    )

    # === AgentSession =======================================================
    from smai_core.plugins import NormalizedMessage, TextContent  # noqa: PLC0415

    llm_providers: dict[TaskRole, LlmProvider] = {_TECHNIQUE_IMPLEMENTER_ROLE: llm}
    if supervisor_llm is not None:
        llm_providers["supervisor"] = supervisor_llm

    session = AgentSession(
        system_prompt=cfg.system_prompt,
        messages=[
            NormalizedMessage(
                role="user",
                content=[TextContent(text=initial_user)],
            )
        ],
        tools=tools,
        llm_providers=llm_providers,
        current_role=_TECHNIQUE_IMPLEMENTER_ROLE,
        workspace_path=workspace_path,
        artifact_store=artifact_store,
        compute=compute,
        status_artifact_path=status_artifact_path,
        nudge_artifact_path=nudge_artifact_path,
        # Round-6 item 5: persist this attempt's trace under the same
        # key ``retry_context.load_retry_context`` reads on the *next*
        # attempt — this finally makes the DEC-023 retry-context feature
        # work end-to-end (it was always ``ArtifactNotFound`` before).
        conversation_trace_artifact_path=DEFAULT_PREV_TRACE_KEY_TEMPLATE.format(
            cg_id=cg_id, entry_id=entry_id
        ),
        progress_sink=progress_sink,
        # The technique implementer threads ``implementation_attempt``
        # into its prompt; mirror it onto the session so the status
        # payload + the supervisor see "which retry is this".
        attempt_index=implementation_attempt,
        config=config or AgentLoopConfig(),
    )

    actual_runner = runner if runner is not None else run_loop
    return await actual_runner(session)


# === Orchestrator dispatch handler ==========================================


def make_dispatch_technique_implementation(
    *,
    workspace_root: Path,
    agent_image: str = DEFAULT_AGENT_IMAGE,
    run_experiment_image: str | None = None,
    run_experiment_gpu: bool = True,
    technique_contract_artifact_path: Callable[[str, str], str] | None = None,
    harness_contract_artifact_path: Callable[[str], str] | None = None,
    harness_manifest_artifact_path: Callable[[str], str] | None = None,
    prev_trace_artifact_path: Callable[[str, str], str] | None = None,
    status_artifact_path: Callable[[str, str], str] | None = None,
    nudge_artifact_path: Callable[[str, str], str] | None = None,
    inline_runner: SessionRunner | None = None,
) -> Callable[[Any], Awaitable[Any]]:
    """Build a :data:`DispatchHandler` for the technique-implementer role (§2.3).

    The factory returns an async callable matching
    :data:`smai_orchestrator.engine.types.DispatchHandler`. Per DEC-017
    / DEC-013 the handler **skips dispatch for additive baselines**
    (entries with ``technique_id is None`` and ``is_baseline is True``)
    by returning a :class:`DispatchOutcome` with no submitted handles
    and no error — the engine's gate-rule machinery is expected to mark
    such entries ``implemented`` directly per DEC-017 sub-decision #5.
    The skip is enforced both at the contract level
    (:func:`smai_runtime.is_additive_baseline`) and via the engine-level
    pre-filter the orchestrator's ``implementing`` composite handler
    applies (Task 2.C4 carry-forward).

    Args mirror :func:`make_dispatch_harness_build` — per-CG path
    callables here take ``(cg_id, entry_id)`` so the keys are
    parameterized per entry.
    """

    def _default_technique_contract_path(cg_id: str, entry_id: str) -> str:
        return DEFAULT_TECHNIQUE_CONTRACT_KEY_TEMPLATE.format(cg_id=cg_id, entry_id=entry_id)

    def _default_harness_contract_path(cg_id: str) -> str:
        return DEFAULT_HARNESS_CONTRACT_KEY_TEMPLATE.format(cg_id=cg_id)

    def _default_manifest_path(cg_id: str) -> str:
        return DEFAULT_HARNESS_MANIFEST_KEY_TEMPLATE.format(cg_id=cg_id)

    def _default_prev_trace_path(cg_id: str, entry_id: str) -> str:
        return DEFAULT_PREV_TRACE_KEY_TEMPLATE.format(cg_id=cg_id, entry_id=entry_id)

    def _default_status_path(cg_id: str, entry_id: str) -> str:
        return DEFAULT_STATUS_KEY_TEMPLATE.format(cg_id=cg_id, entry_id=entry_id)

    def _default_nudge_path(cg_id: str, entry_id: str) -> str:
        return DEFAULT_NUDGE_KEY_TEMPLATE.format(cg_id=cg_id, entry_id=entry_id)

    technique_contract_path_fn = (
        technique_contract_artifact_path or _default_technique_contract_path
    )
    harness_contract_path_fn = harness_contract_artifact_path or _default_harness_contract_path
    manifest_path_fn = harness_manifest_artifact_path or _default_manifest_path
    prev_trace_path_fn = prev_trace_artifact_path or _default_prev_trace_path
    status_path_fn = status_artifact_path or _default_status_path
    nudge_path_fn = nudge_artifact_path or _default_nudge_path

    async def _dispatch(ctx: Any) -> Any:
        from smai_orchestrator.engine.types import DispatchOutcome  # noqa: PLC0415

        # The dispatched entity is an Entry; its parent CG id and per-
        # entry identity are read off the metadata store. The C2 worker
        # constructs DispatchContext with entity_kind=entry — the
        # ctx.entity_id is the entry id.
        entry_id = ctx.entity_id
        entry_record = await ctx.metadata_store.get_entry(entry_id)
        if entry_record is None:
            return DispatchOutcome(
                submitted_handles=[],
                error=f"EntryRecord {entry_id!r} not found in MetadataStore",
            )
        cg_id = entry_record.cg_id

        contract_key = technique_contract_path_fn(cg_id, entry_id)
        try:
            raw = await ctx.artifact_store.get(contract_key)
        except ArtifactNotFound:
            return DispatchOutcome(
                submitted_handles=[],
                error=(
                    f"TechniqueContract not found at ArtifactStore key "
                    f"{contract_key!r}; the upstream pipeline-spec gate must "
                    "have written it before the implementing state was entered."
                ),
            )
        try:
            technique_contract = TechniqueContract.model_validate_json(raw)
        except ValueError as exc:
            return DispatchOutcome(
                submitted_handles=[],
                error=(f"TechniqueContract at {contract_key!r} failed Pydantic validation: {exc}"),
            )

        # Per DEC-013 / DEC-017: additive baselines are not implemented
        # by the agent. The orchestrator's composite ``implementing``
        # gate skips dispatch and marks them ``implemented`` directly
        # (Task 2.C4). This handler returns a no-op outcome so a
        # spec-spec author who forgot the pre-filter still observes the
        # skip rather than burning compute.
        if is_additive_baseline(technique_contract):
            return DispatchOutcome(submitted_handles=[], error=None)

        llm = ctx.llm
        if llm is None:
            return DispatchOutcome(
                submitted_handles=[],
                error=(
                    "DispatchContext.llm is None; technique implementer requires an LlmProvider."
                ),
            )

        manifest_key = manifest_path_fn(cg_id)
        prev_trace_key = (
            prev_trace_path_fn(cg_id, entry_id) if entry_record.implementation_attempt > 0 else None
        )
        status_key = status_path_fn(cg_id, entry_id)
        nudge_key = nudge_path_fn(cg_id, entry_id)

        if inline_runner is None:
            # Production path: the agent-side container loads the workspace
            # and re-derives the session via the same inputs. The dispatch
            # handler stays light here — it submits the job and returns
            # the handle. The container's entrypoint module is configured
            # by the deployment.
            # TODO(observability): agent_sessions for the container path —
            # the container has no MetadataStore handle, so the row would
            # have to be created here at submit-time and closed when the
            # phase-1 job-succeeded/failed handler observes the terminal
            # job. That spans two code sites; left for a follow-up. The
            # inline path (below) and the planner are fully covered.
            command = [
                "python",
                "-m",
                "smai_agents.agents.technique_implementer",
                "--cg-id",
                cg_id,
                "--entry-id",
                entry_id,
                "--manifest-artifact-path",
                manifest_key,
                "--technique-contract-key",
                contract_key,
            ]
            env = {
                "SMAI_CG_ID": cg_id,
                "SMAI_ENTRY_ID": entry_id,
                "SMAI_HARNESS_MANIFEST_KEY": manifest_key,
            }
            try:
                handle = await ctx.compute.submit(
                    image=agent_image,
                    command=command,
                    env=env,
                    gpu=False,
                )
            except Exception as exc:  # noqa: BLE001
                return DispatchOutcome(
                    submitted_handles=[],
                    error=f"{type(exc).__name__}: {exc}",
                )
            return DispatchOutcome(submitted_handles=[handle], error=None)

        # Test path: run the agent loop in-process. The handler reads
        # the harness contract + manifest + harness/* + baseline module
        # so that workspace materialization completes before the loop.
        harness_contract_key = harness_contract_path_fn(cg_id)
        try:
            raw_h = await ctx.artifact_store.get(harness_contract_key)
            raw_m = await ctx.artifact_store.get(manifest_key)
        except ArtifactNotFound as exc:
            return DispatchOutcome(
                submitted_handles=[],
                error=f"missing artifact: {exc}",
            )
        try:
            harness_contract = HarnessContract.model_validate_json(raw_h)
            manifest = HarnessAPIManifest.model_validate_json(raw_m)
        except ValueError as exc:
            return DispatchOutcome(
                submitted_handles=[],
                error=f"contract validation failed: {exc}",
            )

        harness_files, baseline_module = await _load_harness_outputs(
            artifact_store=ctx.artifact_store,
            cg_id=cg_id,
        )

        technique_id = technique_contract.body.technique_id
        # technique_name fallback: use the technique id when available;
        # for additive baselines we'd return early above. Production
        # deployments may override by passing a different
        # `technique_name_resolver` once that surface exists.
        technique_name = technique_id or "baseline"

        workspace_path = workspace_root / cg_id / entry_id

        # Translate :class:`EngineConfig` supervisor settings (Task
        # 3.G4) into the loop's :class:`AgentLoopConfig`. Some unit
        # tests leave ``ctx.config = None``; treat that as "supervisor
        # off" so those tests don't have to wire a full config in.
        engine_config = getattr(ctx, "config", None)
        if (
            engine_config is not None
            and getattr(engine_config, "supervisor_enabled", False) is True
        ):
            supervisor_on = True
            supervisor_cadence = int(getattr(engine_config, "supervisor_check_every_n_turns", 0))
        else:
            supervisor_on = False
            supervisor_cadence = 0
        loop_config = AgentLoopConfig(supervisor_check_every_turns=supervisor_cadence)
        session_id = await open_agent_session(
            ctx.metadata_store,
            parent_kind="entry",
            parent_id=entry_id,
            agent_role="technique_implementer",
            llm=llm,
        )
        # Round-6 item D: close the ``agent_sessions`` row on EVERY exit
        # path (including a raised session).
        outcome = None
        try:
            outcome = await run_technique_implementer_session(
                workspace_path=workspace_path,
                cg_id=cg_id,
                entry_id=entry_id,
                technique_id=technique_id,
                technique_name=technique_name,
                factor_dimension=harness_contract.body.factor.name,
                factor_type=harness_contract.body.factor.type,
                context_kind=_resolve_context_kind(technique_contract),
                grounding_path=None,
                harness_contract=harness_contract,
                technique_contract=technique_contract,
                manifest=manifest,
                harness_files=harness_files,
                baseline_module=baseline_module,
                llm=llm,
                artifact_store=ctx.artifact_store,
                compute=ctx.compute,
                status_artifact_path=status_key,
                nudge_artifact_path=nudge_key,
                run_experiment_image=run_experiment_image,
                run_experiment_gpu=run_experiment_gpu,
                prev_conversation_trace_path=prev_trace_key,
                implementation_attempt=entry_record.implementation_attempt,
                runner=inline_runner,
                supervisor_llm=llm if supervisor_on else None,
                config=loop_config,
                progress_sink=make_progress_sink(ctx.metadata_store, session_id),
            )
        finally:
            await close_agent_session(ctx.metadata_store, session_id, outcome)
        return DispatchOutcome(
            submitted_handles=[
                JobHandle(plugin="inline", handle=f"inline-{entry_id}"),
            ],
            error=None,
        )

    return _dispatch


# === Helpers ================================================================


def _resolve_context_kind(technique_contract: TechniqueContract) -> ContextKind:
    """Map the contract's grounding shape to the prompt's ``context_kind``.

    Per DEC-017 the three input shapes the technique implementer
    consumes are ``method_description`` (novel-technique pipeline,
    primary case under DEC-032), ``description_only`` (standard
    techniques per DEC-015), and ``paper_extract`` (paper-sourced
    techniques). The contract's ``standard`` flag plus the
    ``fidelity_anchor`` discriminate cleanly:
    """
    body = technique_contract.body
    if body.standard:
        return "description_only"
    anchor = body.fidelity_anchor
    if anchor is None:
        # No anchor — best effort default to ``method_description`` for
        # the novel-technique-pipeline primary case. Production deployments
        # may override by supplying an explicit ``context_kind`` on the
        # contract once that surface exists.
        return "method_description"
    kind = getattr(anchor, "kind", None)
    if kind == "paper":
        return "paper_extract"
    return "method_description"


# Path inside ArtifactStore for the harness builder's harness/* outputs.
# The default mirrors v1's `comparison-groups/{cg_id}/harness/*` layout —
# every file under that prefix (excluding `contract.json`, `manifest.json`,
# and `status.json`) is treated as a harness module.
_HARNESS_OUTPUT_PREFIX_TEMPLATE = "comparison-groups/{cg_id}/harness/"
_HARNESS_OUTPUT_RESERVED_NAMES: frozenset[str] = frozenset(
    {"contract.json", "manifest.json", "status.json", "nudge.txt"}
)


async def _load_harness_outputs(
    *,
    artifact_store: ArtifactStore,
    cg_id: str,
) -> tuple[dict[str, bytes], bytes | None]:
    """Read the harness builder's harness/* outputs + optional baseline.

    Returns ``(harness_files, baseline_module)`` where ``harness_files``
    is the ``{relative_path: bytes}`` mapping the in-process runner
    expects and ``baseline_module`` is the ``techniques/baseline.py``
    bytes if the harness builder shipped one.

    The default key convention is the v1 carry-forward; deployments
    that store the harness builder's outputs under different keys may
    override the dispatch-handler's path-resolver callables.
    """
    prefix = _HARNESS_OUTPUT_PREFIX_TEMPLATE.format(cg_id=cg_id)
    iterator = await artifact_store.list(prefix)

    harness_files: dict[str, bytes] = {}
    baseline_module: bytes | None = None

    async for key in iterator:
        rel = key[len(prefix) :]
        if not rel:
            continue
        # Skip the harness contract / manifest / status artifacts —
        # those are already loaded separately.
        if rel in _HARNESS_OUTPUT_RESERVED_NAMES:
            continue
        if rel == "techniques/baseline.py":
            baseline_module = await artifact_store.get(key)
            continue
        body = await artifact_store.get(key)
        # Only carry harness/* into the harness_files dict; everything
        # else (config.yaml, validation_results.json, etc.) lives at
        # the top of the prefix and is irrelevant to the no-go-zone
        # hash check the technique implementer's run will face.
        if rel.startswith("harness/"):
            harness_files[rel[len("harness/") :]] = body

    return harness_files, baseline_module


__all__ = [
    "ContextKind",
    "DEFAULT_AGENT_IMAGE",
    "DEFAULT_HARNESS_CONTRACT_KEY_TEMPLATE",
    "DEFAULT_HARNESS_MANIFEST_KEY_TEMPLATE",
    "DEFAULT_NUDGE_KEY_TEMPLATE",
    "DEFAULT_PREV_TRACE_KEY_TEMPLATE",
    "DEFAULT_STATUS_KEY_TEMPLATE",
    "DEFAULT_TECHNIQUE_CONTRACT_KEY_TEMPLATE",
    "make_dispatch_technique_implementation",
    "run_technique_implementer_session",
]
