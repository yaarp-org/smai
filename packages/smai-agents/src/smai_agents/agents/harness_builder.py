"""Harness builder dispatch handler — ``04-agents.md`` §2.2.

Two surfaces:

* :func:`run_harness_builder_session` — the in-process agent runner.
  Materializes the workspace skeleton + ``harness_contract.json``,
  builds the :class:`AgentSession` with the harness-builder prompt
  config, the standard tool inventory, and the harness-builder-only
  ``emit_harness_manifest`` tool, then drives the loop via
  :func:`run_loop`.

* :func:`dispatch_harness_build` — the
  :data:`smai_orchestrator.engine.types.DispatchHandler`-shaped wrapper
  the engine dispatches when a CG enters the ``implementing`` state per
  ``03-state-machine.md`` §3. Reads the :class:`HarnessContract` from
  :class:`ArtifactStore`, populates the workspace, and runs the agent
  loop **in-process in the worker** — exactly as the planner dispatch
  does (``smai_agents.agents.planner.make_dispatch_planner``). Round 14
  removed the abandoned containerized-agent path. After the session the
  handler publishes the agent's workspace outputs to
  :class:`ArtifactStore`
  (:func:`smai_agents.agents.artifact_publish.publish_workspace_outputs`)
  and synthesizes an ``inline-<cg_id>`` :class:`JobHandle`; the
  ``implementing`` state's edges are ``dispatch_time`` (not
  ``job_succeeded`` — there is no external job to poll), so the next
  worker cycle re-evaluates them against the published artifacts.

Two production-hardening costs are knowingly deferred (acceptable for
``smai dev``; open items for ``smai start``): (a) the agent's
``execute`` tool runs shell subprocesses on the worker host rather than
in a container — no isolation; (b) the multi-turn agent loop runs
synchronously and blocks the worker's poll cycle for its full
duration — the dispatch is not made non-blocking. Both are tracked for
the agent-layer refactor's sandboxed-execution path.

Factor-type-aware framing (DEC-017 / §9 of ``10-runtime-and-templates.md``):
the harness builder always builds the same code shape; what differs is
whether the extension point has a working default (``additive``) or is
a mandatory slot (``substitutive``). The factor type drops into the
rendered initial user message so the agent is aware of the constraint
and the prompt's variant slot can pivot. v1 ships base only; variants
land per §10 OQ.

Per ``05-orchestrator.md`` §1.4: agents may not write to
:class:`MetadataStore`. The ``harness_api_manifest_hash`` per-entry
write per DEC-033 #3 is the orchestrator's gate-rule territory (Task
2.C4 / DEC-016 carry-forward), not this handler's.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from smai_core import HarnessContract, TechniqueContract
from smai_core.plugins import (
    ArtifactNotFound,
    ArtifactStore,
    Compute,
    JobHandle,
    LlmProvider,
)
from smai_runtime import (
    HARNESS_CONTRACT_FILENAME,
    TECHNIQUE_CONTRACT_FILENAME,
    create_workspace_skeleton,
    write_template_files,
)

from smai_agents.agent_session_telemetry import (
    close_agent_session,
    make_progress_sink,
    open_agent_session,
)
from smai_agents.agents.artifact_publish import publish_workspace_outputs
from smai_agents.agents.harness_api_reference import (
    WORKSPACE_HARNESS_API_REFERENCE_PATH,
    build_harness_api_reference,
)
from smai_agents.agents.manifest_tool import make_emit_harness_manifest_tool
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
from smai_agents.std_tools import (
    DEFAULT_RUN_EXPERIMENT_CPU_IMAGE,
    DEFAULT_RUN_EXPERIMENT_IMAGE,
    build_standard_tool_registry,
)
from smai_agents.tools import ToolRegistry

_HARNESS_BUILDER_ROLE: TaskRole = "harness_builder"

# Path inside the workspace where the materialized HarnessContract lives.
# Mirrors :data:`smai_runtime.HARNESS_CONTRACT_FILENAME` ("harness_contract.json")
# under ``contracts/``. Local-relative form lets the agent find it via
# ``read_file("contracts/harness_contract.json")``.
WORKSPACE_HARNESS_CONTRACT_PATH = "contracts/" + HARNESS_CONTRACT_FILENAME

# v1 path convention — mirrors ``CLAUDE.md`` /
# ``designs/yaarp/...`` Artifact Layout for ``comparison-groups/{cg_id}/``.
# Carrying the literal here as the default; deployments may override via
# ``dispatch_harness_build`` factory args (e.g., a hashed-content-addressed
# path for the Postgres-+-S3 production stack).
DEFAULT_HARNESS_CONTRACT_KEY_TEMPLATE = "comparison-groups/{cg_id}/harness/contract.json"
DEFAULT_HARNESS_MANIFEST_KEY_TEMPLATE = "comparison-groups/{cg_id}/harness/manifest.json"
DEFAULT_HARNESS_STATUS_KEY_TEMPLATE = "comparison-groups/{cg_id}/harness/status.json"
DEFAULT_HARNESS_NUDGE_KEY_TEMPLATE = "comparison-groups/{cg_id}/harness/nudge.txt"
DEFAULT_HARNESS_TRACE_KEY_TEMPLATE = "comparison-groups/{cg_id}/harness/conversation-trace.json"

# Per-entry technique-contract key — same template the technique implementer
# (and the orchestrator's cg_execution / proposal specs) use. Re-stated
# here as a literal rather than imported from ``smai_agents.agents.technique_implementer``
# because that module imports from this one (SessionRunner), and an
# import-back-edge would create a cycle.
DEFAULT_TECHNIQUE_CONTRACT_KEY_TEMPLATE = (
    "comparison-groups/{cg_id}/entries/{entry_id}/technique_contract.json"
)

# Workspace-relative path the runtime's ``load_contracts`` reads when the
# runner starts. The harness builder stages the BASELINE entry's contract
# here so the in-loop ``run_experiment`` validation has the
# ``technique_params`` / ``level_value`` ``build_runtime_config`` needs
# (the manifest stays absent — the runtime's validation mode synthesizes a
# stub; the agent emits the real manifest after a passing validation).
WORKSPACE_TECHNIQUE_CONTRACT_PATH = "contracts/" + TECHNIQUE_CONTRACT_FILENAME


# Test-only seam: a ``runner`` callable that takes the assembled
# :class:`AgentSession` and returns an :class:`AgentOutcome`. Production
# deployments leave this ``None``; tests pass :func:`run_loop` (or a
# stub that records the session for later assertions). The seam exists
# because the in-process runner does not return a :class:`JobHandle` —
# the dispatch handler synthesizes one when the agent loop ran inline.
SessionRunner = Callable[[AgentSession], Awaitable[AgentOutcome]]


# === In-process runner ======================================================


async def run_harness_builder_session(
    *,
    workspace_path: Path,
    harness_contract: HarnessContract,
    baseline_technique_contract: TechniqueContract,
    cg_id: str,
    llm: LlmProvider,
    artifact_store: ArtifactStore,
    compute: Compute,
    manifest_artifact_path: str,
    status_artifact_path: str | None = None,
    nudge_artifact_path: str | None = None,
    runtime_image: str | None = None,
    runtime_cpu_image: str | None = None,
    prompt_config: PromptConfig | None = None,
    config: AgentLoopConfig | None = None,
    runner: SessionRunner | None = None,
    supervisor_llm: LlmProvider | None = None,
    progress_sink: Callable[[Any], Awaitable[None]] | None = None,
) -> AgentOutcome:
    """Drive the harness-builder agent loop in-process.

    Materializes the workspace skeleton + fixed templates + the loaded
    :class:`HarnessContract` (per ``10-runtime-and-templates.md`` §2),
    composes the :class:`AgentSession` with the harness-builder prompt
    config + the standard tool registry + the harness-builder-only
    ``emit_harness_manifest`` tool, and returns the loop's
    :class:`AgentOutcome`.

    Args:
        workspace_path: Local-disk workspace the agent operates in.
        harness_contract: The loaded contract (already round-tripped
            from :class:`ArtifactStore`); referenced by the manifest
            tool's hash check + factor-type-aware framing.
        baseline_technique_contract: The CG's baseline-entry technique
            contract (``is_baseline=True``). Staged at
            ``contracts/technique_contract.json`` so the runtime's
            ``load_contracts`` finds it during the agent's in-loop
            ``run_experiment`` validation — ``build_runtime_config``
            reads ``technique_params`` / ``level_value`` off it to
            assemble the harness config dict. The dispatch handler
            fetches it from :class:`ArtifactStore` (see
            :func:`make_dispatch_harness_build`).
        cg_id: The CG identity, threaded into the rendered initial
            user message.
        llm: :class:`LlmProvider` for the harness-builder role.
        artifact_store: Storage seam for status writes and the manifest.
        compute: Submitted ``run_experiment`` validation jobs go here.
        manifest_artifact_path: ArtifactStore key under which the
            ``emit_harness_manifest`` handler writes the frozen manifest.
        status_artifact_path: Optional ArtifactStore key for between-turn
            status writes (§3.2). ``None`` disables.
        nudge_artifact_path: Optional ArtifactStore key for supervisor
            nudges (§2.6 / §3.2).
        runtime_image: GPU container image for ``run_experiment``
            validation jobs. Selected when
            :attr:`HarnessContractBody.compute.gpu` is ``True``.
            Defaults to :data:`smai_agents.std_tools.DEFAULT_RUN_EXPERIMENT_IMAGE`
            when ``None`` (``smai-runtime:dev``).
        runtime_cpu_image: CPU container image for ``run_experiment``
            validation jobs. Selected when
            :attr:`HarnessContractBody.compute.gpu` is ``False`` —
            mirrors the seed-run dispatcher's
            ``image = gpu_image if gpu else cpu_image`` selection in
            :mod:`smai_orchestrator.specs.run_record` so a CPU-only
            experiment cannot dispatch a GPU-only image the operator
            never built. Defaults to
            :data:`smai_agents.std_tools.DEFAULT_RUN_EXPERIMENT_CPU_IMAGE`
            when ``None`` (``smai-runtime-cpu:dev``).
        prompt_config: Optional pre-loaded :class:`PromptConfig`. When
            ``None`` the function calls
            :func:`smai_agents.load_prompt_config` for the
            ``harness_builder`` role. The variant slot (factor-type-keyed
            per §10 OQ) ships base-only in v1.
        config: Optional :class:`AgentLoopConfig`; defaults to the
            stock value.
        runner: Test-only seam — when ``None``, calls :func:`run_loop`.
            Tests inject a stub that asserts on the session shape
            without driving the loop.

    Returns:
        The loop's terminal :class:`AgentOutcome`.
    """
    # === Workspace materialization ==========================================
    # The harness builder writes harness/* and techniques/baseline.py against
    # an empty workspace skeleton + fixed templates. The HarnessContract
    # drops in here (via write_text) since `materialize_workspace` requires
    # a manifest for its full materialize path — which the harness builder
    # has not yet produced.
    create_workspace_skeleton(workspace_path)
    write_template_files(workspace_path)
    contract_path = workspace_path / WORKSPACE_HARNESS_CONTRACT_PATH
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(harness_contract.model_dump_json(indent=2))

    # Stage the baseline entry's technique contract so the runtime's
    # ``load_contracts`` succeeds during the agent's ``run_experiment``
    # validation. The runtime's validation mode tolerates a missing
    # ``harness_api_manifest.json`` (synthesizes a stub — manifest is the
    # OUTCOME of a passing validation per ``04-agents.md`` §9 step 5), but
    # the technique contract stays required: ``build_runtime_config`` reads
    # ``technique_params`` / ``level_value`` off it. The agent runs
    # ``run_experiment(technique="baseline", ...)`` so the baseline entry's
    # contract is the right one to stage.
    technique_contract_path = workspace_path / WORKSPACE_TECHNIQUE_CONTRACT_PATH
    technique_contract_path.parent.mkdir(parents=True, exist_ok=True)
    technique_contract_path.write_text(baseline_technique_contract.model_dump_json(indent=2))

    # Stage the harness-API reference so the agent can learn the ABI it
    # must implement via ``read_file``. The smai_runtime framework source
    # is outside the workspace sandbox — the file tools reject every
    # out-of-workspace path — so without this staged reference the agent
    # loops trying (and failing) to read ``smai_runtime/*.py``. The
    # reference is generated from the live framework so it cannot drift.
    api_reference_path = workspace_path / WORKSPACE_HARNESS_API_REFERENCE_PATH
    api_reference_path.parent.mkdir(parents=True, exist_ok=True)
    api_reference_path.write_text(build_harness_api_reference())

    # === Prompt config + tool registry ======================================
    cfg = (
        prompt_config
        if prompt_config is not None
        else load_prompt_config(role=_HARNESS_BUILDER_ROLE)
    )

    # Derive the ``run_experiment`` GPU flag AND the matching image from
    # the locked contract (`02-dsl-and-contracts.md` §7.4). The harness
    # contract's ``body.compute.gpu`` is the source of truth — the same
    # field the seed-run dispatcher reads (specs/run_record.py
    # _load_compute_requirements_gpu) and uses to pick between the GPU
    # and CPU images (``image = gpu_image if gpu else cpu_image``). The
    # selection lives here at the session boundary, NOT inside the tool
    # factory, so the gpu flag and the chosen image cannot drift apart;
    # round 16 fixed the gpu flag but left the image hardcoded, which
    # made CPU-only experiments on Docker Desktop macOS attempt to pull
    # ``smai-runtime:dev`` (a GPU image the operator never built) and
    # fail with ``JobImageInvalid``.
    gpu = harness_contract.body.compute.gpu
    selected_run_experiment_image = (
        (runtime_image if runtime_image is not None else DEFAULT_RUN_EXPERIMENT_IMAGE)
        if gpu
        else (
            runtime_cpu_image if runtime_cpu_image is not None else DEFAULT_RUN_EXPERIMENT_CPU_IMAGE
        )
    )
    tools: ToolRegistry = build_standard_tool_registry(
        run_experiment_image=selected_run_experiment_image,
        run_experiment_gpu=gpu,
    )
    tools.register(
        make_emit_harness_manifest_tool(
            harness_contract=harness_contract,
            artifact_path=manifest_artifact_path,
        )
    )

    # === Initial user message ===============================================
    initial_user = render_initial_user_message(
        cfg,
        cg_id=cg_id,
        workspace_path=str(workspace_path),
        factor_type=harness_contract.body.factor.type,
        factor_dimension=harness_contract.body.factor.name,
        manifest_artifact_path=manifest_artifact_path,
    )

    # === AgentSession =======================================================
    from smai_core.plugins import NormalizedMessage, TextContent  # noqa: PLC0415

    llm_providers: dict[TaskRole, LlmProvider] = {_HARNESS_BUILDER_ROLE: llm}
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
        current_role=_HARNESS_BUILDER_ROLE,
        workspace_path=workspace_path,
        artifact_store=artifact_store,
        compute=compute,
        status_artifact_path=status_artifact_path,
        nudge_artifact_path=nudge_artifact_path,
        conversation_trace_artifact_path=DEFAULT_HARNESS_TRACE_KEY_TEMPLATE.format(cg_id=cg_id),
        progress_sink=progress_sink,
        # Harness builder has no attempt counter — leave attempt_index None.
        config=config or AgentLoopConfig(),
    )

    actual_runner = runner if runner is not None else run_loop
    return await actual_runner(session)


# === Orchestrator dispatch handler ==========================================


def make_dispatch_harness_build(
    *,
    workspace_root: Path,
    runtime_image: str | None = None,
    runtime_cpu_image: str | None = None,
    harness_contract_artifact_path: Callable[[str], str] | None = None,
    harness_manifest_artifact_path: Callable[[str], str] | None = None,
    technique_contract_artifact_path: Callable[[str, str], str] | None = None,
    status_artifact_path: Callable[[str], str] | None = None,
    nudge_artifact_path: Callable[[str], str] | None = None,
    inline_runner: SessionRunner | None = None,
) -> Callable[[Any], Awaitable[Any]]:
    """Build a :data:`DispatchHandler` for the harness-builder role (§2.2).

    Production deployments construct one of these per pipeline-spec
    instance and bind it on the ``implementing`` state's
    :class:`smai_orchestrator.engine.types.DispatchAction.handler` slot.

    The handler runs the harness-builder agent loop **in-process in the
    worker**, structurally mirroring
    :func:`smai_agents.agents.planner.make_dispatch_planner`: it calls
    :func:`run_harness_builder_session` unconditionally (``runner``
    defaults to :func:`run_loop`), publishes the agent's workspace
    outputs to :class:`ArtifactStore`, and synthesizes an
    ``inline-<cg_id>`` :class:`JobHandle`. There is no containerized
    branch — round 14 removed the unfinished agent-container path.

    Deferred production-hardening (see the module docstring): the
    ``execute`` tool's shell subprocesses run on the worker host (no
    container isolation), and the multi-turn loop blocks the worker
    poll cycle for its full duration (the dispatch is not non-blocking).
    Both are acceptable for ``smai dev`` and open for ``smai start``.

    Args:
        workspace_root: Filesystem root under which per-CG workspaces
            land at ``<workspace_root>/<cg_id>/``.
        runtime_image: GPU container image for ``run_experiment``
            validation jobs *inside* the agent's tool surface.
        runtime_cpu_image: CPU container image for ``run_experiment``
            validation jobs. Both images are threaded to
            :func:`run_harness_builder_session`; neither the GPU flag
            nor the image is a knob on this factory — both are derived
            from the CG's :class:`HarnessContract` (``body.compute.gpu``)
            inside the session runner, mirroring the seed-run dispatcher's
            ``_load_compute_requirements_gpu`` + ``image = gpu_image
            if gpu else cpu_image`` so the agent never needs to think
            about hardware and the gpu flag and chosen image cannot
            drift apart.
        harness_contract_artifact_path: Callable resolving
            ``cg_id → ArtifactStore key`` for the loaded contract.
            Defaults to
            :data:`DEFAULT_HARNESS_CONTRACT_KEY_TEMPLATE`.
        harness_manifest_artifact_path: Callable resolving
            ``cg_id → ArtifactStore key`` for the manifest write.
        status_artifact_path: Callable resolving the per-job status JSON
            key.
        nudge_artifact_path: Callable resolving the supervisor-nudge key.
        inline_runner: Test-only runner override — when non-``None`` it
            replaces :func:`run_loop` as the session runner. Production
            leaves it ``None``. Mirrors
            :func:`make_dispatch_planner`'s ``inline_runner`` seam.

    Returns:
        A :data:`DispatchHandler`-shaped async callable suitable for
        :class:`smai_orchestrator.engine.types.DispatchAction.handler`.
        Typed as ``Callable[[Any], Awaitable[Any]]`` here to keep this
        module's import surface free of the orchestrator-engine types
        per the smai-agents → smai-orchestrator dependency direction
        (the engine imports the handler shape; the handler module need
        not import the engine).
    """

    def _default_contract_path(cg_id: str) -> str:
        return DEFAULT_HARNESS_CONTRACT_KEY_TEMPLATE.format(cg_id=cg_id)

    def _default_manifest_path(cg_id: str) -> str:
        return DEFAULT_HARNESS_MANIFEST_KEY_TEMPLATE.format(cg_id=cg_id)

    def _default_status_path(cg_id: str) -> str:
        return DEFAULT_HARNESS_STATUS_KEY_TEMPLATE.format(cg_id=cg_id)

    def _default_nudge_path(cg_id: str) -> str:
        return DEFAULT_HARNESS_NUDGE_KEY_TEMPLATE.format(cg_id=cg_id)

    def _default_technique_contract_path(cg_id: str, entry_id: str) -> str:
        return DEFAULT_TECHNIQUE_CONTRACT_KEY_TEMPLATE.format(cg_id=cg_id, entry_id=entry_id)

    contract_path_fn = harness_contract_artifact_path or _default_contract_path
    manifest_path_fn = harness_manifest_artifact_path or _default_manifest_path
    status_path_fn = status_artifact_path or _default_status_path
    nudge_path_fn = nudge_artifact_path or _default_nudge_path
    technique_contract_path_fn = (
        technique_contract_artifact_path or _default_technique_contract_path
    )

    async def _dispatch(ctx: Any) -> Any:  # ctx: DispatchContext, returns DispatchOutcome
        from smai_orchestrator.engine.types import DispatchOutcome  # noqa: PLC0415

        cg_id = ctx.entity_id

        # Read the loaded HarnessContract from ArtifactStore.
        contract_key = contract_path_fn(cg_id)
        try:
            raw = await ctx.artifact_store.get(contract_key)
        except ArtifactNotFound:
            return DispatchOutcome(
                submitted_handles=[],
                error=(
                    f"HarnessContract not found at ArtifactStore key "
                    f"{contract_key!r}; the upstream pipeline-spec gate must "
                    "have written it before the implementing state was entered."
                ),
            )
        try:
            harness_contract = HarnessContract.model_validate_json(raw)
        except ValueError as exc:
            return DispatchOutcome(
                submitted_handles=[],
                error=(f"HarnessContract at {contract_key!r} failed Pydantic validation: {exc}"),
            )

        # Find the baseline entry and fetch its technique contract — the
        # runtime's ``load_contracts`` (called every ``run_experiment``
        # validation) needs ``contracts/technique_contract.json`` even in
        # validation mode (``build_runtime_config`` reads ``technique_params``
        # / ``level_value`` off it). The baseline is the technique the
        # harness builder validates against.
        baseline_entry = await _find_baseline_entry(ctx.metadata_store, cg_id)
        if baseline_entry is None:
            return DispatchOutcome(
                submitted_handles=[],
                error=(
                    f"No baseline entry (is_baseline=True) found for CG {cg_id!r}; "
                    "every CG materialized via the planner / orchestrator should have "
                    "exactly one — the harness builder needs its technique_contract.json "
                    "staged for the runtime's load_contracts to succeed in validation mode."
                ),
            )
        baseline_contract_key = technique_contract_path_fn(cg_id, baseline_entry.id)
        try:
            raw_t = await ctx.artifact_store.get(baseline_contract_key)
        except ArtifactNotFound:
            return DispatchOutcome(
                submitted_handles=[],
                error=(
                    f"baseline TechniqueContract not found at ArtifactStore key "
                    f"{baseline_contract_key!r}; the CG-registration transaction must "
                    "have written every entry's technique_contract.json before the "
                    "implementing state was entered."
                ),
            )
        try:
            baseline_technique_contract = TechniqueContract.model_validate_json(raw_t)
        except ValueError as exc:
            return DispatchOutcome(
                submitted_handles=[],
                error=(
                    f"baseline TechniqueContract at {baseline_contract_key!r} "
                    f"failed Pydantic validation: {exc}"
                ),
            )

        manifest_key = manifest_path_fn(cg_id)
        status_key = status_path_fn(cg_id)
        nudge_key = nudge_path_fn(cg_id)

        # Per-CG workspace under the configured root.
        workspace_path = workspace_root / cg_id

        # Resolve the per-task LlmProvider. The ctx.llm seam carries a
        # single provider for simple deployments; per-task swap is the
        # dispatch handler's territory in deployments where the
        # ``EngineConfig`` carries a model dict. C1 ships single-provider.
        llm = ctx.llm
        if llm is None:
            return DispatchOutcome(
                submitted_handles=[],
                error=(
                    "DispatchContext.llm is None; harness builder requires "
                    "an LlmProvider. The deployment must wire one through "
                    "the engine config or through a per-role LlmProvider map "
                    "the dispatch handler resolves."
                ),
            )

        # === Run the harness-builder agent loop in-process =================
        # Translate :class:`EngineConfig` supervisor settings (Task 3.G4)
        # into the loop's :class:`AgentLoopConfig`. The representative
        # ``ctx.llm`` provider doubles as the supervisor LLM in the
        # single-LLM-per-context shape. Some unit tests leave
        # ``ctx.config = None``; treat that as "supervisor off".
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

        # Open the ``agent_sessions`` telemetry row so the run is
        # inspectable while it executes (DEC-033 #2). Round-6 item D:
        # close it on EVERY exit path, including a raised session.
        session_id = await open_agent_session(
            ctx.metadata_store,
            parent_kind="cg",
            parent_id=cg_id,
            agent_role="harness_builder",
            llm=llm,
        )
        outcome: AgentOutcome | None = None
        try:
            outcome = await run_harness_builder_session(
                workspace_path=workspace_path,
                harness_contract=harness_contract,
                baseline_technique_contract=baseline_technique_contract,
                cg_id=cg_id,
                llm=llm,
                artifact_store=ctx.artifact_store,
                compute=ctx.compute,
                manifest_artifact_path=manifest_key,
                status_artifact_path=status_key,
                nudge_artifact_path=nudge_key,
                runtime_image=runtime_image,
                runtime_cpu_image=runtime_cpu_image,
                runner=inline_runner,
                supervisor_llm=llm if supervisor_on else None,
                config=loop_config,
                progress_sink=make_progress_sink(ctx.metadata_store, session_id),
            )
        finally:
            await close_agent_session(ctx.metadata_store, session_id, outcome)

        # Publish the agent's workspace outputs to ArtifactStore so the
        # orchestrator's ``implementing`` gates (and the downstream
        # technique-implementer dispatch) can read them — the agent ran
        # on the worker's local disk, not in a container that pushed to
        # the store. Keyed under the harness namespace (the manifest
        # key's parent) so the layout matches what
        # ``_load_harness_outputs`` + the code reviewer already expect.
        harness_key_prefix = manifest_key.rsplit("/", 1)[0]
        await publish_workspace_outputs(
            artifact_store=ctx.artifact_store,
            workspace_path=workspace_path,
            key_prefix=harness_key_prefix,
            roots=["harness", "techniques", "validation_results.json"],
        )

        # Completeness check (mirrors the planner's ``buffer.finalized``
        # check): the harness builder succeeds iff it emitted the
        # manifest — the ``emit_harness_manifest`` tool only writes it
        # after a passing in-workspace validation run (manifest_tool §9
        # step 5), so manifest-presence IS the harness-build-succeeded
        # signal. Absent => the in-process agent did not complete its
        # contract; surface a dispatch error so the engine's
        # ``_handle_dispatch_failure`` + the ``implementing`` state's
        # RetryPolicy retry or terminate it, rather than parking the CG
        # in ``implementing`` with no edge able to fire.
        if not await ctx.artifact_store.exists(manifest_key):
            return DispatchOutcome(
                submitted_handles=[],
                error=(
                    f"harness builder produced no manifest at {manifest_key!r} "
                    f"(agent outcome: {outcome.kind if outcome is not None else 'unknown'})"
                ),
            )
        return DispatchOutcome(
            submitted_handles=[JobHandle(plugin="inline", handle=f"inline-{cg_id}")],
            error=None,
        )

    return _dispatch


async def _find_baseline_entry(metadata_store: Any, cg_id: str) -> Any:
    """Return the ``is_baseline=True`` :class:`EntryRecord` for ``cg_id`` (or
    ``None`` if no such entry exists).

    The runtime's ``load_contracts`` needs the baseline's
    ``technique_contract.json`` staged at ``contracts/technique_contract.json``
    so the harness builder's in-loop ``run_experiment`` validation can read
    its ``technique_params`` / ``level_value``. Every CG materialized via
    the planner / orchestrator carries exactly one baseline entry by
    construction; returning ``None`` here surfaces the corruption rather
    than failing later inside the runtime.

    Drains all pages of :meth:`MetadataStore.list_entries_for_cg` —
    matches the orchestrator's
    :func:`smai_orchestrator.specs.cg_execution._list_all_entries_for_cg`
    drain pattern.
    """
    cursor: str | None = None
    while True:
        page = await metadata_store.list_entries_for_cg(cg_id, limit=100, cursor=cursor)
        for entry in page.items:
            if entry.is_baseline:
                return entry
        if page.next_cursor is None:
            return None
        cursor = page.next_cursor


__all__ = [
    "DEFAULT_HARNESS_CONTRACT_KEY_TEMPLATE",
    "DEFAULT_HARNESS_MANIFEST_KEY_TEMPLATE",
    "DEFAULT_HARNESS_NUDGE_KEY_TEMPLATE",
    "DEFAULT_HARNESS_STATUS_KEY_TEMPLATE",
    "DEFAULT_HARNESS_TRACE_KEY_TEMPLATE",
    "DEFAULT_TECHNIQUE_CONTRACT_KEY_TEMPLATE",
    "WORKSPACE_HARNESS_CONTRACT_PATH",
    "WORKSPACE_TECHNIQUE_CONTRACT_PATH",
    "SessionRunner",
    "make_dispatch_harness_build",
    "run_harness_builder_session",
]
