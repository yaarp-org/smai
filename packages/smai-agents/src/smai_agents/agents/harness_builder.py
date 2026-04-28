"""Harness builder dispatch handler — ``04-agents.md`` §2.2.

Two surfaces:

* :func:`run_harness_builder_session` — the in-process agent runner.
  Materializes the workspace skeleton + ``harness_contract.json``,
  builds the :class:`AgentSession` with the harness-builder prompt
  config, the standard tool inventory, and the harness-builder-only
  ``emit_harness_manifest`` tool, then drives the loop via
  :func:`run_loop`. Tests exercise this path directly per the Task 2.B3
  brief: "your tests can drive the agent loop in-process for speed."
  In production it is the entry point a containerized job calls after
  :func:`dispatch_harness_build` submits it.

* :func:`dispatch_harness_build` — the
  :data:`smai_orchestrator.engine.types.DispatchHandler`-shaped wrapper
  the engine dispatches when a CG enters the ``implementing`` state per
  ``03-state-machine.md`` §3. Reads the :class:`HarnessContract` from
  :class:`ArtifactStore` (path resolved via the configurable
  ``harness_contract_path_for_cg`` callable on the factory), populates
  the workspace, and returns a :class:`DispatchOutcome` whose
  ``submitted_handles`` carries the :class:`JobHandle` from
  :meth:`Compute.submit`. Phase-1 polling per
  ``05-orchestrator.md`` §3.1 detects completion and fires the matching
  ``fires_on=job_succeeded`` / ``job_failed`` edge per
  Task 2.C4's CG-execution pipeline-spec.

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

from smai_core import HarnessContract
from smai_core.plugins import (
    ArtifactNotFound,
    ArtifactStore,
    Compute,
    JobHandle,
    LlmProvider,
)
from smai_runtime import (
    HARNESS_CONTRACT_FILENAME,
    create_workspace_skeleton,
    write_template_files,
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
from smai_agents.std_tools import build_standard_tool_registry
from smai_agents.tools import ToolRegistry

_HARNESS_BUILDER_ROLE: TaskRole = "harness_builder"

# Default substrate image for the agent-side container. Production
# deployments override at registration time; v1 settled on
# ``smai-agent:dev`` per Task 2.A4 / DEC-026.
DEFAULT_AGENT_IMAGE = "smai-agent:dev"

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
    cg_id: str,
    llm: LlmProvider,
    artifact_store: ArtifactStore,
    compute: Compute,
    manifest_artifact_path: str,
    status_artifact_path: str | None = None,
    nudge_artifact_path: str | None = None,
    run_experiment_image: str | None = None,
    run_experiment_gpu: bool = True,
    prompt_config: PromptConfig | None = None,
    config: AgentLoopConfig | None = None,
    runner: SessionRunner | None = None,
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
        run_experiment_image: Container image for ``run_experiment``
            validation jobs. Threaded through to
            :func:`build_standard_tool_registry`. Defaults to
            :data:`smai_agents.std_tools.DEFAULT_RUN_EXPERIMENT_IMAGE`.
        run_experiment_gpu: GPU flag for validation jobs.
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

    # === Prompt config + tool registry ======================================
    cfg = (
        prompt_config
        if prompt_config is not None
        else load_prompt_config(role=_HARNESS_BUILDER_ROLE)
    )

    tools: ToolRegistry = build_standard_tool_registry(
        run_experiment_image=run_experiment_image
        if run_experiment_image is not None
        else "smai-runtime:dev",
        run_experiment_gpu=run_experiment_gpu,
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

    session = AgentSession(
        system_prompt=cfg.system_prompt,
        messages=[
            NormalizedMessage(
                role="user",
                content=[TextContent(text=initial_user)],
            )
        ],
        tools=tools,
        llm_providers={_HARNESS_BUILDER_ROLE: llm},
        current_role=_HARNESS_BUILDER_ROLE,
        workspace_path=workspace_path,
        artifact_store=artifact_store,
        compute=compute,
        status_artifact_path=status_artifact_path,
        nudge_artifact_path=nudge_artifact_path,
        config=config or AgentLoopConfig(),
    )

    actual_runner = runner if runner is not None else run_loop
    return await actual_runner(session)


# === Orchestrator dispatch handler ==========================================


def make_dispatch_harness_build(
    *,
    workspace_root: Path,
    agent_image: str = DEFAULT_AGENT_IMAGE,
    run_experiment_image: str | None = None,
    run_experiment_gpu: bool = True,
    harness_contract_artifact_path: Callable[[str], str] | None = None,
    harness_manifest_artifact_path: Callable[[str], str] | None = None,
    status_artifact_path: Callable[[str], str] | None = None,
    nudge_artifact_path: Callable[[str], str] | None = None,
    inline_runner: SessionRunner | None = None,
) -> Callable[[Any], Awaitable[Any]]:
    """Build a :data:`DispatchHandler` for the harness-builder role (§2.2).

    Production deployments construct one of these per pipeline-spec
    instance and bind it on the ``implementing`` state's
    :class:`smai_orchestrator.engine.types.DispatchAction.handler` slot.
    The factory shape lets deployments parameterize the substrate-side
    image, the per-CG path conventions, and the test-only inline-runner
    seam.

    Args:
        workspace_root: Filesystem root under which per-CG workspaces
            land. The handler creates ``<workspace_root>/<cg_id>/``
            before calling :func:`Compute.submit`.
        agent_image: Container image for the agent-side job. Defaults
            to :data:`DEFAULT_AGENT_IMAGE`.
        run_experiment_image: Container image for ``run_experiment``
            validation jobs *inside* the agent's tool surface. Threaded
            through to :func:`run_harness_builder_session`.
        run_experiment_gpu: GPU flag for the validation jobs.
        harness_contract_artifact_path: Callable resolving
            ``cg_id → ArtifactStore key`` for the loaded contract.
            Defaults to
            :data:`DEFAULT_HARNESS_CONTRACT_KEY_TEMPLATE`.
        harness_manifest_artifact_path: Callable resolving
            ``cg_id → ArtifactStore key`` for the manifest write.
        status_artifact_path: Callable resolving the per-job status JSON
            key.
        nudge_artifact_path: Callable resolving the supervisor-nudge key.
        inline_runner: Test-only seam — when non-``None``, the dispatch
            handler runs the agent loop in-process via this runner
            instead of submitting a Compute job. Returns a synthetic
            :class:`JobHandle` whose ``handle`` is ``"inline-<cg_id>"``
            so the orchestrator's phase-1 polling sees a stable id; the
            test asserts on workspace post-state and on
            :func:`Compute.submit` having NOT been called.

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

    contract_path_fn = harness_contract_artifact_path or _default_contract_path
    manifest_path_fn = harness_manifest_artifact_path or _default_manifest_path
    status_path_fn = status_artifact_path or _default_status_path
    nudge_path_fn = nudge_artifact_path or _default_nudge_path

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

        # Test-only inline runner — runs the agent loop in-process and
        # synthesizes a JobHandle. Production leaves this None and the
        # branch below submits via Compute.
        if inline_runner is not None:
            await run_harness_builder_session(
                workspace_path=workspace_path,
                harness_contract=harness_contract,
                cg_id=cg_id,
                llm=llm,
                artifact_store=ctx.artifact_store,
                compute=ctx.compute,
                manifest_artifact_path=manifest_key,
                status_artifact_path=status_key,
                nudge_artifact_path=nudge_key,
                run_experiment_image=run_experiment_image,
                run_experiment_gpu=run_experiment_gpu,
                runner=inline_runner,
            )
            return DispatchOutcome(
                submitted_handles=[
                    JobHandle(plugin="inline", handle=f"inline-{cg_id}"),
                ],
                error=None,
            )

        # Production path: lay down the workspace + contract for the
        # agent-side container, then submit. The container's entrypoint
        # invokes :func:`run_harness_builder_session` against the
        # workspace; the entrypoint module wiring is the deployment's
        # concern (the agent.Dockerfile per Task 2.A4 ships the smai-agents
        # CLI that calls into this function).
        create_workspace_skeleton(workspace_path)
        write_template_files(workspace_path)
        contract_path = workspace_path / WORKSPACE_HARNESS_CONTRACT_PATH
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        contract_path.write_text(harness_contract.model_dump_json(indent=2))

        command = [
            "python",
            "-m",
            "smai_agents.agents.harness_builder",
            "--workspace",
            str(workspace_path),
            "--cg-id",
            cg_id,
            "--manifest-artifact-path",
            manifest_key,
        ]
        env = {
            "SMAI_CG_ID": cg_id,
            "SMAI_HARNESS_MANIFEST_KEY": manifest_key,
            "SMAI_HARNESS_STATUS_KEY": status_key,
            "SMAI_HARNESS_NUDGE_KEY": nudge_key,
        }
        try:
            handle = await ctx.compute.submit(
                image=agent_image,
                command=command,
                env=env,
                gpu=False,
            )
        except Exception as exc:  # noqa: BLE001 — engine surfaces every error as DispatchOutcome.error
            return DispatchOutcome(
                submitted_handles=[],
                error=f"{type(exc).__name__}: {exc}",
            )
        return DispatchOutcome(submitted_handles=[handle], error=None)

    return _dispatch


__all__ = [
    "DEFAULT_AGENT_IMAGE",
    "DEFAULT_HARNESS_CONTRACT_KEY_TEMPLATE",
    "DEFAULT_HARNESS_MANIFEST_KEY_TEMPLATE",
    "DEFAULT_HARNESS_NUDGE_KEY_TEMPLATE",
    "DEFAULT_HARNESS_STATUS_KEY_TEMPLATE",
    "WORKSPACE_HARNESS_CONTRACT_PATH",
    "SessionRunner",
    "make_dispatch_harness_build",
    "run_harness_builder_session",
]
