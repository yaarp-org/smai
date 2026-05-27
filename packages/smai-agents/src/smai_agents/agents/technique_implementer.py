"""Technique implementer dispatch handler — ``04-agents.md`` §2.3.

Step 7 of the agent-layer refactor (2026-05-27) ports this role onto
the sandboxed mini-orchestrator pattern that Step 4 + sub-PR F landed
for harness_builder. The host-side surface is a thin wrapper around
:func:`smai_orchestrator.dispatch.make_compute_dispatcher`. The
materialize-workspace + run-agent-loop + publish-outputs sequence
that the round-14-vintage in-process predecessor handled is now split:
the host materializes the per-entry workspace (parent harness contract
+ manifest + harness/ files + baseline source + per-entry technique
contract + grounding); the sandbox (``smai-agent-runtime`` container)
runs PydanticAI Agent calls + lint + validation + diagnose; the host
harvests + publishes the agent's outputs on terminal exit.

Per :doc:`agent_refactor/architectural_decisions.md` §7 ("Sandbox
produces, host attests-and-persists"): the host writes to
:class:`ArtifactStore` using host-side credentials; the sandbox never
has ArtifactStore write access.

Per DEC-013 / DEC-017: additive baselines (entries with
``technique_id is None`` and ``is_baseline is True``) skip dispatch
entirely — the handler returns a no-op
:class:`DispatchOutcome`. The orchestrator's composite ``implementing``
gate marks them ``implemented`` directly without burning compute.

History note: through sub-PR-E (sub-PR-D for harness_builder) this
module hosted a parallel in-process scaffolding (~700 LoC of session
runner + tool registry + prompt loading + the run-loop integration).
Step 7 deletes that scaffolding and flips production dispatch onto
the sandboxed factory below. The old ``inline_runner`` test seam is
gone; tests that exercise technique-implementer dispatch now mock at
the :class:`Compute` boundary (the ``RecordingCompute`` pattern from
harness_builder's test fixtures).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from smai_core import HarnessContract, TechniqueContract
from smai_core.plugins import ArtifactNotFound, ArtifactStore, LlmProvider
from smai_runtime import (
    HARNESS_API_MANIFEST_FILENAME,
    HARNESS_CONTRACT_FILENAME,
    TECHNIQUE_CONTRACT_FILENAME,
    is_additive_baseline,
)

from smai_agents.agent_session_telemetry import open_agent_session
from smai_agents.agents.artifact_publish import publish_workspace_outputs

# Default ArtifactStore key conventions — same per-CG layout as the
# harness builder. Per-entry paths use ``entries/{entry_id}/...`` per
# v1's `comparison-groups/{cg_id}/entries/{entry_id}/...` layout.
DEFAULT_TECHNIQUE_CONTRACT_KEY_TEMPLATE = (
    "comparison-groups/{cg_id}/entries/{entry_id}/technique_contract.json"
)
DEFAULT_HARNESS_CONTRACT_KEY_TEMPLATE = "comparison-groups/{cg_id}/harness/contract.json"
DEFAULT_HARNESS_MANIFEST_KEY_TEMPLATE = "comparison-groups/{cg_id}/harness/manifest.json"
# ArtifactStore namespace the sandboxed technique implementer's
# post-terminal handler publishes its workspace outputs under — the
# per-entry ``code/`` prefix the CG-entries spec's validation gate +
# the code reviewer read (mirrors ``TECHNIQUE_CODE_KEY_TEMPLATE`` /
# ``TECHNIQUE_VALIDATION_KEY_TEMPLATE``).
DEFAULT_ENTRY_CODE_PREFIX_TEMPLATE = "comparison-groups/{cg_id}/entries/{entry_id}/code"
DEFAULT_ENTRY_VALIDATION_KEY_TEMPLATE = (
    "comparison-groups/{cg_id}/entries/{entry_id}/code/validation_results.json"
)
# Optional staged grounding-context file (DEC-017): orchestrator writes
# the per-entry grounding JSON if available (paper extraction, proposal
# text, reviewer-attested spec). When absent the mini-orchestrator falls
# back to a standard-library grounding from the technique_id.
DEFAULT_TECHNIQUE_GROUNDING_KEY_TEMPLATE = (
    "comparison-groups/{cg_id}/entries/{entry_id}/grounding/technique_grounding.json"
)

# Default agent-runtime image. Mirrors
# :data:`smai_orchestrator.engine.config.DEFAULT_AGENT_RUNTIME_IMAGE`;
# duplicated as a literal here to avoid an import-back-edge through
# the orchestrator package (this module sits in smai-agents which
# smai-orchestrator already depends on transitively).
_DEFAULT_AGENT_RUNTIME_IMAGE = "smai-agent-runtime:dev"

# Workspace-relative paths the host-side stager writes to.
_WORKSPACE_HARNESS_CONTRACT_PATH = "contracts/" + HARNESS_CONTRACT_FILENAME
_WORKSPACE_TECHNIQUE_CONTRACT_PATH = "contracts/" + TECHNIQUE_CONTRACT_FILENAME
_WORKSPACE_HARNESS_MANIFEST_PATH = HARNESS_API_MANIFEST_FILENAME
_WORKSPACE_BASELINE_SOURCE_PATH = "techniques/baseline.py"
_WORKSPACE_TECHNIQUE_GROUNDING_PATH = "grounding/technique_grounding.json"


def make_dispatch_technique_implementation_sandboxed(
    *,
    workspace_root: Path,
    agent_runtime_image: str | None = None,
    technique_contract_artifact_path: Callable[[str, str], str] | None = None,
    harness_contract_artifact_path: Callable[[str], str] | None = None,
    harness_manifest_artifact_path: Callable[[str], str] | None = None,
    technique_grounding_artifact_path: Callable[[str, str], str] | None = None,
    entry_code_prefix: Callable[[str, str], str] | None = None,
    retry_policy: Any | None = None,
    extra_env: Mapping[str, str] | None = None,
    llm_for_credentials: LlmProvider | None = None,
) -> _SandboxedTechniqueImplementerBundle:
    """Build the sandboxed technique-implementer dispatcher bundle (Step 7).

    The Step-7 host-side surface mirrors
    :func:`smai_agents.agents.harness_builder.make_dispatch_harness_build_sandboxed`'s
    shape:

    * :attr:`handler` — pre-filters additive baselines (DEC-013 / DEC-017:
      those skip agent dispatch), then materializes the per-entry
      workspace (parent harness contract + harness API manifest +
      harness/ files + baseline source + per-entry technique contract +
      optional grounding) and submits the ``smai-agent-runtime``
      container running ``python -m smai_agent_runtime --role
      technique_implementer --entry-id <id>`` via
      :func:`smai_orchestrator.dispatch.make_compute_dispatcher`.
    * :attr:`post_terminal_handler` — calls
      :meth:`Compute.harvest_workspace` (no-op under bind-mount, volume
      read-back under upload-download) and publishes the sandbox's
      ``techniques/`` + ``validation_results.json`` +
      ``conversation_traces/`` + ``status/`` outputs to
      :class:`ArtifactStore` under the per-entry ``code/`` prefix.

    Args:
        workspace_root: Filesystem root under which per-entry agent
            workspaces land (one subdir per entry_id).
        agent_runtime_image: Optional override for the agent-runtime
            container image. Defaults to ``smai-agent-runtime:dev``.
        technique_contract_artifact_path,
        harness_contract_artifact_path,
        harness_manifest_artifact_path,
        technique_grounding_artifact_path,
        entry_code_prefix: Optional overrides for the ArtifactStore
            key conventions; all default to the
            ``comparison-groups/{cg_id}/...`` layout v1 ships.
        retry_policy: Round-10 declarative :class:`RetryPolicy` for the
            entry-level dispatcher. Propagated verbatim onto the
            :class:`DispatchAction` the caller wires this handler into.
        extra_env: Optional env-var overrides projected into the
            sandbox container's environment (D3 per-role-per-step
            model selection via ``SMAI_MODEL_TECHNIQUE_IMPLEMENTER...``).
        llm_for_credentials: Sub-PR F mechanism: when set, the
            dispatcher calls
            :meth:`LlmProvider.credentials_for_subprocess` per-dispatch
            and merges the result into the container env so the
            sandbox's PydanticAI / provider SDK calls authenticate.
            Optional only so fake-LLM tests can skip it; production
            deployments MUST set it via :class:`smai_cli.runtime.Runtime`
            wiring.

    Returns:
        A :class:`_SandboxedTechniqueImplementerBundle` exposing
        ``.handler`` + ``.post_terminal_handler`` for the spec author
        to wire onto :class:`DispatchAction`.
    """
    from smai_orchestrator.dispatch import (  # noqa: PLC0415
        CommandSpec,
        WorkspaceInputs,
        WorkspaceOutputs,
        make_compute_dispatcher,
    )

    def _default_technique_contract_path(cg_id: str, entry_id: str) -> str:
        return DEFAULT_TECHNIQUE_CONTRACT_KEY_TEMPLATE.format(cg_id=cg_id, entry_id=entry_id)

    def _default_harness_contract_path(cg_id: str) -> str:
        return DEFAULT_HARNESS_CONTRACT_KEY_TEMPLATE.format(cg_id=cg_id)

    def _default_manifest_path(cg_id: str) -> str:
        return DEFAULT_HARNESS_MANIFEST_KEY_TEMPLATE.format(cg_id=cg_id)

    def _default_grounding_path(cg_id: str, entry_id: str) -> str:
        return DEFAULT_TECHNIQUE_GROUNDING_KEY_TEMPLATE.format(cg_id=cg_id, entry_id=entry_id)

    def _default_code_prefix(cg_id: str, entry_id: str) -> str:
        return DEFAULT_ENTRY_CODE_PREFIX_TEMPLATE.format(cg_id=cg_id, entry_id=entry_id)

    technique_contract_path_fn = (
        technique_contract_artifact_path or _default_technique_contract_path
    )
    harness_contract_path_fn = harness_contract_artifact_path or _default_harness_contract_path
    manifest_path_fn = harness_manifest_artifact_path or _default_manifest_path
    grounding_path_fn = technique_grounding_artifact_path or _default_grounding_path
    code_prefix_fn = entry_code_prefix or _default_code_prefix
    image_name = agent_runtime_image or _DEFAULT_AGENT_RUNTIME_IMAGE

    # Per-dispatch shared state. The image_resolver runs first; it
    # records (cg_id, entry_id, is_additive_baseline) so the
    # command_builder + workspace_input_resolver + post_terminal_handler
    # downstream get consistent values without re-reading the contract.
    # Keyed by entry_id since the entity dispatched here is the entry.
    _dispatch_state: dict[str, _DispatchState] = {}

    async def _resolve_image(ctx: Any) -> str:
        """Image resolver doubles as the workspace-materialization site.

        Mirrors the harness_builder pattern: the engine invokes
        ``image_resolver`` exactly once per dispatch and runs it
        BEFORE ``inputs.resolver`` (which calls
        :meth:`Compute.stage_workspace`), so writing the contract +
        harness + baseline files here guarantees they land on disk
        before the substrate-side staging reads them. The
        :class:`ArtifactNotFound` raises during materialization
        propagate to the engine's ``_handle_dispatch_failure`` which
        the round-10 :class:`RetryPolicy` bounds.
        """
        entry_id = ctx.entity_id
        entry_record = await ctx.metadata_store.get_entry(entry_id)
        if entry_record is None:
            raise LookupError(f"EntryRecord {entry_id!r} not found in MetadataStore")
        cg_id = entry_record.cg_id

        contract_key = technique_contract_path_fn(cg_id, entry_id)
        raw_t = await ctx.artifact_store.get(contract_key)
        technique_contract = TechniqueContract.model_validate_json(raw_t)
        skip = is_additive_baseline(technique_contract)
        workspace_path = workspace_root / entry_id
        _dispatch_state[entry_id] = _DispatchState(
            cg_id=cg_id,
            entry_id=entry_id,
            workspace_path=workspace_path,
            skip_dispatch=skip,
        )
        if skip:
            return image_name

        # Stage workspace files. The harness builder's outputs (parent
        # harness contract + manifest + harness/ + techniques/baseline.py)
        # were published to ArtifactStore when its dispatch completed;
        # we re-materialize them per-entry so each technique's sandbox
        # starts from a known good state.
        await _materialize_technique_implementer_workspace(
            workspace_path=workspace_path,
            cg_id=cg_id,
            entry_id=entry_id,
            artifact_store=ctx.artifact_store,
            technique_contract=technique_contract,
            harness_contract_key=harness_contract_path_fn(cg_id),
            manifest_key=manifest_path_fn(cg_id),
            grounding_key=grounding_path_fn(cg_id, entry_id),
        )
        return image_name

    async def _build_command(ctx: Any) -> Any:
        entry_id = ctx.entity_id
        state = _dispatch_state.get(entry_id)
        # No state means image_resolver raised before this point; the
        # engine never invokes the rest of the pipeline. Defensive
        # placeholder command so pyright sees an assignment regardless.
        command = [
            "python",
            "-m",
            "smai_agent_runtime",
            "--role",
            "technique_implementer",
            "--entry-id",
            entry_id,
            "--workspace",
            "/workspace",
        ]
        env: dict[str, str] = {"SMAI_ENTRY_ID": entry_id}
        if state is not None:
            env["SMAI_CG_ID"] = state.cg_id
        # Sub-PR F mechanism: per-dispatch credential resolution. Same
        # treatment as harness_builder. extra_env (model selection)
        # merges last so explicit overrides win.
        if llm_for_credentials is not None:
            env.update(await llm_for_credentials.credentials_for_subprocess())
        if extra_env:
            env.update(extra_env)
        return CommandSpec(
            command=command,
            env=env,
            gpu=False,
            timeout_seconds=3600,
        )

    async def _resolve_workspace_input(ctx: Any) -> Path:
        return workspace_root / ctx.entity_id

    async def _resolve_workspace_output(ctx: Any) -> Path:
        return workspace_root / ctx.entity_id

    inner_bundle = make_compute_dispatcher(
        role="technique_implementer",
        image_resolver=_resolve_image,
        command_builder=_build_command,
        inputs=WorkspaceInputs(resolver=_resolve_workspace_input),
        outputs=WorkspaceOutputs(destination=_resolve_workspace_output),
        retry_policy=retry_policy,
    )

    inner_handler = inner_bundle.handler
    inner_post_terminal = inner_bundle.post_terminal_handler

    async def _handler_with_skip_and_session_open(ctx: Any) -> Any:
        """Wrap the inner handler with the additive-baseline skip + session open.

        Skip semantics: per DEC-013 / DEC-017 the additive-baseline
        case returns a no-op :class:`DispatchOutcome` and does NOT
        submit a Compute job. The engine's gate-rule machinery is
        expected to mark such entries ``implemented`` directly. To
        observe the skip the state machine writes the no-submit
        outcome (zero handles, no error) and the orchestrator's
        composite ``implementing`` gate sees the marker.
        """
        from smai_orchestrator.engine.types import DispatchOutcome  # noqa: PLC0415

        outcome = await inner_handler(ctx)
        # The image resolver populated _dispatch_state; if the skip
        # flag is set we override the outcome to "no dispatch, no
        # error" so the engine's no-survivors path runs.
        state = _dispatch_state.get(ctx.entity_id)
        if state is not None and state.skip_dispatch:
            return DispatchOutcome(submitted_handles=[], error=None)

        if outcome.error is None and outcome.submitted_handles:
            await open_agent_session(
                ctx.metadata_store,
                parent_kind="entry",
                parent_id=ctx.entity_id,
                agent_role="technique_implementer",
                llm=ctx.llm,
                compute_job_handle=outcome.submitted_handles[0],
            )
        return outcome

    async def _post_terminal_with_publish(ctx: Any) -> None:
        if inner_post_terminal is not None:
            await inner_post_terminal(ctx)
        entry_id = ctx.entity_id
        state = _dispatch_state.get(entry_id)
        # Additive-baseline skips wrote nothing to disk; nothing to
        # publish. The gate machinery already handles the no-survivors
        # path.
        if state is None or state.skip_dispatch:
            return
        workspace_path = state.workspace_path
        if not workspace_path.exists():
            return
        await publish_workspace_outputs(
            artifact_store=ctx.artifact_store,
            workspace_path=workspace_path,
            key_prefix=code_prefix_fn(state.cg_id, entry_id),
            roots=[
                "techniques",
                "validation_results.json",
                "conversation_traces",
                "status",
            ],
        )

    return _SandboxedTechniqueImplementerBundle(
        handler=_handler_with_skip_and_session_open,
        post_terminal_handler=_post_terminal_with_publish,
    )


class _DispatchState:
    """Per-dispatch state carried across the image-resolver →
    command-builder → post-terminal sequence."""

    __slots__ = ("cg_id", "entry_id", "skip_dispatch", "workspace_path")

    def __init__(
        self,
        *,
        cg_id: str,
        entry_id: str,
        workspace_path: Path,
        skip_dispatch: bool,
    ) -> None:
        self.cg_id = cg_id
        self.entry_id = entry_id
        self.workspace_path = workspace_path
        self.skip_dispatch = skip_dispatch


class _SandboxedTechniqueImplementerBundle:
    """Mirror of :class:`smai_orchestrator.dispatch.DispatcherBundle`.

    Constructed here so callers need not import the orchestrator-side
    bundle type. Field shape identical to
    :class:`_SandboxedHarnessBuilderBundle` per the cross-role
    consistency the Step-7 cutover lands.
    """

    __slots__ = ("handler", "post_terminal_handler")

    def __init__(
        self,
        *,
        handler: Callable[[Any], Awaitable[Any]],
        post_terminal_handler: Callable[[Any], Awaitable[None]] | None,
    ) -> None:
        self.handler = handler
        self.post_terminal_handler = post_terminal_handler


async def _materialize_technique_implementer_workspace(  # noqa: PLR0913
    *,
    workspace_path: Path,
    cg_id: str,
    entry_id: str,
    artifact_store: ArtifactStore,
    technique_contract: TechniqueContract,
    harness_contract_key: str,
    manifest_key: str,
    grounding_key: str,
) -> None:
    """Stage the technique-implementer sandbox's workspace contents.

    Writes the per-entry technique contract, the parent harness
    contract + manifest, the harness sources (``harness/`` files), the
    baseline source (``techniques/baseline.py``), and the optional
    grounding-context JSON into ``workspace_path`` so
    :meth:`Compute.stage_workspace` reads from a ready-to-run
    workspace.

    Raises :class:`smai_core.plugins.ArtifactNotFound` if the parent
    harness contract or manifest is missing — these are
    pre-requisites the harness builder's prior dispatch must have
    written. The optional grounding file is best-effort (absent →
    sandbox falls back to a standard-library grounding).
    """
    del cg_id, entry_id  # unused; included for future per-key parameterization

    workspace_path.mkdir(parents=True, exist_ok=True)

    # Per-entry technique contract — primary artifact for this role.
    technique_path = workspace_path / _WORKSPACE_TECHNIQUE_CONTRACT_PATH
    technique_path.parent.mkdir(parents=True, exist_ok=True)
    technique_path.write_text(technique_contract.model_dump_json(indent=2))

    # Parent harness contract — grounding for factor + extension points.
    raw_h = await artifact_store.get(harness_contract_key)
    harness_contract = HarnessContract.model_validate_json(raw_h)
    harness_path = workspace_path / _WORKSPACE_HARNESS_CONTRACT_PATH
    harness_path.parent.mkdir(parents=True, exist_ok=True)
    harness_path.write_text(harness_contract.model_dump_json(indent=2))

    # Harness API manifest — declares the extension points the
    # technique implementation must populate.
    raw_m = await artifact_store.get(manifest_key)
    manifest_path = workspace_path / _WORKSPACE_HARNESS_MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(raw_m)

    # Harness sources — the agent reads these to learn the call shape
    # the technique must satisfy.
    await _materialize_harness_sources(
        artifact_store=artifact_store,
        cg_id=harness_contract.envelope.parent_experiment_id,
        workspace_path=workspace_path,
    )

    # Baseline source — reference shape for the technique module.
    # Absent for substitutive-factor experiments without an explicit
    # baseline; the sandbox treats absence as "no baseline reference".
    cg_id_resolved = harness_contract.envelope.parent_experiment_id
    baseline_key = f"comparison-groups/{cg_id_resolved}/harness/techniques/baseline.py"
    try:
        raw_b = await artifact_store.get(baseline_key)
    except ArtifactNotFound:
        raw_b = None
    if raw_b is not None:
        baseline_path = workspace_path / _WORKSPACE_BASELINE_SOURCE_PATH
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_bytes(raw_b)

    # Optional grounding-context JSON. Best-effort: absent → sandbox
    # falls back to standard-library grounding.
    try:
        raw_g = await artifact_store.get(grounding_key)
    except ArtifactNotFound:
        raw_g = None
    if raw_g is not None:
        grounding_path = workspace_path / _WORKSPACE_TECHNIQUE_GROUNDING_PATH
        grounding_path.parent.mkdir(parents=True, exist_ok=True)
        grounding_path.write_bytes(raw_g)


async def _materialize_harness_sources(
    *,
    artifact_store: ArtifactStore,
    cg_id: str,
    workspace_path: Path,
) -> None:
    """Pull all ``harness/*.py`` files the harness builder published.

    The harness builder's post-terminal handler wrote files under
    ``comparison-groups/{cg_id}/harness/harness/`` (per the publish
    layer in :mod:`smai_agents.agents.artifact_publish`). Walk the
    prefix via ``ArtifactStore.list`` (streaming, plugin-defined order)
    and write each file to ``workspace_path/harness/<rel>``.
    """
    prefix = f"comparison-groups/{cg_id}/harness/harness/"
    iterator = await artifact_store.list(prefix)
    async for key in iterator:
        rel = key[len(prefix) :]
        if not rel:
            continue
        data = await artifact_store.get(key)
        local = workspace_path / "harness" / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)


__all__ = [
    "DEFAULT_ENTRY_CODE_PREFIX_TEMPLATE",
    "DEFAULT_ENTRY_VALIDATION_KEY_TEMPLATE",
    "DEFAULT_HARNESS_CONTRACT_KEY_TEMPLATE",
    "DEFAULT_HARNESS_MANIFEST_KEY_TEMPLATE",
    "DEFAULT_TECHNIQUE_CONTRACT_KEY_TEMPLATE",
    "DEFAULT_TECHNIQUE_GROUNDING_KEY_TEMPLATE",
    "make_dispatch_technique_implementation_sandboxed",
]
