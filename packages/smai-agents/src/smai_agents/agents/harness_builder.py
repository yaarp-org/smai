"""Sandboxed harness-builder dispatch handler — ``04-agents.md`` §2.2.

The host-side surface is a thin wrapper around
:func:`smai_orchestrator.dispatch.make_compute_dispatcher`. Per
:doc:`agent_refactor/architectural_decisions.md` §7
("Sandbox produces, host attests-and-persists"): the host materializes
the per-CG workspace contents (harness contract, baseline technique
contract, harness API reference, runtime templates), submits the
``smai-agent-runtime`` container running ``python -m smai_agent_runtime
--role harness_builder --cg-id <id>``, and on terminal success harvests
the workspace + publishes the agent's outputs (``harness/``,
``techniques/``, ``validation_results.json``, ``manifest.json``,
``conversation_traces/``) to :class:`ArtifactStore`.

History note: through sub-PR D this module hosted a parallel
``make_dispatch_harness_build`` factory carrying the round-14
in-process scaffolding (~700 LoC of session runner + tool registry +
prompt loading). Sub-PR E flipped the production dispatch entry in
:mod:`smai_orchestrator.specs.cg_execution` over to the sandboxed
factory below and deleted the in-process scaffolding. The
``inline_runner`` test seam is gone; tests that exercise harness
builder dispatch now mock at the :class:`Compute` boundary
(``RecordingCompute`` pattern in
``tests/_harness_builder_sandboxed_fixtures.py``).

Factor-type-aware framing (DEC-017 / §9 of ``10-runtime-and-templates.md``)
is the same: the harness builder always builds the same code shape;
what differs is whether the extension point has a working default
(``additive``) or is a mandatory slot (``substitutive``). The
factor-type signal lives in the staged ``harness_contract.json``; the
sandbox-side mini-orchestrator's per-step prompts pivot on it.

Per ``05-orchestrator.md`` §1.4: agents may not write to
:class:`MetadataStore`. The ``harness_api_manifest_hash`` per-entry
write per DEC-033 #3 is the orchestrator's gate-rule territory (the
``implementing → implemented`` gate body's fanout), not this handler's.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from smai_core import HarnessContract, TechniqueContract
from smai_core.plugins import ArtifactStore, LlmProvider
from smai_inline_agents.agent_session_telemetry import open_agent_session
from smai_runtime import (
    HARNESS_CONTRACT_FILENAME,
    TECHNIQUE_CONTRACT_FILENAME,
    create_workspace_skeleton,
    write_template_files,
)

from smai_agents.agents.artifact_publish import publish_workspace_outputs
from smai_agents.agents.harness_api_reference import (
    WORKSPACE_HARNESS_API_REFERENCE_PATH,
    build_harness_api_reference,
)

# Path inside the workspace where the materialized HarnessContract lives.
# Mirrors :data:`smai_runtime.HARNESS_CONTRACT_FILENAME` ("harness_contract.json")
# under ``contracts/``. Local-relative form lets the agent find it via
# ``read_file("contracts/harness_contract.json")``.
WORKSPACE_HARNESS_CONTRACT_PATH = "contracts/" + HARNESS_CONTRACT_FILENAME

# v1 path convention — mirrors ``CLAUDE.md`` /
# ``designs/yaarp/...`` Artifact Layout for ``comparison-groups/{cg_id}/``.
# Carrying the literal here as the default; deployments may override via
# ``make_dispatch_harness_build_sandboxed`` factory args (e.g., a
# hashed-content-addressed path for the Postgres-+-S3 production stack).
DEFAULT_HARNESS_CONTRACT_KEY_TEMPLATE = "comparison-groups/{cg_id}/harness/contract.json"
DEFAULT_HARNESS_MANIFEST_KEY_TEMPLATE = "comparison-groups/{cg_id}/harness/manifest.json"

# Per-entry technique-contract key — same template the technique implementer
# (and the orchestrator's cg_execution / proposal specs) use. Re-stated
# here as a literal rather than imported from ``smai_agents.agents.technique_implementer``
# because that module imports from this one historically; the literal
# avoids an import-back-edge cycle when the cycle is reintroduced.
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


# === Sandboxed dispatch handler (agent_refactor Step 4 sub-PR B) =============


def make_dispatch_harness_build_sandboxed(
    *,
    workspace_root: Path,
    agent_runtime_image: str | None = None,
    harness_contract_artifact_path: Callable[[str], str] | None = None,
    technique_contract_artifact_path: Callable[[str, str], str] | None = None,
    harness_publish_key_prefix: Callable[[str], str] | None = None,
    retry_policy: Any | None = None,
    extra_env: Mapping[str, str] | None = None,
    llm_for_credentials: LlmProvider | None = None,
) -> _SandboxedHarnessBuilderBundle:
    """Build the sandboxed harness-builder dispatcher bundle (sub-PR B).

    The new dispatch shape introduced by agent_refactor Step 4. Returns a
    :class:`_SandboxedHarnessBuilderBundle` (mirroring
    :class:`smai_orchestrator.dispatch.DispatcherBundle`) the spec author
    can wire onto :class:`DispatchAction.handler` /
    :attr:`DispatchAction.post_terminal_handler`:

    * :attr:`handler` — pre-materializes the per-CG workspace
      (harness contract, baseline technique contract, harness API
      reference, runtime templates) and submits the
      ``smai-agent-runtime`` container running
      ``python -m smai_agent_runtime --role harness_builder --cg-id <id>``
      via :func:`smai_orchestrator.dispatch.make_compute_dispatcher`. On
      successful submit also opens an ``agent_sessions`` telemetry row
      with the :class:`JobHandle` cross-reference (D2).
    * :attr:`post_terminal_handler` — invoked by phase-1 on terminal
      observation (sub-PR E phase1 reorder — fires BEFORE the
      gate evaluation so the success gate sees the published manifest).
      Calls :meth:`Compute.harvest_workspace` (no-op under bind-mount
      semantics, volume read-back under upload-download) and publishes
      the resulting workspace files (``harness/``, ``techniques/``,
      ``validation_results.json``, ``manifest.json``,
      ``conversation_traces/``) to :class:`ArtifactStore` via
      :func:`publish_workspace_outputs`.

    Sub-PR E cutover: this is the sole harness-builder dispatch path on
    main — the round-14 in-process predecessor (``run_harness_builder_session``
    + ``make_dispatch_harness_build``) was deleted alongside the cutover.

    Args:
        ...
        llm_for_credentials: Optional :class:`LlmProvider` whose
            :meth:`credentials_for_subprocess` is called per-dispatch to
            project provider credentials into the sandbox container env.
            Sub-PR F (round-21 dogfooding finding): without this the
            sandboxed PydanticAI / boto3 inside the container has no
            credentials and fails at client construction. Optional only
            so existing fake-LLM tests that don't need real provider
            auth can omit it; production deployments MUST set it via
            :func:`smai_cli.runtime.Runtime` wiring.
    """
    from smai_orchestrator.dispatch import (  # noqa: PLC0415
        CommandSpec,
        WorkspaceInputs,
        WorkspaceOutputs,
        make_compute_dispatcher,
    )

    def _default_contract_path(cg_id: str) -> str:
        return DEFAULT_HARNESS_CONTRACT_KEY_TEMPLATE.format(cg_id=cg_id)

    def _default_technique_contract_path(cg_id: str, entry_id: str) -> str:
        return DEFAULT_TECHNIQUE_CONTRACT_KEY_TEMPLATE.format(cg_id=cg_id, entry_id=entry_id)

    def _default_publish_prefix(cg_id: str) -> str:
        return DEFAULT_HARNESS_MANIFEST_KEY_TEMPLATE.format(cg_id=cg_id).rsplit("/", 1)[0]

    contract_path_fn = harness_contract_artifact_path or _default_contract_path
    technique_contract_path_fn = (
        technique_contract_artifact_path or _default_technique_contract_path
    )
    publish_prefix_fn = harness_publish_key_prefix or _default_publish_prefix
    image_name = agent_runtime_image or _DEFAULT_AGENT_RUNTIME_IMAGE

    async def _resolve_image(ctx: Any) -> str:
        # Stateless image lookup, but doubles as the per-dispatch
        # workspace-materialization site: the engine invokes
        # ``image_resolver`` exactly once per dispatch and runs it
        # BEFORE ``inputs.resolver`` (which calls
        # ``Compute.stage_workspace``), so writing the contract /
        # template files here guarantees they land on disk before the
        # substrate-side staging reads them. The engine has no
        # separate "before-stage" hook today, so piggybacking on
        # ``image_resolver`` is the right seam.
        cg_id = ctx.entity_id
        workspace_path = workspace_root / cg_id
        await _materialize_harness_builder_workspace(
            workspace_path=workspace_path,
            cg_id=cg_id,
            artifact_store=ctx.artifact_store,
            metadata_store=ctx.metadata_store,
            contract_key=contract_path_fn(cg_id),
            technique_contract_path_fn=technique_contract_path_fn,
        )
        return image_name

    async def _build_command(ctx: Any) -> Any:
        cg_id = ctx.entity_id
        command = [
            "python",
            "-m",
            "smai_agent_runtime",
            "--role",
            "harness_builder",
            "--cg-id",
            cg_id,
            "--workspace",
            "/workspace",
        ]
        env: dict[str, str] = {"SMAI_CG_ID": cg_id}
        # Sub-PR F (credential-flow gap): project the configured
        # :class:`LlmProvider`'s subprocess credentials into the sandbox
        # env so PydanticAI / the provider SDK inside the container can
        # authenticate. Called per-dispatch so chain-derived credentials
        # (boto3 SSO tokens, STS session tokens) refresh as they rotate
        # on the host. Per ``compute_dispatch_decisions.md`` §4: LLM
        # credentials are explicitly part of the sandbox's trust envelope.
        if llm_for_credentials is not None:
            env.update(await llm_for_credentials.credentials_for_subprocess())
        # D3 per-role-per-step model env projection (sub-PR D thread 2):
        # the host computes ``SMAI_MODEL_<ROLE>__<STEP>`` env vars from
        # ``EngineConfig.role_models`` + the configured ``llm_provider``
        # and threads them in at factory-construction time. The
        # sandbox-side :func:`get_model_for_step` picks them up natively.
        # ``extra_env`` wins over credentials so an operator-supplied
        # override (e.g., a test fixture's explicit AWS_REGION) takes
        # precedence over the chain-resolved value.
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

    bundle = make_compute_dispatcher(
        role="harness_builder",
        image_resolver=_resolve_image,
        command_builder=_build_command,
        inputs=WorkspaceInputs(resolver=_resolve_workspace_input),
        outputs=WorkspaceOutputs(destination=_resolve_workspace_output),
        retry_policy=retry_policy,
    )

    inner_handler = bundle.handler
    inner_post_terminal = bundle.post_terminal_handler

    async def _handler_with_session_open(ctx: Any) -> Any:
        outcome = await inner_handler(ctx)
        if outcome.error is None and outcome.submitted_handles:
            # D2: cross-reference the cost-ledger row with the
            # sandbox JobHandle so post-hoc operator queries can
            # walk from `agent_sessions` to `Compute.logs(handle)`
            # even after the parent record's ``harness_job_handle``
            # is overwritten by a later re-dispatch.
            await open_agent_session(
                ctx.metadata_store,
                parent_kind="cg",
                parent_id=ctx.entity_id,
                agent_role="harness_builder",
                llm=ctx.llm,
                compute_job_handle=outcome.submitted_handles[0],
            )
        return outcome

    async def _post_terminal_with_publish(ctx: Any) -> None:
        if inner_post_terminal is not None:
            await inner_post_terminal(ctx)
        cg_id = ctx.entity_id
        workspace_path = workspace_root / cg_id
        if not workspace_path.exists():
            return
        await publish_workspace_outputs(
            artifact_store=ctx.artifact_store,
            workspace_path=workspace_path,
            key_prefix=publish_prefix_fn(cg_id),
            roots=[
                "harness",
                "techniques",
                "validation_results.json",
                "manifest.json",
                "conversation_traces",
            ],
        )

    return _SandboxedHarnessBuilderBundle(
        handler=_handler_with_session_open,
        post_terminal_handler=_post_terminal_with_publish,
    )


# Default agent-runtime image. Mirrors
# :data:`smai_orchestrator.engine.config.DEFAULT_AGENT_RUNTIME_IMAGE`;
# duplicated as a literal here to avoid an import-back-edge through
# the orchestrator package (this module sits in smai-agents which
# smai-orchestrator already depends on transitively).
_DEFAULT_AGENT_RUNTIME_IMAGE = "smai-agent-runtime:dev"


class _SandboxedHarnessBuilderBundle:
    """Mirror of :class:`smai_orchestrator.dispatch.DispatcherBundle`.

    Constructed here so callers of
    :func:`make_dispatch_harness_build_sandboxed` need not import the
    orchestrator-side bundle type. Field shape identical: spec author
    destructures ``.handler`` and ``.post_terminal_handler`` and wires
    them onto :class:`DispatchAction`.
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


async def _materialize_harness_builder_workspace(  # noqa: PLR0913
    *,
    workspace_path: Path,
    cg_id: str,
    artifact_store: Any,
    metadata_store: Any,
    contract_key: str,
    technique_contract_path_fn: Callable[[str, str], str],
) -> None:
    """Stage the harness-builder sandbox's workspace contents.

    Writes the harness contract, baseline technique contract, runtime
    templates, and harness-API reference into ``workspace_path`` so
    :meth:`Compute.stage_workspace` reads from a ready-to-run
    workspace. Skips the prompt-config / tool-registry shape — those
    live inside the sandbox now (sub-PR E cutover).

    Raises :class:`smai_core.plugins.ArtifactNotFound` if the contract
    or baseline technique contract is missing, and :class:`LookupError`
    if no baseline entry exists. The engine's
    ``_handle_dispatch_failure`` catches the raise and rolls the CG
    back; the round-10 retry policy bounds the loop.
    """
    raw = await artifact_store.get(contract_key)
    harness_contract = HarnessContract.model_validate_json(raw)

    baseline_entry = await _find_baseline_entry(metadata_store, cg_id)
    if baseline_entry is None:
        raise LookupError(
            f"No baseline entry (is_baseline=True) found for CG {cg_id!r}; "
            "every CG materialized via the planner / orchestrator should "
            "have exactly one."
        )
    baseline_contract_key = technique_contract_path_fn(cg_id, baseline_entry.id)
    raw_t = await artifact_store.get(baseline_contract_key)
    baseline_technique_contract = TechniqueContract.model_validate_json(raw_t)

    create_workspace_skeleton(workspace_path)
    write_template_files(workspace_path)

    contract_path = workspace_path / WORKSPACE_HARNESS_CONTRACT_PATH
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(harness_contract.model_dump_json(indent=2))

    technique_contract_path = workspace_path / WORKSPACE_TECHNIQUE_CONTRACT_PATH
    technique_contract_path.parent.mkdir(parents=True, exist_ok=True)
    technique_contract_path.write_text(baseline_technique_contract.model_dump_json(indent=2))

    api_reference_path = workspace_path / WORKSPACE_HARNESS_API_REFERENCE_PATH
    api_reference_path.parent.mkdir(parents=True, exist_ok=True)
    api_reference_path.write_text(build_harness_api_reference())


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


# ArtifactStore is held on the type annotation surface above (parameter
# names use ``Any`` to keep the host-side module importable in contexts
# without smai-core fully loaded); the import below ensures the type is
# resolvable for static-analysis consumers of the export list.
_ = ArtifactStore


__all__ = [
    "DEFAULT_HARNESS_CONTRACT_KEY_TEMPLATE",
    "DEFAULT_HARNESS_MANIFEST_KEY_TEMPLATE",
    "DEFAULT_TECHNIQUE_CONTRACT_KEY_TEMPLATE",
    "WORKSPACE_HARNESS_CONTRACT_PATH",
    "WORKSPACE_TECHNIQUE_CONTRACT_PATH",
    "make_dispatch_harness_build_sandboxed",
]
