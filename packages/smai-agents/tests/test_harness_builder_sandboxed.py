"""Unit tests for :func:`make_dispatch_harness_build_sandboxed` (sub-PR B).

Sub-PR B of agent-refactor Step 4. Asserts the new sandboxed-dispatch
factory wires :func:`smai_orchestrator.dispatch.make_compute_dispatcher`
with the expected arg shape:

* image_resolver returns the agent-runtime image (configurable via
  ``agent_runtime_image`` factory arg).
* command_builder returns ``python -m smai_agent_runtime --role
  harness_builder --cg-id <id> --workspace /workspace``.
* inputs.resolver returns the per-CG host workspace path; the factory
  stages that into the substrate via ``Compute.stage_workspace``.
* outputs.destination returns the same per-CG host path (bind-mount
  semantics).
* retry_policy is threaded through verbatim.
* The image_resolver also pre-materializes the workspace (contract +
  baseline technique contract + harness API reference + runtime
  templates) so the sandbox sees a ready-to-run workspace.
* On successful submit, an ``agent_sessions`` row is opened with the
  compute job handle cross-referenced (per D2).
* The bundle's post_terminal_handler harvests the workspace and
  publishes outputs to ArtifactStore.

The sub-PR B dispatch round-trip works on FAKE handlers — these tests
mock :class:`Compute` and assert the call shape; they do NOT exercise
the mini-orchestrator container (that's the
:mod:`smai_agent_runtime.tests.test_entry` surface).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _harness_builder_sandboxed_fixtures import (  # type: ignore[import-not-found]
    RecordingCompute,
    _RecordingArtifactStore,
    _StubDispatchContext,
    _StubEngineConfig,
    _StubLlm,
    _StubMetadataStore,
    make_contract,
    make_technique_contract,
)
from smai_agents.agents.harness_builder import (
    DEFAULT_HARNESS_CONTRACT_KEY_TEMPLATE,
    DEFAULT_TECHNIQUE_CONTRACT_KEY_TEMPLATE,
    make_dispatch_harness_build_sandboxed,
)
from smai_core.plugins import JobHandle


@pytest.mark.asyncio
async def test_factory_returns_bundle_with_handler_and_post_terminal(
    tmp_path: Path,
) -> None:
    """The sandboxed factory returns an object exposing both .handler
    and .post_terminal_handler — the spec author wires both onto
    ``DispatchAction``.
    """
    bundle = make_dispatch_harness_build_sandboxed(workspace_root=tmp_path)
    assert callable(bundle.handler)
    assert callable(bundle.post_terminal_handler)


@pytest.mark.asyncio
async def test_handler_submits_compute_with_expected_argv(tmp_path: Path) -> None:
    """The handler stages the workspace via Compute.stage_workspace,
    then submits with the expected image / command / env / workspace=
    arguments. Validates the sub-PR B dispatch wiring end-to-end on a
    recording fake Compute (no real container).
    """
    artifact_store = _RecordingArtifactStore()
    cg_id = "cg-sandbox-001"
    contract_key = DEFAULT_HARNESS_CONTRACT_KEY_TEMPLATE.format(cg_id=cg_id)
    baseline_key = DEFAULT_TECHNIQUE_CONTRACT_KEY_TEMPLATE.format(
        cg_id=cg_id, entry_id="entry-baseline"
    )
    await artifact_store.put(contract_key, make_contract().model_dump_json().encode())
    await artifact_store.put(baseline_key, make_technique_contract().model_dump_json().encode())

    submit_handle = JobHandle(plugin="recording-compute", handle="agent-job-001")
    compute = RecordingCompute(submit_handle=submit_handle)
    metadata_store = _StubMetadataStore()

    bundle = make_dispatch_harness_build_sandboxed(workspace_root=tmp_path)
    ctx = _StubDispatchContext(
        entity_kind="cg",
        entity_id=cg_id,
        entity_state="implementing",
        entity_version=1,
        metadata_store=metadata_store,
        artifact_store=artifact_store,
        compute=compute,
        llm=_StubLlm(),
        config=_StubEngineConfig(),
    )

    outcome = await bundle.handler(ctx)

    assert outcome.error is None
    assert outcome.submitted_handles == [submit_handle]

    # Substrate call order: stage_workspace first, then submit
    kinds = [c.kind for c in compute.calls]
    assert kinds == ["stage_workspace", "submit"], kinds

    submit_payload = compute.calls[1].payload
    assert submit_payload["image"] == "smai-agent-runtime:dev"
    assert submit_payload["command"] == [
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
    assert submit_payload["env"] == {"SMAI_CG_ID": cg_id}
    assert submit_payload["gpu"] is False
    assert "workspace" in submit_payload["plugin_options"]


@pytest.mark.asyncio
async def test_handler_materializes_workspace_before_stage(tmp_path: Path) -> None:
    """The image_resolver pre-materializes the workspace
    (harness_contract.json + technique_contract.json + harness API
    reference) BEFORE Compute.stage_workspace reads from the path.
    Asserted by checking the workspace contents exist after the
    handler returns.
    """
    artifact_store = _RecordingArtifactStore()
    cg_id = "cg-sandbox-002"
    await artifact_store.put(
        DEFAULT_HARNESS_CONTRACT_KEY_TEMPLATE.format(cg_id=cg_id),
        make_contract().model_dump_json().encode(),
    )
    await artifact_store.put(
        DEFAULT_TECHNIQUE_CONTRACT_KEY_TEMPLATE.format(cg_id=cg_id, entry_id="entry-baseline"),
        make_technique_contract().model_dump_json().encode(),
    )

    compute = RecordingCompute()
    bundle = make_dispatch_harness_build_sandboxed(workspace_root=tmp_path)
    ctx = _StubDispatchContext(
        entity_kind="cg",
        entity_id=cg_id,
        entity_state="implementing",
        entity_version=1,
        metadata_store=_StubMetadataStore(),
        artifact_store=artifact_store,
        compute=compute,
        llm=_StubLlm(),
        config=_StubEngineConfig(),
    )
    await bundle.handler(ctx)

    workspace = tmp_path / cg_id
    assert (workspace / "contracts" / "harness_contract.json").exists()
    assert (workspace / "contracts" / "technique_contract.json").exists()
    assert (workspace / "contracts" / "harness_api_reference.md").exists()


@pytest.mark.asyncio
async def test_handler_opens_agent_session_with_job_handle(tmp_path: Path) -> None:
    """After a successful submit, the handler opens an ``agent_sessions``
    row with ``compute_job_handle=<the returned JobHandle>`` (per D2's
    cross-reference column).
    """
    artifact_store = _RecordingArtifactStore()
    cg_id = "cg-sandbox-003"
    await artifact_store.put(
        DEFAULT_HARNESS_CONTRACT_KEY_TEMPLATE.format(cg_id=cg_id),
        make_contract().model_dump_json().encode(),
    )
    await artifact_store.put(
        DEFAULT_TECHNIQUE_CONTRACT_KEY_TEMPLATE.format(cg_id=cg_id, entry_id="entry-baseline"),
        make_technique_contract().model_dump_json().encode(),
    )

    handle = JobHandle(plugin="recording-compute", handle="agent-job-d2")
    compute = RecordingCompute(submit_handle=handle)
    metadata_store = _StubMetadataStore()
    bundle = make_dispatch_harness_build_sandboxed(workspace_root=tmp_path)
    ctx = _StubDispatchContext(
        entity_kind="cg",
        entity_id=cg_id,
        entity_state="implementing",
        entity_version=1,
        metadata_store=metadata_store,
        artifact_store=artifact_store,
        compute=compute,
        llm=_StubLlm(),
        config=_StubEngineConfig(),
    )
    await bundle.handler(ctx)

    assert len(metadata_store.sessions) == 1
    session_kwargs = metadata_store.sessions[0]
    assert session_kwargs["parent_kind"] == "cg"
    assert session_kwargs["parent_id"] == cg_id
    assert session_kwargs["agent_role"] == "harness_builder"
    assert session_kwargs["compute_job_handle"] == handle


@pytest.mark.asyncio
async def test_custom_agent_runtime_image(tmp_path: Path) -> None:
    """A non-default image flows through to Compute.submit."""
    artifact_store = _RecordingArtifactStore()
    cg_id = "cg-sandbox-img"
    await artifact_store.put(
        DEFAULT_HARNESS_CONTRACT_KEY_TEMPLATE.format(cg_id=cg_id),
        make_contract().model_dump_json().encode(),
    )
    await artifact_store.put(
        DEFAULT_TECHNIQUE_CONTRACT_KEY_TEMPLATE.format(cg_id=cg_id, entry_id="entry-baseline"),
        make_technique_contract().model_dump_json().encode(),
    )

    compute = RecordingCompute()
    bundle = make_dispatch_harness_build_sandboxed(
        workspace_root=tmp_path,
        agent_runtime_image="my-custom-agent-runtime:v42",
    )
    ctx = _StubDispatchContext(
        entity_kind="cg",
        entity_id=cg_id,
        entity_state="implementing",
        entity_version=1,
        metadata_store=_StubMetadataStore(),
        artifact_store=artifact_store,
        compute=compute,
        llm=_StubLlm(),
        config=_StubEngineConfig(),
    )
    await bundle.handler(ctx)

    submit_payload = next(c.payload for c in compute.calls if c.kind == "submit")
    assert submit_payload["image"] == "my-custom-agent-runtime:v42"


@pytest.mark.asyncio
async def test_post_terminal_handler_publishes_workspace(tmp_path: Path) -> None:
    """The bundle's post_terminal_handler harvests the workspace
    (no-op under bind-mount) and publishes the canned files the
    sandbox wrote (harness/, techniques/, validation_results.json,
    manifest.json, conversation_traces/) to ArtifactStore. Tests the
    "host attests-and-persists" half of architectural_decisions §7.
    """
    artifact_store = _RecordingArtifactStore()
    cg_id = "cg-sandbox-harvest"

    # Stand in for the sandbox writing canned outputs to the host
    # workspace (since this test doesn't run the actual container —
    # the sub-PR B mini-orchestrator unit test covers that). Pre-
    # populating mimics what the post-terminal hook will see after the
    # real round-trip.
    workspace = tmp_path / cg_id
    (workspace / "harness").mkdir(parents=True, exist_ok=True)
    (workspace / "harness" / "__init__.py").write_text("# stub harness body\n")
    (workspace / "techniques").mkdir(parents=True, exist_ok=True)
    (workspace / "techniques" / "baseline.py").write_text("# stub baseline body\n")
    (workspace / "validation_results.json").write_text('{"succeeded": true}')
    (workspace / "manifest.json").write_text('{"runtime_template_version": "1.0.0"}')

    compute = RecordingCompute()
    bundle = make_dispatch_harness_build_sandboxed(workspace_root=tmp_path)

    # The post-terminal handler reads ``entity_id``, ``compute``,
    # ``artifact_store``, and ``dispatch_context`` off the context.
    # PostTerminalContext is a Pydantic model that validates plugin
    # instances — we bypass validation here by passing a duck-typed
    # stub since the handler doesn't use Pydantic introspection.
    from types import SimpleNamespace  # noqa: PLC0415

    dispatch_ctx = _StubDispatchContext(
        entity_kind="cg",
        entity_id=cg_id,
        entity_state="implementing",
        entity_version=1,
        metadata_store=_StubMetadataStore(),
        artifact_store=artifact_store,
        compute=compute,
        llm=None,
        config=_StubEngineConfig(),
    )
    pt_ctx = SimpleNamespace(
        entity_kind="cg",
        entity_id=cg_id,
        compute=compute,
        metadata_store=dispatch_ctx.metadata_store,
        artifact_store=artifact_store,
        config=dispatch_ctx.config,
        job_handle=JobHandle(plugin="recording-compute", handle="agent-job-harvest"),
        dispatch_context=dispatch_ctx,
    )
    assert bundle.post_terminal_handler is not None
    await bundle.post_terminal_handler(pt_ctx)

    keys = artifact_store.all_keys()
    # Expect at least the harness body, baseline, validation results,
    # and manifest to land in the store under the harness key prefix.
    assert any(k.endswith("/harness/__init__.py") for k in keys), keys
    assert any(k.endswith("/techniques/baseline.py") for k in keys), keys
    assert any(k.endswith("/validation_results.json") for k in keys), keys
    assert any(k.endswith("/manifest.json") for k in keys), keys

    # The harvest_workspace call also fired (bind-mount no-op but
    # exercised for Protocol uniformity).
    kinds = [c.kind for c in compute.calls]
    assert "stage_workspace" in kinds  # invoked by the inner harvest
    assert "harvest_workspace" in kinds
