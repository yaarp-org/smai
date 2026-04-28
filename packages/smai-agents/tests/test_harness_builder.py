"""Tests for the harness builder dispatch handler (§2.2).

Two surfaces under test:

* :func:`run_harness_builder_session` — the in-process runner. Drives
  the loop end-to-end against a stubbed :class:`LlmProvider` returning
  canned tool calls; asserts the workspace ends up with valid
  ``harness/``, ``techniques/baseline.py``, ``validation_results.json``,
  and ``harness_api_manifest.json``.
* :func:`make_dispatch_harness_build` — the dispatch handler factory.
  Exercises the production path via a fake :class:`Compute` (asserts
  Compute.submit was called with the right shape) and the
  inline-runner test seam (asserts the agent loop ran in-process).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _agent_fakes import StubArtifactStore, StubLlmProvider  # type: ignore[import-not-found]
from _agent_helpers import model_response  # type: ignore[import-not-found]
from _b2_fakes import FakeCompute  # type: ignore[import-not-found]
from _b3_fakes import (  # type: ignore[import-not-found]
    SAMPLE_HARNESS_FILES,
    make_harness_contract,
    make_minimal_manifest,
)
from smai_agents import (
    AgentLoopConfig,
    AgentOutcome,
    AgentSession,
    make_dispatch_harness_build,
    run_harness_builder_session,
)
from smai_core.plugins import JobStatus
from smai_runtime import (
    HARNESS_API_MANIFEST_FILENAME,
    VALIDATION_RESULTS_FILENAME,
    HarnessAPIManifest,
)

# === In-process runner — happy path =========================================


@pytest.mark.asyncio
async def test_run_harness_builder_session_produces_all_four_artifacts(
    tmp_path: Path,
) -> None:
    """End-to-end fixture per the §3.3 acceptance criterion: the loop
    runs against canned tool calls and the workspace ends up with
    valid harness/, techniques/baseline.py, validation_results.json,
    and harness_api_manifest.json."""
    contract = make_harness_contract(factor_type="additive")
    manifest = make_minimal_manifest(
        parent_harness_contract_hash=contract.envelope.content_hash,
        factor_type="additive",
    )

    workspace = tmp_path / "ws"
    artifact_store = StubArtifactStore()

    # Canned LLM responses: write harness, write baseline, run validation,
    # emit manifest, finish. The validation tool's metrics.json + the
    # validation_results.json are seeded by the FakeCompute.on_submit
    # callback so the manifest tool's pass-check fires.
    async def _on_submit(ws: Path) -> None:
        # Write metrics.json (so run_experiment tool can harvest it) plus
        # validation_results.json (so emit_harness_manifest's pass check
        # passes).
        (ws / "metrics.json").write_text(json.dumps({"loss": 0.1}))
        (ws / VALIDATION_RESULTS_FILENAME).write_text(
            json.dumps({"passed": True, "notes": "ok"}),
        )

    fake_compute = FakeCompute(
        status_queue=[
            JobStatus(
                state="succeeded",
                exit_code=0,
                started_at=None,
                finished_at=None,
                failure_reason=None,
            )
        ],
        on_submit=_on_submit,
    )
    fake_compute.set_workspace(workspace)

    write_calls = [
        ("tu-w-init", "write_file", {
            "path": "harness/__init__.py",
            "content": SAMPLE_HARNESS_FILES["__init__.py"].decode(),
        }),
        ("tu-w-trainer", "write_file", {
            "path": "harness/trainer.py",
            "content": SAMPLE_HARNESS_FILES["trainer.py"].decode(),
        }),
    ]
    canned = [
        # Turn 1: write harness/__init__.py + harness/trainer.py.
        model_response(tool_uses=write_calls, stop_reason="tool_use"),
        # Turn 2: write techniques/baseline.py.
        model_response(
            tool_uses=[(
                "tu-w-baseline",
                "write_file",
                {
                    "path": "techniques/baseline.py",
                    "content": "def apply(config):\n    return {}\n",
                },
            )],
            stop_reason="tool_use",
        ),
        # Turn 3: run_experiment validation.
        model_response(
            tool_uses=[(
                "tu-run",
                "run_experiment",
                {"technique": "baseline", "seed": 42, "epochs": 1, "subset_fraction": 0.1},
            )],
            stop_reason="tool_use",
        ),
        # Turn 4: emit_harness_manifest.
        model_response(
            tool_uses=[(
                "tu-emit",
                "emit_harness_manifest",
                manifest.model_dump(mode="json"),
            )],
            stop_reason="tool_use",
        ),
        # Turn 5: finish.
        model_response(
            tool_uses=[(
                "tu-fin",
                "finish",
                {"success": True, "summary": "harness built and validated"},
            )],
            stop_reason="tool_use",
        ),
    ]
    llm = StubLlmProvider(canned)

    outcome = await run_harness_builder_session(
        workspace_path=workspace,
        harness_contract=contract,
        cg_id="cg-fixture",
        llm=llm,
        artifact_store=artifact_store,
        compute=fake_compute,
        manifest_artifact_path="cg-fixture/harness/manifest.json",
        config=AgentLoopConfig(status_write_every_turns=0),
    )

    assert outcome.kind == "finished"
    assert outcome.finish_success is True

    # Workspace post-state: all four expected artifacts.
    assert (workspace / "harness" / "__init__.py").is_file()
    assert (workspace / "harness" / "trainer.py").is_file()
    assert (workspace / "techniques" / "baseline.py").is_file()
    assert (workspace / VALIDATION_RESULTS_FILENAME).is_file()
    assert (workspace / "contracts" / HARNESS_API_MANIFEST_FILENAME).is_file()

    # ArtifactStore got the manifest at the configured key.
    body = await artifact_store.get("cg-fixture/harness/manifest.json")
    revived = HarnessAPIManifest.model_validate_json(body)
    assert revived.content_hash != ""
    assert revived.parent_harness_contract_hash == contract.envelope.content_hash


# === In-process runner — workspace skeleton + contracts =====================


@pytest.mark.asyncio
async def test_run_harness_builder_session_lays_down_contract_for_agent(
    tmp_path: Path,
) -> None:
    """Before invoking the loop, the workspace has the harness contract
    available at contracts/harness_contract.json so the agent can
    read_file it."""
    contract = make_harness_contract(factor_type="substitutive")
    workspace = tmp_path / "ws"
    artifact_store = StubArtifactStore()

    # Capture the AgentSession at loop entry so we can assert without
    # actually driving the loop.
    captured: list[AgentSession] = []

    async def _capture_runner(session: AgentSession) -> AgentOutcome:
        captured.append(session)
        return AgentOutcome(
            kind="finished",
            turn_count=0,
            usage_total=session.usage_total,
            finish_success=True,
            finish_summary="captured",
        )

    fake_compute = FakeCompute()
    llm = StubLlmProvider([])  # never called

    await run_harness_builder_session(
        workspace_path=workspace,
        harness_contract=contract,
        cg_id="cg-substitutive",
        llm=llm,
        artifact_store=artifact_store,
        compute=fake_compute,
        manifest_artifact_path="cg-substitutive/harness/manifest.json",
        config=AgentLoopConfig(status_write_every_turns=0),
        runner=_capture_runner,
    )

    assert (workspace / "experiment.py").is_file()
    assert (workspace / "techniques" / "__init__.py").is_file()
    assert (workspace / "contracts" / "harness_contract.json").is_file()

    assert len(captured) == 1
    session = captured[0]
    assert session.current_role == "harness_builder"
    # The standard inventory + emit_harness_manifest + finish gives 9 tools.
    assert len(session.tools) == 9
    assert "emit_harness_manifest" in session.tools


# === Dispatch handler — production path uses Compute.submit =================


@pytest.mark.asyncio
async def test_dispatch_harness_build_submits_through_compute(
    tmp_path: Path,
) -> None:
    """In the production code path the handler reads the contract
    from ArtifactStore, lays down the workspace, and submits a
    Compute job. The returned :class:`DispatchOutcome` carries the
    handle from :meth:`Compute.submit`."""
    contract = make_harness_contract(factor_type="additive")
    artifact_store = StubArtifactStore()
    contract_key = "comparison-groups/cg-1/harness/contract.json"
    await artifact_store.put(
        contract_key,
        contract.model_dump_json(indent=2).encode("utf-8"),
        content_type="application/json",
    )

    fake_compute = FakeCompute()
    fake_compute.set_workspace(tmp_path / "cg-1")

    handler = make_dispatch_harness_build(workspace_root=tmp_path)

    # Construct a minimal DispatchContext-shaped duck-typed object
    # rather than importing the engine type — the dispatch handler
    # accesses only ctx.entity_id, ctx.artifact_store, ctx.compute, ctx.llm.
    class _StubContext:
        def __init__(self) -> None:
            self.entity_id = "cg-1"
            self.entity_kind = "cg"
            self.entity_state = "implementing"
            self.entity_version = 1
            self.artifact_store = artifact_store
            self.compute = fake_compute
            self.llm = StubLlmProvider([])
            self.metadata_store = None
            self.config = None
            self.checkpointer = None

    outcome = await handler(_StubContext())

    assert outcome.error is None
    assert len(outcome.submitted_handles) == 1
    handle = outcome.submitted_handles[0]
    assert handle.handle == "fake-job-1"
    # Compute.submit was called with the harness-builder entry point.
    assert len(fake_compute.submit_calls) == 1
    call = fake_compute.submit_calls[0]
    assert "smai_agents.agents.harness_builder" in call["command"]
    assert call["env"]["SMAI_CG_ID"] == "cg-1"
    # Workspace was materialized before submit so the container's
    # entrypoint can read contracts/harness_contract.json on start.
    assert (tmp_path / "cg-1" / "contracts" / "harness_contract.json").is_file()


# === Dispatch handler — inline runner exercises the loop in-process =========


@pytest.mark.asyncio
async def test_dispatch_harness_build_inline_runner_drives_loop(
    tmp_path: Path,
) -> None:
    """The ``inline_runner`` seam lets tests skip Compute.submit and
    exercise the agent loop in-process; the production-path branch
    is *not* taken (Compute.submit is never called)."""
    contract = make_harness_contract(factor_type="additive")
    artifact_store = StubArtifactStore()
    contract_key = "comparison-groups/cg-1/harness/contract.json"
    await artifact_store.put(
        contract_key,
        contract.model_dump_json(indent=2).encode("utf-8"),
        content_type="application/json",
    )

    fake_compute = FakeCompute()

    captured: list[AgentSession] = []

    async def _capture_runner(session: AgentSession) -> AgentOutcome:
        captured.append(session)
        return AgentOutcome(
            kind="finished",
            turn_count=0,
            usage_total=session.usage_total,
            finish_success=True,
            finish_summary="inline",
        )

    handler = make_dispatch_harness_build(
        workspace_root=tmp_path,
        inline_runner=_capture_runner,
    )

    class _StubContext:
        def __init__(self) -> None:
            self.entity_id = "cg-1"
            self.entity_kind = "cg"
            self.entity_state = "implementing"
            self.entity_version = 1
            self.artifact_store = artifact_store
            self.compute = fake_compute
            self.llm = StubLlmProvider([])
            self.metadata_store = None
            self.config = None
            self.checkpointer = None

    outcome = await handler(_StubContext())

    assert outcome.error is None
    assert len(outcome.submitted_handles) == 1
    assert outcome.submitted_handles[0].plugin == "inline"
    assert outcome.submitted_handles[0].handle == "inline-cg-1"
    # Compute.submit is *not* invoked on the inline-runner path.
    assert fake_compute.submit_calls == []
    # The runner saw a fully-assembled session.
    assert len(captured) == 1


# === Dispatch handler — error paths =========================================


@pytest.mark.asyncio
async def test_dispatch_harness_build_returns_error_on_missing_contract(
    tmp_path: Path,
) -> None:
    """No contract under the configured ArtifactStore key → DispatchOutcome
    carries the error string and submitted_handles is empty."""
    handler = make_dispatch_harness_build(workspace_root=tmp_path)

    class _StubContext:
        def __init__(self) -> None:
            self.entity_id = "cg-missing"
            self.entity_kind = "cg"
            self.entity_state = "implementing"
            self.entity_version = 1
            self.artifact_store = StubArtifactStore()
            self.compute = FakeCompute()
            self.llm = StubLlmProvider([])
            self.metadata_store = None
            self.config = None
            self.checkpointer = None

    outcome = await handler(_StubContext())
    assert outcome.error is not None
    assert "HarnessContract not found" in outcome.error
    assert outcome.submitted_handles == []


@pytest.mark.asyncio
async def test_dispatch_harness_build_returns_error_when_llm_missing(
    tmp_path: Path,
) -> None:
    """ctx.llm is None → DispatchOutcome carries the error string."""
    contract = make_harness_contract(factor_type="additive")
    artifact_store = StubArtifactStore()
    await artifact_store.put(
        "comparison-groups/cg-1/harness/contract.json",
        contract.model_dump_json(indent=2).encode("utf-8"),
        content_type="application/json",
    )

    handler = make_dispatch_harness_build(workspace_root=tmp_path)

    class _StubContext:
        def __init__(self) -> None:
            self.entity_id = "cg-1"
            self.entity_kind = "cg"
            self.entity_state = "implementing"
            self.entity_version = 1
            self.artifact_store = artifact_store
            self.compute = FakeCompute()
            self.llm = None
            self.metadata_store = None
            self.config = None
            self.checkpointer = None

    outcome = await handler(_StubContext())
    assert outcome.error is not None
    assert "LlmProvider" in outcome.error
    assert outcome.submitted_handles == []
