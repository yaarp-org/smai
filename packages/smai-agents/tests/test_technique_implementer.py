"""Tests for the technique implementer dispatch handler (§2.3).

Two surfaces:

* :func:`run_technique_implementer_session` — happy-path end-to-end run
  with canned tool calls writing the technique module + validating it +
  finishing.
* :func:`make_dispatch_technique_implementation` — additive-baseline
  skip path (DEC-013 / DEC-017) and the round-14 in-process dispatch
  (the agent loop runs in the worker, no Compute job; the handler
  synthesizes an ``inline-<entry_id>`` handle).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from _agent_fakes import StubArtifactStore, StubLlmProvider  # type: ignore[import-not-found]
from _agent_helpers import model_response  # type: ignore[import-not-found]
from _b2_fakes import FakeCompute  # type: ignore[import-not-found]
from _b3_fakes import (  # type: ignore[import-not-found]
    SAMPLE_HARNESS_FILES,
    make_additive_baseline_technique_contract,
    make_harness_contract,
    make_minimal_manifest,
    make_substitutive_baseline_technique_contract,
    make_technique_contract,
)
from smai_agents import (
    PREV_CONVERSATION_TRACE_FILENAME,
    AgentLoopConfig,
    AgentOutcome,
    AgentSession,
    make_dispatch_technique_implementation,
    run_technique_implementer_session,
)
from smai_core.plugins import JobStatus
from smai_orchestrator.entities.tracking import EntryRecord
from smai_runtime import VALIDATION_RESULTS_FILENAME


def _now() -> datetime:
    return datetime.now(tz=UTC)


# === In-process runner — happy path =========================================


@pytest.mark.asyncio
async def test_run_technique_implementer_session_produces_technique_module(
    tmp_path: Path,
) -> None:
    """End-to-end: canned tool calls write techniques/<name>.py and
    run validation; workspace ends up with the technique module +
    validation_results.json (the no-go-zone hash check covers
    harness/* / experiment.py / techniques/__init__.py at run time —
    those are byte-replayed from the harness builder's outputs)."""
    contract = make_harness_contract(factor_type="additive")
    technique_contract = make_technique_contract(
        parent_harness_contract_hash=contract.envelope.content_hash,
        entry_id="entry-1",
        technique_id="tq-cutout",
    )
    manifest = make_minimal_manifest(
        parent_harness_contract_hash=contract.envelope.content_hash,
        factor_type="additive",
    )

    workspace = tmp_path / "ws"
    artifact_store = StubArtifactStore()

    async def _on_submit(ws: Path) -> None:
        (ws / "metrics.json").write_text(json.dumps({"loss": 0.1}))
        (ws / VALIDATION_RESULTS_FILENAME).write_text(
            json.dumps({"passed": True}),
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

    canned = [
        # Turn 1: write techniques/<name>.py.
        model_response(
            tool_uses=[
                (
                    "tu-w",
                    "write_file",
                    {
                        "path": "techniques/cutout.py",
                        "content": "def apply(config):\n    return {}\n",
                    },
                )
            ],
            stop_reason="tool_use",
        ),
        # Turn 2: validate.
        model_response(
            tool_uses=[
                (
                    "tu-run",
                    "run_experiment",
                    {"technique": "cutout", "seed": 42, "epochs": 1, "subset_fraction": 0.1},
                )
            ],
            stop_reason="tool_use",
        ),
        # Turn 3: finish.
        model_response(
            tool_uses=[
                (
                    "tu-fin",
                    "finish",
                    {"success": True, "summary": "ok"},
                )
            ],
            stop_reason="tool_use",
        ),
    ]
    llm = StubLlmProvider(canned)

    outcome = await run_technique_implementer_session(
        workspace_path=workspace,
        cg_id="cg-1",
        entry_id="entry-1",
        technique_id="tq-cutout",
        technique_name="cutout",
        factor_dimension="augmentation",
        factor_type="additive",
        context_kind="method_description",
        grounding_path="techniques/tq-cutout/method_description.json",
        harness_contract=contract,
        technique_contract=technique_contract,
        manifest=manifest,
        harness_files=SAMPLE_HARNESS_FILES,
        baseline_module=b"def apply(config):\n    return {}\n",
        llm=llm,
        artifact_store=artifact_store,
        compute=fake_compute,
        config=AgentLoopConfig(status_write_every_turns=0),
    )

    assert outcome.kind == "finished"
    # Workspace post-state: technique module + replayed harness +
    # validation_results.json.
    assert (workspace / "techniques" / "cutout.py").is_file()
    assert (workspace / "techniques" / "baseline.py").is_file()
    assert (workspace / "harness" / "trainer.py").is_file()
    # All three contract artifacts dropped under contracts/.
    assert (workspace / "contracts" / "harness_contract.json").is_file()
    assert (workspace / "contracts" / "technique_contract.json").is_file()
    assert (workspace / "contracts" / "harness_api_manifest.json").is_file()
    assert (workspace / VALIDATION_RESULTS_FILENAME).is_file()


# === DEC-023 retry-context propagation ======================================


@pytest.mark.asyncio
async def test_run_technique_implementer_session_loads_prev_conversation_trace(
    tmp_path: Path,
) -> None:
    """On retry (implementation_attempt > 0), the previous attempt's
    conversation-trace.json is copied into the workspace at
    prev-conversation-trace.json before the loop runs."""
    contract = make_harness_contract(factor_type="additive")
    technique_contract = make_technique_contract(
        parent_harness_contract_hash=contract.envelope.content_hash,
        entry_id="entry-retry",
        technique_id="tq-retry",
    )
    manifest = make_minimal_manifest(
        parent_harness_contract_hash=contract.envelope.content_hash,
        factor_type="additive",
    )

    workspace = tmp_path / "ws-retry"
    artifact_store = StubArtifactStore()

    # Seed the prior attempt's conversation trace under the canonical key.
    prior_trace = b'{"turn": 1, "role": "technique_implementer", "content": "..."}'
    prev_trace_key = "comparison-groups/cg-1/entries/entry-retry/conversation-trace.json"
    await artifact_store.put(prev_trace_key, prior_trace)

    captured: list[AgentSession] = []

    async def _capture(session: AgentSession) -> AgentOutcome:
        captured.append(session)
        return AgentOutcome(
            kind="finished",
            turn_count=0,
            usage_total=session.usage_total,
            finish_success=True,
            finish_summary="x",
        )

    await run_technique_implementer_session(
        workspace_path=workspace,
        cg_id="cg-1",
        entry_id="entry-retry",
        technique_id="tq-retry",
        technique_name="retry",
        factor_dimension="augmentation",
        factor_type="additive",
        context_kind="method_description",
        grounding_path=None,
        harness_contract=contract,
        technique_contract=technique_contract,
        manifest=manifest,
        harness_files=SAMPLE_HARNESS_FILES,
        baseline_module=None,
        llm=StubLlmProvider([]),
        artifact_store=artifact_store,
        compute=FakeCompute(),
        prev_conversation_trace_path=prev_trace_key,
        implementation_attempt=1,
        runner=_capture,
    )

    prev_path = workspace / PREV_CONVERSATION_TRACE_FILENAME
    assert prev_path.is_file()
    assert prev_path.read_bytes() == prior_trace
    # The captured session was constructed against the workspace that
    # already has prev-conversation-trace.json.
    assert len(captured) == 1


# === Additive baseline skip (DEC-013 / DEC-017) =============================


@pytest.mark.asyncio
async def test_dispatch_technique_implementation_skips_additive_baseline(
    tmp_path: Path,
) -> None:
    """An entry whose contract is the additive baseline (technique_id=None,
    is_baseline=True) is NOT dispatched — the handler returns a no-op
    DispatchOutcome and Compute.submit is never called."""
    contract = make_harness_contract(factor_type="additive")
    technique_contract = make_additive_baseline_technique_contract(
        parent_harness_contract_hash=contract.envelope.content_hash,
    )

    artifact_store = StubArtifactStore()
    contract_key = "comparison-groups/cg-1/entries/entry-baseline/technique_contract.json"
    await artifact_store.put(
        contract_key,
        technique_contract.model_dump_json(indent=2).encode("utf-8"),
    )

    fake_compute = FakeCompute()

    handler = make_dispatch_technique_implementation(workspace_root=tmp_path)

    class _StubMetadataStore:
        async def get_entry(self, entry_id: str) -> EntryRecord:
            return EntryRecord(
                id=entry_id,
                cg_id="cg-1",
                technique_id=None,
                is_baseline=True,
                entry_id=entry_id,
                state="implementing",
                implementation_attempt=0,
                created_at=_now(),
                updated_at=_now(),
            )

    class _StubContext:
        def __init__(self) -> None:
            self.entity_id = "entry-baseline"
            self.entity_kind = "entry"
            self.entity_state = "implementing"
            self.entity_version = 1
            self.artifact_store = artifact_store
            self.compute = fake_compute
            self.llm = StubLlmProvider([])
            self.metadata_store = _StubMetadataStore()
            self.config = None
            self.checkpointer = None

    outcome = await handler(_StubContext())
    assert outcome.error is None
    assert outcome.submitted_handles == []
    assert fake_compute.submit_calls == []


# === Substitutive baseline IS dispatched — runs in-process (round 14) =======


@pytest.mark.asyncio
async def test_dispatch_technique_implementation_runs_session_in_process(
    tmp_path: Path,
) -> None:
    """Substitutive baselines have real technique modules and ARE
    dispatched (DEC-017): round 14 runs the technique-implementer loop
    in-process (no Compute.submit) and synthesizes an ``inline-<entry>``
    handle. The ``inline_runner`` seam swaps the runner; the
    ``validation_results.json`` the completeness check requires is the
    runner's side effect."""
    contract = make_harness_contract(factor_type="substitutive")
    technique_contract = make_substitutive_baseline_technique_contract(
        parent_harness_contract_hash=contract.envelope.content_hash,
    )
    manifest = make_minimal_manifest(
        parent_harness_contract_hash=contract.envelope.content_hash,
        factor_type="substitutive",
    )

    artifact_store = StubArtifactStore()
    await artifact_store.put(
        "comparison-groups/cg-1/entries/entry-vgg/technique_contract.json",
        technique_contract.model_dump_json(indent=2).encode("utf-8"),
    )
    await artifact_store.put(
        "comparison-groups/cg-1/harness/contract.json",
        contract.model_dump_json(indent=2).encode("utf-8"),
    )
    await artifact_store.put(
        "comparison-groups/cg-1/harness/manifest.json",
        manifest.model_dump_json(indent=2).encode("utf-8"),
    )

    validation_key = "comparison-groups/cg-1/entries/entry-vgg/code/validation_results.json"

    async def _capture_runner(session: AgentSession) -> AgentOutcome:
        # Stand in for the agent's validation run: the completeness
        # check requires validation_results.json in the store.
        await artifact_store.put(validation_key, json.dumps({"passed": True}).encode("utf-8"))
        return AgentOutcome(
            kind="finished",
            turn_count=0,
            usage_total=session.usage_total,
            finish_success=True,
        )

    fake_compute = FakeCompute()
    handler = make_dispatch_technique_implementation(
        workspace_root=tmp_path, inline_runner=_capture_runner
    )

    class _StubMetadataStore:
        async def get_entry(self, entry_id: str) -> EntryRecord:
            return EntryRecord(
                id=entry_id,
                cg_id="cg-1",
                technique_id="tq-vgg",
                is_baseline=True,
                entry_id=entry_id,
                state="implementing",
                implementation_attempt=0,
                created_at=_now(),
                updated_at=_now(),
            )

    class _StubContext:
        def __init__(self) -> None:
            self.entity_id = "entry-vgg"
            self.entity_kind = "entry"
            self.entity_state = "implementing"
            self.entity_version = 1
            self.artifact_store = artifact_store
            self.compute = fake_compute
            self.llm = StubLlmProvider([])
            self.metadata_store = _StubMetadataStore()
            self.config = None
            self.checkpointer = None

    outcome = await handler(_StubContext())
    assert outcome.error is None
    assert len(outcome.submitted_handles) == 1
    assert outcome.submitted_handles[0].plugin == "inline"
    assert outcome.submitted_handles[0].handle == "inline-entry-vgg"
    # The agent loop ran in-process — Compute.submit is never called.
    assert fake_compute.submit_calls == []


# === Missing entry record → error path ======================================


@pytest.mark.asyncio
async def test_dispatch_technique_implementation_returns_error_on_missing_entry(
    tmp_path: Path,
) -> None:
    """MetadataStore returns None for the entry → DispatchOutcome with error."""
    handler = make_dispatch_technique_implementation(workspace_root=tmp_path)

    class _StubMetadataStore:
        async def get_entry(self, entry_id: str) -> EntryRecord | None:
            del entry_id
            return None

    class _StubContext:
        def __init__(self) -> None:
            self.entity_id = "entry-missing"
            self.entity_kind = "entry"
            self.entity_state = "implementing"
            self.entity_version = 1
            self.artifact_store = StubArtifactStore()
            self.compute = FakeCompute()
            self.llm = StubLlmProvider([])
            self.metadata_store = _StubMetadataStore()
            self.config = None
            self.checkpointer = None

    outcome = await handler(_StubContext())
    assert outcome.error is not None
    assert "not found" in outcome.error
    assert outcome.submitted_handles == []
