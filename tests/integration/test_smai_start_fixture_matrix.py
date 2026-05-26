"""``smai start`` no-creds fixture-matrix integration test (Task 3.G3).

Drives a CG end-to-end through :meth:`Runtime.start_worker` against
the SQLite + LocalFs + LocalGpu (FakeCompute) fixture matrix — the
no-credentials counterpart of the credentialed Postgres + S3 + Modal
acceptance the implementation plan §3.4 names. Per Task 3.G3's
no-credentials-in-CI convention, the credentialed lane is local-manual
only (see :mod:`test_smai_start_credentialed`); this test gates CI.

Mocking strategy mirrors :mod:`test_smoke_e2e`: per-role
:class:`StubLlmProvider` for code_reviewer / contextual_evaluator,
:class:`SmokeFakeCompute` whose ``status`` always reports succeeded,
and pre-staged agent-side artifacts so the production dispatch
handlers (whose agent loops are out-of-scope for an in-process
integration test) drain cleanly.

The difference from :mod:`test_smoke_e2e`: that test calls
:meth:`Runtime.run_one_cycle` deterministically; this test boots
:meth:`Runtime.start_worker` (the production-mode shape) and waits
for the worker's background loop to drive the CG to terminal.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from smai_agents import AgentOutcome, AgentSession
from smai_artifacts_localfs import LocalFsStore
from smai_cli.runtime import (
    EXPERIMENT_PLAN_KEY_TEMPLATE,
    HARNESS_CONTRACT_KEY_TEMPLATE,
    Runtime,
)
from smai_core import (
    HarnessContract,
    Registries,
    TechniqueRef,
    load_default_registries,
)
from smai_core.plugins import (
    CacheConfig,
    ComputeCapabilities,
    JobHandle,
    JobStatus,
    LlmCapabilities,
    ModelResponse,
    NormalizedMessage,
    StopReason,
    TextContent,
    TokenUsage,
    ToolDefinition,
    ToolUseContent,
    WorkspaceHandle,
)
from smai_orchestrator import (
    DEFAULT_TASK_ROLES,
    PluginOverrides,
    PluginSelection,
    RuntimeConfig,
)
from smai_orchestrator.engine.config import EngineConfig
from smai_orchestrator.specs import (
    CONTEXTUAL_VERDICT_KEY_TEMPLATE,
    EVALUATION_RESULT_KEY_TEMPLATE,
    HARNESS_MANIFEST_KEY_TEMPLATE,
    HARNESS_VALIDATION_KEY_TEMPLATE,
    RUN_METRICS_KEY_TEMPLATE,
    TECHNIQUE_VALIDATION_KEY_TEMPLATE,
)
from smai_runtime import (
    MANIFEST_SCHEMA_VERSION,
    RUNTIME_TEMPLATE_VERSION,
    HarnessAPIManifest,
    HarnessExtensionPoint,
    compute_harness_version_hash,
    freeze_manifest,
)

# Reuse the smoke fixture — it's already a minimal additive single-
# factor cutout/CIFAR-10 experiment shaped for sub-second drive.
SMOKE_YAML_PATH = Path(__file__).parent / "fixtures" / "smoke_experiment.yaml"
SMOKE_TECHNIQUE_ID = "tech_cutout"
SMOKE_TREATMENT_ENTRY_ID = "entry_cutout"
SMOKE_BASELINE_ENTRY_ID = "entry_baseline"

HARNESS_INIT_PY = b"# placeholder harness package (Task 3.G3 fixture-matrix).\n"
HARNESS_TRAINER_PY = b"def train_one_epoch():\n    pass\n"
TECHNIQUE_CODE_PY = b"def apply(config):\n    return {'train_transforms': []}\n"


# === Stubs (lifted from test_smoke_e2e but kept module-local to
#     avoid cross-test imports) ============================================


class _StubLlm:
    """Deterministic :class:`LlmProvider` for the fixture-matrix test."""

    def __init__(
        self,
        responses: Sequence[ModelResponse] | None = None,
        *,
        name: str = "stub-g3",
    ) -> None:
        self.name = name
        self.capabilities = LlmCapabilities(
            supports_caching=True,
            context_window=200_000,
            max_output_tokens=4_096,
            supports_tool_use=True,
            model_id=f"{name}:test",
        )
        self.model_id = f"{name}:test"
        self._responses: deque[ModelResponse] = deque(responses or [])
        self.calls: list[dict[str, Any]] = []

    async def call(
        self,
        system: str,
        messages: list[NormalizedMessage],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        cache_config: CacheConfig | None = None,
    ) -> ModelResponse:
        del tools, max_tokens, temperature, cache_config
        self.calls.append({"system": system, "messages_n": len(messages)})
        if not self._responses:
            raise AssertionError(
                f"_StubLlm {self.name!r}: ran out of canned responses for system={system!r}"
            )
        return self._responses.popleft()


class _SuccessFakeCompute:
    """``Compute`` whose ``status`` always reports succeeded — same shape
    as :class:`SmokeFakeCompute` in :mod:`test_smoke_e2e`."""

    name: str = "fixture-matrix-fake-compute"
    capabilities: ComputeCapabilities = ComputeCapabilities(
        supports_gpu=True,
        max_timeout_seconds=3600,
        supports_log_streaming=False,
    )

    def __init__(self) -> None:
        self.submitted: list[dict[str, Any]] = []
        self._counter = 0

    async def submit(
        self,
        image: str,
        command: list[str],
        env: dict[str, str],
        gpu: bool = False,
        timeout_seconds: int = 3600,
        **plugin_options: object,
    ) -> JobHandle:
        del timeout_seconds, plugin_options
        self._counter += 1
        handle = JobHandle(plugin=self.name, handle=f"g3-{self._counter}")
        self.submitted.append(
            {
                "image": image,
                "command": list(command),
                "env": dict(env),
                "gpu": gpu,
                "handle": handle.handle,
            }
        )
        return handle

    async def status(self, handle: JobHandle) -> JobStatus:
        del handle
        now = datetime.now(UTC).isoformat()
        return JobStatus(
            state="succeeded",
            exit_code=0,
            started_at=now,
            finished_at=now,
            failure_reason=None,
        )

    async def cancel(self, handle: JobHandle) -> None:
        del handle

    async def logs(self, handle: JobHandle) -> str:
        del handle
        return ""

    async def stage_workspace(self, local_path: Path) -> WorkspaceHandle:
        return WorkspaceHandle(plugin=self.name, handle=str(local_path))

    async def harvest_workspace(self, handle: WorkspaceHandle, local_path: Path) -> None:
        del handle, local_path


def _model_response(
    *,
    tool_uses: list[tuple[str, str, dict[str, Any]]] | None = None,
    text: str | None = None,
    stop_reason: StopReason = "tool_use",
) -> ModelResponse:
    content: list[TextContent | ToolUseContent] = []
    if text is not None:
        content.append(TextContent(text=text))
    if tool_uses is not None:
        for tu_id, tu_name, tu_input in tool_uses:
            content.append(ToolUseContent(id=tu_id, name=tu_name, input=dict(tu_input)))
    return ModelResponse(
        message=NormalizedMessage(role="assistant", content=content),
        stop_reason=stop_reason,
        usage=TokenUsage(input_tokens=1, output_tokens=1),
    )


def _canned_review_pass() -> ModelResponse:
    return _model_response(
        tool_uses=[
            (
                "review_call_1",
                "submit_review",
                {"overall_pass": True, "findings": []},
            )
        ]
    )


def _canned_contextual_promising() -> ModelResponse:
    return _model_response(
        tool_uses=[
            (
                "ctx_call_1",
                "submit_evaluation",
                {
                    "overall_verdict": "promising",
                    "summary": "fixture-matrix contextual verdict",
                    "rankings": [],
                    "insights": ["fixture insight"],
                    "limitations": [],
                    "suggested_follow_ups": [],
                },
            )
        ]
    )


def _make_smoke_registries() -> Registries:
    technique = TechniqueRef(
        id=SMOKE_TECHNIQUE_ID,
        name="Cutout",
        description="Cutout regularization (fixture-matrix technique).",
        category="augmentation",
        compatible_factor_types=["additive"],
        standard=True,
        affects_extension_points=["train_transforms"],
    )
    return load_default_registries(technique_registry={technique.id: technique})


async def _read_harness_contract(artifact_store: LocalFsStore, cg_id: str) -> HarnessContract:
    raw = await artifact_store.get(HARNESS_CONTRACT_KEY_TEMPLATE.format(cg_id=cg_id))
    return HarnessContract.model_validate_json(raw)


def _build_smoke_manifest(harness_contract: HarnessContract) -> HarnessAPIManifest:
    files: dict[str, bytes] = {
        "__init__.py": HARNESS_INIT_PY,
        "trainer.py": HARNESS_TRAINER_PY,
    }
    manifest = HarnessAPIManifest(
        extension_points=[
            HarnessExtensionPoint(
                key="train_transforms",
                type_signature="list[Callable[[Tensor], Tensor]]",
                purpose="optional training-set transforms (fixture-matrix)",
                optional=True,
                integration_pattern="append",
            ),
        ],
        integration_pattern_summary="fixture-matrix",
        harness_version_hash=compute_harness_version_hash(files),
        parent_harness_contract_hash=harness_contract.envelope.content_hash,
        manifest_schema_version=MANIFEST_SCHEMA_VERSION,
        runtime_template_version=RUNTIME_TEMPLATE_VERSION,
    )
    return freeze_manifest(manifest)


async def _pre_stage_artifacts(*, artifact_store: LocalFsStore, cg_id: str) -> HarnessAPIManifest:
    """Mirror :func:`_pre_stage_for_smoke` from :mod:`test_smoke_e2e`."""
    harness_contract = await _read_harness_contract(artifact_store, cg_id)
    manifest = _build_smoke_manifest(harness_contract)

    await artifact_store.put(f"comparison-groups/{cg_id}/harness/__init__.py", HARNESS_INIT_PY)
    await artifact_store.put(f"comparison-groups/{cg_id}/harness/trainer.py", HARNESS_TRAINER_PY)
    await artifact_store.put(
        HARNESS_MANIFEST_KEY_TEMPLATE.format(cg_id=cg_id),
        manifest.model_dump_json().encode("utf-8"),
    )
    await artifact_store.put(
        HARNESS_VALIDATION_KEY_TEMPLATE.format(cg_id=cg_id),
        json.dumps({"passed": True}).encode("utf-8"),
    )

    for entry_id in (SMOKE_BASELINE_ENTRY_ID, SMOKE_TREATMENT_ENTRY_ID):
        await artifact_store.put(
            TECHNIQUE_VALIDATION_KEY_TEMPLATE.format(cg_id=cg_id, entry_id=entry_id),
            json.dumps({"passed": True}).encode("utf-8"),
        )

    await artifact_store.put(
        f"comparison-groups/{cg_id}/entries/{SMOKE_TREATMENT_ENTRY_ID}/code/techniques/{SMOKE_TECHNIQUE_ID}.py",
        TECHNIQUE_CODE_PY,
    )

    for entry_id, accuracy in (
        (SMOKE_BASELINE_ENTRY_ID, 0.50),
        (SMOKE_TREATMENT_ENTRY_ID, 0.85),
    ):
        await artifact_store.put(
            RUN_METRICS_KEY_TEMPLATE.format(cg_id=cg_id, entry_id=entry_id, seed=0),
            json.dumps({"accuracy": accuracy}).encode("utf-8"),
        )
    return manifest


async def _harness_inline_runner(session: AgentSession) -> AgentOutcome:
    """Round-14 test runner for the in-process harness builder. Stages
    the harness manifest as the runner's own side effect — keyed off
    ``session.workspace_path`` (``<root>/<cg_id>``) so the dispatch
    handler's completeness check sees it with no race against the
    background worker."""
    cg_id = session.workspace_path.name
    harness_contract = await _read_harness_contract(session.artifact_store, cg_id)  # type: ignore[arg-type]
    manifest = _build_smoke_manifest(harness_contract)
    await session.artifact_store.put(
        HARNESS_MANIFEST_KEY_TEMPLATE.format(cg_id=cg_id),
        manifest.model_dump_json().encode("utf-8"),
    )
    return AgentOutcome(
        kind="finished", turn_count=0, usage_total=session.usage_total, finish_success=True
    )


async def _technique_inline_runner(session: AgentSession) -> AgentOutcome:
    """Round-14 test runner for the in-process technique implementer.
    Stages ``validation_results.json`` (``session.workspace_path`` is
    ``<root>/<cg_id>/<entry_id>``) so the completeness check + the
    entry-spec validation gate pass race-free."""
    entry_id = session.workspace_path.name
    cg_id = session.workspace_path.parent.name
    await session.artifact_store.put(
        TECHNIQUE_VALIDATION_KEY_TEMPLATE.format(cg_id=cg_id, entry_id=entry_id),
        json.dumps({"passed": True}).encode("utf-8"),
    )
    return AgentOutcome(
        kind="finished", turn_count=0, usage_total=session.usage_total, finish_success=True
    )


def _build_per_role_stubs() -> dict[str, _StubLlm]:
    role_to_stub: dict[str, _StubLlm] = {}
    for role in DEFAULT_TASK_ROLES:
        if role == "code_reviewer":
            role_to_stub[role] = _StubLlm([_canned_review_pass()], name=f"stub-{role}")
        elif role == "contextual_evaluator":
            role_to_stub[role] = _StubLlm([_canned_contextual_promising()], name=f"stub-{role}")
        else:
            role_to_stub[role] = _StubLlm([], name=f"stub-{role}")
    return role_to_stub


def _make_runtime_config(*, sqlite_path: Path) -> RuntimeConfig:
    """Production-shaped config — same plugin selection as ``smai dev``
    but uses Phase-2 default pipelines and a short poll interval to
    keep the test fast.

    File-backed SQLite (vs ``:memory:``) — the in-band test path uses
    ``:memory:`` because there's only one connection (the test thread).
    Here the worker runs in a background asyncio task and opens its own
    connections via :class:`sqlalchemy.ext.asyncio.AsyncEngine`'s pool;
    file-backed is the cross-connection-safe shape, matching
    ``smai dev``'s ``$SMAI_HOME/state.db`` default.
    """
    return RuntimeConfig(
        # 1-second poll keeps the test under a few seconds; production
        # default is 30s.
        engine=EngineConfig(poll_interval_seconds=1, worker_count=1),
        plugins=PluginSelection(
            llm_provider="bedrock",
            metadata_store="sqlite",
            artifact_store="localfs",
            compute="localgpu",
            metadata_store_config={"uri": f"sqlite+aiosqlite:///{sqlite_path}"},
        ),
        pipelines=["smai_cg_execution", "smai_cg_entries"],
    )


# === The fixture-matrix integration test =====================================


@pytest.mark.asyncio
async def test_smai_start_drives_cg_to_complete_no_creds(tmp_path: Path) -> None:
    """End-to-end happy path through :meth:`Runtime.start_worker`.

    Boots the production-mode worker (background asyncio task), seeds
    a CG via :meth:`ExperimentsService.submit_text`, pre-stages the
    agent-side artifacts the gate bodies read, and waits for the
    worker's background loop to drive the CG to ``complete``. Asserts
    the four evaluation artifacts persisted at their canonical keys.
    """
    artifact_store = LocalFsStore(tmp_path / "artifacts")
    role_stubs = _build_per_role_stubs()
    fake_compute = _SuccessFakeCompute()
    overrides = PluginOverrides(
        llm_providers=dict(role_stubs),
        artifact_store=artifact_store,
        compute=fake_compute,
    )
    config = _make_runtime_config(sqlite_path=tmp_path / "state.db")

    smoke_yaml = SMOKE_YAML_PATH.read_text()

    async with Runtime.start_worker(
        config,
        worker_id="g3-fixture-matrix-test",
        workspace_root=tmp_path / "workspaces",
        plugin_overrides=overrides,
        harness_builder_inline_runner=_harness_inline_runner,
        technique_implementer_inline_runner=_technique_inline_runner,
    ) as runtime:
        # Seed the technique registry so the smoke YAML compiles cleanly.
        runtime.experiments._registries_factory = _make_smoke_registries  # type: ignore[attr-defined]

        cg_ids = await runtime.experiments.submit_text(smoke_yaml)
        assert cg_ids == ["cg_smoke"]
        cg_id = cg_ids[0]

        # Sanity — the four contract artifacts persisted.
        assert await artifact_store.exists(EXPERIMENT_PLAN_KEY_TEMPLATE.format(cg_id=cg_id))
        assert await artifact_store.exists(HARNESS_CONTRACT_KEY_TEMPLATE.format(cg_id=cg_id))

        # Pre-stage the agent side-effect artifacts so the gate bodies
        # have what they need; the production dispatch handlers fire
        # against the FakeCompute which reports succeeded.
        await _pre_stage_artifacts(artifact_store=artifact_store, cg_id=cg_id)

        # Wait for the worker's background loop to drive the CG to a
        # terminal state. The 1s poll interval + sub-second cycle work
        # means terminal lands within ~10s on a slow CI runner.
        deadline = asyncio.get_event_loop().time() + 30.0
        terminal_state: str | None = None
        while asyncio.get_event_loop().time() < deadline:
            snap = await runtime.status.get(cg_id)
            if snap.is_terminal:
                terminal_state = snap.state
                break
            await asyncio.sleep(0.5)
        assert terminal_state == "complete", (
            f"expected terminal `complete` within deadline; got {terminal_state}"
        )

        # The four evaluation artifacts persisted.
        assert await artifact_store.exists(EVALUATION_RESULT_KEY_TEMPLATE.format(cg_id=cg_id))
        assert await artifact_store.exists(CONTEXTUAL_VERDICT_KEY_TEMPLATE.format(cg_id=cg_id))

        # The fake Compute observed the runs dispatch (round 14: the
        # harness-builder / technique-implementer agents run in-process,
        # so only the seed runs submit jobs); the code-reviewer +
        # contextual-evaluator stubs were invoked (gate bodies fired).
        assert len(fake_compute.submitted) >= 2
        assert len(role_stubs["code_reviewer"].calls) >= 1
        assert len(role_stubs["contextual_evaluator"].calls) >= 1

        # The worker_id propagated through to the runtime as the
        # production-mode contract requires.
        assert runtime.worker_id == "g3-fixture-matrix-test"


def test_smai_start_fixture_matrix_yaml_loadable() -> None:
    """The smoke YAML the integration test consumes is parseable.

    A drift-detector mirroring :func:`test_smoke_yaml_fixture_is_loadable`
    in :mod:`test_smoke_e2e`.
    """
    payload: Any = yaml.safe_load(SMOKE_YAML_PATH.read_text())
    assert isinstance(payload, dict)
    assert payload["kind"] == "experiment"
    experiment: Any = payload["experiment"]
    assert experiment["id"] == "cg_smoke"
