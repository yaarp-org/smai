"""Test-fixture scaffolding for ``packages/smai-api/tests/``.

Per the per-task fixture filename hygiene convention (CLAUDE.md "Per-
task fixture filename hygiene"), this module's name is unique within
the workspace under ``--import-mode=importlib``. Helpers here build:

* :class:`StubLlmProvider` / :class:`FakeCompute` — minimal Protocol
  conforming plugin instances. The smai-cli tests have similar shims
  in ``_cli_fakes`` but we don't reach across packages — a small copy
  here keeps test-isolation from being a sys.path puzzle.
* :func:`make_test_runtime` — builds an in-memory ``SqliteStore`` +
  ``LocalFsStore`` + the four services, returning a duck-typed
  Runtime-shaped object that satisfies the API's
  :func:`smai_api._deps.get_runtime` reader.
* :func:`seed_state` — primes the store with one CG, one entry, one
  run, one proposal, one paper so the per-resource conformance tests
  have known-good IDs to round-trip against.
* :data:`EXPERIMENT_YAML` + :func:`make_registries_with_technique` —
  reused from the smai-cli test playbook (3.D5 / 3.H1 cousin). A
  compilable single-experiment DSL document with the technique
  pre-registered so ``POST /api/v1/experiments/compile`` succeeds in
  the conformance suite (the suite's default placeholder body would
  fail compilation; overriding the ``experiment_definition_text``
  fixture wires our compilable YAML through).
"""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, NamedTuple

import yaml
from smai_artifacts_localfs import LocalFsStore
from smai_cli.runtime import (
    ExperimentsService,
    PapersService,
    ProposalsService,
    StatusService,
)
from smai_core import (
    DslDocument,
    DslDocumentAdapter,
    Registries,
    TechniqueRef,
    load_default_registries,
)
from smai_core.plugins import (
    ArtifactNotFound,
    ArtifactStoreCapabilities,
    CacheConfig,
    ComputeCapabilities,
    JobHandle,
    JobStatus,
    LlmCapabilities,
    LlmProvider,
    ModelResponse,
    NormalizedMessage,
    TextContent,
    TokenUsage,
    ToolDefinition,
    WorkspaceHandle,
)
from smai_orchestrator import (
    ComparisonGroupRecord,
    EntryRecord,
    PaperRecord,
    ProposalRecord,
    RunRecord,
    RuntimeConfig,
)
from smai_orchestrator.engine.config import EngineConfig
from smai_orchestrator.runtime import PluginSelection
from smai_store_sqlite import SqliteStore

# === StubLlmProvider =========================================================


class StubLlmProvider:
    """Deterministic :class:`LlmProvider` for the API tests.

    Issues a one-token canned response on ``call(...)``. The verify
    endpoint's LLM probe runs through this and returns ok=True; tests
    that exercise the agent loop end-to-end (which the API tests
    explicitly do NOT) would need richer scripted responses.
    """

    def __init__(
        self,
        responses: Sequence[ModelResponse] | None = None,
        *,
        capabilities: LlmCapabilities | None = None,
    ) -> None:
        self.name = "stub-api"
        self.capabilities = capabilities or LlmCapabilities(
            supports_caching=True,
            context_window=200_000,
            max_output_tokens=4_096,
            supports_tool_use=True,
            model_id="stub-api:test",
        )
        self.model_id = "stub-api:test"
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
        if self._responses:
            return self._responses.popleft()
        return ModelResponse(
            message=NormalizedMessage(role="assistant", content=[TextContent(text="stub")]),
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=1, output_tokens=1),
        )


# === FakeCompute =============================================================


class FakeCompute:
    """No-op :class:`Compute` Protocol shim."""

    name: str = "fake-api"
    capabilities: ComputeCapabilities = ComputeCapabilities(
        supports_gpu=False,
        max_timeout_seconds=3600,
        supports_log_streaming=False,
    )

    def __init__(self) -> None:
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
        del image, command, env, gpu, timeout_seconds, plugin_options
        self._counter += 1
        return JobHandle(plugin=self.name, handle=f"fake-{self._counter}")

    async def status(self, handle: JobHandle) -> JobStatus:
        del handle
        return JobStatus(
            state="running",
            exit_code=None,
            started_at=None,
            finished_at=None,
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


# === InMemoryArtifactStore (presigned-disabled variant for the 302 tests) ====


class InMemoryArtifactStore:
    """In-memory :class:`ArtifactStore` with HTTP-shaped presigned URLs.

    Used by the artifact-redirect router test to assert the 302 branch
    without needing a real S3 bucket. ``url_for`` returns
    ``http://test-presigned/<key>`` so the router's HTTP-scheme check
    triggers the redirect path.
    """

    def __init__(self, *, supports_presigned: bool = True) -> None:
        self.name = "inmem-api"
        self.capabilities = ArtifactStoreCapabilities(
            supports_presigned_urls=supports_presigned,
            max_object_size_bytes=None,
        )
        self._data: dict[str, bytes] = {}

    async def put(self, key: str, data: bytes, content_type: str | None = None) -> None:
        del content_type
        self._data[key] = data

    async def get(self, key: str) -> bytes:
        if key not in self._data:
            raise ArtifactNotFound(key)
        return self._data[key]

    async def exists(self, key: str) -> bool:
        return key in self._data

    async def list(self, prefix: str) -> AsyncIterator[str]:
        keys = sorted(k for k in self._data if k.startswith(prefix))

        async def _gen() -> AsyncIterator[str]:
            for k in keys:
                yield k

        return _gen()

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def url_for(
        self,
        key: str,
        expires_in: int = 3600,
        method: Literal["GET", "PUT"] = "GET",
    ) -> str:
        del expires_in, method
        return f"http://test-presigned/{key}"


# === Registries with starter techniques =====================================


def make_registries_with_technique(technique_id: str = "tech_cutout") -> Registries:
    technique = TechniqueRef(
        id=technique_id,
        name="Cutout",
        description="Cutout regularization technique.",
        category="augmentation",
        compatible_factor_types=["additive"],
        standard=True,
        affects_extension_points=["train_transforms"],
    )
    return load_default_registries(technique_registry={technique.id: technique})


# === Compilable experiment YAML =============================================

EXPERIMENT_YAML = """\
kind: experiment
experiment:
  id: cg_smai_api_test
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
  validation:
    metric: { kind: atomic, ref: accuracy }
    direction: higher_is_better
    aggregation: { method: mean }
    comparison:
      rule: compare_to_baseline
      threshold: 0.01
    seed_count_required: 3
"""


def parse_experiment_yaml() -> DslDocument:
    payload = yaml.safe_load(EXPERIMENT_YAML)
    return DslDocumentAdapter.validate_python(payload, context={"smai_mode": "dsl"})


# === Plugins bundle + duck-typed Runtime ====================================


class _PluginsBundle(NamedTuple):
    """:class:`InstantiatedPlugins`-shaped tuple for the API tests."""

    llm_providers: dict[str, LlmProvider]
    metadata_store: SqliteStore
    artifact_store: Any  # LocalFsStore | InMemoryArtifactStore
    compute: FakeCompute


@dataclass
class SmaiApiTestRuntime:
    """Duck-typed :class:`smai_cli.runtime.Runtime` for the API tests.

    Exposes the properties the routers + middleware actually read:
    ``config`` / ``plugins`` / ``experiments`` / ``status`` / ``proposals``
    / ``papers``. Not a subclass of the real ``Runtime`` because that
    would force us through ``Runtime.start_in_band``'s context manager
    (which boots the worker loop + per-role LLM providers needing AWS
    creds). Tests run faster + offline-clean against this duck-type.
    """

    config: RuntimeConfig
    plugins: _PluginsBundle
    experiments: ExperimentsService
    status: StatusService
    proposals: ProposalsService
    papers: PapersService
    workspace_root: Path = field(default_factory=lambda: Path("/tmp/smai-api-tests"))


def _make_test_config() -> RuntimeConfig:
    return RuntimeConfig(
        engine=EngineConfig(poll_interval_seconds=10),
        plugins=PluginSelection(
            llm_provider="bedrock",
            metadata_store="sqlite",
            artifact_store="localfs",
            compute="localgpu",
            metadata_store_config={"uri": "sqlite+aiosqlite:///:memory:"},
            llm_provider_config={"region": "us-west-2", "model_id": "stub"},
        ),
        pipelines=["smai_cg_execution", "smai_cg_entries"],
    )


async def make_test_runtime(
    *,
    artifact_root: Path,
    use_inmem_artifact: bool = False,
) -> SmaiApiTestRuntime:
    """Build a duck-typed Runtime backed by an in-memory SqliteStore +
    LocalFsStore (or an InMemoryArtifactStore that exercises the 302
    branch in the artifact-fetch router test)."""
    store = SqliteStore("sqlite+aiosqlite:///:memory:")
    await store.migrate()

    artifact_store: Any
    if use_inmem_artifact:
        artifact_store = InMemoryArtifactStore(supports_presigned=True)
    else:
        artifact_store = LocalFsStore(artifact_root)

    llm_providers: dict[str, LlmProvider] = {
        role: StubLlmProvider()
        for role in (
            "planner",
            "harness_builder",
            "technique_implementer",
            "code_reviewer",
            "contextual_evaluator",
            "supervisor",
            "screener",
            "enricher",
        )
    }
    plugins = _PluginsBundle(
        llm_providers=llm_providers,
        metadata_store=store,
        artifact_store=artifact_store,
        compute=FakeCompute(),
    )
    config = _make_test_config()
    experiments = ExperimentsService(
        plugins=plugins,  # type: ignore[arg-type]
        registries_factory=make_registries_with_technique,
    )
    status = StatusService(plugins=plugins)  # type: ignore[arg-type]
    proposals = ProposalsService(plugins=plugins)  # type: ignore[arg-type]
    papers = PapersService(plugins=plugins)  # type: ignore[arg-type]
    return SmaiApiTestRuntime(
        config=config,
        plugins=plugins,
        experiments=experiments,
        status=status,
        proposals=proposals,
        papers=papers,
        workspace_root=artifact_root.parent,
    )


# === Pre-seeded entity IDs ==================================================


@dataclass(frozen=True)
class SeededState:
    """Snapshot of pre-seeded entity IDs for the conformance fixtures."""

    cg_id: str
    entry_id: str
    run_id: str
    proposal_id: str
    paper_id: str


_SEED_NOW = datetime(2026, 5, 3, 12, 0, 0, tzinfo=UTC)


async def seed_state(runtime: SmaiApiTestRuntime) -> SeededState:
    """Insert one CG, one entry, one run, one proposal, one paper.

    States are picked so each entity sits in its respective ``list_active``
    aggregator's coverage:

    * CG in ``draft``  → covered by ``get_ready_for_harness_build``
    * Entry in ``pending`` → covered by ``list_entries_for_cg``
    * Run in ``running`` → covered by ``get_in_flight_runs``
    * Proposal in ``proposal_submitted`` → covered by
      ``get_ready_for_proposal_design``
    * Paper in ``submitted`` → covered by ``get_ready_for_paper_fetch``

    Pre-seeded artifact: a harness ``status.json`` blob so the
    artifact-redirect tests have something to fetch + the agent-status
    composite returns a populated harness payload.
    """
    cg_id = "cg-k1-test-001"
    entry_id = "entry-k1-test-001"
    run_id = "run-k1-test-001"
    proposal_id = "prop-k1-test-001"
    paper_id = "2501.10001"
    store = runtime.plugins.metadata_store

    await store.create_proposal(
        ProposalRecord(
            id=proposal_id,
            submission_kind="novel_technique",
            state="proposal_submitted",
            created_at=_SEED_NOW,
            updated_at=_SEED_NOW,
        )
    )
    await store.create_paper(
        PaperRecord(
            arxiv_id=paper_id,
            title="Test Paper for smai-api conformance",
            state="submitted",
            created_at=_SEED_NOW,
            updated_at=_SEED_NOW,
        )
    )
    await store.create_cg(
        ComparisonGroupRecord(
            id=cg_id,
            proposal_id=proposal_id,
            experiment_definition_id=cg_id,
            state="draft",
            created_at=_SEED_NOW,
            updated_at=_SEED_NOW,
        )
    )
    await store.create_entry(
        EntryRecord(
            id=entry_id,
            cg_id=cg_id,
            technique_id=None,
            is_baseline=True,
            entry_id=entry_id,
            state="pending",
            created_at=_SEED_NOW,
            updated_at=_SEED_NOW,
        )
    )
    await store.create_run(
        RunRecord(
            id=run_id,
            cg_id=cg_id,
            entry_id=entry_id,
            seed=1,
            state="running",
            created_at=_SEED_NOW,
            updated_at=_SEED_NOW,
        )
    )
    # Seed a small harness/status.json artifact under the CG's namespace
    # so the artifact-fetch + agent-status tests have something to read.
    # Shape matches ``smai_agents.between_turn.write_status``'s payload.
    await runtime.plugins.artifact_store.put(
        f"comparison-groups/{cg_id}/harness/status.json",
        (
            b'{"role": "harness_builder", "turn_count": 1, "monotonic_timestamp": 12.5, '
            b'"wall_clock_utc": "2026-05-11T00:00:00+00:00", '
            b'"usage_total": {"input_tokens": 10, "output_tokens": 2}, '
            b'"nudges_consumed": 0, "truncations_fired": 0, '
            b'"last_tool_call": "write_file", "last_tool_error": null, '
            b'"tool_errors_fired": 0, "attempt_index": null, "supervisor_aborted": false}'
        ),
    )
    return SeededState(
        cg_id=cg_id,
        entry_id=entry_id,
        run_id=run_id,
        proposal_id=proposal_id,
        paper_id=paper_id,
    )


__all__ = [
    "EXPERIMENT_YAML",
    "FakeCompute",
    "InMemoryArtifactStore",
    "SeededState",
    "StubLlmProvider",
    "SmaiApiTestRuntime",
    "make_registries_with_technique",
    "make_test_runtime",
    "parse_experiment_yaml",
    "seed_state",
]
