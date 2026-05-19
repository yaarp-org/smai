"""End-to-end :meth:`Runtime.start_in_band` tests.

Per Task 2.D2 acceptance: a programmatic Tier-A consumer constructs a
:class:`Runtime`, submits an experiment, and reads its state via
:class:`StatusService`. This is the surface Task 2.D3 wraps for the
canonical smoke test; here we verify the API contract and basic
round-trip semantics.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _cli_fakes import (  # type: ignore[import-not-found]
    EXPERIMENT_YAML,
    FakeCompute,
    StubLlmProvider,
    make_registries_with_technique,
)
from smai_artifacts_localfs import LocalFsStore
from smai_cli.runtime import (
    EXPERIMENT_PLAN_KEY_TEMPLATE,
    HARNESS_CONTRACT_KEY_TEMPLATE,
    VALIDATION_CONFIG_KEY_TEMPLATE,
    ExperimentsService,
    Runtime,
)
from smai_orchestrator import (
    DEFAULT_TASK_ROLES,
    PluginOverrides,
    PluginSelection,
    RuntimeConfig,
)
from smai_orchestrator.engine.config import EngineConfig


def _make_runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        engine=EngineConfig(poll_interval_seconds=10),
        plugins=PluginSelection(
            llm_provider="bedrock",
            metadata_store="sqlite",
            artifact_store="localfs",
            compute="localgpu",
            metadata_store_config={"uri": "sqlite+aiosqlite:///:memory:"},
        ),
        pipelines=["smai_cg_execution", "smai_cg_entries"],
    )


@pytest.mark.asyncio
async def test_runtime_start_in_band_yields_constructed_instance(
    tmp_path: Path,
) -> None:
    """The async-context-managed runtime yields a :class:`Runtime` with
    sub-services bound."""
    overrides = PluginOverrides(
        llm_providers={role: StubLlmProvider() for role in DEFAULT_TASK_ROLES},
        metadata_store=None,  # use real SqliteStore via discovery
        artifact_store=LocalFsStore(tmp_path / "artifacts"),
        compute=FakeCompute(),
    )
    config = _make_runtime_config()
    async with Runtime.start_in_band(
        config,
        workspace_root=tmp_path / "workspaces",
        plugin_overrides=overrides,
        run_worker=False,
    ) as runtime:
        assert runtime.experiments is not None
        assert runtime.status is not None
        assert runtime.workspace_root == tmp_path / "workspaces"
        assert (tmp_path / "workspaces").is_dir()


@pytest.mark.asyncio
async def test_submit_text_creates_cg_and_persists_artifacts(
    tmp_path: Path,
) -> None:
    """``submit_text`` round-trips the four contract artifacts to
    :class:`ArtifactStore` and creates a ``draft`` CG row in
    :class:`MetadataStore`.
    """
    artifact_store = LocalFsStore(tmp_path / "artifacts")
    overrides = PluginOverrides(
        llm_providers={role: StubLlmProvider() for role in DEFAULT_TASK_ROLES},
        artifact_store=artifact_store,
        compute=FakeCompute(),
    )
    config = _make_runtime_config()
    async with Runtime.start_in_band(
        config,
        workspace_root=tmp_path / "workspaces",
        plugin_overrides=overrides,
        run_worker=False,
    ) as runtime:
        # Replace the registries factory on the experiments service so
        # the technique resolves cleanly.
        runtime.experiments._registries_factory = make_registries_with_technique  # type: ignore[attr-defined]

        cg_ids = await runtime.experiments.submit_text(EXPERIMENT_YAML)
        assert cg_ids == ["cg_example"]

        # Status round-trip.
        snap = await runtime.status.get("cg_example")
        assert snap.cg_id == "cg_example"
        assert snap.state == "draft"
        assert snap.is_terminal is False

        # Artifacts present.
        assert await artifact_store.exists(EXPERIMENT_PLAN_KEY_TEMPLATE.format(cg_id="cg_example"))
        assert await artifact_store.exists(HARNESS_CONTRACT_KEY_TEMPLATE.format(cg_id="cg_example"))
        assert await artifact_store.exists(
            VALIDATION_CONFIG_KEY_TEMPLATE.format(cg_id="cg_example")
        )


@pytest.mark.asyncio
async def test_submit_text_resolves_techniques_from_metadata_store(
    tmp_path: Path,
) -> None:
    """With no injected ``registries_factory``, ``submit_text`` builds
    the compiler's technique registry from the store's technique mirror.
    A ``tech_cutout`` row upserted into the store makes the
    ``tech_cutout``-referencing experiment compile cleanly (round-3
    friction (A) — previously this failed ``technique.id_registered``).
    """
    from smai_core import TechniqueRef
    from smai_store_sqlite import SqliteStore

    store = SqliteStore(uri="sqlite+aiosqlite:///:memory:")
    await store.migrate()
    await store.upsert_technique(
        TechniqueRef(
            id="tech_cutout",
            name="Cutout",
            description="Cutout regularization technique.",
            category="augmentation",
            compatible_factor_types=["additive"],
            standard=True,
            affects_extension_points=["train_transforms"],
        )
    )
    artifact_store = LocalFsStore(tmp_path / "artifacts")
    overrides = PluginOverrides(
        llm_providers={role: StubLlmProvider() for role in DEFAULT_TASK_ROLES},
        metadata_store=store,
        artifact_store=artifact_store,
        compute=FakeCompute(),
    )
    config = _make_runtime_config()
    async with Runtime.start_in_band(
        config,
        workspace_root=tmp_path / "workspaces",
        plugin_overrides=overrides,
        run_worker=False,
    ) as runtime:
        # No registries-factory override — the store-backed path is exercised.
        cg_ids = await runtime.experiments.submit_text(EXPERIMENT_YAML)
        assert cg_ids == ["cg_example"]
        snap = await runtime.status.get("cg_example")
        assert snap.state == "draft"


@pytest.mark.asyncio
async def test_submit_text_unregistered_technique_still_fails(tmp_path: Path) -> None:
    """Sanity: the store-backed registry doesn't paper over a genuinely
    missing technique — an empty store still fails verification."""
    from smai_core.verification import VerificationError

    artifact_store = LocalFsStore(tmp_path / "artifacts")
    overrides = PluginOverrides(
        llm_providers={role: StubLlmProvider() for role in DEFAULT_TASK_ROLES},
        artifact_store=artifact_store,
        compute=FakeCompute(),
    )
    config = _make_runtime_config()
    async with Runtime.start_in_band(
        config,
        workspace_root=tmp_path / "workspaces",
        plugin_overrides=overrides,
        run_worker=False,
    ) as runtime:
        with pytest.raises(VerificationError):
            await runtime.experiments.submit_text(EXPERIMENT_YAML)


@pytest.mark.asyncio
async def test_status_get_raises_for_unknown_cg(tmp_path: Path) -> None:
    overrides = PluginOverrides(
        llm_providers={role: StubLlmProvider() for role in DEFAULT_TASK_ROLES},
        artifact_store=LocalFsStore(tmp_path / "artifacts"),
        compute=FakeCompute(),
    )
    config = _make_runtime_config()
    async with Runtime.start_in_band(
        config,
        workspace_root=tmp_path / "workspaces",
        plugin_overrides=overrides,
        run_worker=False,
    ) as runtime:
        from smai_cli.runtime import CGNotFoundError  # noqa: PLC0415

        with pytest.raises(CGNotFoundError):
            await runtime.status.get("does-not-exist")


@pytest.mark.asyncio
async def test_workspace_root_is_created_on_entry(tmp_path: Path) -> None:
    workspace_root = tmp_path / "nested" / "workspaces"
    overrides = PluginOverrides(
        llm_providers={role: StubLlmProvider() for role in DEFAULT_TASK_ROLES},
        artifact_store=LocalFsStore(tmp_path / "artifacts"),
        compute=FakeCompute(),
    )
    config = _make_runtime_config()
    async with Runtime.start_in_band(
        config,
        workspace_root=workspace_root,
        plugin_overrides=overrides,
        run_worker=False,
    ) as runtime:
        del runtime
        assert workspace_root.is_dir()


@pytest.mark.asyncio
async def test_runtime_can_be_re_entered_after_exit(tmp_path: Path) -> None:
    """Successive ``start_in_band`` calls succeed (the spec registry
    is reset on entry/exit)."""
    overrides = PluginOverrides(
        llm_providers={role: StubLlmProvider() for role in DEFAULT_TASK_ROLES},
        artifact_store=LocalFsStore(tmp_path / "artifacts"),
        compute=FakeCompute(),
    )
    config = _make_runtime_config()
    for _ in range(2):
        async with Runtime.start_in_band(
            config,
            workspace_root=tmp_path / "workspaces",
            plugin_overrides=overrides,
            run_worker=False,
        ) as _runtime:
            del _runtime


@pytest.mark.asyncio
async def test_engine_role_models_selects_per_role_llm_model(tmp_path: Path) -> None:
    """``engine.role_models`` flows into the per-role :class:`LlmProvider`
    build — the planner instance resolves to the configured model id while
    sibling roles stay on the :data:`TASK_DEFAULTS` model."""
    config = RuntimeConfig(
        engine=EngineConfig(
            poll_interval_seconds=10,
            role_models={"planner": "us.anthropic.claude-sonnet-4-6"},
        ),
        plugins=PluginSelection(
            llm_provider="bedrock",
            metadata_store="sqlite",
            artifact_store="localfs",
            compute="localgpu",
            metadata_store_config={"uri": "sqlite+aiosqlite:///:memory:"},
            llm_provider_config={"region": "us-east-1"},
        ),
        pipelines=["smai_cg_execution", "smai_cg_entries"],
    )
    # llm_providers left None -> the real per-role build path runs (it
    # constructs BedrockProvider instances, no network at construction).
    overrides = PluginOverrides(
        artifact_store=LocalFsStore(tmp_path / "artifacts"),
        compute=FakeCompute(),
    )
    async with Runtime.start_in_band(
        config,
        workspace_root=tmp_path / "workspaces",
        plugin_overrides=overrides,
        run_worker=False,
        env={},
    ) as runtime:
        providers = runtime._state.plugins.llm_providers  # noqa: SLF001 — test introspection
        assert getattr(providers["planner"], "model_id", None) == "us.anthropic.claude-sonnet-4-6"
        # harness_builder is not in role_models -> still on the Opus default.
        assert getattr(providers["harness_builder"], "model_id", None) == (
            "us.anthropic.claude-opus-4-6-v1"
        )


@pytest.mark.asyncio
async def test_smai_model_env_overrides_engine_role_models(tmp_path: Path) -> None:
    """``SMAI_MODEL_<ROLE>`` wins over ``engine.role_models`` (round-7
    precedence chain: env > config override > TASK_DEFAULTS)."""
    config = RuntimeConfig(
        engine=EngineConfig(
            poll_interval_seconds=10,
            role_models={"planner": "us.anthropic.claude-sonnet-4-6"},
        ),
        plugins=PluginSelection(
            llm_provider="bedrock",
            metadata_store="sqlite",
            artifact_store="localfs",
            compute="localgpu",
            metadata_store_config={"uri": "sqlite+aiosqlite:///:memory:"},
            llm_provider_config={"region": "us-east-1"},
        ),
        pipelines=["smai_cg_execution", "smai_cg_entries"],
    )
    overrides = PluginOverrides(
        artifact_store=LocalFsStore(tmp_path / "artifacts"),
        compute=FakeCompute(),
    )
    async with Runtime.start_in_band(
        config,
        workspace_root=tmp_path / "workspaces",
        plugin_overrides=overrides,
        run_worker=False,
        env={"SMAI_MODEL_PLANNER": "bedrock:env-wins-model"},
    ) as runtime:
        providers = runtime._state.plugins.llm_providers  # noqa: SLF001 — test introspection
        assert getattr(providers["planner"], "model_id", None) == "env-wins-model"


def test_experiments_service_constructed_via_runtime_property() -> None:
    """Property access exposes the bound :class:`ExperimentsService`."""
    # Instance-level smoke check; full lifecycle in async tests above.
    # This module-level test ensures the import surface is intact.
    assert ExperimentsService is not None


@pytest.mark.asyncio
async def test_run_one_cycle_drives_every_registered_spec(tmp_path: Path) -> None:
    """``run_one_cycle()`` returns one :class:`WorkerCycleStats` per
    registered spec — both ``cg_entries`` and ``cg_execution`` are
    advanced each cycle (R3 / F1).

    Before R3, only the CG-execution spec was driven; entries sat in
    ``pending`` indefinitely unless a test manually fired
    :meth:`MetadataStore.transition_entry_state`. The smoke test (Task
    2.D3) papered over the gap; this assertion is the regression
    pin.
    """
    overrides = PluginOverrides(
        llm_providers={role: StubLlmProvider() for role in DEFAULT_TASK_ROLES},
        artifact_store=LocalFsStore(tmp_path / "artifacts"),
        compute=FakeCompute(),
    )
    config = _make_runtime_config()
    async with Runtime.start_in_band(
        config,
        workspace_root=tmp_path / "workspaces",
        plugin_overrides=overrides,
        run_worker=False,
    ) as runtime:
        runtime.experiments._registries_factory = make_registries_with_technique  # type: ignore[attr-defined]
        await runtime.experiments.submit_text(EXPERIMENT_YAML)
        stats_per_spec = await runtime.run_one_cycle()
        # Phase-3 expanded the registered SMAI spec set: cg_entries +
        # cg_execution (Phase-2) + run_record (Task 3.E3) + proposal
        # pipeline (Task 3.E1) + paper-ingestion (Task 3.E2). Each
        # registered spec produces one :class:`WorkerCycleStats` per
        # cycle.
        assert isinstance(stats_per_spec, list)
        assert len(stats_per_spec) == 5


async def _stage_manifest_for(artifact_store: LocalFsStore, cg_id: str) -> None:
    """Pre-stage a minimal valid harness manifest — round 14's
    in-process technique-implementer dispatch reads it before running
    the agent, and this test elides the harness-build path that would
    normally produce it."""
    from smai_core import HarnessContract
    from smai_orchestrator.specs import HARNESS_MANIFEST_KEY_TEMPLATE
    from smai_runtime import (
        MANIFEST_SCHEMA_VERSION,
        RUNTIME_TEMPLATE_VERSION,
        HarnessAPIManifest,
        HarnessExtensionPoint,
        compute_harness_version_hash,
        freeze_manifest,
    )

    raw = await artifact_store.get(HARNESS_CONTRACT_KEY_TEMPLATE.format(cg_id=cg_id))
    harness_contract = HarnessContract.model_validate_json(raw)
    manifest = freeze_manifest(
        HarnessAPIManifest(
            extension_points=[
                HarnessExtensionPoint(
                    key="train_transforms",
                    type_signature="list[Callable]",
                    purpose="optional training-set transforms",
                    optional=True,
                    integration_pattern="append",
                )
            ],
            integration_pattern_summary="round-14 entry-spec test fixture",
            harness_version_hash=compute_harness_version_hash({"__init__.py": b""}),
            parent_harness_contract_hash=harness_contract.envelope.content_hash,
            manifest_schema_version=MANIFEST_SCHEMA_VERSION,
            runtime_template_version=RUNTIME_TEMPLATE_VERSION,
        )
    )
    await artifact_store.put(
        HARNESS_MANIFEST_KEY_TEMPLATE.format(cg_id=cg_id),
        manifest.model_dump_json().encode("utf-8"),
    )


async def _entry_validation_runner(session: object) -> object:
    """Round-14 test runner for the in-process technique implementer:
    stages ``validation_results.json`` as its own side effect so the
    dispatch handler's completeness check passes."""
    from smai_agents import AgentOutcome
    from smai_orchestrator.specs import TECHNIQUE_VALIDATION_KEY_TEMPLATE

    ws = session.workspace_path  # type: ignore[attr-defined]
    await session.artifact_store.put(  # type: ignore[attr-defined]
        TECHNIQUE_VALIDATION_KEY_TEMPLATE.format(cg_id=ws.parent.name, entry_id=ws.name),
        b'{"passed": true}',
    )
    return AgentOutcome(
        kind="finished",
        turn_count=0,
        usage_total=session.usage_total,  # type: ignore[attr-defined]
        finish_success=True,
    )


@pytest.mark.asyncio
async def test_run_one_cycle_advances_treatment_entry_through_entry_spec(
    tmp_path: Path,
) -> None:
    """Per R3 / F1 — ``run_one_cycle`` advances a treatment entry
    through the ``cg_entries`` spec without manual
    :meth:`MetadataStore.transition_entry_state` intervention.

    Setup elides the harness-build path (which would normally drive
    ``draft → implementing`` on the CG) by transitioning the CG
    directly to ``implementing`` and pre-staging the harness manifest.
    The treatment entry is then visible to
    ``get_ready_to_implement_entry``; after one ``run_one_cycle``, the
    entry-spec's phase-3 dispatch fires, runs the technique-implementer
    agent in-process (round 14), and the entry advances ``pending →
    implementing`` with the synthetic ``inline-<entry_id>``
    :class:`JobHandle` recorded on ``EntryRecord.implementation_job_handle``.
    """
    artifact_store = LocalFsStore(tmp_path / "artifacts")
    overrides = PluginOverrides(
        llm_providers={role: StubLlmProvider() for role in DEFAULT_TASK_ROLES},
        artifact_store=artifact_store,
        compute=FakeCompute(),
    )
    config = _make_runtime_config()
    async with Runtime.start_in_band(
        config,
        workspace_root=tmp_path / "workspaces",
        plugin_overrides=overrides,
        run_worker=False,
        technique_implementer_inline_runner=_entry_validation_runner,
    ) as runtime:
        runtime.experiments._registries_factory = make_registries_with_technique  # type: ignore[attr-defined]
        await runtime.experiments.submit_text(EXPERIMENT_YAML)

        # Move the CG into ``implementing`` so the entry spec's phase-2
        # query (gated on parent_state=implementing) discovers the
        # treatment entry, and pre-stage the harness manifest the
        # in-process technique-implementer dispatch reads. These are the
        # only manual setup steps — the entry-spec drive itself is what
        # we are asserting on.
        cg = await runtime.plugins.metadata_store.get_cg("cg_example")
        assert cg is not None
        await runtime.plugins.metadata_store.transition_cg_state(
            "cg_example", cg.version, "implementing"
        )
        await _stage_manifest_for(artifact_store, "cg_example")

        stats_per_spec = await runtime.run_one_cycle()
        # Five specs after Tasks 3.E1 + 3.E2 + 3.E3: proposal + paper +
        # cg_entries + cg_execution + run_record.
        assert len(stats_per_spec) == 5

        treatment = await runtime.plugins.metadata_store.get_entry("entry_cutout")
        assert treatment is not None
        assert treatment.state == "implementing", (
            f"expected entry-spec to advance treatment entry to "
            f"'implementing'; got state={treatment.state}"
        )
        assert treatment.implementation_job_handle is not None
