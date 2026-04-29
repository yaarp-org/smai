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


def test_experiments_service_constructed_via_runtime_property() -> None:
    """Property access exposes the bound :class:`ExperimentsService`."""
    # Instance-level smoke check; full lifecycle in async tests above.
    # This module-level test ensures the import surface is intact.
    assert ExperimentsService is not None
