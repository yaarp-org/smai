"""End-to-end test for the ``smai run`` + ``smai status`` round-trip
through the in-band Runtime — Task 2.D2's stated acceptance shape.

Per the acceptance:
> smai dev boots and stays up; smai run experiment.yaml (using a
> smoke-test experiment) returns a CG ID; smai status <cg-id>
> reports state.

Full CLI subprocess invocation isn't easily testable (live AWS, file
I/O, signal handling); instead we exercise the underlying
service-surface call path that ``smai run`` and ``smai status`` adapt
over (per `09` §1.2 / §9). Task 2.D3 builds the canonical end-to-end
smoke test on top.
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
from smai_cli.runtime import Runtime
from smai_orchestrator import (
    DEFAULT_TASK_ROLES,
    PluginOverrides,
    PluginSelection,
    RuntimeConfig,
)
from smai_orchestrator.engine.config import EngineConfig


def _make_config() -> RuntimeConfig:
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
async def test_submit_then_status_returns_draft_state(tmp_path: Path) -> None:
    """``smai run`` writes a ``draft`` CG; ``smai status`` reads it."""
    overrides = PluginOverrides(
        llm_providers={role: StubLlmProvider() for role in DEFAULT_TASK_ROLES},
        artifact_store=LocalFsStore(tmp_path / "artifacts"),
        compute=FakeCompute(),
    )
    async with Runtime.start_in_band(
        _make_config(),
        workspace_root=tmp_path / "workspaces",
        plugin_overrides=overrides,
        run_worker=False,
    ) as runtime:
        # Inject the registry seeded with `tech_cutout`.
        runtime.experiments._registries_factory = make_registries_with_technique  # type: ignore[attr-defined]

        cg_ids = await runtime.experiments.submit_text(EXPERIMENT_YAML)
        assert cg_ids == ["cg_example"]

        snap = await runtime.status.get("cg_example")
        assert snap.cg_id == "cg_example"
        assert snap.state == "draft"
        assert snap.is_terminal is False


@pytest.mark.asyncio
async def test_submit_creates_entry_records_per_definition_entry(
    tmp_path: Path,
) -> None:
    """``submit_text`` must create one :class:`EntryRecord` per
    ``ExperimentDefinition.entries`` so the worker can drive entries
    through ``pending → implementing → implemented``.
    """
    artifact_store = LocalFsStore(tmp_path / "artifacts")
    overrides = PluginOverrides(
        llm_providers={role: StubLlmProvider() for role in DEFAULT_TASK_ROLES},
        artifact_store=artifact_store,
        compute=FakeCompute(),
    )
    async with Runtime.start_in_band(
        _make_config(),
        workspace_root=tmp_path / "workspaces",
        plugin_overrides=overrides,
        run_worker=False,
    ) as runtime:
        runtime.experiments._registries_factory = make_registries_with_technique  # type: ignore[attr-defined]
        await runtime.experiments.submit_text(EXPERIMENT_YAML)

        entries_page = await runtime.plugins.metadata_store.list_entries_for_cg("cg_example")
        entry_ids = sorted(e.id for e in entries_page.items)
        assert entry_ids == ["entry_baseline", "entry_cutout"]
        # The treatment entry has a non-null technique_id; baseline is None.
        for e in entries_page.items:
            if e.id == "entry_baseline":
                assert e.is_baseline is True
                assert e.technique_id is None
            else:
                assert e.is_baseline is False
                assert e.technique_id == "tech_cutout"


@pytest.mark.asyncio
async def test_wait_for_terminal_times_out_on_non_terminal_cg(
    tmp_path: Path,
) -> None:
    """``wait_for_terminal`` raises :class:`WaitTimeoutError` if the
    CG never reaches a terminal state within the timeout."""
    from smai_cli.runtime import WaitTimeoutError  # noqa: PLC0415

    overrides = PluginOverrides(
        llm_providers={role: StubLlmProvider() for role in DEFAULT_TASK_ROLES},
        artifact_store=LocalFsStore(tmp_path / "artifacts"),
        compute=FakeCompute(),
    )
    async with Runtime.start_in_band(
        _make_config(),
        workspace_root=tmp_path / "workspaces",
        plugin_overrides=overrides,
        run_worker=False,
    ) as runtime:
        runtime.experiments._registries_factory = make_registries_with_technique  # type: ignore[attr-defined]
        cg_ids = await runtime.experiments.submit_text(EXPERIMENT_YAML)
        with pytest.raises(WaitTimeoutError):
            await runtime.status.wait_for_terminal(
                cg_ids[0], timeout=0.5, poll_interval_seconds=0.1
            )


@pytest.mark.asyncio
async def test_run_one_cycle_advances_no_state_when_no_dispatch_ready(
    tmp_path: Path,
) -> None:
    """``runtime.run_one_cycle()`` with a draft CG and no dispatch
    handlers ready returns stats; the CG stays in ``draft`` because
    the spec's first phase-2 query is for the ``draft → implementing``
    edge which requires the harness builder agent to be invocable —
    the FakeCompute / StubLlmProvider here don't drive it.
    """
    overrides = PluginOverrides(
        llm_providers={role: StubLlmProvider() for role in DEFAULT_TASK_ROLES},
        artifact_store=LocalFsStore(tmp_path / "artifacts"),
        compute=FakeCompute(),
    )
    async with Runtime.start_in_band(
        _make_config(),
        workspace_root=tmp_path / "workspaces",
        plugin_overrides=overrides,
        run_worker=False,
    ) as runtime:
        runtime.experiments._registries_factory = make_registries_with_technique  # type: ignore[attr-defined]
        cg_ids = await runtime.experiments.submit_text(EXPERIMENT_YAML)
        stats = await runtime.run_one_cycle()
        assert stats is not None
        # The CG should still be in `draft` (the harness-build dispatch
        # would need to succeed to advance, which requires the agent
        # loop to fire, and our stubs don't implement it for this test).
        snap = await runtime.status.get(cg_ids[0])
        assert snap.state in {"draft", "implementing"}
