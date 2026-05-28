"""Tests for ``smai run --techniques`` (round-13 friction fix).

``smai run`` resolves techniques only by reading the
:class:`MetadataStore`; an experiment referencing a hand-authored
``technique_id`` therefore failed ``technique.id_registered``
verification with no user-facing way to register it. The
``--techniques FILE`` flag upserts supplied :class:`TechniqueRef`
objects into the store before the experiment is compiled.

The happy path is exercised at the :meth:`ExperimentsService.submit_text`
service level (the verb body is a thin parse-and-delegate over it, and
``smai run`` itself instantiates the full dev plugin set, which needs
credentials). The malformed-file path is exercised through the CLI verb
because its parse happens before any plugin is touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _cli_fakes import (  # type: ignore[import-not-found]
    EXPERIMENT_YAML,
    FakeCompute,
    StubLlmProvider,
)
from smai_artifacts_localfs import LocalFsStore
from smai_cli.main import app
from smai_cli.runtime import Runtime
from smai_core import TechniqueRef
from smai_orchestrator import (
    DEFAULT_TASK_ROLES,
    PluginOverrides,
    PluginSelection,
    RuntimeConfig,
)
from smai_orchestrator.engine.config import EngineConfig
from typer.testing import CliRunner


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


def _cutout_technique_ref(technique_id: str = "tech_cutout") -> TechniqueRef:
    """A valid :class:`TechniqueRef` for the ``tech_cutout`` the
    :data:`EXPERIMENT_YAML` fixture references."""
    return TechniqueRef(
        id=technique_id,
        name="Cutout",
        description="Cutout regularization technique.",
        category="augmentation",
        compatible_factor_types=["additive"],
        standard=True,
        affects_extension_points=["train_transforms"],
        context_kind="standard",
    )


def _overrides(tmp_path: Path) -> PluginOverrides:
    return PluginOverrides(
        llm_providers={role: StubLlmProvider() for role in DEFAULT_TASK_ROLES},
        artifact_store=LocalFsStore(tmp_path / "artifacts"),
        compute=FakeCompute(),
    )


@pytest.mark.asyncio
async def test_submit_text_with_techniques_upserts_and_registers(tmp_path: Path) -> None:
    """``submit_text(..., techniques=[ref])`` upserts the technique into
    the store, so the ``tech_cutout``-referencing experiment clears
    ``technique.id_registered`` and the CG registers in ``draft``."""
    config = _make_runtime_config()
    async with Runtime.start_in_band(
        config,
        workspace_root=tmp_path / "workspaces",
        plugin_overrides=_overrides(tmp_path),
        run_worker=False,
    ) as runtime:
        cg_ids = await runtime.experiments.submit_text(
            EXPERIMENT_YAML, techniques=[_cutout_technique_ref()]
        )
        assert cg_ids == ["cg_example"]
        assert (await runtime.status.get("cg_example")).state == "draft"

        # The technique landed in the store's mirror.
        page = await runtime.plugins.metadata_store.list_techniques()
        assert {t.id for t in page.items} == {"tech_cutout"}


@pytest.mark.asyncio
async def test_submit_text_without_techniques_still_fails(tmp_path: Path) -> None:
    """The gap is only closed when the flag is used: the same experiment
    submitted with no ``techniques`` against an empty store still fails
    cleanly with the ``technique.id_registered`` verification error."""
    from smai_core.verification import VerificationError

    config = _make_runtime_config()
    async with Runtime.start_in_band(
        config,
        workspace_root=tmp_path / "workspaces",
        plugin_overrides=_overrides(tmp_path),
        run_worker=False,
    ) as runtime:
        with pytest.raises(VerificationError) as excinfo:
            await runtime.experiments.submit_text(EXPERIMENT_YAML)
        assert "technique.id_registered" in str(excinfo.value)


@pytest.mark.asyncio
async def test_submit_text_techniques_idempotent(tmp_path: Path) -> None:
    """Re-upserting the same technique is harmless: two submissions of
    distinct experiments, each passing the same ``techniques``, both
    succeed (a re-run with the same ``--techniques`` re-upserts cleanly).
    """
    config = _make_runtime_config()
    async with Runtime.start_in_band(
        config,
        workspace_root=tmp_path / "workspaces",
        plugin_overrides=_overrides(tmp_path),
        run_worker=False,
    ) as runtime:
        ref = _cutout_technique_ref()
        first = await runtime.experiments.submit_text(EXPERIMENT_YAML, techniques=[ref])
        assert first == ["cg_example"]

        # A second, fully distinct experiment (CG + entry ids renamed)
        # re-supplying the same technique — the re-upsert must not error.
        second_yaml = (
            EXPERIMENT_YAML.replace("id: cg_example", "id: cg_two")
            .replace("id: entry_baseline", "id: entry_two_baseline")
            .replace("id: entry_cutout", "id: entry_two_cutout")
        )
        second = await runtime.experiments.submit_text(second_yaml, techniques=[ref])
        assert second == ["cg_two"]

        # Still exactly one row — the second upsert updated, not duplicated.
        page = await runtime.plugins.metadata_store.list_techniques()
        assert [t.id for t in page.items] == ["tech_cutout"]


@pytest.mark.asyncio
async def test_submit_text_multiple_techniques(tmp_path: Path) -> None:
    """Multiple techniques are all upserted — the one the experiment
    references and an extra unreferenced one both reach the store."""
    config = _make_runtime_config()
    async with Runtime.start_in_band(
        config,
        workspace_root=tmp_path / "workspaces",
        plugin_overrides=_overrides(tmp_path),
        run_worker=False,
    ) as runtime:
        refs = [_cutout_technique_ref(), _cutout_technique_ref("tech_mixup")]
        cg_ids = await runtime.experiments.submit_text(EXPERIMENT_YAML, techniques=refs)
        assert cg_ids == ["cg_example"]

        page = await runtime.plugins.metadata_store.list_techniques()
        assert {t.id for t in page.items} == {"tech_cutout", "tech_mixup"}


def test_run_malformed_techniques_file_surfaces_clear_error(tmp_path: Path) -> None:
    """A malformed ``--techniques`` JSON file fails at parse time with a
    clear message, before any plugin is instantiated."""
    yaml_path = tmp_path / "experiment.yaml"
    yaml_path.write_text(EXPERIMENT_YAML, encoding="utf-8")
    tech_path = tmp_path / "techniques.json"
    tech_path.write_text("{ this is not valid json", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["run", str(yaml_path), "--techniques", str(tech_path)])
    assert result.exit_code != 0
    combined = (result.output or "") + (str(result.exception) if result.exception else "")
    assert "--techniques" in combined
    assert "could not read JSON" in combined


def test_run_invalid_technique_ref_surfaces_clear_error(tmp_path: Path) -> None:
    """A syntactically valid JSON file that is not a valid TechniqueRef
    is rejected with a TechniqueRef-shaped error message."""
    yaml_path = tmp_path / "experiment.yaml"
    yaml_path.write_text(EXPERIMENT_YAML, encoding="utf-8")
    tech_path = tmp_path / "techniques.json"
    tech_path.write_text(json.dumps([{"id": "tech_cutout"}]), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["run", str(yaml_path), "--techniques", str(tech_path)])
    assert result.exit_code != 0
    combined = (result.output or "") + (str(result.exception) if result.exception else "")
    assert "invalid TechniqueRef" in combined
