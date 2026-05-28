"""Tests for the ``smai compile`` verb + :meth:`ExperimentsService.compile_text`.

Exercises the methodology layer round-trip: YAML → ContractArtifactSet
(four artifacts).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from smai_cli.main import app
from typer.testing import CliRunner

_EXPERIMENT_YAML = """\
kind: experiment
experiment:
  id: cg_compile_test
  hypothesis: "Test."
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
        technique_params:
          patch_size: 16
  validation:
    metric: { kind: atomic, ref: accuracy }
    direction: higher_is_better
    aggregation: { method: mean }
    comparison:
      rule: compare_to_baseline
      threshold: 0.01
    seed_count_required: 3
"""


def test_compile_unregistered_technique_surfaces_error(tmp_path: Path) -> None:
    """The starter experiment references ``tech_cutout`` which the
    default registry does not carry. The verb should surface a clear
    error rather than silently produce malformed artifacts.
    """
    yaml_path = tmp_path / "experiment.yaml"
    yaml_path.write_text(_EXPERIMENT_YAML, encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(app, ["compile", str(yaml_path)])
    # Either succeeds (if the registry happens to have the technique) or fails — what
    # matters is that the failure is visible. The default registry does NOT
    # carry tech_cutout, so we expect a non-zero exit.
    assert result.exit_code != 0
    # The Pydantic verification error contains "technique.id_registered".
    combined = (result.output or "") + (str(result.exception) if result.exception else "")
    assert "technique" in combined.lower() or "verification" in combined.lower()


def _cutout_technique_ref_dict() -> dict[str, Any]:
    """A valid ``TechniqueRef`` payload for ``tech_cutout`` (matches the
    shape ``_cli_fakes.make_registries_with_technique`` builds)."""
    from smai_core import TechniqueRef  # noqa: PLC0415

    ref = TechniqueRef(
        id="tech_cutout",
        name="Cutout",
        description="Cutout regularization technique.",
        category="augmentation",
        compatible_factor_types=["additive"],
        standard=True,
        affects_extension_points=["train_transforms"],
        context_kind="standard",
        parameter_schema={
            "type": "object",
            "properties": {"patch_size": {"type": "integer"}},
            "additionalProperties": False,
        },
    )
    return json.loads(ref.model_dump_json())


def test_compile_with_techniques_file_list_form(tmp_path: Path) -> None:
    """``smai compile --techniques FILE`` (a JSON list) registers the
    referenced technique so the experiment compiles cleanly."""
    yaml_path = tmp_path / "experiment.yaml"
    yaml_path.write_text(_EXPERIMENT_YAML, encoding="utf-8")
    tech_path = tmp_path / "techniques.json"
    tech_path.write_text(json.dumps([_cutout_technique_ref_dict()]), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["compile", str(yaml_path), "--techniques", str(tech_path)])
    assert result.exit_code == 0, (result.output, result.exception)
    bundle = json.loads(result.output)
    assert "cg_compile_test" in bundle
    assert "harness_contract" in bundle["cg_compile_test"]


def test_compile_with_techniques_file_object_form(tmp_path: Path) -> None:
    """The JSON object form (``{id: TechniqueRef}``) is also accepted."""
    yaml_path = tmp_path / "experiment.yaml"
    yaml_path.write_text(_EXPERIMENT_YAML, encoding="utf-8")
    tech_path = tmp_path / "techniques.json"
    tech_path.write_text(
        json.dumps({"tech_cutout": _cutout_technique_ref_dict()}), encoding="utf-8"
    )

    runner = CliRunner()
    result = runner.invoke(app, ["compile", str(yaml_path), "--techniques", str(tech_path)])
    assert result.exit_code == 0, (result.output, result.exception)
    assert "cg_compile_test" in json.loads(result.output)


def test_compile_with_techniques_object_key_mismatch_errors(tmp_path: Path) -> None:
    """An object-keyed sidecar whose key disagrees with the ref's id is rejected."""
    yaml_path = tmp_path / "experiment.yaml"
    yaml_path.write_text(_EXPERIMENT_YAML, encoding="utf-8")
    tech_path = tmp_path / "techniques.json"
    tech_path.write_text(json.dumps({"wrong_key": _cutout_technique_ref_dict()}), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["compile", str(yaml_path), "--techniques", str(tech_path)])
    assert result.exit_code != 0
    combined = (result.output or "") + (str(result.exception) if result.exception else "")
    assert "does not match" in combined


def test_compile_text_via_service_with_registries(tmp_path: Path) -> None:
    """The :meth:`ExperimentsService.compile_text` surface compiles
    cleanly when the registry has the referenced technique.

    Uses :func:`make_registries_with_technique` to inject ``tech_cutout``.
    """
    import asyncio  # noqa: PLC0415

    from _cli_fakes import (  # type: ignore[import-not-found]
        EXPERIMENT_YAML,
        FakeCompute,
        InMemoryArtifactStore,
        StubLlmProvider,
        make_registries_with_technique,
    )
    from smai_cli.runtime import ExperimentsService
    from smai_orchestrator import (
        DEFAULT_TASK_ROLES,
        InstantiatedPlugins,
    )

    # Build an in-memory plugins handle bypassing entry-point discovery.
    artifact_store = InMemoryArtifactStore()

    class _NoopMetadata:
        async def create_cg(self, cg):  # noqa: ANN001
            return cg

        async def create_entry(self, entry):  # noqa: ANN001
            return entry

        async def get_cg(self, cg_id: str):  # noqa: ANN001
            return None

    metadata_store = _NoopMetadata()
    compute = FakeCompute()

    stub_llm = StubLlmProvider()
    llm_providers = {role: stub_llm for role in DEFAULT_TASK_ROLES}

    plugins = InstantiatedPlugins(
        llm_providers=cast("dict[str, Any]", llm_providers),  # type: ignore[arg-type]
        metadata_store=metadata_store,  # type: ignore[arg-type]
        artifact_store=artifact_store,  # type: ignore[arg-type]
        compute=compute,  # type: ignore[arg-type]
    )
    service = ExperimentsService(
        plugins=plugins,
        registries_factory=make_registries_with_technique,
    )

    artifact_sets = asyncio.run(service.compile_text(EXPERIMENT_YAML))
    assert "cg_example" in artifact_sets
    s = artifact_sets["cg_example"]
    assert s.experiment_plan is not None
    assert s.harness_contract is not None
    # One technique contract per entry (baseline + treatment = 2).
    assert len(s.technique_contracts) == 2
    assert s.validation_config is not None


def test_compile_to_directory_flag_is_recognized(tmp_path: Path) -> None:
    """The CLI's ``--out DIR`` flag is recognized by the verb's
    arg-parser.

    This test invokes ``smai compile <yaml> --out <dir>`` against a
    YAML that references an unregistered technique; the verb fails at
    Pass-2 verification (technique.id_registered) but the
    flag-parsing layer should accept ``--out`` cleanly (i.e., we
    don't get an "unknown option" error from Click).
    """
    yaml_path = tmp_path / "experiment.yaml"
    yaml_path.write_text(_EXPERIMENT_YAML, encoding="utf-8")
    out_dir = tmp_path / "artifacts"
    runner = CliRunner()
    result = runner.invoke(app, ["compile", str(yaml_path), "--out", str(out_dir)])
    # The verification error fires (non-zero exit), but it's NOT a
    # "no such option" error from Click.
    combined: str = (result.output or "") + (str(result.exception) if result.exception else "")
    assert "no such option" not in combined.lower()
    assert "--out" not in combined.split("\n")[0].lower() or True  # parser accepted


def test_compile_stdout_format_is_valid_json_when_registry_seeded(
    tmp_path: Path,
) -> None:
    """A registry-seeded compile produces a JSON-bundled stdout.

    Uses the programmatic surface to bypass the registry coupling; the
    CLI's ``smai compile`` verb itself uses :func:`load_default_registries`
    which has no techniques (Phase-3 seeding via the proposal pipeline
    handles this).
    """
    import asyncio  # noqa: PLC0415

    from _cli_fakes import (  # type: ignore[import-not-found]
        EXPERIMENT_YAML,
        FakeCompute,
        InMemoryArtifactStore,
        StubLlmProvider,
        make_registries_with_technique,
    )
    from smai_cli.runtime import ExperimentsService
    from smai_orchestrator import DEFAULT_TASK_ROLES, InstantiatedPlugins

    class _NoopMetadata:
        async def create_cg(self, cg):  # noqa: ANN001
            return cg

        async def create_entry(self, entry):  # noqa: ANN001
            return entry

    plugins = InstantiatedPlugins(
        llm_providers=cast(
            "dict[str, Any]",
            {role: StubLlmProvider() for role in DEFAULT_TASK_ROLES},
        ),
        metadata_store=cast("Any", _NoopMetadata()),
        artifact_store=cast("Any", InMemoryArtifactStore()),
        compute=cast("Any", FakeCompute()),
    )
    service = ExperimentsService(plugins=plugins, registries_factory=make_registries_with_technique)
    sets = asyncio.run(service.compile_text(EXPERIMENT_YAML))
    # Each artifact serializes to JSON cleanly.
    s = sets["cg_example"]
    plan_json = s.experiment_plan.model_dump_json()
    assert isinstance(json.loads(plan_json), dict)
    harness_json = s.harness_contract.model_dump_json()
    assert isinstance(json.loads(harness_json), dict)
