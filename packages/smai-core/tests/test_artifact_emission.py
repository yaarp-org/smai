"""End-to-end emission tests against the five worked-example fixtures.

Per ``02-dsl-and-contracts.md`` §7 + §8. Each fixture compiles to a complete
:class:`ContractArtifactSet`; per-artifact bodies populate per spec; surface
maps are non-empty and well-formed; parent-hash references thread through.
"""

from __future__ import annotations

import pytest
from _emit_helpers import (  # type: ignore[import-not-found]
    EXPERIMENT_FIXTURES,
    compile_experiment_fixture,
)


@pytest.mark.parametrize("name", EXPERIMENT_FIXTURES)
def test_fixture_compiles(name: str) -> None:
    art_set, _report = compile_experiment_fixture(name)
    assert art_set.experiment_plan.envelope.artifact_kind == "experiment_plan"
    assert art_set.harness_contract.envelope.artifact_kind == "harness_contract"
    assert art_set.validation_config.envelope.artifact_kind == "validation_config"
    assert art_set.technique_contracts, "expected at least one technique contract"


@pytest.mark.parametrize("name", EXPERIMENT_FIXTURES)
def test_fixture_envelope_provenance(name: str) -> None:
    art_set, _ = compile_experiment_fixture(name)
    plan = art_set.experiment_plan
    assert plan.envelope.compiler_version == "0.1.0"
    assert plan.envelope.schema_version == 1
    assert set(plan.envelope.registry_hashes) == {"factor_types", "metrics", "techniques"}
    for digest in plan.envelope.registry_hashes.values():
        assert len(digest) == 64


@pytest.mark.parametrize("name", EXPERIMENT_FIXTURES)
def test_fixture_parent_hashes_thread_through(name: str) -> None:
    art_set, _ = compile_experiment_fixture(name)
    plan_hash = art_set.experiment_plan.envelope.content_hash
    harness_hash = art_set.harness_contract.envelope.content_hash
    # HarnessContract references plan
    assert art_set.harness_contract.body.parent_experiment_hash == plan_hash
    # TechniqueContracts reference plan + harness
    for tc in art_set.technique_contracts:
        assert tc.body.parent_experiment_hash == plan_hash
        assert tc.body.parent_harness_contract_hash == harness_hash
    # ValidationConfig references plan
    assert art_set.validation_config.body.parent_experiment_hash == plan_hash


@pytest.mark.parametrize("name", EXPERIMENT_FIXTURES)
def test_fixture_surface_maps_populated(name: str) -> None:
    art_set, _ = compile_experiment_fixture(name)
    assert art_set.experiment_plan.envelope.surface_map
    assert art_set.harness_contract.envelope.surface_map
    assert art_set.validation_config.envelope.surface_map
    for tc in art_set.technique_contracts:
        assert tc.envelope.surface_map
        assert tc.envelope.surface_map["body.technique_params"] == "correctness_iterable"


def test_resnet_vs_vgg_baseline_entry_id_filled() -> None:
    art_set, _ = compile_experiment_fixture("resnet50_vs_vgg16_cifar10")
    assert art_set.validation_config.body.comparison.baseline_entry_id == "vgg16_reference"


def test_cutout_baseline_technique_contract_is_minimal() -> None:
    art_set, _ = compile_experiment_fixture("cutout_on_cifar10")
    baselines = [tc for tc in art_set.technique_contracts if tc.body.is_baseline]
    assert len(baselines) == 1
    bt = baselines[0]
    assert bt.body.technique_id is None
    assert bt.body.technique_params is None
    assert bt.body.fidelity_anchor is None


def test_pruning_sweep_emits_five_technique_contracts() -> None:
    art_set, _ = compile_experiment_fixture("pruning_sparsity_sweep")
    assert len(art_set.technique_contracts) == 5
    treatments = [tc for tc in art_set.technique_contracts if not tc.body.is_baseline]
    for tc in treatments:
        assert tc.body.level_value is not None
        assert tc.body.level_value.kind == "continuous"
        assert tc.body.technique_params is not None
        assert "sparsity" in tc.body.technique_params


def test_position_embeddings_validation_uses_lower_is_better() -> None:
    art_set, _ = compile_experiment_fixture("position_embeddings_wikitext103")
    assert art_set.validation_config.body.direction == "lower_is_better"
    assert art_set.validation_config.body.metric.kind == "atomic"
    assert (
        art_set.validation_config.body.metric.ref == "perplexity"  # type: ignore[union-attr]
    )


def test_factor_model_emits_one_set_per_child() -> None:
    from _emit_helpers import compile_factor_model_fixture  # type: ignore[import-not-found]

    sets, _ = compile_factor_model_fixture("factor_model_resnet50_imagenet")
    assert set(sets.keys()) == {
        "cg_activation_function",
        "cg_normalization_layer",
        "cg_training_augmentation",
    }
    # Every artifact in every set carries parent_factor_model_id
    for art_set in sets.values():
        assert (
            art_set.experiment_plan.envelope.parent_factor_model_id
            == "resnet50_imagenet_arch_improvements_2024"
        )
        assert (
            art_set.harness_contract.envelope.parent_factor_model_id
            == "resnet50_imagenet_arch_improvements_2024"
        )
        assert (
            art_set.validation_config.envelope.parent_factor_model_id
            == "resnet50_imagenet_arch_improvements_2024"
        )


def test_factor_model_per_cg_self_consistent() -> None:
    from _emit_helpers import compile_factor_model_fixture  # type: ignore[import-not-found]

    sets, _ = compile_factor_model_fixture("factor_model_resnet50_imagenet")
    for cg_id, art_set in sets.items():
        plan_hash = art_set.experiment_plan.envelope.content_hash
        harness_hash = art_set.harness_contract.envelope.content_hash
        assert art_set.harness_contract.body.parent_experiment_hash == plan_hash
        for tc in art_set.technique_contracts:
            assert tc.body.parent_experiment_hash == plan_hash
            assert tc.body.parent_harness_contract_hash == harness_hash
        assert art_set.validation_config.body.parent_experiment_hash == plan_hash
        # body.factor_model_id matches the FactorModel id
        assert (
            art_set.experiment_plan.body.factor_model_id
            == "resnet50_imagenet_arch_improvements_2024"
        )
        # Extra guard: the experiment id matches the dict key
        assert art_set.experiment_plan.envelope.parent_experiment_id == cg_id


def test_harness_contract_seeds_lifted_out_of_controlled_conditions() -> None:
    art_set, _ = compile_experiment_fixture("resnet50_vs_vgg16_cifar10")
    seeds = art_set.harness_contract.body.seeds
    assert sorted(seeds) == [42, 55, 1337, 2024, 9999]
    paths = {fv.path for fv in art_set.harness_contract.body.fixed_variables}
    assert "seeds" not in paths


def test_harness_contract_factor_carried_through() -> None:
    art_set, _ = compile_experiment_fixture("resnet50_vs_vgg16_cifar10")
    factor = art_set.harness_contract.body.factor
    assert factor.name == "architecture"
    assert factor.type == "substitutive"


def test_harness_contract_compute_defaults_to_gpu_true() -> None:
    """Fixtures without a ``compute`` block compile to ``gpu=True`` (the
    pre-Phase-1 hardcoded dispatch behavior)."""
    art_set, _ = compile_experiment_fixture("resnet50_vs_vgg16_cifar10")
    assert art_set.harness_contract.body.compute.gpu is True
    assert art_set.harness_contract.envelope.surface_map["body.compute.gpu"] == "locked"


def test_harness_contract_compute_propagates_from_controlled_conditions() -> None:
    """A ``controlled_conditions.compute.gpu: false`` declaration threads
    through the ExperimentPlan into the HarnessContract body (round-3
    friction (C) — CPU-only / macOS-LocalGpu experiments)."""
    from _emit_helpers import (  # type: ignore[import-not-found]
        load_fixture_payload,
    )
    from _verification_helpers import fixture_registries  # type: ignore[import-not-found]
    from smai_core import (
        DslDocumentAdapter,
        ExperimentDocument,
        VerifiedExperimentDefinition,
        emit_artifacts,
        verify,
    )

    payload = load_fixture_payload("cutout_on_cifar10")
    payload["experiment"]["controlled_conditions"]["compute"] = {"gpu": False}
    doc = DslDocumentAdapter.validate_python(payload, context={"smai_mode": "dsl"})
    assert isinstance(doc, ExperimentDocument)
    verified = verify(doc, fixture_registries())
    assert isinstance(verified, VerifiedExperimentDefinition)
    art_set, _report = emit_artifacts(verified, fixture_registries())
    assert art_set.experiment_plan.body.controlled_conditions.compute.gpu is False
    assert art_set.harness_contract.body.compute.gpu is False


def test_harness_contract_no_go_zones_have_v1_defaults() -> None:
    art_set, _ = compile_experiment_fixture("resnet50_vs_vgg16_cifar10")
    assert "experiment.py" in art_set.harness_contract.body.no_go_zones
    assert "techniques/__init__.py" in art_set.harness_contract.body.no_go_zones


def test_harness_contract_required_metric_matches_validation() -> None:
    art_set, _ = compile_experiment_fixture("resnet50_vs_vgg16_cifar10")
    required = art_set.harness_contract.body.required_metrics
    assert len(required) == 1
    assert required[0] == art_set.validation_config.body.metric


def test_harness_contract_fixed_variables_paths_are_dot_separated() -> None:
    art_set, _ = compile_experiment_fixture("resnet50_vs_vgg16_cifar10")
    fvs = art_set.harness_contract.body.fixed_variables
    assert any(fv.path == "dataset.name" for fv in fvs)
    assert any(fv.path == "optimization.optimizer" for fv in fvs)
    assert any(fv.path == "optimization.learning_rate" for fv in fvs)


def test_pruning_fixture_pulls_extra_field_into_fixed_variables() -> None:
    """``pruning_method: rigl`` lives at top level via ControlledConditions's extras."""
    art_set, _ = compile_experiment_fixture("pruning_sparsity_sweep")
    fvs = {fv.path: fv.value for fv in art_set.harness_contract.body.fixed_variables}
    assert fvs.get("pruning_method") == "rigl"


def test_factor_model_extras_flatten_into_fixed_variables() -> None:
    """Factor-model mixup CG holds activation_function + normalization_layer."""
    from _emit_helpers import compile_factor_model_fixture  # type: ignore[import-not-found]

    sets, _ = compile_factor_model_fixture("factor_model_resnet50_imagenet")
    mixup = sets["cg_training_augmentation"]
    fvs = {fv.path: fv.value for fv in mixup.harness_contract.body.fixed_variables}
    assert fvs.get("activation_function") == "ReLU"
    assert fvs.get("normalization_layer") == "BatchNorm"
