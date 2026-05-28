"""Behavioral tests for the ``additive`` and ``substitutive`` plugins.

Per ``designs/smai/02-dsl-and-contracts.md`` §3.2 and §3.3.

Each test exercises one branch of the plugin's check list and confirms the
expected ``ValidationError.code`` is emitted (or that the code list is
empty on the happy path).
"""

from __future__ import annotations

from smai_core import (
    AdditivePlugin,
    AggregationRule,
    AtomicMetricRef,
    ComparisonRule,
    ControlledConditions,
    Entry,
    ExperimentDefinition,
    Factor,
    Level,
    MetricRegistry,
    Registries,
    SubstitutivePlugin,
    TechniqueRef,
    ValidationCriteria,
)


def _empty_registries(techniques: dict[str, TechniqueRef] | None = None) -> Registries:
    return Registries(
        technique_registry=techniques or {},
        metric_registry=MetricRegistry(atomic={}, parametric={}),
        factor_type_plugins={},
    )


def _validation() -> ValidationCriteria:
    return ValidationCriteria(
        metric=AtomicMetricRef(ref="accuracy"),
        direction="higher_is_better",
        aggregation=AggregationRule(method="mean"),
        comparison=ComparisonRule(
            rule="compare_to_baseline", threshold=0.01, baseline_entry_id="baseline"
        ),
        seed_count_required=3,
    )


def _conditions(extra: dict[str, str | int | float | bool] | None = None) -> ControlledConditions:
    payload: dict[str, object] = {
        "dataset": {"name": "cifar10"},
        "optimization": {"lr": 0.01, "epochs": 100},
        "seeds": [1, 2, 3],
    }
    if extra:
        payload.update(extra)
    return ControlledConditions.model_validate(payload)


# ---------------------------------------------------------------------------
# Additive — happy path and per-branch failures
# ---------------------------------------------------------------------------


def test_additive_happy_path_clean() -> None:
    factor = Factor(
        name="dropout",
        type="additive",
        description="Whether dropout is used.",
    )
    entries = [
        Entry(
            id="baseline",
            is_baseline=True,
            level=Level(factor="dropout", name="absent"),
        ),
        Entry(
            id="treatment",
            is_baseline=False,
            level=Level(factor="dropout", name="present", technique_id="dropout_p05"),
        ),
    ]
    experiment = ExperimentDefinition(
        id="cg_test",
        hypothesis="Dropout helps.",
        factors=[factor],
        controlled_conditions=_conditions(),
        entries=entries,
        validation=_validation(),
    )
    errors = AdditivePlugin().validate(experiment, _empty_registries())
    assert errors == []


def test_additive_factor_too_few_entries() -> None:
    factor = Factor(name="dropout", type="additive", description="x")
    entries = [
        Entry(
            id="only",
            is_baseline=True,
            level=Level(factor="dropout", name="absent"),
        ),
    ]
    experiment = ExperimentDefinition(
        id="cg_test",
        hypothesis="x",
        factors=[factor],
        controlled_conditions=_conditions(),
        entries=entries,
        validation=_validation(),
    )
    codes = {e.code for e in AdditivePlugin().validate(experiment, _empty_registries())}
    assert "additive.factor_too_few_entries" in codes


def test_additive_baseline_count_zero() -> None:
    factor = Factor(name="dropout", type="additive", description="x")
    entries = [
        Entry(
            id="t1",
            is_baseline=False,
            level=Level(factor="dropout", name="p05", technique_id="dropout_p05"),
        ),
        Entry(
            id="t2",
            is_baseline=False,
            level=Level(factor="dropout", name="p10", technique_id="dropout_p10"),
        ),
    ]
    experiment = ExperimentDefinition(
        id="cg_test",
        hypothesis="x",
        factors=[factor],
        controlled_conditions=_conditions(),
        entries=entries,
        validation=_validation(),
    )
    codes = {e.code for e in AdditivePlugin().validate(experiment, _empty_registries())}
    assert "additive.baseline_count" in codes


def test_additive_baseline_count_two() -> None:
    factor = Factor(name="dropout", type="additive", description="x")
    entries = [
        Entry(id="b1", is_baseline=True, level=Level(factor="dropout", name="absent")),
        Entry(id="b2", is_baseline=True, level=Level(factor="dropout", name="absent")),
        Entry(
            id="t1",
            is_baseline=False,
            level=Level(factor="dropout", name="p05", technique_id="dropout_p05"),
        ),
    ]
    experiment = ExperimentDefinition(
        id="cg_test",
        hypothesis="x",
        factors=[factor],
        controlled_conditions=_conditions(),
        entries=entries,
        validation=_validation(),
    )
    codes = {e.code for e in AdditivePlugin().validate(experiment, _empty_registries())}
    assert "additive.baseline_count" in codes


def test_additive_baseline_with_non_null_technique() -> None:
    factor = Factor(name="dropout", type="additive", description="x")
    entries = [
        Entry(
            id="baseline",
            is_baseline=True,
            level=Level(factor="dropout", name="absent", technique_id="dropout_zero"),
        ),
        Entry(
            id="treatment",
            is_baseline=False,
            level=Level(factor="dropout", name="present", technique_id="dropout_p05"),
        ),
    ]
    experiment = ExperimentDefinition(
        id="cg_test",
        hypothesis="x",
        factors=[factor],
        controlled_conditions=_conditions(),
        entries=entries,
        validation=_validation(),
    )
    codes = {e.code for e in AdditivePlugin().validate(experiment, _empty_registries())}
    assert "additive.baseline_must_be_null_technique" in codes


def test_additive_treatment_with_null_technique() -> None:
    factor = Factor(name="dropout", type="additive", description="x")
    entries = [
        Entry(id="baseline", is_baseline=True, level=Level(factor="dropout", name="absent")),
        Entry(
            id="treatment",
            is_baseline=False,
            level=Level(factor="dropout", name="present"),  # technique_id null!
        ),
    ]
    experiment = ExperimentDefinition(
        id="cg_test",
        hypothesis="x",
        factors=[factor],
        controlled_conditions=_conditions(),
        entries=entries,
        validation=_validation(),
    )
    codes = {e.code for e in AdditivePlugin().validate(experiment, _empty_registries())}
    assert "additive.treatment_must_have_technique" in codes


# ---------------------------------------------------------------------------
# Substitutive — happy path and per-branch failures
# ---------------------------------------------------------------------------


def _arch_techs() -> dict[str, TechniqueRef]:
    return {
        "vgg16": TechniqueRef(
            id="vgg16",
            name="VGG-16",
            description="VGG with 16 layers.",
            category="architecture",
            compatible_factor_types=["substitutive"],
            standard=True,
            affects_extension_points=["model"],
            context_kind="standard",
        ),
        "resnet50": TechniqueRef(
            id="resnet50",
            name="ResNet-50",
            description="Residual network with 50 layers.",
            category="architecture",
            compatible_factor_types=["substitutive"],
            standard=True,
            affects_extension_points=["model"],
            context_kind="standard",
        ),
    }


def test_substitutive_happy_path_clean() -> None:
    factor = Factor(name="architecture", type="substitutive", description="x")
    entries = [
        Entry(
            id="baseline",
            is_baseline=True,
            level=Level(factor="architecture", name="vgg16", technique_id="vgg16"),
        ),
        Entry(
            id="alt",
            is_baseline=False,
            level=Level(factor="architecture", name="resnet50", technique_id="resnet50"),
        ),
    ]
    experiment = ExperimentDefinition(
        id="cg_test",
        hypothesis="x",
        factors=[factor],
        controlled_conditions=_conditions(),
        entries=entries,
        validation=_validation(),
    )
    errors = SubstitutivePlugin().validate(experiment, _empty_registries(_arch_techs()))
    assert errors == []


def test_substitutive_factor_too_few_entries() -> None:
    factor = Factor(name="architecture", type="substitutive", description="x")
    entries = [
        Entry(
            id="only",
            is_baseline=True,
            level=Level(factor="architecture", name="vgg16", technique_id="vgg16"),
        ),
    ]
    experiment = ExperimentDefinition(
        id="cg_test",
        hypothesis="x",
        factors=[factor],
        controlled_conditions=_conditions(),
        entries=entries,
        validation=_validation(),
    )
    plugin = SubstitutivePlugin()
    codes = {e.code for e in plugin.validate(experiment, _empty_registries(_arch_techs()))}
    assert "substitutive.factor_too_few_entries" in codes


def test_substitutive_baseline_count_zero() -> None:
    factor = Factor(name="architecture", type="substitutive", description="x")
    entries = [
        Entry(
            id="a",
            is_baseline=False,
            level=Level(factor="architecture", name="vgg16", technique_id="vgg16"),
        ),
        Entry(
            id="b",
            is_baseline=False,
            level=Level(factor="architecture", name="resnet50", technique_id="resnet50"),
        ),
    ]
    experiment = ExperimentDefinition(
        id="cg_test",
        hypothesis="x",
        factors=[factor],
        controlled_conditions=_conditions(),
        entries=entries,
        validation=_validation(),
    )
    plugin = SubstitutivePlugin()
    codes = {e.code for e in plugin.validate(experiment, _empty_registries(_arch_techs()))}
    assert "substitutive.baseline_count" in codes


def test_substitutive_null_technique_rejected() -> None:
    factor = Factor(name="architecture", type="substitutive", description="x")
    entries = [
        Entry(
            id="baseline",
            is_baseline=True,
            level=Level(factor="architecture", name="vgg16", technique_id="vgg16"),
        ),
        Entry(
            id="alt",
            is_baseline=False,
            level=Level(factor="architecture", name="absent"),  # null technique!
        ),
    ]
    experiment = ExperimentDefinition(
        id="cg_test",
        hypothesis="x",
        factors=[factor],
        controlled_conditions=_conditions(),
        entries=entries,
        validation=_validation(),
    )
    plugin = SubstitutivePlugin()
    codes = {e.code for e in plugin.validate(experiment, _empty_registries(_arch_techs()))}
    assert "substitutive.all_techniques_required" in codes


def test_substitutive_factor_in_controls_by_factor_name() -> None:
    """``factor.name`` appearing in controlled_conditions is contradictory."""
    factor = Factor(name="optimization", type="substitutive", description="x")
    entries = [
        Entry(
            id="baseline",
            is_baseline=True,
            level=Level(factor="optimization", name="vgg16", technique_id="vgg16"),
        ),
        Entry(
            id="alt",
            is_baseline=False,
            level=Level(factor="optimization", name="resnet50", technique_id="resnet50"),
        ),
    ]
    # ControlledConditions has ``optimization`` as a declared core field —
    # the factor name collides with it.
    experiment = ExperimentDefinition(
        id="cg_test",
        hypothesis="x",
        factors=[factor],
        controlled_conditions=_conditions(),
        entries=entries,
        validation=_validation(),
    )
    plugin = SubstitutivePlugin()
    codes = {e.code for e in plugin.validate(experiment, _empty_registries(_arch_techs()))}
    assert "substitutive.factor_not_in_controls" in codes


def test_substitutive_factor_in_controls_by_shared_category() -> None:
    """When all in-use techniques share a category and that category appears
    in controls, the plugin reports the contradiction."""
    factor = Factor(name="architecture", type="substitutive", description="x")
    entries = [
        Entry(
            id="baseline",
            is_baseline=True,
            level=Level(factor="architecture", name="vgg16", technique_id="vgg16"),
        ),
        Entry(
            id="alt",
            is_baseline=False,
            level=Level(factor="architecture", name="resnet50", technique_id="resnet50"),
        ),
    ]
    experiment = ExperimentDefinition(
        id="cg_test",
        hypothesis="x",
        factors=[factor],
        # The shared category of the entries' techniques is "architecture";
        # planting that as an extra controlled condition triggers the rule.
        controlled_conditions=_conditions(extra={"architecture": "vgg16"}),
        entries=entries,
        validation=_validation(),
    )
    plugin = SubstitutivePlugin()
    codes = {e.code for e in plugin.validate(experiment, _empty_registries(_arch_techs()))}
    assert "substitutive.factor_not_in_controls" in codes


# ---------------------------------------------------------------------------
# Behavioral asymmetry between additive and substitutive
# ---------------------------------------------------------------------------


def test_additive_admits_null_baseline_substitutive_does_not() -> None:
    """The defining asymmetry: additive baseline = null technique; substitutive
    rejects any null technique."""
    factor_add = Factor(name="dropout", type="additive", description="x")
    factor_sub = Factor(name="architecture", type="substitutive", description="x")

    null_baseline_entries = [
        Entry(id="baseline", is_baseline=True, level=Level(factor="dropout", name="absent")),
        Entry(
            id="treatment",
            is_baseline=False,
            level=Level(factor="dropout", name="p05", technique_id="dropout_p05"),
        ),
    ]
    additive_exp = ExperimentDefinition(
        id="cg_test",
        hypothesis="x",
        factors=[factor_add],
        controlled_conditions=_conditions(),
        entries=null_baseline_entries,
        validation=_validation(),
    )
    additive_codes = {e.code for e in AdditivePlugin().validate(additive_exp, _empty_registries())}
    assert "additive.baseline_must_be_null_technique" not in additive_codes
    assert "additive.treatment_must_have_technique" not in additive_codes

    # Same null-baseline shape under substitutive — rejected.
    sub_entries = [
        Entry(id="baseline", is_baseline=True, level=Level(factor="architecture", name="absent")),
        Entry(
            id="alt",
            is_baseline=False,
            level=Level(factor="architecture", name="resnet50", technique_id="resnet50"),
        ),
    ]
    sub_exp = ExperimentDefinition(
        id="cg_test",
        hypothesis="x",
        factors=[factor_sub],
        controlled_conditions=_conditions(),
        entries=sub_entries,
        validation=_validation(),
    )
    sub_plugin = SubstitutivePlugin()
    sub_codes = {e.code for e in sub_plugin.validate(sub_exp, _empty_registries(_arch_techs()))}
    assert "substitutive.all_techniques_required" in sub_codes


def test_validation_returns_list_of_validation_error() -> None:
    """Both plugins return ``list[ValidationError]`` (Protocol contract)."""
    from smai_core import ValidationError

    factor = Factor(name="dropout", type="additive", description="x")
    experiment = ExperimentDefinition(
        id="cg_test",
        hypothesis="x",
        factors=[factor],
        controlled_conditions=_conditions(),
        entries=[
            Entry(id="only", is_baseline=True, level=Level(factor="dropout", name="absent")),
        ],
        validation=_validation(),
    )
    errors = AdditivePlugin().validate(experiment, _empty_registries())
    assert isinstance(errors, list)
    assert all(isinstance(e, ValidationError) for e in errors)
