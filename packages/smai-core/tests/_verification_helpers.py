"""Shared helpers for verification (Task 1.5) tests.

A central ``make_experiment`` builder constructs a minimal valid
``ExperimentDefinition`` with kwargs to inject specific violations. Keeps
per-rule tests short and uniform.
"""

from __future__ import annotations

from typing import Any

from smai_core import (
    AggregationRule,
    AtomicMetricRef,
    ComparisonRule,
    ControlledConditions,
    Entry,
    ExperimentDefinition,
    Factor,
    FactorModel,
    Level,
    MetricRef,
    NumericValue,
    PaperFidelityAnchor,
    Registries,
    TechniqueParams,
    TechniqueRef,
    TrendCheck,
    ValidationCriteria,
    load_default_registries,
)


def standard_technique(
    tech_id: str,
    *,
    category: str = "architecture",
    compatible: list[str] | None = None,
    standard: bool = True,
    parameter_schema: dict[str, Any] | None = None,
    implies_controlled: list[str] | None = None,
    affects_extension_points: list[str] | None = None,
    fidelity_anchor: object | None = None,
) -> TechniqueRef:
    """Build a ``TechniqueRef`` with sensible defaults for tests."""
    return TechniqueRef(
        id=tech_id,
        name=tech_id.replace("tech_", ""),
        description=f"Test technique {tech_id}.",
        category=category,
        compatible_factor_types=compatible or ["substitutive"],  # type: ignore[arg-type]
        standard=standard,
        fidelity_anchor=fidelity_anchor,  # type: ignore[arg-type]
        affects_extension_points=affects_extension_points or ["model"],
        implies_controlled=implies_controlled or [],
        parameter_schema=parameter_schema,
    )


def fixture_technique_registry() -> dict[str, TechniqueRef]:
    """Technique registry covering every fixture reference under
    ``tests/fixtures/experiments/``.

    All techniques are ``standard=True`` so the
    ``technique.fidelity_anchor_present_or_standard`` rule passes by default;
    individual tests that exercise non-standard / paper-anchored techniques
    construct their own.
    """
    return {
        # resnet50_vs_vgg16_cifar10.yaml
        "tech_vgg16_cifar10": standard_technique(
            "tech_vgg16_cifar10", category="architecture", implies_controlled=["dataset"]
        ),
        "tech_resnet50_cifar10": standard_technique(
            "tech_resnet50_cifar10", category="architecture", implies_controlled=["dataset"]
        ),
        # cutout_on_cifar10.yaml — additive
        "tech_cutout": standard_technique(
            "tech_cutout",
            category="augmentation",
            compatible=["additive"],
            implies_controlled=["architecture"],
            parameter_schema={
                "type": "object",
                "properties": {"patch_size": {"type": "integer", "minimum": 1}},
                "required": ["patch_size"],
                "additionalProperties": False,
            },
        ),
        # pruning_sparsity_sweep.yaml — additive, parametric
        "tech_rigl": standard_technique(
            "tech_rigl",
            category="pruning",
            compatible=["additive"],
            implies_controlled=["architecture"],
            parameter_schema={
                "type": "object",
                "properties": {"sparsity": {"type": "number", "minimum": 0.0, "maximum": 1.0}},
                "required": ["sparsity"],
                "additionalProperties": False,
            },
        ),
        # position_embeddings_wikitext103.yaml — substitutive heterogeneous
        "tech_no_position_embedding": standard_technique(
            "tech_no_position_embedding", category="position_embedding"
        ),
        "tech_absolute_pe": standard_technique("tech_absolute_pe", category="position_embedding"),
        "tech_rope": standard_technique("tech_rope", category="position_embedding"),
        "tech_alibi": standard_technique("tech_alibi", category="position_embedding"),
        # factor_model_resnet50_imagenet.yaml
        "tech_relu": standard_technique("tech_relu", category="activation"),
        "tech_gelu": standard_technique("tech_gelu", category="activation"),
        "tech_silu": standard_technique("tech_silu", category="activation"),
        "tech_batchnorm": standard_technique("tech_batchnorm", category="normalization"),
        "tech_layernorm": standard_technique("tech_layernorm", category="normalization"),
        "tech_groupnorm": standard_technique("tech_groupnorm", category="normalization"),
        "tech_mixup": standard_technique(
            "tech_mixup",
            category="augmentation",
            compatible=["additive"],
            parameter_schema={
                "type": "object",
                "properties": {"alpha": {"type": "number"}},
                "required": ["alpha"],
                "additionalProperties": False,
            },
        ),
    }


def fixture_registries() -> Registries:
    """Default registries plus the fixture technique registry seeded in."""
    base = load_default_registries()
    return base.model_copy(update={"technique_registry": fixture_technique_registry()})


def basic_validation(
    *,
    metric: MetricRef | None = None,
    direction: str = "higher_is_better",
    threshold: float = 0.01,
    seed_count_required: int = 3,
    optional_telemetry: list[MetricRef] | None = None,
    trend_check: TrendCheck | None = None,
    rule: str = "compare_to_baseline",
    target_value: float | None = None,
) -> ValidationCriteria:
    """Build a ValidationCriteria with DSL-mode context so
    ``ComparisonRule.baseline_entry_id`` is allowed to remain unfilled —
    Pass-2 verification (Task 1.5) runs on DSL input, before
    compile-emit fills compiler-fill fields per §2.5."""
    payload: dict[str, Any] = {
        "metric": (metric or AtomicMetricRef(ref="accuracy")).model_dump(),
        "direction": direction,
        "aggregation": {"method": "mean"},
        "comparison": {
            "rule": rule,
            "threshold": threshold,
            "target_value": target_value,
        },
        "seed_count_required": seed_count_required,
        "optional_telemetry": (
            None if optional_telemetry is None else [r.model_dump() for r in optional_telemetry]
        ),
        "trend_check": None if trend_check is None else trend_check.model_dump(),
    }
    return ValidationCriteria.model_validate(payload, context={"smai_mode": "dsl"})


def basic_conditions(
    *,
    seeds: list[int] | None = None,
    extra: dict[str, Any] | None = None,
) -> ControlledConditions:
    payload: dict[str, Any] = {
        "dataset": {"name": "cifar10", "split": "standard"},
        "optimization": {"optimizer": "adamw", "lr": 0.001, "epochs": 100},
        "seeds": seeds if seeds is not None else [1, 2, 3, 4, 5],
    }
    if extra:
        payload.update(extra)
    return ControlledConditions.model_validate(payload)


def make_experiment(
    *,
    cg_id: str = "cg_test",
    hypothesis: str = "Test hypothesis.",
    factor: Factor | None = None,
    entries: list[Entry] | None = None,
    controlled_conditions: ControlledConditions | None = None,
    validation: ValidationCriteria | None = None,
    factor_model_id: str | None = None,
) -> ExperimentDefinition:
    """Construct a minimally valid ExperimentDefinition for tests.

    Default shape: substitutive ``architecture`` factor with two entries
    (``vgg16`` baseline, ``resnet50`` treatment), matching standard CIFAR-10
    controlled conditions. Tests inject specific violations via overrides.
    """
    if factor is None:
        factor = Factor(
            name="architecture",
            type="substitutive",
            description="Image classifier backbone.",
        )
    if entries is None:
        entries = [
            Entry(
                id="vgg16_baseline",
                is_baseline=True,
                level=Level(
                    factor="architecture",
                    name="VGG-16",
                    technique_id="tech_vgg16_cifar10",
                ),
            ),
            Entry(
                id="resnet50_treatment",
                is_baseline=False,
                level=Level(
                    factor="architecture",
                    name="ResNet-50",
                    technique_id="tech_resnet50_cifar10",
                ),
            ),
        ]
    if controlled_conditions is None:
        controlled_conditions = basic_conditions()
    if validation is None:
        validation = basic_validation()
    return ExperimentDefinition(
        id=cg_id,
        hypothesis=hypothesis,
        factor_model_id=factor_model_id,
        factors=[factor],
        controlled_conditions=controlled_conditions,
        entries=entries,
        validation=validation,
    )


def codes(findings: list[Any]) -> set[str]:
    return {f.code for f in findings}


def force_set(obj: object, attr: str, value: object) -> None:
    """Bypass Pydantic's typecheck for negative-path tests.

    Pyright treats ``BaseModel.__dict__`` as read-only ``MappingProxyType``;
    Pydantic v2's ``validate_assignment=False`` default means ``setattr`` is a
    direct write through ``object.__setattr__``. This helper centralizes that
    pattern so tests don't litter ``# type: ignore`` annotations and keeps
    pyright in strict mode for the source tree.
    """
    object.__setattr__(obj, attr, value)


__all__ = [
    "AggregationRule",
    "AtomicMetricRef",
    "ComparisonRule",
    "ControlledConditions",
    "Entry",
    "ExperimentDefinition",
    "Factor",
    "FactorModel",
    "Level",
    "MetricRef",
    "NumericValue",
    "PaperFidelityAnchor",
    "Registries",
    "TechniqueParams",
    "TechniqueRef",
    "TrendCheck",
    "ValidationCriteria",
    "basic_conditions",
    "basic_validation",
    "codes",
    "fixture_registries",
    "fixture_technique_registry",
    "force_set",
    "make_experiment",
    "standard_technique",
]
