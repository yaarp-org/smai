"""Per-rule tests for Category G — Continuous/ordinal value rules (§5.8)."""

from __future__ import annotations

from _verification_helpers import (
    Entry,
    Factor,
    Level,
    NumericValue,
    codes,
    fixture_registries,
    force_set,
    make_experiment,
)
from smai_core.verification.category_g_continuous_ordinal_values import (
    value_in_declared_range,
    value_kind_consistency_within_factor,
    value_technique_params_alignment,
    value_unit_consistency_within_factor,
)


def _additive_with_values(*values: NumericValue):  # type: ignore[no-untyped-def]
    """Helper: build an additive ``pruning_sparsity`` experiment with the
    given continuous/ordinal level values on each treatment entry."""
    entries = [
        Entry(
            id="b",
            is_baseline=True,
            level=Level(factor="pruning_sparsity", name="dense"),
        ),
    ]
    for idx, value in enumerate(values):
        entries.append(
            Entry(
                id=f"t{idx}",
                is_baseline=False,
                level=Level(
                    factor="pruning_sparsity",
                    name=f"sparsity_{idx}",
                    technique_id="tech_rigl",
                    technique_params={"sparsity": value.value},
                    value=value,
                ),
            )
        )
    return make_experiment(
        factor=Factor(name="pruning_sparsity", type="additive", description="x"),
        entries=entries,
    )


# value.in_declared_range ---------------------------------------------------


def test_in_range_passes_when_within_bounds() -> None:
    experiment = _additive_with_values(
        NumericValue(value=0.5, kind="continuous", min=0.0, max=1.0),
    )
    assert value_in_declared_range(experiment, fixture_registries()) == []


def test_in_range_fails_when_above_max() -> None:
    experiment = _additive_with_values(
        NumericValue(value=1.5, kind="continuous", min=0.0, max=1.0),
    )
    assert "value.in_declared_range" in codes(
        value_in_declared_range(experiment, fixture_registries())
    )


# value.kind_consistency_within_factor -------------------------------------


def test_kind_consistency_passes_for_uniform_continuous() -> None:
    experiment = _additive_with_values(
        NumericValue(value=0.3, kind="continuous"),
        NumericValue(value=0.5, kind="continuous"),
    )
    assert value_kind_consistency_within_factor(experiment, fixture_registries()) == []


def test_kind_consistency_fails_when_mixing_kinds() -> None:
    experiment = _additive_with_values(
        NumericValue(value=0.3, kind="continuous"),
        NumericValue(value=1, kind="ordinal"),
    )
    assert "value.kind_consistency_within_factor" in codes(
        value_kind_consistency_within_factor(experiment, fixture_registries())
    )


# value.unit_consistency_within_factor -------------------------------------


def test_unit_consistency_passes_when_matching() -> None:
    experiment = _additive_with_values(
        NumericValue(value=0.3, kind="continuous", unit="fraction"),
        NumericValue(value=0.5, kind="continuous", unit="fraction"),
    )
    assert value_unit_consistency_within_factor(experiment, fixture_registries()) == []


def test_unit_consistency_warns_when_mixed() -> None:
    experiment = _additive_with_values(
        NumericValue(value=0.3, kind="continuous", unit="fraction"),
        NumericValue(value=50, kind="continuous", unit="percent"),
    )
    findings = value_unit_consistency_within_factor(experiment, fixture_registries())
    assert "value.unit_consistency_within_factor" in codes(findings)
    assert all(f.severity == "warning" for f in findings)


# value.technique_params_alignment -----------------------------------------


def test_alignment_passes_when_value_matches_param() -> None:
    experiment = _additive_with_values(
        NumericValue(value=0.5, kind="continuous"),
    )
    # ``tech_rigl`` has parameter_schema with single numeric param ``sparsity``
    # whose value matches level.value.value.
    assert value_technique_params_alignment(experiment, fixture_registries()) == []


def test_alignment_advisories_on_mismatch() -> None:
    experiment = _additive_with_values(
        NumericValue(value=0.5, kind="continuous"),
    )
    # Patch the technique_params to disagree with the level value.
    force_set(experiment.entries[1].level, "technique_params", {"sparsity": 0.9})
    findings = value_technique_params_alignment(experiment, fixture_registries())
    assert "value.technique_params_alignment" in codes(findings)
    assert all(f.severity == "advisory" for f in findings)
