"""Tests for §5.10 — Pipeline / sequence rejection (advisory)."""

from __future__ import annotations

from _verification_helpers import (
    Entry,
    Factor,
    Level,
    codes,
    fixture_registries,
    make_experiment,
    standard_technique,
)
from smai_core.verification.category_pipeline_rejection import (
    factor_suspected_pipeline_encoding,
    factor_suspected_sequence_encoding,
)


def _ten_levels(level_names: list[str]):  # type: ignore[no-untyped-def]
    """Build a substitutive experiment with ``len(level_names)`` entries.

    The verifier dispatches advisories on > 8 levels (per §5.10's heuristic);
    callers pass 10 names to trigger the count gate.
    """
    techs = {
        f"tech_pipeline_{i}": standard_technique(
            f"tech_pipeline_{i}", category="other", compatible=["substitutive"]
        )
        for i in range(len(level_names))
    }
    entries = [
        Entry(
            id=f"e{i}",
            is_baseline=(i == 0),
            level=Level(
                factor="recipe",
                name=level_names[i],
                technique_id=f"tech_pipeline_{i}",
            ),
        )
        for i in range(len(level_names))
    ]
    experiment = make_experiment(
        factor=Factor(name="recipe", type="substitutive", description="x"),
        entries=entries,
    )
    registries = fixture_registries().model_copy(update={"technique_registry": {**techs}})
    return experiment, registries


# factor.suspected_pipeline_encoding ---------------------------------------


def test_pipeline_advisory_does_not_fire_for_simple_names() -> None:
    experiment, registries = _ten_levels([f"recipe_{i}" for i in range(10)])
    assert factor_suspected_pipeline_encoding(experiment, registries) == []


def test_pipeline_advisory_fires_on_dot_partitioned_names() -> None:
    experiment, registries = _ten_levels(
        [
            "stage1.stage2",
            "stageA.stageB",
            "rec3",
            "rec4",
            "rec5",
            "rec6",
            "rec7",
            "rec8",
            "rec9",
            "rec10",
        ]
    )
    findings = factor_suspected_pipeline_encoding(experiment, registries)
    assert "factor.suspected_pipeline_encoding" in codes(findings)
    assert all(f.severity == "advisory" for f in findings)


# factor.suspected_sequence_encoding ---------------------------------------


def test_sequence_advisory_does_not_fire_for_simple_names() -> None:
    experiment = make_experiment()
    assert factor_suspected_sequence_encoding(experiment, fixture_registries()) == []


def test_sequence_advisory_fires_on_then_marker() -> None:
    experiment, registries = _ten_levels(
        ["pretrain_then_finetune"] + [f"rec_{i}" for i in range(9)]
    )
    findings = factor_suspected_sequence_encoding(experiment, registries)
    assert "factor.suspected_sequence_encoding" in codes(findings)
    assert all(f.severity == "advisory" for f in findings)
