"""Integration tests for ``verify(...)`` — composition + dispatch + fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from _verification_helpers import (
    Entry,
    Factor,
    Level,
    basic_validation,
    codes,
    fixture_registries,
    fixture_technique_registry,
    force_set,
    make_experiment,
    standard_technique,
)
from smai_core import (
    DslDocumentAdapter,
    ExperimentDocument,
    FactorModelDocument,
    VerificationError,
    VerifiedExperimentDefinition,
    verify,
    verify_experiment,
    verify_to_report,
)

_FIXTURE_DIR = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "experiments"


def _load_fixture(name: str):  # type: ignore[no-untyped-def]
    return DslDocumentAdapter.validate_python(
        yaml.safe_load((_FIXTURE_DIR / name).read_text(encoding="utf-8")),
        context={"smai_mode": "dsl"},
    )


# Worked-example fixtures -------------------------------------------------


@pytest.mark.parametrize(
    "fixture_name",
    [
        "resnet50_vs_vgg16_cifar10.yaml",
        "cutout_on_cifar10.yaml",
        "pruning_sparsity_sweep.yaml",
        "position_embeddings_wikitext103.yaml",
    ],
)
def test_worked_example_cg_verifies(fixture_name: str) -> None:
    doc = _load_fixture(fixture_name)
    assert isinstance(doc, ExperimentDocument)
    verified = verify(doc, fixture_registries())
    assert isinstance(verified, VerifiedExperimentDefinition)


def test_worked_example_factor_model_verifies() -> None:
    doc = _load_fixture("factor_model_resnet50_imagenet.yaml")
    assert isinstance(doc, FactorModelDocument)
    verified = verify(doc, fixture_registries())
    assert isinstance(verified, list)
    assert len(verified) == len(doc.experiments)
    assert all(isinstance(v, VerifiedExperimentDefinition) for v in verified)


def test_worked_example_factor_model_is_advisory_clean() -> None:
    """Canonical fixtures are golden examples — they should produce no
    advisories. Downstream users copying this fixture inherit its shape, so
    cross-CG factor / controlled-condition values must match exactly (e.g.,
    ``activation_function: ReLU`` matches the baseline level name ``ReLU``
    in ``cg_activation_function``). The advisory rule's behavior is exercised
    independently in ``test_verification_category_h.py``.
    """
    doc = _load_fixture("factor_model_resnet50_imagenet.yaml")
    report = verify_to_report(doc, fixture_registries())
    assert report.success is True
    assert report.advisories == [], (
        f"canonical FactorModel fixture should be advisory-clean, got: "
        f"{[a.code for a in report.advisories]}"
    )


# verify(...) raises on errors --------------------------------------------


def test_verify_raises_on_unknown_technique() -> None:
    """Empty technique registry surfaces ``technique.id_registered`` as error."""
    doc = _load_fixture("resnet50_vs_vgg16_cifar10.yaml")
    base = fixture_registries().model_copy(update={"technique_registry": {}})
    with pytest.raises(VerificationError) as excinfo:
        verify(doc, base)
    err_codes = codes(excinfo.value.report.errors)
    assert "technique.id_registered" in err_codes


def test_verify_aggregates_multiple_findings() -> None:
    """A handcrafted broken definition raises with all errors populated."""
    techs = fixture_technique_registry()
    techs["tech_resnet50_cifar10"] = standard_technique(
        "tech_resnet50_cifar10",
        category="not_a_real_category",
        standard=False,  # also missing fidelity_anchor
        context_kind="paper_extract",
    )
    registries = fixture_registries().model_copy(update={"technique_registry": techs})
    experiment = make_experiment()
    force_set(experiment.entries[1].level, "technique_id", "tech_unknown")
    force_set(experiment.controlled_conditions, "seeds", [1, 1, 1])
    bad_validation = basic_validation(threshold=-0.5)  # validation.threshold_sign
    force_set(experiment, "validation", bad_validation)

    with pytest.raises(VerificationError) as excinfo:
        verify_experiment(experiment, registries)
    err_codes = codes(excinfo.value.report.errors)
    assert "technique.id_registered" in err_codes
    assert "validation.threshold_sign_matches_direction" in err_codes
    assert "controls.seeds_unique" in err_codes


# Severity partition -----------------------------------------------------


def test_warnings_do_not_block_success() -> None:
    """``seed_count_required=2`` triggers the warning rule but still passes."""
    experiment = make_experiment(validation=basic_validation(seed_count_required=2))
    report = verify_to_report(ExperimentDocument(experiment=experiment), fixture_registries())
    assert report.success is True
    assert "validation.seed_count_recommended_minimum" in {w.code for w in report.warnings}


# Idempotency -------------------------------------------------------------


def test_verify_idempotent_on_verified_experiment() -> None:
    experiment = make_experiment()
    verified = verify_experiment(experiment, fixture_registries())
    re_verified = verify_experiment(verified, fixture_registries())
    assert isinstance(re_verified, VerifiedExperimentDefinition)
    assert re_verified.model_dump() == verified.model_dump()


# Determinism -------------------------------------------------------------


def test_verify_deterministic_finding_order() -> None:
    """Same input + same registries → same finding order on every call."""
    techs = fixture_technique_registry()
    techs["tech_resnet50_cifar10"] = standard_technique(
        "tech_resnet50_cifar10",
        category="zzz_invalid",
        standard=False,
        context_kind="paper_extract",
    )
    registries = fixture_registries().model_copy(update={"technique_registry": techs})
    experiment = make_experiment()
    force_set(experiment.entries[1].level, "technique_id", "tech_missing")

    report_a = verify_to_report(ExperimentDocument(experiment=experiment), registries)
    report_b = verify_to_report(ExperimentDocument(experiment=experiment), registries)
    assert [f.code for f in report_a.errors] == [f.code for f in report_b.errors]
    assert [f.code for f in report_a.warnings] == [f.code for f in report_b.warnings]


# Plugin-delegation dispatch shape ---------------------------------------


def test_plugin_delegated_additive_baseline_must_be_null_technique() -> None:
    experiment = make_experiment(
        factor=Factor(name="aug", type="additive", description="x"),
        entries=[
            Entry(
                id="b",
                is_baseline=True,
                level=Level(
                    factor="aug",
                    name="absent",
                    technique_id="tech_cutout",  # additive baseline must be null
                ),
            ),
            Entry(
                id="t",
                is_baseline=False,
                level=Level(
                    factor="aug",
                    name="cutout",
                    technique_id="tech_cutout",
                    technique_params={"patch_size": 16},
                ),
            ),
        ],
    )
    with pytest.raises(VerificationError) as excinfo:
        verify_experiment(experiment, fixture_registries())
    assert "additive.baseline_must_be_null_technique" in codes(excinfo.value.report.errors)


def test_plugin_delegated_substitutive_all_techniques_required() -> None:
    experiment = make_experiment()
    force_set(experiment.entries[1].level, "technique_id", None)
    with pytest.raises(VerificationError) as excinfo:
        verify_experiment(experiment, fixture_registries())
    assert "substitutive.all_techniques_required" in codes(excinfo.value.report.errors)


# Public surface -----------------------------------------------------------


def test_public_surface_re_exports() -> None:
    """``verify``, ``verify_to_report``, ``VerificationError``,
    ``VerifiedExperimentDefinition`` resolve through ``smai_core``."""
    import smai_core

    assert smai_core.verify is verify
    assert smai_core.verify_to_report is verify_to_report
    assert smai_core.VerificationError is VerificationError
    assert smai_core.VerifiedExperimentDefinition is VerifiedExperimentDefinition
