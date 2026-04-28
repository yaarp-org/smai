"""Factor-type-aware harness construction tests (§9)."""

from __future__ import annotations

import pytest
from smai_core import HarnessContract, TechniqueContract
from smai_runtime import (
    HarnessAPIManifest,
    HarnessExtensionPoint,
    assert_additive_baseline_has_no_required_extension_points,
    assert_substitutive_entry_provides_value,
    freeze_manifest,
    is_additive_baseline,
)


def test_additive_baseline_detected(
    additive_baseline_technique_contract: TechniqueContract,
) -> None:
    assert is_additive_baseline(additive_baseline_technique_contract) is True


def test_non_baseline_not_additive_baseline(
    additive_technique_contract: TechniqueContract,
) -> None:
    assert is_additive_baseline(additive_technique_contract) is False


def test_substitutive_entry_requires_value_pass(
    substitutive_technique_contract: TechniqueContract,
) -> None:
    assert_substitutive_entry_provides_value(substitutive_technique_contract)


def test_substitutive_entry_requires_value_fail(
    substitutive_technique_contract: TechniqueContract,
) -> None:
    """A substitutive entry with null technique_id is a methodology bug,
    but the runtime double-checks it before agents enter the workspace.
    """
    body = substitutive_technique_contract.body.model_copy(update={"technique_id": None})
    null_sub = substitutive_technique_contract.model_copy(update={"body": body})
    with pytest.raises(ValueError):
        assert_substitutive_entry_provides_value(null_sub)


def test_additive_manifest_has_no_required_extension_points(
    additive_harness_contract: HarnessContract,
    additive_manifest: HarnessAPIManifest,
) -> None:
    """The fixture additive manifest is opt-in everywhere — sanity check
    that the validator passes.
    """
    assert_additive_baseline_has_no_required_extension_points(
        additive_harness_contract, additive_manifest
    )


def test_additive_manifest_with_required_key_rejected(
    additive_harness_contract: HarnessContract,
) -> None:
    bad_manifest = freeze_manifest(
        HarnessAPIManifest(
            extension_points=[
                HarnessExtensionPoint(
                    key="model_wrapper",
                    type_signature="Callable[[nn.Module], nn.Module]",
                    purpose="x",
                    optional=False,  # required on an additive factor — illegal.
                    integration_pattern="replace",
                )
            ],
            integration_pattern_summary="bad",
            harness_version_hash="bad",
            parent_harness_contract_hash=additive_harness_contract.envelope.content_hash,
            manifest_schema_version=1,
            runtime_template_version="1.0.0",
        )
    )
    with pytest.raises(ValueError):
        assert_additive_baseline_has_no_required_extension_points(
            additive_harness_contract, bad_manifest
        )
