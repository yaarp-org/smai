"""Factor-type-aware harness construction helpers.

Per ``10-runtime-and-templates.md`` §9. The harness's extension-point shape
varies by ``factor_type``: ``additive`` factors default to identity / no-op
(so the baseline runs the harness as-is); ``substitutive`` factors require
a value at every entry.

This module exposes static helpers that the harness builder agent (and the
fixed template's pre-flight check) can call to validate that an
``ExperimentDefinition`` × ``HarnessAPIManifest`` pair is structurally
coherent against the loaded ``smai_core.factor_types.FactorTypePlugin``.
"""

from __future__ import annotations

from smai_core import HarnessContract, TechniqueContract

from smai_runtime.manifest import HarnessAPIManifest


def is_additive_baseline(technique_contract: TechniqueContract) -> bool:
    """True iff the entry is the additive-factor baseline (§9.1).

    Per DEC-013 / ``01-data-model.md`` §4.6, the additive baseline carries
    ``technique_id == None`` and ``is_baseline == True``. That entry's
    ``apply()`` returns an empty dict and the integrator splices nothing —
    the harness runs as-is.
    """
    body = technique_contract.body
    return body.is_baseline and body.technique_id is None


def assert_substitutive_entry_provides_value(
    technique_contract: TechniqueContract,
) -> None:
    """Raise ``ValueError`` if a substitutive entry has a null ``technique_id`` (§9.2).

    For substitutive factors every entry — including the baseline — must
    populate the slot. The methodology layer's verification (Task 1.5)
    already enforces this; this helper is the runtime-side belt-and-braces
    check before agents enter the workspace.
    """
    if technique_contract.body.technique_id is None:
        raise ValueError(
            "substitutive entry must provide a non-null technique_id; "
            f"entry_id={technique_contract.body.entry_id!r} is null."
        )


def assert_additive_baseline_has_no_required_extension_points(
    harness_contract: HarnessContract,
    manifest: HarnessAPIManifest,
) -> None:
    """Sanity check: an additive factor's manifest should not declare
    ``optional=False`` extension points (§9.1).

    Per §9.1, additive factors expose only opt-in extension points — the
    harness must run as-is when the additive baseline returns ``{}``. A
    manifest declaring a required (``optional=False``) extension point on
    an additive factor is internally inconsistent: the additive baseline
    cannot satisfy a required key.
    """
    factor_type = harness_contract.body.factor.type
    if factor_type != "additive":
        return
    required = [ep.key for ep in manifest.extension_points if not ep.optional]
    if required:
        raise ValueError(
            f"additive factor manifest declares required extension points "
            f"{sorted(required)}; additive baselines must run with no "
            f"contribution and therefore cannot satisfy required keys."
        )
