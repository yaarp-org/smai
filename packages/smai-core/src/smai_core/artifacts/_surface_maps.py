"""Hard-coded ``surface_map`` annotations per artifact kind.

Per ``02-dsl-and-contracts.md`` §7.3.1 / §7.4.1 / §7.5.1 / §7.6.1 + DEC-031
sub-decision #9: each artifact's per-field surface annotations live in one
canonical place (this module) and are applied at emission time.

Design choice (per the task spec's "implementation choice" guidance): the
maps are static ``dict[str, SurfaceKind]`` constants rather than derived from
Pydantic ``Field`` metadata. The canonical doc explicitly picks the
single-source-of-truth form for the maps, and the surface annotation does
not always line up with a single Pydantic field (e.g., ``body.factors[0].*``
includes nested struct paths).
"""

from __future__ import annotations

from smai_core.artifacts._envelope import SurfaceKind

EXPERIMENT_PLAN_SURFACE_MAP: dict[str, SurfaceKind] = {
    "body.hypothesis": "recorded",
    "body.factor_model_id": "locked",
    "body.factors[0].name": "locked",
    "body.factors[0].type": "locked",
    "body.factors[0].description": "recorded",
    "body.controlled_conditions.*": "locked",
    "body.entries[*].id": "locked",
    "body.entries[*].is_baseline": "locked",
    "body.entries[*].level.factor": "locked",
    "body.entries[*].level.name": "locked",
    "body.entries[*].level.technique_id": "locked",
    "body.entries[*].level.technique_params": "locked",
    "body.entries[*].level.value": "locked",
    "body.entries[*].level.description": "recorded",
    "body.validation.*": "locked",
    "body.validation.rationale": "recorded",
}
"""Per ``02-dsl-and-contracts.md`` §7.3.1."""


HARNESS_CONTRACT_SURFACE_MAP: dict[str, SurfaceKind] = {
    "body.parent_experiment_hash": "locked",
    "body.factor": "locked",
    "body.seeds": "locked",
    "body.fixed_variables[*]": "locked",
    "body.required_metrics[*]": "locked",
    "body.optional_telemetry": "locked",
    "body.no_go_zones[*]": "locked",
    "body.compute.gpu": "locked",
}
"""Per ``02-dsl-and-contracts.md`` §7.4.1.

Notably, *everything* in :class:`HarnessContract` is locked — the harness
builder agent's freedom is "how to write code that satisfies these
constraints," not "what constraints to satisfy." The runtime wiring contract
(extension-point signatures) lives in the pipeline-layer ``HarnessAPIManifest``,
not here.
"""


TECHNIQUE_CONTRACT_SURFACE_MAP: dict[str, SurfaceKind] = {
    "body.entry_id": "locked",
    "body.parent_experiment_id": "locked",
    "body.parent_experiment_hash": "locked",
    "body.parent_harness_contract_hash": "locked",
    "body.technique_id": "locked",
    "body.is_baseline": "locked",
    "body.level_value": "locked",
    "body.standard": "locked",
    "body.technique_params": "correctness_iterable",
    "body.fidelity_anchor": "recorded",
}
"""Per ``02-dsl-and-contracts.md`` §7.5.1.

This is the only artifact whose ``surface_map`` carries a
``correctness_iterable`` annotation in v1. The technique implementer agent
iterates against ``technique_params`` until validation tests pass; the
iteration is bounded by the locked surface around it.
"""


VALIDATION_CONFIG_SURFACE_MAP: dict[str, SurfaceKind] = {
    "body.parent_experiment_hash": "locked",
    "body.metric": "locked",
    "body.direction": "locked",
    "body.aggregation": "locked",
    "body.comparison": "locked",
    "body.seed_count_required": "locked",
    "body.trend_check.expected_direction": "locked",
    "body.trend_check.alert_on_violation": "locked",
    "body.rationale": "recorded",
}
"""Per ``02-dsl-and-contracts.md`` §7.6.1."""
