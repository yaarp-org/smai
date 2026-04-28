"""Tests for canonical JSON serialization, hashing, and ``freeze_with_hash``.

Per ``02-dsl-and-contracts.md`` §8.2.
"""

from __future__ import annotations

from typing import Any

from smai_core import (
    AtomicMetricRef,
    ParametricMetricRef,
    artifact_hash,
    canonical_json,
    metric_ref_to_runtime_key,
)
from smai_core.artifacts._canonical import (
    freeze_with_hash,
    hash_registry_dict,
    hash_string_set,
)


def test_canonical_json_sorts_keys() -> None:
    a: dict[str, Any] = {"b": 1, "a": 2}
    b: dict[str, Any] = {"a": 2, "b": 1}
    assert canonical_json(a) == canonical_json(b)


def test_canonical_json_uses_tight_separators() -> None:
    raw = canonical_json({"a": 1, "b": [1, 2]})
    assert b" " not in raw
    assert b"\n" not in raw


def test_canonical_json_handles_pydantic_models() -> None:
    ref = AtomicMetricRef(ref="accuracy")
    raw = canonical_json(ref)
    assert b'"kind":"atomic"' in raw
    assert b'"ref":"accuracy"' in raw


def test_artifact_hash_is_64_hex_chars() -> None:
    digest = artifact_hash({"a": 1})
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_artifact_hash_stable_across_dict_order() -> None:
    a = artifact_hash({"alpha": 1, "beta": 2})
    b = artifact_hash({"beta": 2, "alpha": 1})
    assert a == b


def test_hash_registry_dict_stable() -> None:
    a = hash_registry_dict({"alpha": {"x": 1}, "beta": {"y": 2}})
    b = hash_registry_dict({"beta": {"y": 2}, "alpha": {"x": 1}})
    assert a == b


def test_hash_string_set_sorts_input() -> None:
    a = hash_string_set(["b", "a"])
    b = hash_string_set(["a", "b"])
    assert a == b


def test_freeze_with_hash_populates_content_hash() -> None:
    from smai_core import (
        ControlledConditions,
        Entry,
        Factor,
        Level,
        ValidationCriteria,
    )
    from smai_core.artifacts._envelope import ArtifactEnvelope
    from smai_core.artifacts.experiment_plan import ExperimentPlan, ExperimentPlanBody

    plan = ExperimentPlan(
        envelope=ArtifactEnvelope(
            artifact_kind="experiment_plan",
            artifact_id="x",
            schema_version=1,
            compiler_version="0.1.0",
            parent_experiment_id="x",
            registry_hashes={"factor_types": "a", "metrics": "b", "techniques": "c"},
            surface_map={"body.hypothesis": "recorded"},
        ),
        body=ExperimentPlanBody(
            hypothesis="x",
            factors=[Factor(name="architecture", type="substitutive", description="d")],
            controlled_conditions=ControlledConditions(
                dataset={"name": "cifar10"},
                optimization={"optimizer": "adamw"},
                seeds=[1, 2, 3],
            ),
            entries=[
                Entry(
                    id="b",
                    is_baseline=True,
                    level=Level(factor="architecture", name="VGG", technique_id="t"),
                ),
                Entry(
                    id="t",
                    is_baseline=False,
                    level=Level(factor="architecture", name="ResNet", technique_id="t2"),
                ),
            ],
            validation=ValidationCriteria.model_validate(
                {
                    "metric": {"kind": "atomic", "ref": "accuracy"},
                    "direction": "higher_is_better",
                    "aggregation": {"method": "mean"},
                    "comparison": {
                        "rule": "compare_to_baseline",
                        "threshold": 0.01,
                        "baseline_entry_id": "b",
                    },
                    "seed_count_required": 3,
                }
            ),
        ),
    )
    assert plan.envelope.content_hash == ""
    frozen = freeze_with_hash(plan)
    assert frozen.envelope.content_hash != ""
    assert len(frozen.envelope.content_hash) == 64
    # Re-freezing produces the same hash (idempotency for equivalent input,
    # using the post-freeze artifact's body).
    refrozen = freeze_with_hash(frozen)
    assert refrozen.envelope.content_hash == frozen.envelope.content_hash


def test_metric_ref_to_runtime_key_atomic() -> None:
    assert metric_ref_to_runtime_key(AtomicMetricRef(ref="accuracy")) == "accuracy"


def test_metric_ref_to_runtime_key_parametric_single() -> None:
    ref = ParametricMetricRef(family="top_k_accuracy", parameters={"k": 5})
    assert metric_ref_to_runtime_key(ref) == "top_k_accuracy__k=5"


def test_metric_ref_to_runtime_key_parametric_multi_sorted() -> None:
    ref = ParametricMetricRef(
        family="min_traj_error_at_k",
        parameters={"k": 6, "error_kind": "minADE"},
    )
    # Keys sorted lexicographically: error_kind < k
    assert metric_ref_to_runtime_key(ref) == "min_traj_error_at_k__error_kind=minADE__k=6"


def test_metric_ref_to_runtime_key_parameter_order_invariant() -> None:
    a = ParametricMetricRef(
        family="f",
        parameters={"a": 1, "b": 2},
    )
    b = ParametricMetricRef(
        family="f",
        parameters={"b": 2, "a": 1},
    )
    assert metric_ref_to_runtime_key(a) == metric_ref_to_runtime_key(b)
