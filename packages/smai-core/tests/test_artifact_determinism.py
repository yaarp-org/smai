"""Determinism tests for the contract emission pipeline.

* Golden-hash regression: each fixture's artifact-set hashes are pinned.
* Property-based: shuffling equivalent fields in the input does not change
  artifact hashes (per ``02-dsl-and-contracts.md`` §8.2).
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from _emit_helpers import (  # type: ignore[import-not-found]
    EXPERIMENT_FIXTURES,
    FACTOR_MODEL_FIXTURES,
    compile_experiment_fixture,
    compile_factor_model_fixture,
    load_fixture_payload,
)
from _verification_helpers import fixture_registries  # type: ignore[import-not-found]
from hypothesis import given, settings
from hypothesis import strategies as st
from smai_core import (
    DslDocumentAdapter,
    ExperimentDocument,
    Registries,
    emit_artifacts,
    verify,
)
from smai_core.artifacts._set import ContractArtifactSet

# Pinned hashes for the five worked-example fixtures. If any of these change,
# the canonical-form serialization or the emission pipeline has shifted —
# investigate before updating the constants.
GOLDEN_EXPERIMENT_HASHES: dict[str, dict[str, str]] = {
    "resnet50_vs_vgg16_cifar10": {
        "experiment_plan": ("cd2d9c025649b0cab7cdd5d1aa1356fe82ef0e259170e882908bc6541af120ab"),
        "harness_contract": ("83aaadc6c9e564bdea54f68b55931441a76b768bd6b5dbf8304d71de938013e5"),
        "validation_config": ("f177f28b006524dc9b82513f7e75dee77784253354354423ba5ea8468144b84b"),
    },
    "cutout_on_cifar10": {
        "experiment_plan": ("8f1458ce4c0261f221330735d4977aacf5e42db9b0b55befe6b886595e82374b"),
        "harness_contract": ("d96964620e409c1c256d6a6b9f14b092a628f68730975775c15be3de8962cd5d"),
        "validation_config": ("9cb5cbe9361f63091ee92a002a572df48751192fc50910314c6ebcd3d3f66017"),
    },
    "pruning_sparsity_sweep": {
        "experiment_plan": ("b05731478ffb6b6508bf01e82ab0944d8abcdc558701d8057a5337dd6aa18fb0"),
        "harness_contract": ("489ff675c06c21b278356147ce6c7fd12e612e68f676bc329bd7f653eac2f46a"),
        "validation_config": ("869f0757ed9a4c4e1f2c0a49ca022f7e209eb5231e475865a2e9e7805919701a"),
    },
    "position_embeddings_wikitext103": {
        "experiment_plan": ("985a2c2f0dee67577c6ea25e8e6ffd20a55772d0962a5c3a51c00c1c9059a53e"),
        "harness_contract": ("422092a93e4536a2b56751e55ddb03d65e85d6772c17faee7af1fcc156f700b0"),
        "validation_config": ("378e43ea3c2edfd69ea77950a66f2522dc3e9146a6950bab7b20780d10600e06"),
    },
}


def _set_hashes(art_set: ContractArtifactSet) -> dict[str, str]:
    return {
        "experiment_plan": art_set.experiment_plan.envelope.content_hash,
        "harness_contract": art_set.harness_contract.envelope.content_hash,
        "validation_config": art_set.validation_config.envelope.content_hash,
    }


@pytest.mark.parametrize("name", EXPERIMENT_FIXTURES)
def test_emission_is_deterministic_within_run(name: str) -> None:
    """Compiling the same fixture twice produces byte-equal artifact hashes."""
    a, _ = compile_experiment_fixture(name)
    b, _ = compile_experiment_fixture(name)
    assert _set_hashes(a) == _set_hashes(b)
    for ta, tb in zip(a.technique_contracts, b.technique_contracts, strict=True):
        assert ta.envelope.content_hash == tb.envelope.content_hash


@pytest.mark.parametrize("name", EXPERIMENT_FIXTURES)
def test_golden_hashes_pinned(name: str) -> None:
    """Each fixture's hashes are documented constants; mismatch flags drift."""
    art_set, _ = compile_experiment_fixture(name)
    hashes = _set_hashes(art_set)
    expected = GOLDEN_EXPERIMENT_HASHES[name]
    for kind, digest in hashes.items():
        assert digest == expected[kind], (
            f"{name}.{kind} hash drift: expected {expected[kind]} got {digest}"
        )


GOLDEN_TECHNIQUE_HASHES: dict[str, dict[str, str]] = {
    "resnet50_vs_vgg16_cifar10": {
        "resnet50_treatment": ("ee793b9c69660bc179783e2fdfd33c251675a04eebafeb92ca3416db92c12d43"),
        "vgg16_reference": ("c8cdf2bf7d1aaf4f2b3ac12d33491f4525cf6c5434ef8c99ba5b61c2da1e488a"),
    },
    "cutout_on_cifar10": {
        "cutout_treatment": ("3f24ce31aaa781656e706d4cfb1b8c0a54693944e5e2722fd52c8d7ae4428b0b"),
        "no_aug_baseline": ("7965901065ce229aade1d13ad081ad9331c0a0bac35f44c0b9081f7a66967e8c"),
    },
    "pruning_sparsity_sweep": {
        "dense_baseline": ("e1f9812d4fab30b5a6560d71bd1cbc9789a11373c7a996588f50866eba9035b1"),
        "sparsity_50": ("ac3b9eb6dc07236fdd58703413d542b75294f335123d5118e56c2d54096dcff8"),
        "sparsity_70": ("ff3b003db1ca9db6448aac517c9b4ef0391fa6c15480e0e444981f4b7388e11b"),
        "sparsity_90": ("c836807a59c018abe2e7e4af97e20c0650ca96d13d173531ffcda82fad4969a5"),
        "sparsity_99": ("68f0863f0e5de860cb1c68dad8438d0eb984e3d7919aa9e2c9ca17a91a3854a1"),
    },
    "position_embeddings_wikitext103": {
        "absolute_pe": ("769f23404d60e9950deed69ea9d8d560eb9b5d883f15836c45d945e1e17e7776"),
        "alibi": ("b2ca416a40d5915163dc45c0eba377d67ae48f12571e45d2bae77646bac3fb62"),
        "no_pe_reference": ("c5168e9d8e31759de9b5cdda8b9e24a1ecd9c83e007e08d62e397243a9fa00c7"),
        "rope": ("99812dffe33e5175cc86abbd744443662c1ea41f1b1c0537537416959a7b47c9"),
    },
}


@pytest.mark.parametrize("name", EXPERIMENT_FIXTURES)
def test_technique_contract_hashes_pinned(name: str) -> None:
    art_set, _ = compile_experiment_fixture(name)
    expected = GOLDEN_TECHNIQUE_HASHES[name]
    actual = {tc.body.entry_id: tc.envelope.content_hash for tc in art_set.technique_contracts}
    assert actual == expected


@pytest.mark.parametrize("name", FACTOR_MODEL_FIXTURES)
def test_factor_model_emission_deterministic_within_run(name: str) -> None:
    a, _ = compile_factor_model_fixture(name)
    b, _ = compile_factor_model_fixture(name)
    assert {k: _set_hashes(v) for k, v in a.items()} == {k: _set_hashes(v) for k, v in b.items()}


# ---------------------------------------------------------------------------
# Property-based: shuffling equivalent fields does not change hashes
# ---------------------------------------------------------------------------


def _baseline_resnet_payload() -> dict[str, Any]:
    return load_fixture_payload("resnet50_vs_vgg16_cifar10")


def _compile_payload(payload: dict[str, Any], registries: Registries) -> ContractArtifactSet:
    doc = DslDocumentAdapter.validate_python(payload, context={"smai_mode": "dsl"})
    assert isinstance(doc, ExperimentDocument)
    verified = verify(doc, registries)
    art_set, _ = emit_artifacts(verified, registries)
    return art_set


@given(seed=st.integers(min_value=0, max_value=10**6))
@settings(max_examples=20, deadline=None)
def test_shuffling_entry_order_does_not_change_hashes(seed: int) -> None:
    """Per §8.2: entries with distinct ids are set-semantic — order-invariant."""
    import random

    registries = fixture_registries()
    rng = random.Random(seed)
    payload = _baseline_resnet_payload()
    entries = list(cast(list[dict[str, Any]], payload["experiment"]["entries"]))
    rng.shuffle(entries)
    payload["experiment"]["entries"] = entries
    a = _compile_payload(payload, registries)

    payload2 = _baseline_resnet_payload()  # canonical fixture order
    b = _compile_payload(payload2, registries)
    assert _set_hashes(a) == _set_hashes(b)


@given(seed=st.integers(min_value=0, max_value=10**6))
@settings(max_examples=20, deadline=None)
def test_shuffling_optimization_dict_keys_does_not_change_hashes(seed: int) -> None:
    """Per §8.2: object keys are sorted lexicographically before hashing."""
    import random

    registries = fixture_registries()
    rng = random.Random(seed)
    payload = _baseline_resnet_payload()
    opt = payload["experiment"]["controlled_conditions"]["optimization"]
    items = list(opt.items())
    rng.shuffle(items)
    payload["experiment"]["controlled_conditions"]["optimization"] = dict(items)
    a = _compile_payload(payload, registries)

    payload2 = _baseline_resnet_payload()
    b = _compile_payload(payload2, registries)
    assert _set_hashes(a) == _set_hashes(b)


@given(seed=st.integers(min_value=0, max_value=10**6))
@settings(max_examples=10, deadline=None)
def test_shuffling_user_optional_telemetry_order_does_not_change_hashes(seed: int) -> None:
    """User-declared ``optional_telemetry`` is a set-semantic field (§6.6)."""
    import random

    registries = fixture_registries()
    rng = random.Random(seed)
    payload = _baseline_resnet_payload()
    payload["experiment"]["validation"]["optional_telemetry"] = [
        {"kind": "atomic", "ref": "perplexity"},
        {"kind": "atomic", "ref": "params"},  # also a cost-tag dup
    ]
    a = _compile_payload(payload, registries)

    payload2 = _baseline_resnet_payload()
    payload2["experiment"]["validation"]["optional_telemetry"] = [
        {"kind": "atomic", "ref": "params"},
        {"kind": "atomic", "ref": "perplexity"},
    ]
    rng.shuffle(payload2["experiment"]["validation"]["optional_telemetry"])
    b = _compile_payload(payload2, registries)
    assert _set_hashes(a) == _set_hashes(b)
