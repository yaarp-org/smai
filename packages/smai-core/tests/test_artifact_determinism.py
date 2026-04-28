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
        "experiment_plan": ("b4f98fcf1ee8081072ec02b6393cbb77c1eaa5a1a0a513e5c9e35cad185a0fd0"),
        "harness_contract": ("c9c71723808a7109e7def6c29fca663e40d386884bf6b4fdbe4bc194d1f97b06"),
        "validation_config": ("1d2c17d6bc079bb2ba982c9bbc93ba82bed412249076c95fe3520eb0c81bedc8"),
    },
    "cutout_on_cifar10": {
        "experiment_plan": ("f5774de59565381f7728bc92496a6d4f332728b2b3ab7fda3cbd44819c554294"),
        "harness_contract": ("765b3b240bf31d14020c08ba16dd7743a8e4c07236d4df276562cb99ec389350"),
        "validation_config": ("700548b32e1563ed6da0325c03146cf4a64e9196dce55ee584aed3fc175174bb"),
    },
    "pruning_sparsity_sweep": {
        "experiment_plan": ("4c8fa861cde9b07655e900e5869c07b441dccfc272754f43b526cd218658dbb7"),
        "harness_contract": ("6cb8950b82f74a65857eb40400c3121b3f5d4db0020124f589197760d5b91638"),
        "validation_config": ("f5b6110b29011569dc3115f03474ffdcc18703dec5e91aebc373adb9fa386ce2"),
    },
    "position_embeddings_wikitext103": {
        "experiment_plan": ("46caab1213b177dfc2ddf9d8d5173e9a1c31de5eb73eab552214fc638f5b05a3"),
        "harness_contract": ("615ba975214a07cacc148cb35b1bd58c4b120f99221fe4bfd7aa98dda04a519d"),
        "validation_config": ("e542785450f4d8ca315c60a9016a92ae51e9259be8436e4582c92c5a907fb810"),
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
        "resnet50_treatment": ("4e5f008e462e7056069df6d77937387eb91264169563aace03d52efc2505fedd"),
        "vgg16_reference": ("629ee6c9711d095cde878ad479757827cb6ac597bccb6e6a522eeb6ac2e6bfdd"),
    },
    "cutout_on_cifar10": {
        "cutout_treatment": ("a3d5786259d634514f6c40c0ab4ff17879d3ca8ddf5426eb1e9b747c29c556c4"),
        "no_aug_baseline": ("e923e9d12e884a48139b966bd2b42f45a6a8964602d5d56abb886070beb831eb"),
    },
    "pruning_sparsity_sweep": {
        "dense_baseline": ("c6695c3e2c381ddef432111309786e8ef1e9591188e650f28277bdcc689b49af"),
        "sparsity_50": ("3e837b7f8e2cc9db4a16f65d33992d3d421df8b403c100f79bac8eb6ad64e214"),
        "sparsity_70": ("ab9fa7777e8f5f99fcbc43d12cf65f7d1facc303f1ca8cde6d2d77daa7a789e9"),
        "sparsity_90": ("5e4782e9a72f293e5a28b264ce1c69b3ef64660ba5a2bbf05332fa8c6ac7d115"),
        "sparsity_99": ("0623a30ca33bc34239a9e60e06ab9516cdf48a81344762df58ad14334d8904fe"),
    },
    "position_embeddings_wikitext103": {
        "absolute_pe": ("0a1dfd07dc8d262db9a0045f5aa54a7fda55b52efdb53bb27675dfddbc6de1e1"),
        "alibi": ("319b888705462e8e91bf99530b8267fe22fc89ce57be8ab0a8d85fc12c41a04d"),
        "no_pe_reference": ("7d736d38cbb9a905de8e9cd37aeae4fef1c900681459bc558cb5d210da714d8c"),
        "rope": ("c82ade601a7365040a970cd9cf699f70d9ad1c2269f313d215fe546b9ebe7f1c"),
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
