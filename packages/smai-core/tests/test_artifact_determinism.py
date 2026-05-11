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
        "experiment_plan": ("5ed8716168380ad8a0de9a9cdf8cad7e13498f4da218001a0574b9ae4829a2d2"),
        "harness_contract": ("8b0a617737bcca71a02b98887b44ff4a0b8e5c1ebd75a7e4d13525f300a4d475"),
        "validation_config": ("2b96df360fb8a31a9f84ba98a2c05486e2bf470b8a3423905dd1eca61dd19059"),
    },
    "cutout_on_cifar10": {
        "experiment_plan": ("cab6072a45611f051a62eb69656b2419578b91cf2a6934eb1fafd477075d88ee"),
        "harness_contract": ("3a84c26d3b35fb27a45aa11a9eefb9f9347cdc5b4918313a070d1c4898467b1f"),
        "validation_config": ("f9052ad8e8a54f25ab6728c53561e1dd02c718b4350542c16c570632b1c83eeb"),
    },
    "pruning_sparsity_sweep": {
        "experiment_plan": ("07e8fcbac92d4c2aa9132ad1b0bd0dfb0aa31e275fbbc7336334c00be7dbe831"),
        "harness_contract": ("866b1ff92a7bf4327cba676c00d4a876c37a75dca6cd25fa9d6ddb0fcffdc562"),
        "validation_config": ("67ae213cab295dd5ae4b651e78b865a12d991e03dc8db8e194bfb1d89d4cd598"),
    },
    "position_embeddings_wikitext103": {
        "experiment_plan": ("3e9116595ff3d10ec86bb49fea371ee7513b73adfeb392d096bc7f14b4f2277f"),
        "harness_contract": ("b6034ff556290f25b2ba737417880ad2c6a78ee68358ab27ebdf281e268dac5f"),
        "validation_config": ("75b9ff13b5608da6cab42587f4509257608a451704760a8f71599c638daf85d5"),
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
        "resnet50_treatment": ("840fb35cc99f3f0bcae1bb0caccb4c93e231c1be885bec9bd918bbe166b3717c"),
        "vgg16_reference": ("6901753c20b91045e28c47d101d7153f52c09a9b5d0c8f5530ae45f6e51fcd05"),
    },
    "cutout_on_cifar10": {
        "cutout_treatment": ("879dd2d91d013142ec4bbfefdbd81ca0e766232b2519a0987b48e3cb1ecc339d"),
        "no_aug_baseline": ("579d7a2741f6036965dda39271ed4f3735c19fa94eae92f23f47ae7b62061ed0"),
    },
    "pruning_sparsity_sweep": {
        "dense_baseline": ("ba676f0e7ba6e68e0b6c0b7b496d4b547d33893fbe3fc356af2320abf8671f23"),
        "sparsity_50": ("3cf2feb9c979ca7fc5d7b1874459d88c31fd01cf38868ce8aaf8ba595fcd4425"),
        "sparsity_70": ("66bab29d688753fdc1e90e7fd4039d5fed6c956c5ddb14af7684cb22b3d0b5dc"),
        "sparsity_90": ("4f6d187230723bbd2fa40afdc8ff611c48db0adcb359d7212f4974fb4b97d4e0"),
        "sparsity_99": ("f593c37950297423f14d4603f81ba62ee053de5f20a37fd3addd628579b43d52"),
    },
    "position_embeddings_wikitext103": {
        "absolute_pe": ("50ffd752c3e510ab59cc3c229c29173624598563d4840090a5cf109673906bbc"),
        "alibi": ("e9172be4c4c86081faa29e53fc68c3493d0f361aac85c6bc9db6f512d85e63af"),
        "no_pe_reference": ("9a3ba0837d63541594ab1769e036793ec45736121c69d008bbba41c2631dcdb5"),
        "rope": ("53675bc682951aa6a8d0709713890e00534f726a8af96dc78ca133bb941553bf"),
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
