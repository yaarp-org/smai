"""Manifest schema, canonicalization, and content-hash determinism."""

from __future__ import annotations

from smai_runtime import (
    HarnessAPIManifest,
    HarnessExtensionPoint,
    compute_harness_version_hash,
    compute_manifest_hash,
    freeze_manifest,
    manifest_canonical_form,
)


def _manifest(eps: list[HarnessExtensionPoint]) -> HarnessAPIManifest:
    return HarnessAPIManifest(
        extension_points=eps,
        integration_pattern_summary="t",
        harness_version_hash="hv",
        parent_harness_contract_hash="pc",
        manifest_schema_version=1,
        runtime_template_version="1.0.0",
    )


def test_freeze_manifest_populates_content_hash() -> None:
    m = freeze_manifest(_manifest([]))
    assert m.content_hash != ""
    assert len(m.content_hash) == 64  # sha256 hex


def test_manifest_hash_independent_of_extension_point_order() -> None:
    a = _manifest(
        [
            HarnessExtensionPoint(
                key="callbacks",
                type_signature="list[Callable]",
                purpose="cb",
                optional=True,
                integration_pattern="append",
            ),
            HarnessExtensionPoint(
                key="train_transforms",
                type_signature="list[Callable]",
                purpose="tt",
                optional=True,
                integration_pattern="append",
            ),
        ]
    )
    b = _manifest(list(reversed(a.extension_points)))
    assert compute_manifest_hash(a) == compute_manifest_hash(b)


def test_manifest_canonical_form_strips_content_hash() -> None:
    m = freeze_manifest(_manifest([]))
    body = manifest_canonical_form(m)
    assert "content_hash" not in body


def test_freeze_manifest_is_idempotent() -> None:
    m1 = freeze_manifest(_manifest([]))
    m2 = freeze_manifest(m1)
    assert m1.content_hash == m2.content_hash


def test_compute_harness_version_hash_is_deterministic() -> None:
    files = {
        "trainer.py": b"def run(): pass\n",
        "data_loader.py": b"def load(): pass\n",
    }
    h1 = compute_harness_version_hash(files)
    h2 = compute_harness_version_hash(files)
    assert h1 == h2

    # Reversing dict insertion order does not change the hash (sorted internally).
    reversed_files = {"data_loader.py": files["data_loader.py"], "trainer.py": files["trainer.py"]}
    assert compute_harness_version_hash(reversed_files) == h1


def test_compute_harness_version_hash_changes_on_content_change() -> None:
    files_a = {"a.py": b"x"}
    files_b = {"a.py": b"y"}
    assert compute_harness_version_hash(files_a) != compute_harness_version_hash(files_b)
