"""``HarnessAPIManifest`` — pipeline-layer post-build contract surface.

Per ``10-runtime-and-templates.md`` §5. Produced by the harness builder agent
after the harness code is written; consumed by the technique implementer
(which needs to know the typed shape of every extension point it might
populate) and by the integration template's runtime type check (§6).

Hashing reuses ``smai_core.canonical_json`` / ``smai_core.artifact_hash`` so
every smai artifact (methodology contracts, manifest, raw metrics) hashes
through the same canonical-JSON path (§5.3).
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict
from smai_core import canonical_json

IntegrationPattern: TypeAlias = Literal[
    "append",
    "replace",
    "wrap",
    "override_dict",
]
"""Closed v1 set of integration patterns (§3.5 / §5.1)."""

MANIFEST_SCHEMA_VERSION: int = 1
"""Bumped on backward-incompatible changes to the manifest schema (§5.1)."""


class HarnessExtensionPoint(BaseModel):
    """One typed extension point in the harness API (§5.1)."""

    model_config = ConfigDict(extra="forbid")

    key: str
    type_signature: str
    purpose: str
    optional: bool
    integration_pattern: IntegrationPattern


class HarnessAPIManifest(BaseModel):
    """The harness's runtime extension-point declaration (§5.1).

    ``content_hash`` is populated by :func:`freeze_manifest` (analogous to
    ``smai_core.freeze_with_hash`` for methodology artifacts) and is what
    pipeline-tracking entities reference per DEC-033 #3.
    """

    model_config = ConfigDict(extra="forbid")

    extension_points: list[HarnessExtensionPoint]
    integration_pattern_summary: str
    harness_version_hash: str
    parent_harness_contract_hash: str
    manifest_schema_version: int
    runtime_template_version: str
    content_hash: str = ""


def manifest_canonical_form(manifest: HarnessAPIManifest) -> dict[str, Any]:
    """Canonical body for hashing (§5.3).

    Strips ``content_hash`` (self-reference) and lexicographically sorts
    ``extension_points`` by ``key``. ``canonical_json`` then sorts dict
    keys and emits tight separators.
    """
    body: dict[str, Any] = manifest.model_dump(mode="json")
    body.pop("content_hash", None)
    sorted_eps: list[dict[str, Any]] = sorted(
        body.get("extension_points", []),
        key=lambda ep: str(ep.get("key", "")),
    )
    body["extension_points"] = sorted_eps
    return body


def compute_manifest_hash(manifest: HarnessAPIManifest) -> str:
    """SHA-256 hex digest of the manifest's canonical form (§5.3)."""
    return hashlib.sha256(canonical_json(manifest_canonical_form(manifest))).hexdigest()


def freeze_manifest(manifest: HarnessAPIManifest) -> HarnessAPIManifest:
    """Return a copy of ``manifest`` with ``content_hash`` populated.

    Two semantically identical manifests produce byte-equal hashes (§5.3
    canonicalization).
    """
    digest = compute_manifest_hash(manifest)
    return manifest.model_copy(update={"content_hash": digest})


def compute_harness_version_hash(harness_files: dict[str, bytes]) -> str:
    """SHA-256 over the harness directory contents (§5.1).

    ``harness_files`` maps relative path → file bytes. Sorted by path,
    byte-concatenated with a path separator marker, then hashed. This is
    the canonical input the harness builder agent uses to fill
    ``HarnessAPIManifest.harness_version_hash``.
    """
    h = hashlib.sha256()
    for path in sorted(harness_files):
        h.update(path.encode("utf-8"))
        h.update(b"\x00")
        h.update(harness_files[path])
        h.update(b"\x00")
    return h.hexdigest()
