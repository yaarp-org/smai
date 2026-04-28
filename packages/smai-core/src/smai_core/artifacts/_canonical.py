"""Canonical JSON serialization, content hashing, and ``freeze_with_hash``.

Per ``02-dsl-and-contracts.md`` §8.2:

* Object keys sorted lexicographically.
* Tight separators (no whitespace).
* Floats serialized with Python's shortest round-trip repr (``repr(float)``).
* Sha256 over the canonical bytes is the artifact's ``content_hash``.

Pydantic v2's ``model_dump_json`` is *not* canonical by default — it preserves
field-declaration order and inserts no whitespace, but does not sort keys
inside ``dict`` values. We use ``model_dump(mode="json")`` to get a JSON-shaped
``dict``, then ``json.dumps(..., sort_keys=True, separators=(",", ":"))`` to
canonicalize.

The hash is computed over the artifact's full envelope+body shape *minus* the
envelope's ``content_hash`` field (which would otherwise self-reference). All
other envelope fields — including ``parent_factor_model_id``,
``registry_hashes``, and ``surface_map`` — are part of the hash so changes to
provenance, registry state, or surface annotations propagate to a new hash.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, TypeVar, cast

from pydantic import BaseModel

ArtifactT = TypeVar("ArtifactT", bound=BaseModel)


def canonical_json(payload: BaseModel | dict[str, Any] | list[Any]) -> bytes:
    """Return the canonical-form JSON encoding of ``payload``.

    Pydantic models are dumped via ``model_dump(mode="json")`` first so that
    nested types (e.g., ``MetricRef`` discriminated union, ``NumericValue``)
    are reduced to JSON-shaped primitives before the ``json.dumps`` step
    sorts keys. The resulting bytes are byte-stable across runs and machines
    (modulo the underlying Python ``str(float)`` shortest-round-trip
    behavior, which is fixed in CPython).
    """
    if isinstance(payload, BaseModel):
        data: object = payload.model_dump(mode="json")
    else:
        data = payload
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def artifact_hash(payload: BaseModel | dict[str, Any] | list[Any]) -> str:
    """Sha256 hex digest of :func:`canonical_json` of ``payload``."""
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def freeze_with_hash(artifact: ArtifactT) -> ArtifactT:
    """Return a copy of ``artifact`` with ``envelope.content_hash`` populated.

    The hash is computed over the artifact's canonical JSON with the
    ``content_hash`` field stripped from the envelope (otherwise the hash
    would self-reference). Reapplied as a sha256 hex digest on the copy via
    ``model_copy``, which avoids a re-validation pass — important because
    embedded ``ComparisonRule`` instances on ``ExperimentPlan`` carry the
    DSL-mode (unfilled) shape and re-running the artifact-mode validator on
    them would fail.

    This function is the single source of truth for the §8.2 hashing
    contract; emit_artifacts calls it once per artifact in the sequential
    build (§8.1).
    """
    raw: dict[str, Any] = artifact.model_dump(mode="json")
    envelope_payload_obj = raw.get("envelope")
    if not isinstance(envelope_payload_obj, dict):
        raise TypeError("freeze_with_hash expected artifact to expose an 'envelope' dict")
    envelope_payload: dict[str, Any] = cast("dict[str, Any]", envelope_payload_obj)
    envelope_for_hash: dict[str, Any] = {
        k: v for k, v in envelope_payload.items() if k != "content_hash"
    }
    raw_for_hash: dict[str, Any] = {**raw, "envelope": envelope_for_hash}
    digest = artifact_hash(raw_for_hash)

    envelope_attr = getattr(artifact, "envelope", None)
    if envelope_attr is None:
        raise TypeError("freeze_with_hash expected artifact to expose an 'envelope' attribute")
    new_envelope = envelope_attr.model_copy(update={"content_hash": digest})
    return artifact.model_copy(update={"envelope": new_envelope})


def hash_registry_dict(payload: dict[str, Any]) -> str:
    """Sha256 hex digest of a registry-shaped dict.

    Used for ``registry_hashes`` envelope provenance: the dict is reduced to
    a canonical JSON representation (sorted keys, tight separators), then
    hashed. Pydantic models inside ``payload`` are dumped via
    ``model_dump(mode="json")`` already by the caller.
    """
    return artifact_hash(payload)


def hash_string_set(values: list[str]) -> str:
    """Sha256 hex digest of a sorted list of strings.

    Used to hash the factor-type-plugin registry — only the plugin names are
    load-bearing for v1 since plugin Protocols are not directly
    JSON-serializable.
    """
    return artifact_hash(sorted(values))
