"""Common envelope shared by the four contract artifacts.

Per ``02-dsl-and-contracts.md`` §7.1 + §7.2. Each emitted artifact carries
identity, versioning, registry-hash provenance, and a per-field ``surface_map``
keyed by JSONPath into the artifact body.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

ArtifactKind: TypeAlias = Literal[
    "experiment_plan",
    "harness_contract",
    "technique_contract",
    "validation_config",
]
"""Discriminator value on every artifact's envelope (§7.1)."""

SurfaceKind: TypeAlias = Literal["locked", "correctness_iterable", "recorded"]
"""The three v1 surface annotations (§7.2 + DEC-031 sub-decision #2/#9)."""

EMPTY_CONTENT_HASH: str = ""
"""Sentinel populated into ``ArtifactEnvelope.content_hash`` when the artifact
is first constructed; replaced by a sha256 hex digest when the artifact is
frozen via :func:`smai_core.artifacts._canonical.freeze_with_hash`."""


class ArtifactEnvelope(BaseModel):
    """Common metadata on every contract artifact (§7.1).

    ``content_hash`` is populated post-construction by
    :func:`freeze_with_hash` — the canonical-form hash spans every field of
    the artifact except this one (which would otherwise self-reference).

    ``surface_map`` is the per-field surface decomposition (DEC-031
    sub-decision #9): keys are JSONPaths into the artifact body, values are
    one of ``locked`` / ``correctness_iterable`` / ``recorded``.
    """

    model_config = ConfigDict(extra="forbid")

    artifact_kind: ArtifactKind
    artifact_id: str
    schema_version: int
    compiler_version: str
    content_hash: str = EMPTY_CONTENT_HASH
    parent_experiment_id: str
    parent_factor_model_id: str | None = None
    registry_hashes: dict[str, str] = Field(default_factory=dict)
    surface_map: dict[str, SurfaceKind] = Field(default_factory=dict)
