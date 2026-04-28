"""Shared base classes and conventions for pipeline-tracking records.

Per ``designs/smai/01-data-model.md`` §5.2 — the six cross-cutting topics
that apply uniformly across every pipeline-tracking record:

* §5.2.1 ``version`` field for compare-and-swap on every state-affecting
  write.
* §5.2.2 methodology references by stable ID (ULID-shaped in v2;
  format-validated as a permissive non-empty no-whitespace ASCII string
  here, since ``01`` §5.2.2 explicitly defers strict format settling to
  implementation). Content-hash references are 64-char hex strings.
* §5.2.4 ``leased_by`` / ``lease_expires_at`` / ``lease_nonce`` triple on
  every state-driven entity. ``FactorModelRecord`` (§5.8) does not carry
  these per DEC-031 #5.
* §5.2.6 ``created_at`` / ``updated_at`` / ``last_error`` audit fields on
  every record. The transition history lives in a separate
  ``transition_log`` table (DEC-033 #1) — not a field on the record.

The base class structure uses a single ``BasePipelineRecord`` for the
identity / version / audit fields shared by every record, and a
``LeaseableRecord`` subclass that adds the §5.2.4 lease triple.
``FactorModelRecord`` extends the bare base; the five state-driven
records extend ``LeaseableRecord``.
"""

from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel, ConfigDict

# === Format conventions =====================================================

# Permissive ID-format validator for methodology-entity references and
# pipeline-record IDs. ``01`` §5.2.2 says "ULID-shaped strings in v2;
# format settled in implementation" — strict 26-char Crockford-base32
# would lock the shape down before the ID-generation code lands and would
# reject the human-readable test fixtures the conformance suite uses. We
# instead require non-empty ASCII without internal whitespace, length ≤
# 64 (well above ULID's 26). When ULID generation lands the validator can
# tighten without breaking call sites that already produce conformant IDs.
_ID_PATTERN = re.compile(r"^[A-Za-z0-9_\-./:]{1,64}$")

# 64-char lowercase hex (sha256 content hash, per ``01`` §5.2.2 / ``02``
# §8.2's "byte-deterministic JSON, sha256 over the canonical form").
_CONTENT_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def validate_id_format(value: object) -> object:
    """Reject obviously malformed ID strings (per §5.2.2).

    Used by every record module's ``@field_validator`` on its ``id`` and
    ``*_id`` foreign-key fields. The format is permissive to admit both
    ULIDs and the human-readable IDs that conformance fixtures and CLI
    callers produce; tightening to strict ULID is a forward-compatible
    change once ID generation lands.

    ``None`` passes through unchanged so the same validator can be reused
    on optional foreign-key columns (e.g. ``ComparisonGroupRecord.factor_model_id``).
    The parameter is typed ``object`` because Pydantic v2 invokes
    ``field_validator`` with the raw input *before* the field's static
    type annotation is enforced — non-string values reaching here are a
    real possibility that we want to reject explicitly with ``TypeError``.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"id field must be a string; got {type(value).__name__}")
    if not _ID_PATTERN.match(value):
        raise ValueError(
            f"id field {value!r} does not match the ULID-shaped format "
            f"(non-empty ASCII, no whitespace, length 1-64)"
        )
    return value


def validate_content_hash(value: object) -> object:
    """Reject content-hash values that are not 64-char lowercase hex (§5.2.2).

    ``None`` passes through unchanged — every content-hash field is
    nullable per the relevant §5 record sections (the hash is populated
    by a downstream dispatch handler). Parameter typed ``object`` for
    the same reason as :func:`validate_id_format`.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"content-hash field must be a string; got {type(value).__name__}")
    if not _CONTENT_HASH_PATTERN.match(value):
        raise ValueError(
            f"content-hash field {value!r} is not 64-char lowercase hex "
            f"(per `01` §5.2.2 / `02` §8.2 — sha256 over the canonical form)"
        )
    return value


# === Base classes ===========================================================


class BasePipelineRecord(BaseModel):
    """Common identity / version / audit fields (§5.2.1 + §5.2.6).

    Every pipeline-tracking record carries:

    * an ``id`` (or ``arxiv_id`` for :class:`PaperRecord`, which provides
      its own primary key),
    * a monotone ``version`` for CAS (per §5.2.1),
    * ``created_at`` / ``updated_at`` timestamps,
    * an optional ``last_error`` populated on failure-edge transitions.

    Records that are driven through a state machine (everything except
    :class:`FactorModelRecord`) extend :class:`LeaseableRecord` instead,
    which adds the §5.2.4 lease triple.
    """

    model_config = ConfigDict(extra="forbid")

    # `version` defaults to 0 per §5.2.1 — newly-created entities start at
    # 0; the first transition takes them to 1.
    version: int = 0
    created_at: datetime
    updated_at: datetime
    last_error: str | None = None


class LeaseableRecord(BasePipelineRecord):
    """Base for records driven through orchestrator-dispatched state machines.

    Adds the §5.2.4 lease triple. Per §5.2.4 lease ops are independent of
    entity-state CAS — the ``version`` field is unrelated to lease
    acquisition. ``lease_nonce`` is the ABA-protection token round-tripped
    via release / extend.
    """

    leased_by: str | None = None
    lease_expires_at: datetime | None = None
    lease_nonce: str | None = None


__all__ = [
    "BasePipelineRecord",
    "LeaseableRecord",
    "validate_content_hash",
    "validate_id_format",
]
