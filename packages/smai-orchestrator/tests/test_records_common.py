"""Tests for the shared base classes / format validators in
:mod:`smai_orchestrator.entities.tracking._common` (Task 1.10).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from smai_orchestrator.entities.tracking._common import (
    BasePipelineRecord,
    LeaseableRecord,
    validate_content_hash,
    validate_id_format,
)


def _now() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


# === id-format validator =====================================================


@pytest.mark.parametrize(
    "value",
    [
        "cg_x",
        "cg_round_trip",
        "01HQABCDEFGHJKMNPQRSTVWXYZ",  # 26-char Crockford-style
        "prop-default",
        "tech.id.with.dots",
        "ns:scoped/id",
        "X",
    ],
)
def test_id_format_accepts_valid(value: str) -> None:
    assert validate_id_format(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "",
        " has space ",
        "with whitespace mid",
        "tab\there",
        "id\nwith\nnewline",
        "x" * 65,
        "weird?char",
        "weird*char",
        "weird@char",
        "naïve",
    ],
)
def test_id_format_rejects_invalid(value: str) -> None:
    with pytest.raises(ValueError):
        validate_id_format(value)


def test_id_format_passes_through_none() -> None:
    assert validate_id_format(None) is None


def test_id_format_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        validate_id_format(42)  # type: ignore[arg-type]


# === content-hash validator ==================================================


def test_content_hash_accepts_64_char_lowercase_hex() -> None:
    sha = "a" * 64
    assert validate_content_hash(sha) == sha


def test_content_hash_accepts_realistic_sha256() -> None:
    sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert validate_content_hash(sha) == sha


@pytest.mark.parametrize(
    "value",
    [
        "",
        "abcd",  # too short
        "z" * 64,  # 'z' is not hex
        "A" * 64,  # uppercase not allowed
        "a" * 63,
        "a" * 65,
        "0" * 63 + "G",  # 'G' not hex
        " " + "a" * 63,
    ],
)
def test_content_hash_rejects_invalid(value: str) -> None:
    with pytest.raises(ValueError):
        validate_content_hash(value)


def test_content_hash_passes_through_none() -> None:
    assert validate_content_hash(None) is None


# === BasePipelineRecord ======================================================


def test_base_record_requires_timestamps() -> None:
    """``created_at`` / ``updated_at`` are required (per §5.2.6)."""
    with pytest.raises(ValidationError):
        BasePipelineRecord()  # type: ignore[call-arg]


def test_base_record_default_version_zero() -> None:
    """Newly-created entities start at version 0 (per §5.2.1)."""
    rec = BasePipelineRecord(created_at=_now(), updated_at=_now())
    assert rec.version == 0
    assert rec.last_error is None


def test_base_record_extra_forbid() -> None:
    """``extra='forbid'`` rejects unknown fields (Task 1.10 supersedes 1.8's
    ``extra='allow'`` stub)."""
    with pytest.raises(ValidationError):
        BasePipelineRecord.model_validate(
            {"created_at": _now(), "updated_at": _now(), "rogue": "field"}
        )


# === LeaseableRecord =========================================================


def test_leaseable_default_lease_fields_none() -> None:
    """The §5.2.4 lease triple defaults to ``None`` on construction."""
    rec = LeaseableRecord(created_at=_now(), updated_at=_now())
    assert rec.leased_by is None
    assert rec.lease_expires_at is None
    assert rec.lease_nonce is None


def test_leaseable_accepts_full_lease_triple() -> None:
    later = datetime(2026, 1, 1, 1, 0, tzinfo=UTC)
    rec = LeaseableRecord(
        created_at=_now(),
        updated_at=_now(),
        leased_by="worker-a",
        lease_expires_at=later,
        lease_nonce="00000000-0000-0000-0000-000000000000",
    )
    assert rec.leased_by == "worker-a"
    assert rec.lease_expires_at == later
    assert rec.lease_nonce is not None
