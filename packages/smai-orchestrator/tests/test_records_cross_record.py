"""Cross-record sanity tests (Task 1.10).

Verify that methodology-entity ID references and content-hash references
across records share the same format-validation rules — and that the
inventory of state-driven entities matches §5.9.
"""

from __future__ import annotations

from datetime import UTC, datetime

from smai_orchestrator.entities.tracking import (
    ComparisonGroupRecord,
    EntryRecord,
    FactorModelRecord,
    PaperRecord,
    ProposalRecord,
    RunRecord,
)


def _now() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


def test_state_driven_records_carry_lease_triple() -> None:
    """Per §5.9 — every state-driven record (CG / Entry / Run / Proposal /
    Paper) carries the §5.2.4 lease triple. ``FactorModelRecord`` does not."""
    for cls in (ComparisonGroupRecord, EntryRecord, RunRecord, ProposalRecord, PaperRecord):
        fields = cls.model_fields.keys()
        assert "leased_by" in fields, f"{cls.__name__} missing leased_by"
        assert "lease_expires_at" in fields, f"{cls.__name__} missing lease_expires_at"
        assert "lease_nonce" in fields, f"{cls.__name__} missing lease_nonce"

    # FactorModelRecord must NOT carry the lease triple.
    fm_fields = FactorModelRecord.model_fields.keys()
    assert "leased_by" not in fm_fields
    assert "lease_expires_at" not in fm_fields
    assert "lease_nonce" not in fm_fields


def test_every_record_carries_version_and_timestamps() -> None:
    """Per §5.2.1 / §5.2.6 — the four base fields are universal."""
    for cls in (
        ComparisonGroupRecord,
        EntryRecord,
        RunRecord,
        ProposalRecord,
        PaperRecord,
        FactorModelRecord,
    ):
        fields = cls.model_fields.keys()
        assert "version" in fields, f"{cls.__name__} missing version"
        assert "created_at" in fields, f"{cls.__name__} missing created_at"
        assert "updated_at" in fields, f"{cls.__name__} missing updated_at"


def test_state_driven_records_carry_state() -> None:
    """Per §5.9 — every state-driven record carries ``state``;
    ``FactorModelRecord`` does not (degenerate lifecycle)."""
    for cls in (ComparisonGroupRecord, EntryRecord, RunRecord, ProposalRecord, PaperRecord):
        assert "state" in cls.model_fields, f"{cls.__name__} missing state"
    assert "state" not in FactorModelRecord.model_fields


def test_state_driven_records_carry_last_error() -> None:
    """Per §5.2.6 — every record carries ``last_error``."""
    for cls in (
        ComparisonGroupRecord,
        EntryRecord,
        RunRecord,
        ProposalRecord,
        PaperRecord,
        FactorModelRecord,
    ):
        assert "last_error" in cls.model_fields


def test_cg_proposal_id_and_proposal_id_share_format() -> None:
    """A proposal with id=X parents a CG that references X via
    ``proposal_id`` — both fields share the same format validator (both
    accept the same string, neither rejects it)."""
    common_id = "01HQABCDEFGHJKMNPQRSTVWXYZ"
    prop = ProposalRecord.model_validate(
        {
            "id": common_id,
            "submission_kind": "novel_technique",
            "state": "proposal_submitted",
            "created_at": _now(),
            "updated_at": _now(),
        }
    )
    cg = ComparisonGroupRecord.model_validate(
        {
            "id": "cg_y",
            "proposal_id": common_id,
            "experiment_definition_id": "cg_y",
            "state": "draft",
            "created_at": _now(),
            "updated_at": _now(),
        }
    )
    assert cg.proposal_id == prop.id


def test_run_entry_and_cg_id_chain() -> None:
    """``RunRecord.entry_id`` and ``EntryRecord.id`` share the format —
    constructing both with the same id round-trips."""
    entry_id = "entry_round_trip"
    cg_id = "cg_round_trip"
    entry = EntryRecord.model_validate(
        {
            "id": entry_id,
            "cg_id": cg_id,
            "technique_id": None,
            "is_baseline": True,
            "entry_id": entry_id,
            "state": "implemented",
            "created_at": _now(),
            "updated_at": _now(),
        }
    )
    run = RunRecord.model_validate(
        {
            "id": "run_x",
            "cg_id": cg_id,
            "entry_id": entry_id,
            "seed": 0,
            "state": "pending",
            "created_at": _now(),
            "updated_at": _now(),
        }
    )
    assert run.entry_id == entry.id
    assert run.cg_id == entry.cg_id


def test_content_hash_format_is_uniform_across_records() -> None:
    """The same 64-char hex format applies to every content-hash field
    across every record (per §5.2.2)."""
    sha = "a" * 64

    cg = ComparisonGroupRecord.model_validate(
        {
            "id": "cg_h",
            "proposal_id": "prop_h",
            "experiment_definition_id": "cg_h",
            "experiment_plan_hash": sha,
            "harness_contract_hash": sha,
            "validation_config_hash": sha,
            "code_review_result_hash": sha,
            "state": "draft",
            "created_at": _now(),
            "updated_at": _now(),
        }
    )
    assert cg.experiment_plan_hash == sha
    assert cg.code_review_result_hash == sha

    entry = EntryRecord.model_validate(
        {
            "id": "entry_h",
            "cg_id": "cg_h",
            "technique_id": None,
            "is_baseline": True,
            "entry_id": "entry_h",
            "technique_contract_hash": sha,
            "harness_api_manifest_hash": sha,
            "state": "implemented",
            "created_at": _now(),
            "updated_at": _now(),
        }
    )
    assert entry.technique_contract_hash == sha
    assert entry.harness_api_manifest_hash == sha


def test_id_format_consistency_across_records() -> None:
    """A malformed ID (whitespace) is rejected on every record's primary
    ``id`` field."""
    import pytest
    from pydantic import ValidationError

    bad_id = "id with space"
    base_now = {"created_at": _now(), "updated_at": _now()}

    for cls, base_fields in [
        (
            ComparisonGroupRecord,
            {
                "proposal_id": "prop_x",
                "experiment_definition_id": "cg_y",
                "state": "draft",
            },
        ),
        (
            EntryRecord,
            {
                "cg_id": "cg_x",
                "technique_id": None,
                "is_baseline": True,
                "entry_id": "entry_y",
                "state": "pending",
            },
        ),
        (
            RunRecord,
            {
                "cg_id": "cg_x",
                "entry_id": "entry_x",
                "seed": 0,
                "state": "pending",
            },
        ),
        (
            ProposalRecord,
            {
                "submission_kind": "novel_technique",
                "state": "proposal_submitted",
            },
        ),
        (
            FactorModelRecord,
            {
                "factor_model_id": "fm_x",
                "research_question": "q",
            },
        ),
    ]:
        with pytest.raises(ValidationError):
            cls.model_validate({"id": bad_id, **base_now, **base_fields})
