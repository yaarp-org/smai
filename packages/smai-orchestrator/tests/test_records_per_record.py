"""Per-record-type tests (Task 1.10) — one block per record covering:

* fixture validates,
* JSON model_dump / model_validate round-trip,
* malformed-input rejection,
* state-Literal rejects unknown values,
* version field starts at 1 and is monotone,
* lease fields default to ``None`` on construction.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
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


# === ComparisonGroupRecord ===================================================


def _cg_min_payload() -> dict[str, object]:
    return {
        "id": "cg_x",
        "proposal_id": "prop_x",
        "experiment_definition_id": "cg_x",
        "state": "draft",
        "created_at": _now(),
        "updated_at": _now(),
    }


def test_cg_validates_min_payload() -> None:
    cg = ComparisonGroupRecord.model_validate(_cg_min_payload())
    assert cg.id == "cg_x"
    assert cg.state == "draft"
    assert cg.version == 0
    # Lease fields default to None per §5.2.4.
    assert cg.leased_by is None
    assert cg.lease_expires_at is None
    assert cg.lease_nonce is None
    # Hashes default to None per §5.3.
    assert cg.experiment_plan_hash is None
    assert cg.harness_contract_hash is None
    assert cg.validation_config_hash is None
    # Code-review state defaults per §5.3.
    assert cg.code_review_attempt == 0
    assert cg.code_review_result_hash is None


def test_cg_json_round_trip() -> None:
    cg = ComparisonGroupRecord.model_validate(_cg_min_payload())
    payload = cg.model_dump_json()
    cg2 = ComparisonGroupRecord.model_validate_json(payload)
    assert cg2 == cg


def test_cg_rejects_unknown_field() -> None:
    payload = _cg_min_payload() | {"unknown": "rejected"}
    with pytest.raises(ValidationError):
        ComparisonGroupRecord.model_validate(payload)


def test_cg_rejects_unknown_state() -> None:
    payload = _cg_min_payload() | {"state": "frobnicating"}
    with pytest.raises(ValidationError):
        ComparisonGroupRecord.model_validate(payload)


@pytest.mark.parametrize(
    "state",
    [
        "draft",
        "implementing",
        "implemented",
        "running",
        "evaluating",
        "complete",
        "implementation_failed",
        "running_failed",
        "evaluation_failed",
    ],
)
def test_cg_accepts_canonical_states(state: str) -> None:
    payload = _cg_min_payload() | {"state": state}
    cg = ComparisonGroupRecord.model_validate(payload)
    assert cg.state == state


def test_cg_version_monotone() -> None:
    """``version`` accepts increments via model_copy (CAS at the storage
    layer is the plugin's job — record-level monotonicity is implicit)."""
    cg1 = ComparisonGroupRecord.model_validate(_cg_min_payload())
    cg2 = cg1.model_copy(update={"version": cg1.version + 1, "state": "implementing"})
    assert cg2.version == cg1.version + 1
    assert cg2.state == "implementing"


def test_cg_rejects_malformed_id() -> None:
    payload = _cg_min_payload() | {"id": "has space"}
    with pytest.raises(ValidationError):
        ComparisonGroupRecord.model_validate(payload)


def test_cg_rejects_malformed_hash() -> None:
    # 63 chars, not 64.
    payload = _cg_min_payload() | {"experiment_plan_hash": "a" * 63}
    with pytest.raises(ValidationError):
        ComparisonGroupRecord.model_validate(payload)


def test_cg_accepts_optional_factor_model_id_none() -> None:
    cg = ComparisonGroupRecord.model_validate(_cg_min_payload())
    assert cg.factor_model_id is None


def test_cg_accepts_valid_content_hash() -> None:
    payload = _cg_min_payload() | {"experiment_plan_hash": "0" * 64}
    cg = ComparisonGroupRecord.model_validate(payload)
    assert cg.experiment_plan_hash == "0" * 64


# === EntryRecord =============================================================


def _entry_min_payload(*, technique_id: str | None = None) -> dict[str, object]:
    return {
        "id": "entry_x",
        "cg_id": "cg_x",
        "technique_id": technique_id,
        "is_baseline": technique_id is None,
        "entry_id": "entry_x",
        "state": "pending",
        "created_at": _now(),
        "updated_at": _now(),
    }


def test_entry_validates_baseline_with_null_technique() -> None:
    """Per DEC-013 — additive baselines have ``technique_id is None``."""
    entry = EntryRecord.model_validate(_entry_min_payload())
    assert entry.technique_id is None
    assert entry.is_baseline is True


def test_entry_validates_with_technique() -> None:
    entry = EntryRecord.model_validate(_entry_min_payload(technique_id="tech_dropout"))
    assert entry.technique_id == "tech_dropout"
    assert entry.is_baseline is False


def test_entry_json_round_trip() -> None:
    entry = EntryRecord.model_validate(_entry_min_payload(technique_id="tech_dropout"))
    payload = entry.model_dump_json()
    entry2 = EntryRecord.model_validate_json(payload)
    assert entry2 == entry


def test_entry_rejects_unknown_state() -> None:
    payload = _entry_min_payload() | {"state": "unknown"}
    with pytest.raises(ValidationError):
        EntryRecord.model_validate(payload)


@pytest.mark.parametrize(
    "state", ["pending", "implementing", "implemented", "implementation_failed"]
)
def test_entry_accepts_canonical_states(state: str) -> None:
    payload = _entry_min_payload() | {"state": state}
    entry = EntryRecord.model_validate(payload)
    assert entry.state == state


def test_entry_default_implementation_attempt_zero() -> None:
    entry = EntryRecord.model_validate(_entry_min_payload())
    assert entry.implementation_attempt == 0
    assert entry.harness_api_manifest_hash is None
    assert entry.technique_contract_hash is None


def test_entry_rejects_unknown_field() -> None:
    payload = _entry_min_payload() | {"unknown": "rejected"}
    with pytest.raises(ValidationError):
        EntryRecord.model_validate(payload)


def test_entry_default_lease_fields_none() -> None:
    entry = EntryRecord.model_validate(_entry_min_payload())
    assert entry.leased_by is None
    assert entry.lease_expires_at is None
    assert entry.lease_nonce is None


def test_entry_accepts_manifest_hash() -> None:
    payload = _entry_min_payload() | {"harness_api_manifest_hash": "f" * 64}
    entry = EntryRecord.model_validate(payload)
    assert entry.harness_api_manifest_hash == "f" * 64


# === RunRecord ===============================================================


def _run_min_payload() -> dict[str, object]:
    return {
        "id": "run_x",
        "cg_id": "cg_x",
        "entry_id": "entry_x",
        "seed": 42,
        "state": "pending",
        "created_at": _now(),
        "updated_at": _now(),
    }


def test_run_validates_min_payload() -> None:
    run = RunRecord.model_validate(_run_min_payload())
    assert run.id == "run_x"
    assert run.seed == 42
    assert run.state == "pending"
    assert run.run_attempt == 0
    assert run.failure_reason is None


def test_run_json_round_trip() -> None:
    run = RunRecord.model_validate(_run_min_payload())
    payload = run.model_dump_json()
    run2 = RunRecord.model_validate_json(payload)
    assert run2 == run


@pytest.mark.parametrize(
    "state",
    ["pending", "submitted", "running", "succeeded", "failed", "inconclusive"],
)
def test_run_accepts_canonical_states(state: str) -> None:
    payload = _run_min_payload() | {"state": state}
    run = RunRecord.model_validate(payload)
    assert run.state == state


def test_run_rejects_unknown_state() -> None:
    payload = _run_min_payload() | {"state": "queued"}
    with pytest.raises(ValidationError):
        RunRecord.model_validate(payload)


def test_run_telemetry_optional() -> None:
    run = RunRecord.model_validate(_run_min_payload())
    assert run.duration_seconds is None
    assert run.started_at is None
    assert run.completed_at is None


def test_run_accepts_terminal_telemetry() -> None:
    end = datetime(2026, 1, 1, 1, tzinfo=UTC)
    payload = _run_min_payload() | {
        "state": "succeeded",
        "duration_seconds": 3600.5,
        "started_at": _now(),
        "completed_at": end,
        "raw_metrics_artifact_key": "comparison-groups/cg_x/runs/run_x/0.json",
    }
    run = RunRecord.model_validate(payload)
    assert run.duration_seconds == 3600.5
    assert run.completed_at == end
    assert run.raw_metrics_artifact_key is not None


# === ProposalRecord ==========================================================


def _proposal_min_payload(*, kind: str = "novel_technique") -> dict[str, object]:
    return {
        "id": "prop_x",
        "submission_kind": kind,
        "state": "proposal_submitted",
        "created_at": _now(),
        "updated_at": _now(),
    }


def test_proposal_validates_min_payload() -> None:
    prop = ProposalRecord.model_validate(_proposal_min_payload())
    assert prop.id == "prop_x"
    assert prop.submission_kind == "novel_technique"
    assert prop.state == "proposal_submitted"
    assert prop.user_decision is None
    assert prop.user_decided_at is None


def test_proposal_json_round_trip() -> None:
    prop = ProposalRecord.model_validate(_proposal_min_payload())
    payload = prop.model_dump_json()
    prop2 = ProposalRecord.model_validate_json(payload)
    assert prop2 == prop


@pytest.mark.parametrize(
    "state",
    ["proposal_submitted", "designing", "designed", "registered", "rejected", "failed"],
)
def test_proposal_accepts_canonical_states(state: str) -> None:
    payload = _proposal_min_payload() | {"state": state}
    prop = ProposalRecord.model_validate(payload)
    assert prop.state == state


def test_proposal_rejects_unknown_state() -> None:
    payload = _proposal_min_payload() | {"state": "queued"}
    with pytest.raises(ValidationError):
        ProposalRecord.model_validate(payload)


def test_proposal_rejects_unknown_submission_kind() -> None:
    payload = _proposal_min_payload(kind="from_thin_air")
    with pytest.raises(ValidationError):
        ProposalRecord.model_validate(payload)


def test_proposal_accepts_reproduce_paper_kind() -> None:
    payload = _proposal_min_payload(kind="reproduce_paper") | {
        "reproduce_paper_arxiv_id": "2501.00001",
    }
    prop = ProposalRecord.model_validate(payload)
    assert prop.submission_kind == "reproduce_paper"
    assert prop.reproduce_paper_arxiv_id == "2501.00001"


def test_proposal_user_decision_literal() -> None:
    payload = _proposal_min_payload() | {
        "state": "designed",
        "user_decision": "approved",
    }
    prop = ProposalRecord.model_validate(payload)
    assert prop.user_decision == "approved"

    bad = _proposal_min_payload() | {"user_decision": "maybe"}
    with pytest.raises(ValidationError):
        ProposalRecord.model_validate(bad)


def test_proposal_default_lease_fields_none() -> None:
    prop = ProposalRecord.model_validate(_proposal_min_payload())
    assert prop.leased_by is None
    assert prop.lease_expires_at is None
    assert prop.lease_nonce is None


def test_proposal_default_retry_counters_zero() -> None:
    prop = ProposalRecord.model_validate(_proposal_min_payload())
    assert prop.design_attempt == 0
    assert prop.registration_attempt == 0


# === PaperRecord =============================================================


def _paper_min_payload() -> dict[str, object]:
    return {
        "arxiv_id": "2501.00001",
        "state": "submitted",
        "created_at": _now(),
        "updated_at": _now(),
    }


def test_paper_validates_min_payload() -> None:
    paper = PaperRecord.model_validate(_paper_min_payload())
    assert paper.arxiv_id == "2501.00001"
    assert paper.state == "submitted"
    assert paper.title is None
    assert paper.authors == []
    assert paper.categories == []


def test_paper_json_round_trip() -> None:
    paper = PaperRecord.model_validate(_paper_min_payload())
    payload = paper.model_dump_json()
    paper2 = PaperRecord.model_validate_json(payload)
    assert paper2 == paper


@pytest.mark.parametrize(
    "state",
    [
        "submitted",
        "fetching",
        "screening",
        "planning",
        "registered",
        "rejected",
        "failed",
        "partial",
    ],
)
def test_paper_accepts_canonical_states(state: str) -> None:
    payload = _paper_min_payload() | {"state": state}
    paper = PaperRecord.model_validate(payload)
    assert paper.state == state


def test_paper_rejects_unknown_state() -> None:
    payload = _paper_min_payload() | {"state": "ingested"}
    with pytest.raises(ValidationError):
        PaperRecord.model_validate(payload)


def test_paper_rejects_empty_arxiv_id() -> None:
    payload = _paper_min_payload() | {"arxiv_id": ""}
    with pytest.raises(ValidationError):
        PaperRecord.model_validate(payload)


def test_paper_rejects_whitespace_arxiv_id() -> None:
    payload = _paper_min_payload() | {"arxiv_id": " 2501.00001 "}
    with pytest.raises(ValidationError):
        PaperRecord.model_validate(payload)


def test_paper_accepts_legacy_arxiv_id_format() -> None:
    # Legacy archive/YYMM### shape allowed (the spec is intentionally permissive).
    payload = _paper_min_payload() | {"arxiv_id": "math/0001234"}
    paper = PaperRecord.model_validate(payload)
    assert paper.arxiv_id == "math/0001234"


def test_paper_screen_decision_literal() -> None:
    payload = _paper_min_payload() | {
        "state": "screening",
        "screen_result_decision": "accept",
    }
    paper = PaperRecord.model_validate(payload)
    assert paper.screen_result_decision == "accept"

    bad = _paper_min_payload() | {"screen_result_decision": "maybe"}
    with pytest.raises(ValidationError):
        PaperRecord.model_validate(bad)


def test_paper_default_retry_counters_zero() -> None:
    paper = PaperRecord.model_validate(_paper_min_payload())
    assert paper.planning_attempt == 0
    assert paper.screening_attempt == 0
    assert paper.registration_attempt == 0


def test_paper_no_cg_reference_field() -> None:
    """Per DEC-032 — ``PaperRecord`` carries no CG references."""
    fields = PaperRecord.model_fields.keys()
    for f in fields:
        assert "cg" not in f, f"PaperRecord must not carry a CG reference field; got {f!r}"


# === FactorModelRecord =======================================================


def _factor_model_payload() -> dict[str, object]:
    return {
        "id": "fm_x",
        "factor_model_id": "fm_x",
        "research_question": "Does X improve Y under Z?",
        "created_at": _now(),
        "updated_at": _now(),
    }


def test_factor_model_validates_min_payload() -> None:
    fm = FactorModelRecord.model_validate(_factor_model_payload())
    assert fm.id == "fm_x"
    assert fm.research_question.startswith("Does")
    assert fm.version == 0


def test_factor_model_json_round_trip() -> None:
    fm = FactorModelRecord.model_validate(_factor_model_payload())
    payload = fm.model_dump_json()
    fm2 = FactorModelRecord.model_validate_json(payload)
    assert fm2 == fm


def test_factor_model_no_lease_fields() -> None:
    """Per DEC-031 #5 / §5.2.4 — ``FactorModelRecord`` has no lease triple
    (no orchestrator-dispatched lifecycle)."""
    fields = FactorModelRecord.model_fields.keys()
    assert "leased_by" not in fields
    assert "lease_expires_at" not in fields
    assert "lease_nonce" not in fields


def test_factor_model_no_state_field() -> None:
    """Per DEC-031 #5 — degenerate lifecycle, no state machine."""
    assert "state" not in FactorModelRecord.model_fields


def test_factor_model_rejects_unknown_field() -> None:
    payload = _factor_model_payload() | {"unknown": "rejected"}
    with pytest.raises(ValidationError):
        FactorModelRecord.model_validate(payload)
