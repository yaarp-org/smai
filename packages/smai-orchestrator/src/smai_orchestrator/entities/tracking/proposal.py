""":class:`ProposalRecord` — runtime instance of a user / research-agent proposal.

Per ``designs/smai/01-data-model.md`` §5.6 — the **primary input path**
in v2 per DEC-032. Created synchronously by the API on submission
(``08-novel-technique-pipeline.md`` §3.1), driven through the proposal
pipeline-spec.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import field_validator
from smai_core.plugins import JobHandle

from smai_orchestrator.entities.tracking._common import (
    LeaseableRecord,
    validate_id_format,
)

# Catalog (per ``03-state-machine.md`` §4.1 — lowercase snake_case):
#   proposal_submitted, designing, designed, registered, rejected, failed.
ProposalState = Literal[
    "proposal_submitted",
    "designing",
    "designed",
    "registered",
    "rejected",
    "failed",
]

# Submission-kind discriminator (per §5.6).
SubmissionKind = Literal["novel_technique", "reproduce_paper"]

# User-decision discriminator (per §5.6).
UserDecision = Literal["approved", "rejected"]


class ProposalRecord(LeaseableRecord):
    """Runtime instance of a user proposal (§5.6)."""

    # === Identity ===
    id: str
    submitted_by: str | None = None

    # === Submission shape ===
    submission_kind: SubmissionKind
    technique_description_artifact_key: str | None = None
    reproduce_paper_arxiv_id: str | None = None

    # === State ===
    state: ProposalState

    # === Designing-stage agent state ===
    planner_job_handle: JobHandle | None = None
    design_plan_artifact_key: str | None = None
    error_context_artifact_key: str | None = None

    # === Designing retry counter ===
    design_attempt: int = 0

    # === Registration retry counter ===
    registration_attempt: int = 0

    # === User-decision state ===
    user_decision: UserDecision | None = None
    user_decided_at: datetime | None = None

    _validate_ids = field_validator("id")(validate_id_format)


__all__ = ["ProposalRecord", "ProposalState", "SubmissionKind", "UserDecision"]
