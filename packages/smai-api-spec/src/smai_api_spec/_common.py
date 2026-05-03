"""Cross-resource shared shapes for the SMAI HTTP API contract.

Per ``designs/smai/11-api.md`` §5.1.2 / §5.1: the audit-fields mixin
that every entity-detail response inherits, the state Literal
re-declarations, and the configured ``BaseModel`` subclass with
``extra="forbid"``.

The state Literals are intentionally **duplicated** from
``smai_orchestrator.entities.tracking`` rather than re-imported. Per
DEC-037 the ``smai-api-spec`` package depends only on Pydantic — pulling
``smai-orchestrator`` would break that contract and force every consumer
of this package (third-party SDKs, codegen pipelines, Yaarp v2's API)
to install the orchestrator. The duplication is defended by
``tests/test_state_literal_parity.py``, which imports both sides at test
time and asserts equality.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

# === Configured base ========================================================


class APIBaseModel(BaseModel):
    """Common base for every API request / response model.

    ``extra="forbid"`` enforces the schema contract — unexpected keys in
    a payload are an error, not silently ignored. Per ``11`` §5.

    Subclassing this base ensures the ``model_config`` is consistent
    across the package; per-class overrides should be rare.
    """

    model_config = ConfigDict(extra="forbid")


# === Audit fields ===========================================================


class BaseAuditedResponse(APIBaseModel):
    """The four audit fields every entity-detail response carries.

    Mirrors :class:`smai_orchestrator.entities.tracking.BasePipelineRecord`
    minus the lease triple (``leased_by`` / ``lease_expires_at`` /
    ``lease_nonce``), which is orchestrator-internal per ``11`` §5.1.2 and
    NOT exposed on API responses. ``version`` IS exposed for debugging
    convenience but the SPA should not branch on it.
    """

    created_at: datetime
    updated_at: datetime
    last_error: str | None = None
    version: int


# === State Literals (DUPLICATED from smai_orchestrator) =====================
#
# Per DEC-037 / §5.1 above. Defended by tests/test_state_literal_parity.py.
#
# Sources of truth (mirror these character-for-character):
#   - ProposalState — smai_orchestrator/entities/tracking/proposal.py
#   - PaperState    — smai_orchestrator/entities/tracking/paper.py
#   - CGState       — smai_orchestrator/entities/tracking/comparison_group.py
#   - EntryState    — smai_orchestrator/entities/tracking/entry.py
#   - RunState      — smai_orchestrator/entities/tracking/run.py
# Submission / decision discriminators live alongside ProposalState.

ProposalState = Literal[
    "proposal_submitted",
    "designing",
    "designed",
    "registered",
    "rejected",
    "failed",
]

SubmissionKind = Literal["novel_technique", "reproduce_paper"]

UserDecision = Literal["approved", "rejected"]

PaperState = Literal[
    "submitted",
    "fetching",
    "screening",
    "planning",
    "registered",
    "rejected",
    "failed",
    "partial",
]

ScreenDecision = Literal["accept", "reject"]

CGState = Literal[
    "draft",
    "implementing",
    "implemented",
    "running",
    "evaluating",
    "complete",
    "implementation_failed",
    "running_failed",
    "evaluation_failed",
]

EntryState = Literal[
    "pending",
    "implementing",
    "implemented",
    "implementation_failed",
]

RunState = Literal[
    "pending",
    "submitted",
    "running",
    "succeeded",
    "failed",
    "inconclusive",
]

# Entity-kind discriminator — used by the SSE StateChangeEvent and the
# resource-kind framing across the API contract.
EntityKind = Literal["proposal", "paper", "comparison_group", "entry", "run"]


__all__ = [
    "APIBaseModel",
    "BaseAuditedResponse",
    "CGState",
    "EntityKind",
    "EntryState",
    "PaperState",
    "ProposalState",
    "RunState",
    "ScreenDecision",
    "SubmissionKind",
    "UserDecision",
]
