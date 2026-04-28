""":class:`PaperRecord` — runtime instance of paper ingestion.

Per ``designs/smai/01-data-model.md`` §5.7 — the **supporting utility**
per DEC-032. Carries paper-ingestion-pipeline state, arXiv metadata, and
artifact pointers. Per DEC-032 it carries **NO** CG references (paper
ingestion produces ``TechniqueRef``s only; CGs are exclusively
proposal-born).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator
from smai_core.plugins import JobHandle

from smai_orchestrator.entities.tracking._common import (
    LeaseableRecord,
)

# Catalog (per ``03-state-machine.md`` §5.1 — lowercase snake_case):
#   submitted, fetching, screening, planning, registered, rejected, failed, partial.
# ``partial`` is non-terminal (promotable); ``registered`` / ``rejected``
# / ``failed`` are terminals.
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

# Screening-result discriminator (per §5.7).
ScreenDecision = Literal["accept", "reject"]


class PaperRecord(LeaseableRecord):
    """Runtime instance of paper ingestion (§5.7).

    Primary key is ``arxiv_id`` (per ``07-plugin-interfaces.md`` §5.3
    ``get_paper(arxiv_id)``). The arxiv-id format is intentionally not
    locked down here — both the modern ``YYMM.NNNNN[v#]`` shape and the
    legacy ``archive/YYMM###`` shape exist and the plugin layer is
    free to enforce per its substrate. We require non-empty string.
    """

    # === Identity ===
    arxiv_id: str = Field(min_length=1, max_length=64)

    # === Bibliographic metadata (DEC-009) ===
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    published_date: datetime | None = None
    categories: list[str] = Field(default_factory=list)

    # === Artifact pointers ===
    latex_bundle_artifact_key: str | None = None
    expanded_tex_artifact_key: str | None = None
    extracted_text_artifact_key: str | None = None
    figures_artifact_key: str | None = None

    # === Screening result (intermediate state telemetry) ===
    screen_result_decision: ScreenDecision | None = None
    screen_result_reason: str | None = None

    # === State ===
    state: PaperState

    # === Per-stage agent / dispatch state ===
    fetcher_job_handle: JobHandle | None = None
    screener_job_handle: JobHandle | None = None
    planner_job_handle: JobHandle | None = None

    technique_buffer_artifact_key: str | None = None
    error_context_artifact_key: str | None = None

    # === Planning retry counter ===
    planning_attempt: int = 0

    # === Screening retry counter ===
    screening_attempt: int = 0

    # === Registration retry counter ===
    registration_attempt: int = 0

    @field_validator("arxiv_id")
    @classmethod
    def _validate_arxiv_id(cls, value: str) -> str:
        """Reject empty / whitespace-only arxiv ids (per §5.7)."""
        if not value or value.strip() != value:
            raise ValueError(
                f"arxiv_id {value!r} must be a non-empty string with no leading/trailing whitespace"
            )
        return value


__all__ = ["PaperRecord", "PaperState", "ScreenDecision"]
