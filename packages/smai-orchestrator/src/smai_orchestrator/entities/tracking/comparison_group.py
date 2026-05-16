""":class:`ComparisonGroupRecord` — runtime instance of an
:class:`smai_core.entities.ExperimentDefinition`.

Per ``designs/smai/01-data-model.md`` §5.3. Produced by the proposal
pipeline's registration step (``08-novel-technique-pipeline.md`` §3.3)
and driven through the CG-execution pipeline-spec
(``03-state-machine.md`` §3).
"""

from __future__ import annotations

from typing import Literal

from pydantic import field_validator
from smai_core.plugins import JobHandle

from smai_orchestrator.entities.tracking._common import (
    LeaseableRecord,
    validate_content_hash,
    validate_id_format,
)

# Catalog (per ``03-state-machine.md`` §3.1): draft, implementing,
# implemented, running, evaluating, complete, implementation_failed,
# running_failed, evaluation_failed. (No lone ``failed`` terminal —
# phase-specific failure terminals per ``03`` §2.9.)
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


class ComparisonGroupRecord(LeaseableRecord):
    """Runtime instance of an ``ExperimentDefinition`` (§5.3)."""

    # === Identity ===
    id: str
    proposal_id: str
    factor_model_id: str | None = None
    # Human-readable label preserved from the planner buffer's draft CG id.
    # Round-9: the primary ``id`` is now a fresh ULID-shaped string (so a
    # long planner-side symbolic id can't blow past the 64-char id-format
    # cap); this slot keeps the readable name for CLI / SPA listings and
    # ``smai status`` rendering.
    symbolic_id: str | None = None

    # === Methodology references ===
    experiment_definition_id: str
    experiment_plan_hash: str | None = None
    harness_contract_hash: str | None = None
    validation_config_hash: str | None = None
    # TechniqueContract hashes live on EntryRecord (one per entry); see §5.4.

    # === State (per §5.2.1; lease + version + audit on the base classes) ===
    state: CGState

    # === Job handle (typed; opaque-handle-with-plugin-name per ``07`` §7) ===
    harness_job_handle: JobHandle | None = None

    # === Code-review state (DEC-016, edge gate) ===
    code_review_attempt: int = 0
    code_review_result_hash: str | None = None

    # === Implementation-phase retry counter (round 10) ===
    # Bumped by the engine on each (re-)entry into the ``implementing``
    # state's dispatch via the declared :class:`RetryPolicy`; the
    # synthesized retry-exhausted terminal edge transitions the CG to
    # ``implementation_failed`` once it reaches
    # ``max_implementation_phase_attempts``. Distinct from
    # :attr:`code_review_attempt` (which counts code-review retries
    # against the implementer code) and from
    # :attr:`EntryRecord.implementation_attempt` (per-entry).
    implementation_phase_attempt: int = 0

    _validate_ids = field_validator(
        "id",
        "proposal_id",
        "factor_model_id",
        "experiment_definition_id",
    )(validate_id_format)

    _validate_hashes = field_validator(
        "experiment_plan_hash",
        "harness_contract_hash",
        "validation_config_hash",
        "code_review_result_hash",
    )(validate_content_hash)


__all__ = ["CGState", "ComparisonGroupRecord"]
