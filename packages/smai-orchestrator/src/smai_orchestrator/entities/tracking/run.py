""":class:`RunRecord` — runtime instance of one (entry × seed) training run.

Per ``designs/smai/01-data-model.md`` §5.5. The run sub-state-machine is
lifted as a fourth pipeline-spec per ``03-state-machine.md`` §3.9.

Per §5.2.5 / DEC-033, cost data lives in a separate ``RunCost`` table
foreign-keyed to this record (gpu_hours, cost_usd at recorded hourly
rate, container memory if surfaced). No cost fields on the record itself
— aggregate-on-query via SQL JOIN.
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

# Catalog (per ``03-state-machine.md`` §3.9 — sub-state-machine pipeline-spec):
#   pending → submitted → running → succeeded | failed | inconclusive
# `succeeded` matches `Compute.JobState.succeeded` (per `07` §7.2).
# `inconclusive` is terminal: the run completed but emitted no usable
# metric (distinct from `failed`, which means the run errored).
RunState = Literal[
    "pending",
    "submitted",
    "running",
    "succeeded",
    "failed",
    "inconclusive",
]


class RunRecord(LeaseableRecord):
    """Runtime instance of one (entry × seed) training run (§5.5)."""

    # === Identity ===
    id: str
    cg_id: str
    entry_id: str
    seed: int

    # === State ===
    state: RunState

    # === Job handle ===
    compute_job_handle: JobHandle | None = None

    # === Raw-metrics artifact pointer ===
    raw_metrics_artifact_key: str | None = None

    # === Run telemetry ===
    duration_seconds: float | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # === Failure / retry ===
    run_attempt: int = 0
    failure_reason: str | None = None

    _validate_ids = field_validator(
        "id",
        "cg_id",
        "entry_id",
    )(validate_id_format)


__all__ = ["RunRecord", "RunState"]
