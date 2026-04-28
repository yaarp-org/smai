""":class:`EntryRecord` — runtime instance of an
:class:`smai_core.entities.Entry`.

Per ``designs/smai/01-data-model.md`` §5.4. Created during the proposal
pipeline's registration transaction. Additive baselines (per DEC-013)
have ``technique_id is None`` and skip implementation dispatch (per
DEC-017): the orchestrator's composite ``implementing`` dispatcher
pre-filters them and writes ``state="implemented"`` directly.
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

# Catalog (per ``03-state-machine.md`` §3.1 / §3.4): pending,
# implementing, implemented, implementation_failed.
EntryState = Literal[
    "pending",
    "implementing",
    "implemented",
    "implementation_failed",
]


class EntryRecord(LeaseableRecord):
    """Runtime instance of an ``Entry`` (§5.4)."""

    # === Identity ===
    id: str
    cg_id: str
    technique_id: str | None
    is_baseline: bool

    # === Methodology references ===
    entry_id: str
    technique_contract_hash: str | None = None

    # === Manifest reference (settled per §5.2.3) ===
    harness_api_manifest_hash: str | None = None

    # === State ===
    state: EntryState

    # === Job handle ===
    implementation_job_handle: JobHandle | None = None

    # === Implementation-stage retry state ===
    implementation_attempt: int = 0

    _validate_ids = field_validator(
        "id",
        "cg_id",
        "technique_id",
        "entry_id",
    )(validate_id_format)

    _validate_hashes = field_validator(
        "technique_contract_hash",
        "harness_api_manifest_hash",
    )(validate_content_hash)


__all__ = ["EntryRecord", "EntryState"]
