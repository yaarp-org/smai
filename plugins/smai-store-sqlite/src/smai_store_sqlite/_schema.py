"""SQLAlchemy 2.0 Core schema for the :class:`MetadataStore` SQLite plugin.

Per DEC-036 (substrate choice — SQLAlchemy 2.0 async Core; no Alembic at
Task 2.A2), DEC-030 (SQL-shape commitment), DEC-033 (transition log,
normalized cost tables), DEC-035 (cursor pagination, lease semantics,
caller-resolved pool→states).

Schema covers nine tables:

* Six pipeline-tracking tables — one per record type in
  ``01-data-model.md`` §5: ``cgs`` (§5.3), ``entries`` (§5.4), ``runs``
  (§5.5), ``proposals`` (§5.6), ``papers`` (§5.7), ``factor_models``
  (§5.8).
* One methodology lookup mirror — ``techniques`` (per
  ``07-plugin-interfaces.md`` §5.3, the ``TechniqueRef`` registry
  exposed via ``get_technique`` / ``list_techniques`` /
  ``upsert_technique`` for long-running deployments).
* Three normalized cost tables per DEC-033 — ``transition_log`` (§5.2.6
  audit shape), ``agent_sessions`` (token-shaped agent costs),
  ``run_costs`` (GPU / container shaped run costs).

Conventions:

* All ``id`` / ``*_id`` columns are 64-char ``String`` to match the
  permissive ``_ID_PATTERN`` validator in
  ``smai_orchestrator.entities.tracking._common`` (ULID is 26 chars;
  the cap admits human-readable test fixtures).
* All content-hash columns are 64-char ``String`` (sha256 hex per
  ``01`` §5.2.2).
* Job handles serialize as JSON blobs (``JobHandle`` is
  ``{plugin: str, handle: str, metadata: dict[str, Any]}`` per
  ``07`` §7.2 — too plugin-specific to flatten into columns).
* List-shaped Pydantic fields (``PaperRecord.authors``,
  ``PaperRecord.categories``, ``TechniqueRef.compatible_factor_types``
  etc.) serialize as JSON arrays.
* Datetime columns use ``DateTime(timezone=True)``. SQLite renders
  these as ISO-8601 text via SQLAlchemy's adapter; Postgres
  (Task 3.F1) gets ``TIMESTAMP WITH TIME ZONE`` from the same
  declaration.
* Indexes target the hot scheduling-query paths per DEC-035 #3 —
  ``(state, created_at, id)`` for FIFO scheduling-window walks and
  per-state count aggregation; ``(parent_id, created_at, id)`` for
  ``list_*`` CRUD queries by parent.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
)

# Single ``MetaData`` instance shared by all tables in this plugin. The
# ``Postgres`` plugin (Task 3.F1) imports the same module — dialect
# variations are rendered at engine-bind time, not declaration time.
metadata: MetaData = MetaData()


# === ComparisonGroupRecord (§5.3) ============================================

cgs_table: Table = Table(
    "cgs",
    metadata,
    # Identity
    Column("id", String(64), primary_key=True),
    Column("proposal_id", String(64), nullable=False, index=True),
    Column("factor_model_id", String(64), nullable=True),
    # Methodology references
    Column("experiment_definition_id", String(64), nullable=False),
    Column("experiment_plan_hash", String(64), nullable=True),
    Column("harness_contract_hash", String(64), nullable=True),
    Column("validation_config_hash", String(64), nullable=True),
    # State
    Column("state", String(32), nullable=False),
    # Job handle (JSON blob — JobHandle.{plugin, handle, metadata})
    Column("harness_job_handle", JSON, nullable=True),
    # Code-review state (DEC-016)
    Column("code_review_attempt", Integer, nullable=False, default=0),
    Column("code_review_result_hash", String(64), nullable=True),
    # Common (BasePipelineRecord, §5.2.1 + §5.2.6)
    Column("version", Integer, nullable=False, default=0),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("last_error", Text, nullable=True),
    # Lease triple (LeaseableRecord, §5.2.4)
    Column("leased_by", String(128), nullable=True),
    Column("lease_expires_at", DateTime(timezone=True), nullable=True),
    Column("lease_nonce", String(64), nullable=True),
    # Hot scheduling-query path (DEC-035 #3): (state, created_at, id) anchors
    # FIFO walks for ``get_ready_for_*`` / ``get_in_flight_*`` queries.
    Index("ix_cgs_state_created_id", "state", "created_at", "id"),
)


# === EntryRecord (§5.4) ======================================================

entries_table: Table = Table(
    "entries",
    metadata,
    # Identity
    Column("id", String(64), primary_key=True),
    Column("cg_id", String(64), ForeignKey("cgs.id"), nullable=False),
    Column("technique_id", String(64), nullable=True),
    Column("is_baseline", Boolean, nullable=False),
    # Methodology references
    Column("entry_id", String(64), nullable=False),
    Column("technique_contract_hash", String(64), nullable=True),
    # Manifest reference (DEC-033 #3 — manifest hash placement on EntryRecord)
    Column("harness_api_manifest_hash", String(64), nullable=True),
    # State
    Column("state", String(32), nullable=False),
    # Job handle
    Column("implementation_job_handle", JSON, nullable=True),
    # Implementation retry counter
    Column("implementation_attempt", Integer, nullable=False, default=0),
    # Common
    Column("version", Integer, nullable=False, default=0),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("last_error", Text, nullable=True),
    # Lease triple
    Column("leased_by", String(128), nullable=True),
    Column("lease_expires_at", DateTime(timezone=True), nullable=True),
    Column("lease_nonce", String(64), nullable=True),
    Index("ix_entries_state_created_id", "state", "created_at", "id"),
    Index("ix_entries_cg_created_id", "cg_id", "created_at", "id"),
)


# === RunRecord (§5.5) ========================================================

runs_table: Table = Table(
    "runs",
    metadata,
    # Identity
    Column("id", String(64), primary_key=True),
    Column("cg_id", String(64), ForeignKey("cgs.id"), nullable=False),
    Column("entry_id", String(64), ForeignKey("entries.id"), nullable=False),
    Column("seed", Integer, nullable=False),
    # State
    Column("state", String(32), nullable=False),
    # Job handle
    Column("compute_job_handle", JSON, nullable=True),
    # Telemetry
    Column("raw_metrics_artifact_key", String(512), nullable=True),
    Column("duration_seconds", Float(), nullable=True),
    Column("started_at", DateTime(timezone=True), nullable=True),
    Column("completed_at", DateTime(timezone=True), nullable=True),
    # Failure / retry
    Column("run_attempt", Integer, nullable=False, default=0),
    Column("failure_reason", Text, nullable=True),
    # Common
    Column("version", Integer, nullable=False, default=0),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("last_error", Text, nullable=True),
    # Lease triple
    Column("leased_by", String(128), nullable=True),
    Column("lease_expires_at", DateTime(timezone=True), nullable=True),
    Column("lease_nonce", String(64), nullable=True),
    Index("ix_runs_state_created_id", "state", "created_at", "id"),
    Index("ix_runs_entry_created_id", "entry_id", "created_at", "id"),
)


# === ProposalRecord (§5.6) ===================================================

proposals_table: Table = Table(
    "proposals",
    metadata,
    # Identity
    Column("id", String(64), primary_key=True),
    Column("submitted_by", String(128), nullable=True),
    # Submission shape
    Column("submission_kind", String(32), nullable=False),
    Column("technique_description_artifact_key", String(512), nullable=True),
    Column("reproduce_paper_arxiv_id", String(64), nullable=True),
    # State
    Column("state", String(32), nullable=False),
    # Designing-stage agent state
    Column("planner_job_handle", JSON, nullable=True),
    Column("design_plan_artifact_key", String(512), nullable=True),
    Column("error_context_artifact_key", String(512), nullable=True),
    # Retry counters
    Column("design_attempt", Integer, nullable=False, default=0),
    Column("registration_attempt", Integer, nullable=False, default=0),
    # User-decision
    Column("user_decision", String(32), nullable=True),
    Column("user_decided_at", DateTime(timezone=True), nullable=True),
    # Common
    Column("version", Integer, nullable=False, default=0),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("last_error", Text, nullable=True),
    # Lease triple
    Column("leased_by", String(128), nullable=True),
    Column("lease_expires_at", DateTime(timezone=True), nullable=True),
    Column("lease_nonce", String(64), nullable=True),
    Index("ix_proposals_state_created_id", "state", "created_at", "id"),
)


# === PaperRecord (§5.7) ======================================================

# Primary key is ``arxiv_id`` per ``07-plugin-interfaces.md`` §5.3
# (``get_paper(arxiv_id)``) — paper records do not get a separate ULID.
papers_table: Table = Table(
    "papers",
    metadata,
    Column("arxiv_id", String(64), primary_key=True),
    # Bibliographic metadata (DEC-009)
    Column("title", Text, nullable=True),
    Column("authors", JSON, nullable=True),  # list[str]
    Column("abstract", Text, nullable=True),
    Column("published_date", DateTime(timezone=True), nullable=True),
    Column("categories", JSON, nullable=True),  # list[str]
    # Artifact pointers
    Column("latex_bundle_artifact_key", String(512), nullable=True),
    Column("expanded_tex_artifact_key", String(512), nullable=True),
    Column("extracted_text_artifact_key", String(512), nullable=True),
    Column("figures_artifact_key", String(512), nullable=True),
    # Screening result
    Column("screen_result_decision", String(32), nullable=True),
    Column("screen_result_reason", Text, nullable=True),
    # State
    Column("state", String(32), nullable=False),
    # Per-stage job handles
    Column("fetcher_job_handle", JSON, nullable=True),
    Column("screener_job_handle", JSON, nullable=True),
    Column("planner_job_handle", JSON, nullable=True),
    # Per-stage artifact pointers
    Column("technique_buffer_artifact_key", String(512), nullable=True),
    Column("error_context_artifact_key", String(512), nullable=True),
    # Retry counters
    Column("planning_attempt", Integer, nullable=False, default=0),
    Column("screening_attempt", Integer, nullable=False, default=0),
    Column("registration_attempt", Integer, nullable=False, default=0),
    # Common
    Column("version", Integer, nullable=False, default=0),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("last_error", Text, nullable=True),
    # Lease triple
    Column("leased_by", String(128), nullable=True),
    Column("lease_expires_at", DateTime(timezone=True), nullable=True),
    Column("lease_nonce", String(64), nullable=True),
    Index("ix_papers_state_created_arxiv", "state", "created_at", "arxiv_id"),
)


# === FactorModelRecord (§5.8) ================================================
#
# Degenerate lifecycle per DEC-031 #5 — no state machine, no lease, no
# retry counters. Per DEC-033's transition-log exception: no rows ever
# appear in ``transition_log`` for ``factor_model``-kind entities.

factor_models_table: Table = Table(
    "factor_models",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("factor_model_id", String(64), nullable=False),
    Column("research_question", Text, nullable=False),
    # Common (no lease triple per §5.2.4)
    Column("version", Integer, nullable=False, default=0),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("last_error", Text, nullable=True),
)


# === TechniqueRef registry mirror (§5.3 second block) ========================
#
# Methodology entity lookup, exposed via ``get_technique`` /
# ``list_techniques`` / ``upsert_technique``. Stored shape is the
# ``TechniqueRef`` Pydantic model serialized to JSON; the row carries
# enough denormalized fields for the ``list_techniques_for_paper`` query
# (``fidelity_anchor->>arxiv_id``) without needing JSON-path support.

techniques_table: Table = Table(
    "techniques",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("name", String(256), nullable=False),
    Column("description", Text, nullable=False),
    Column("category", String(128), nullable=False),
    Column("compatible_factor_types", JSON, nullable=False),  # list[str]
    Column("standard", Boolean, nullable=False, default=False),
    # Denormalized for ``list_techniques_for_paper`` query — duplicated
    # from ``fidelity_anchor`` JSON when the anchor is a paper.
    Column("fidelity_anchor", JSON, nullable=True),
    Column("fidelity_anchor_kind", String(32), nullable=True),
    Column("fidelity_anchor_arxiv_id", String(64), nullable=True, index=True),
    Column("affects_extension_points", JSON, nullable=False),  # list[str]
    Column("implies_controlled", JSON, nullable=False),  # list[str]
    Column("parameter_schema", JSON, nullable=True),
)


# === transition_log (DEC-033 #1) =============================================

transition_log_table: Table = Table(
    "transition_log",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("entity_kind", String(32), nullable=False),
    Column("entity_id", String(64), nullable=False),
    Column("from_state", String(32), nullable=True),
    Column("to_state", String(32), nullable=False),
    Column("edge_name", String(128), nullable=True),
    Column("gate_outcome_reason", Text, nullable=True),
    Column("worker_id", String(128), nullable=True),
    Column("occurred_at", DateTime(timezone=True), nullable=False),
    # Per ``05-orchestrator.md`` §9 OQ5 carry-forward — operator
    # queries run as ``WHERE entity_kind = ? AND entity_id = ? ORDER BY
    # occurred_at``. The composite index serves both the per-entity
    # replay and the cross-cycle telemetry rollup.
    Index("ix_transition_log_entity", "entity_kind", "entity_id", "occurred_at"),
)


# === AgentSession cost table (DEC-033 #2) ====================================

agent_sessions_table: Table = Table(
    "agent_sessions",
    metadata,
    Column("id", String(64), primary_key=True),
    # Polymorphic FK — single (parent_kind, parent_id) discriminator
    # rather than per-kind nullable FK columns. Reasoning is in the
    # plugin's _records.py docstring; short version: keeps the column
    # set bounded as the entity-kind set evolves, and the (kind, id)
    # composite index gives fast per-parent rollups.
    Column("parent_kind", String(32), nullable=False),
    Column("parent_id", String(64), nullable=False),
    # Role / provider / model
    Column("agent_role", String(64), nullable=False),
    Column("llm_provider", String(64), nullable=True),
    Column("model_id", String(128), nullable=True),
    # Realized totals
    Column("input_tokens", Integer, nullable=False, default=0),
    Column("cached_input_tokens", Integer, nullable=False, default=0),
    Column("output_tokens", Integer, nullable=False, default=0),
    Column("turn_count", Integer, nullable=False, default=0),
    # Rates at session time (USD per 1M tokens)
    Column("input_token_rate_usd_per_million", Float(), nullable=True),
    Column("cached_input_token_rate_usd_per_million", Float(), nullable=True),
    Column("output_token_rate_usd_per_million", Float(), nullable=True),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("ended_at", DateTime(timezone=True), nullable=True),
    Index("ix_agent_sessions_parent", "parent_kind", "parent_id", "started_at"),
)


# === RunCost cost table (DEC-033 #2) =========================================

run_costs_table: Table = Table(
    "run_costs",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.id"), nullable=False, unique=True),
    Column("cg_id", String(64), nullable=False, index=True),
    Column("gpu_hours", Float(), nullable=False, default=0.0),
    Column("gpu_hourly_rate_usd", Float(), nullable=True),
    Column("container_memory_gb", Float(), nullable=True),
    Column("compute_plugin", String(64), nullable=True),
    Column("recorded_at", DateTime(timezone=True), nullable=False),
)


# === Per-entity-kind handle to the right table for generic helpers ===========

# Used by lease ops, count queries, and transition CAS to dispatch on
# ``entity_kind`` without a long ``if/elif`` ladder. The PK column name
# differs for ``papers`` (``arxiv_id``); the ``ENTITY_PK_COLUMN`` map
# encodes this so generic helpers can build ``WHERE pk = :id``
# transparently.

ENTITY_TABLE: dict[str, Table] = {
    "cg": cgs_table,
    "entry": entries_table,
    "run": runs_table,
    "proposal": proposals_table,
    "paper": papers_table,
}

ENTITY_PK_COLUMN: dict[str, str] = {
    "cg": "id",
    "entry": "id",
    "run": "id",
    "proposal": "id",
    "paper": "arxiv_id",
}


__all__ = [
    "ENTITY_PK_COLUMN",
    "ENTITY_TABLE",
    "agent_sessions_table",
    "cgs_table",
    "entries_table",
    "factor_models_table",
    "metadata",
    "papers_table",
    "proposals_table",
    "run_costs_table",
    "runs_table",
    "techniques_table",
    "transition_log_table",
]
