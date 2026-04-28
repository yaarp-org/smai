"""Plugin-internal Pydantic types for the DEC-033 normalized cost tables.

Per ``designs/decisions.md`` DEC-033 (Pipeline-tracking entity architecture)
and Task 2.A2's "Carry-forward from Task 1.10" hand-off.

The three normalized cost tables — ``transition_log`` (event-sourced edge
audit), ``AgentSession`` (token-shaped agent costs), ``RunCost`` (GPU /
container shaped run costs) — are not in ``01-data-model.md`` §5's six
record types because they are *cross-cutting* tables, not per-entity
records. ``MetadataStore`` Protocol §5.3 does NOT yet enumerate CRUD
methods for them (the Protocol is Session-C-pending on this surface per
DEC-033 affected-area note); they are written transactionally as part of
``transition_*_state`` and the agent-loop / run-dispatcher write paths,
and read by aggregation SQL inside the plugin.

These types live **plugin-internal** rather than in
``smai-orchestrator.entities.tracking`` for two reasons:

1. The Protocol surface does not currently reference them — adding them
   to the orchestrator package would be premature commitment to a public
   shape before the corresponding ``MetadataStore`` CRUD methods are
   designed.
2. Keeping them here lets the SQLite reference plugin and the eventual
   Postgres plugin (Task 3.F1) settle column shapes and indexes
   together, then graduate the public Pydantic types when the Protocol
   surface lands the CRUD methods (Session C of ``07-plugin-interfaces.md``).

When the Protocol surface gains CRUD for these tables, this module's
type signatures graduate to ``smai-orchestrator.entities.tracking``
verbatim — the column choices made here are the candidate-default for
that promotion.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Polymorphic-FK discriminator for ``AgentSession.parent_kind``.
# Per the Task 2.A2 spec's open question on ``§5.10`` polymorphic FK
# shape: chose **single ``parent_kind`` + ``parent_id``** columns over
# per-kind FK columns (``cg_id``, ``entry_id``, ``proposal_id``,
# ``paper_id``) gated by a ``CHECK (cardinality(...) = 1)`` constraint.
# Reasoning is documented in the Task 2.A2 report under "DEC-033
# cost-table schemas" — short version: per-kind columns scale with the
# entity count, force a CHECK constraint per dialect, and force every
# read query to ``COALESCE`` across columns; the discriminator pattern
# costs one indexed lookup ``(parent_kind, parent_id)`` and stays correct
# under entity-set evolution.
#
# Note: ``run`` is NOT in this set — agent sessions parent CGs, entries,
# proposals, and papers (the four entity kinds with agent-driven
# dispatch in ``03-state-machine.md`` §3 / §4 / §5). Run-stage cost is
# captured in ``RunCost`` per row, not as agent-session work.
AgentSessionParentKind = Literal["cg", "entry", "proposal", "paper"]


# Agent-role discriminator. The seven agent roles in ``04-agents.md`` §2
# plus the proposal-pipeline planner per ``08`` §3 / §5.3. Open as a
# wide ``str`` for forward-compat — ``Literal`` would lock the set
# before all role-specific cost shapes (e.g., contextual evaluator's
# inline-call shape) have been exercised in production.


class TransitionLogEntry(BaseModel):
    """One state-transition event (DEC-033 #1).

    Append-only audit row, one per successful CAS state transition.
    Foreign-keyed to the entity by ``(entity_kind, entity_id)`` —
    polymorphic in the same shape as :class:`AgentSession`. Per
    DEC-033's reasoning: keeping audit history on a separate
    append-only table preserves the SQL substrate's natural shape for
    transition queries (``WHERE entity_id = :x ORDER BY occurred_at``)
    and avoids the write-amplification of round-tripping a
    ``transition_history: list[Transition]`` column on every CAS
    UPDATE.

    ``from_state`` is ``None`` on creation rows (``create_*`` methods
    insert the entity and append a creation row with
    ``from_state=None``, ``to_state=<initial>``).

    ``edge_name`` is the gate name that fired (e.g.,
    ``cg.dispatch_implementation``), populated by the engine when it
    drives the transition; ``None`` for direct creation.

    ``gate_outcome_reason`` carries the gate's textual reason —
    populated for transitions taken under a gate-rule decision, ``None``
    for direct creation or for transitions that don't go through a gate.
    """

    model_config = ConfigDict(extra="forbid")

    id: int | None = None
    entity_kind: Literal["cg", "entry", "run", "proposal", "paper"]
    entity_id: str
    from_state: str | None = None
    to_state: str
    edge_name: str | None = None
    gate_outcome_reason: str | None = None
    worker_id: str | None = None
    occurred_at: datetime


class AgentSessionRecord(BaseModel):
    """Per-session agent-loop cost record (DEC-033 #2).

    One row per agent invocation; the agent loop writes this on session
    end with the realized token totals plus the rates recorded **at
    session time** (per the v1 principle "pricing is unstable; store
    the factual data" — `token_metering.md` §1). Aggregate-on-query via
    ``SELECT SUM(input_tokens * input_token_rate_usd + ...) FROM
    agent_sessions WHERE parent_kind = ... AND parent_id = ...``.

    Rates are stored as USD-per-1M-tokens (the unit Bedrock /
    Anthropic / OpenAI all surface) — keep multiplicands together to
    avoid floating-point loss on a per-token rate. Cached input tokens
    are tracked separately because their rate is typically 0.1× the
    base input rate (per `prompt_caching.md`) and aggregation queries
    want the breakdown.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    parent_kind: AgentSessionParentKind
    parent_id: str

    # Role + provider + model — informational, also indexable for
    # per-role / per-model aggregations (the `dao_design.md` §2 GSI3 /
    # GSI4 carry-forward).
    agent_role: str
    llm_provider: str | None = None
    model_id: str | None = None

    # Realized token totals.
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    turn_count: int = 0

    # Rates at session time (USD per 1M tokens). Nullable for
    # provider-internal sessions where rate metadata is unavailable
    # (the aggregate query treats ``NULL`` as "no cost contribution",
    # which matches v1's defensive behavior).
    input_token_rate_usd_per_million: float | None = None
    cached_input_token_rate_usd_per_million: float | None = None
    output_token_rate_usd_per_million: float | None = None

    started_at: datetime
    ended_at: datetime | None = None


class RunCostRecord(BaseModel):
    """Per-run GPU / container cost record (DEC-033 #2).

    One row per :class:`RunRecord`; the run dispatcher writes this on
    run completion (``succeeded`` / ``failed`` / ``inconclusive``
    terminals) with the realized GPU-hours + the hourly rate at
    completion time. ``container_memory_gb`` is recorded when the
    plugin surfaces it (Modal does, AWS Batch does for some queue
    families) — nullable for plugins that don't.

    1:1 with :class:`RunRecord` via ``run_id`` UNIQUE; the table exists
    as a separate row (rather than additional columns on ``runs``)
    because the run dispatcher writes cost data on a different edge
    than the state transitions, and the separation makes the run-table
    write-path single-purpose (CAS the state, separately upsert the
    cost row).
    """

    model_config = ConfigDict(extra="forbid")

    id: str  # equals run_id; stored separately for FK indexing
    run_id: str
    cg_id: str

    gpu_hours: float = 0.0
    gpu_hourly_rate_usd: float | None = None
    container_memory_gb: float | None = None
    compute_plugin: str | None = None

    recorded_at: datetime = Field(default_factory=lambda: datetime.fromtimestamp(0))


__all__ = [
    "AgentSessionParentKind",
    "AgentSessionRecord",
    "RunCostRecord",
    "TransitionLogEntry",
]
