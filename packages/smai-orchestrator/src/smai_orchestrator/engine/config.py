"""Engine configuration.

Per ``designs/smai/05-orchestrator.md`` §6. The C1 task settled the
engine-driver fields (clock seams, orphan grace, lease seconds, retry
policies); Task 2.C2 grew this with the worker-loop fields
(``poll_interval_seconds``, ``worker_count``, ``pool_overrides``,
``fair_scheduling``, ``fair_scheduling_weights``) per `05` §6 / DEC-034
#4. C1 fields are unchanged.

The fields fall into three groups:

* **Engine driver** — :attr:`time_provider`, :attr:`wall_clock`,
  :attr:`orphan_grace_seconds`, :attr:`lease_seconds`,
  :attr:`retry_policies` (Task 2.C1).
* **Worker loop** — :attr:`poll_interval_seconds`, :attr:`worker_count`,
  :attr:`pool_overrides`, :attr:`fair_scheduling`,
  :attr:`fair_scheduling_weights` (Task 2.C2).
* **Plugin selection** — separate top-level :class:`RuntimeConfig`,
  Task 2.C3.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from smai_events import EventChannel, NullEventChannel

from smai_orchestrator.engine.clock import (
    TimeProvider,
    WallClockProvider,
    default_time_provider,
    default_wall_clock,
)


class RetryPolicy(BaseModel):
    """Named backoff configuration for retry-edge gate rules (`05` §6).

    Gate rules consult a :class:`RetryPolicy` from
    :attr:`EngineConfig.retry_policies` when deciding whether a failed
    attempt should be retried (advance to the prior state) or abandoned
    (advance to a terminal-fail state). The policy's *application* is
    the gate rule's responsibility — this object just carries the
    knobs.
    """

    model_config = ConfigDict(extra="forbid")

    max_attempts: int
    backoff_seconds: int = 30
    backoff_multiplier: float = 2.0


class EngineConfig(BaseModel):
    """Engine-behavior configuration — separable from plugin selection
    per `05` §6 / DEC-028.

    Per the design doc, ``EngineConfig`` is co-loaded with plugin
    selection through the same config-layering pipeline (env →
    ``smai.yaml`` → flags); the eventual top-level :class:`RuntimeConfig`
    composes both. The C1 subset here is sufficient for the engine
    primitives; Task 2.C3 lifts this into the full shape per `05` §6
    (additive — every C1 field stays).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    time_provider: TimeProvider = default_time_provider
    """Monotonic-style accessor for engine-set elapsed-time arithmetic
    (retry backoff, poll-loop cadence, etc.); tests inject a fake clock
    here per `implementation_plan.md` §6 Q4. Production default is
    :func:`time.monotonic` (`clock.default_time_provider`)."""

    wall_clock: WallClockProvider = default_wall_clock
    """Wall-clock seam for datetime comparisons against record fields
    (specifically ``record.updated_at`` for phase-1 orphan detection
    per `05` §3.1). Separate from :attr:`time_provider` because monotonic
    seconds and ``updated_at`` (a UTC ``datetime``) are not comparable
    types. Tests inject a fake datetime here for orphan-grace cases."""

    orphan_grace_seconds: int = 600
    """Phase-1 orphan detection threshold (`05` §3.1). Entities in an
    in-progress state with a null ``job_handle`` past this many seconds
    are reset to the prior state. The default mirrors `05` §6's stated
    default."""

    lease_seconds: int = 120
    """Lease lifetime for multi-worker contention (`05` §3.5 / DEC-035
    #2). C1 runs single-worker and does not acquire leases; Task 2.C2 /
    3.G1 wire this through the worker loop."""

    extend_lease_interval_seconds: int = 60
    """Heartbeat cadence for long-running inline dispatches (`05` §3.5).
    The dispatch wrapper spawns a background task that calls
    :meth:`MetadataStore.extend_lease` every ``extend_lease_interval_seconds``
    while the dispatch is in flight. Default ``60`` is half of the
    default :attr:`lease_seconds=120`, matching `05` §3.5's "renew at
    half the lease window" sketch — one renewal halfway through the
    window keeps a lease alive across a typical multi-turn agent loop
    without flooding the store with extends. A dispatch that finishes
    in under :attr:`extend_lease_interval_seconds` triggers zero
    extension calls (the heartbeat task is cancelled before its first
    sleep elapses)."""

    retry_policies: dict[str, RetryPolicy] = Field(default_factory=dict)
    """Named retry policies referenced by retry-edge gate rules (`05`
    §6). The spec declares which retry policy each failure edge
    consults; the deployment configures the actual numbers via this
    map."""

    # ---- Worker-loop fields (Task 2.C2) -----------------------------------

    poll_interval_seconds: int = 30
    """Worker-loop cadence in seconds between cycles (`05` §6). The
    `smai dev` deployment uses 10 (lower latency for interactive dev);
    self-hosted production and the hosted backend use 30. Tests inject
    a small value (or 0) and rely on the :attr:`time_provider` seam to
    advance the loop deterministically."""

    worker_count: int = 1
    """Number of parallel worker processes (`05` §3.5 / §6). Default 1
    matches the v1 Lambda-with-reserved-concurrency-1 shape and the
    `smai dev` deployment. ``>1`` implies multi-worker leasing — the
    :class:`MetadataStore` plugin must support it; Task 3.G1 wires
    leasing into the worker loop. C2 ships single-worker only."""

    pool_overrides: dict[str, int] = Field(default_factory=dict)
    """Per-deployment override map for pipeline-spec pool limits (`05`
    §6). Keyed by :class:`ConcurrencyPool.name`; value replaces the
    spec's declared :attr:`ConcurrencyPool.limit`. Spec defaults stand
    if a pool name is absent from the map."""

    fair_scheduling: Literal["off", "round_robin", "weighted"] = "off"
    """Top-level fair-scheduling policy passed into the
    :class:`MetadataStore` plugin's discovery queries (`05` §3.4 / §6).
    ``"off"`` — single-tenant, FIFO discovery. ``"round_robin"`` /
    ``"weighted"`` — multi-tenant; the SQL-shaped plugin implements
    the policy via window functions per DEC-030. C2 plumbs the field
    through the spec's :class:`SchedulingQueryRef` but does not
    interpret it — that's plugin behavior."""

    fair_scheduling_weights: dict[str, float] | None = None
    """Per-tenant weights map for ``fair_scheduling="weighted"`` (`05`
    §6). Tenant id → weight. Only meaningful when
    :attr:`fair_scheduling` is ``"weighted"``; ignored otherwise."""

    # ---- Supervisor (Task 3.G4 / `04-agents.md` §2.6 / §15 OQ2) -----------

    supervisor_enabled: bool = True
    """Master switch for the between-turn supervisor check
    (``04-agents.md`` §2.6). Default-on per the Task 3.G4 brief; the
    multi-turn agent dispatch handlers (harness builder, technique
    implementer, planner) translate this into the supervised
    :class:`smai_agents.AgentLoopConfig.supervisor_check_every_turns`
    field at session-construction time. Set to ``False`` to disable
    the supervisor across the deployment without changing per-handler
    code.

    Per §15 OQ2 the *invocation lifecycle* (where in the worker loop
    the stuck-detection signal evaluation lives, what the
    configurable thresholds look like) is open; v2 ships a
    cadence-driven trigger via :attr:`supervisor_check_every_n_turns`
    and lets richer signal sources land later without a config-shape
    break."""

    supervisor_check_every_n_turns: int = 1
    """Cadence in turns for the between-turn supervisor check
    (``04-agents.md`` §2.6 / §15 OQ2). v2 default ``1`` (every turn)
    matches v1's ``agent_design.md`` §5 default. Tuned per deployment
    via the same config-layering pipeline as the rest of
    :class:`EngineConfig`. Ignored when :attr:`supervisor_enabled` is
    ``False``."""

    # ---- Runtime container images (round-4 friction A) -------------------

    runtime_image: str = "smai-runtime:dev"
    """Container image hosting *experiment seed runs* whose
    ``controlled_conditions.compute.gpu`` is ``true`` (the default). The
    reference build is ``runtime.Dockerfile`` — ``nvidia/cuda``-based,
    needs the NVIDIA Container Toolkit, amd64-only. The run-record
    dispatcher passes this to :meth:`Compute.submit` when the CG's
    :class:`HarnessContract` requests GPU. Override per deployment via
    ``smai.yaml`` (e.g. to point at a published internal image)."""

    runtime_cpu_image: str = "smai-runtime-cpu:dev"
    """Container image hosting experiment seed runs whose
    ``controlled_conditions.compute.gpu`` is ``false`` (round-4 friction
    A). The reference build is ``runtime-cpu.Dockerfile`` — lean,
    multi-arch ``python:3.11-slim-bookworm`` base with CPU torch wheels,
    so CPU-only experiments don't drag in the ~5-8 GB CUDA image and the
    macOS / Apple-Silicon path runs natively. Distinct from
    :attr:`runtime_image` and from the agent-side ``smai-agent:dev``
    (which has no ML stack). Selected automatically by the run-record
    dispatcher at the same read site as the ``gpu`` flag; override here."""

    # ---- Retention sweep (Task 3.H2 / DEC-033 #1, #2) ---------------------

    # ---- Live updates (Task 4.K2 / `12-ui-process.md` §6.4) ---------------

    event_channel: EventChannel = Field(default_factory=NullEventChannel, exclude=True)
    """Where the engine fires state-machine transitions + worker
    heartbeats (`12-ui-process.md` §6.4 RESOLVED 2026-05-03 —
    engine-wraps; ``smai-events`` sibling package OQ1 RESOLVED).

    ``exclude=True``: this is a runtime-injected callable, not config
    data — it must not appear in ``model_dump`` output (the API's
    ``GET /system/config`` route serializes ``RuntimeConfig`` and would
    otherwise raise on the un-serializable ``EventChannel`` type).

    Default :class:`NullEventChannel` (no-op) keeps existing engine
    tests + headless deployments behavior-clean. The in-band Runtime
    (``smai dev`` / ``smai ui --with-worker``) sets this to an
    :class:`InProcessEventChannel` over a process-local
    :class:`EventBroker` so the same-process API can subscribe;
    Task 4.K3 ships ``PgNotifyEventChannel`` against the same Protocol
    for the cross-process case.

    Carry-forward to Task 4.K3: the Postgres variant must run
    ``pg_notify`` inside the active asyncpg transaction (per `12` §6.5)
    so a ROLLBACK suppresses the wire signal. The in-process channel
    here has no such coupling — by the time
    ``transition_state`` returns, the row has already committed.
    """

    retention_policies: dict[str, int] = Field(default_factory=dict)
    """Per-table retention windows in days for ``smai migrate --prune``
    (DEC-033 #1, #2). Keyed by table name (``transition_log``,
    ``agent_sessions``, ``run_costs``); value is the maximum age in
    days. ``0`` disables retention for that table.

    Empty (the default) means "use the conservative defaults shipped
    in :data:`smai_orchestrator.migrations.runner.DEFAULT_RETENTION_DAYS`":
    ``transition_log = 90``, ``agent_sessions = 180``,
    ``run_costs = 365``. Operators override per deployment via this
    map; see ``MIGRATIONS.md`` next to the migrations package for the
    full table.

    The prune sweep is **never automatic at boot** — it runs only when
    ``smai migrate --prune`` is invoked explicitly (`09-cli.md` §1). A
    table not in :data:`RETENTION_TABLES` is rejected with
    :class:`ValueError` at sweep time so a misspelled key is surfaced
    rather than silently skipped."""


__all__ = ["EngineConfig", "RetryPolicy"]
