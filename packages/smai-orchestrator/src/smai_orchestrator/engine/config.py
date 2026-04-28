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


__all__ = ["EngineConfig", "RetryPolicy"]
