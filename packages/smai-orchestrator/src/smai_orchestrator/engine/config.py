"""Engine configuration — the C1 subset.

Per ``designs/smai/05-orchestrator.md`` §6. C1 ships only the fields the
engine substrate consumes directly:

* :attr:`EngineConfig.time_provider` — the wall-clock conformance seam
  per `implementation_plan.md` §6 Q4 (settled here).
* :attr:`EngineConfig.orphan_grace_seconds` — write-first rollback /
  orphan detection per `05` §1.4 / §3.1.
* :attr:`EngineConfig.lease_seconds` — multi-worker leasing tunable
  per `05` §3.5; C1 runs single-worker and never acquires leases (the
  field exists so C2 / 3.G1 can wire it in without re-shaping the
  config).
* :attr:`EngineConfig.retry_policies` — named retry policies referenced
  by retry-edge gate rules per `05` §6.

Out of scope, deferred to later tasks:

* ``poll_interval_seconds`` / ``worker_count`` — Task 2.C2 (worker loop).
* ``pool_overrides`` / ``fair_scheduling`` / ``fair_scheduling_weights``
  — Task 2.C3 (full :class:`RuntimeConfig`).
"""

from __future__ import annotations

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


__all__ = ["EngineConfig", "RetryPolicy"]
