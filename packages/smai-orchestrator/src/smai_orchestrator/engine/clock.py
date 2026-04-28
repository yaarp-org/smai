"""Time-source abstraction for the engine.

Per ``designs/smai/implementation_plan.md`` §6 Q4 (wall-clock conformance
test seam): every clock read in the engine goes through
:attr:`EngineConfig.time_provider` rather than a naked ``time.monotonic()``
call. Tests inject a fake clock through the same seam to verify
time-dependent behavior (orphan-grace expiry, lease renewal cadence,
retry backoff) without sleeping for real wall-clock time.

The seam is intentionally minimal — production code uses
:func:`time.monotonic` (the default), tests construct their own
``FakeClock`` substrate; this module exports only the type alias and
the default callable so production wiring is uncluttered.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime

# Type alias for the monotonic seam. ``float`` (seconds since an opaque
# epoch) is the right shape for elapsed-time arithmetic where the engine
# sets the start point itself (retry backoff, lease-renewal cadence,
# poll-loop timing); we don't promise a wall-clock-correct ``datetime``.
# Tests subclass-and-override to advance the clock without sleeping.
# Matches the conformance-base ``TimeProvider`` shape under
# ``smai_core.plugins.conformance._common`` for consistency.
TimeProvider = Callable[[], float]

# Wall-clock seam, separate from :data:`TimeProvider`. Phase-1 orphan
# detection (`05` §3.1) compares ``orphan_grace_seconds`` against the
# entity's ``updated_at`` field — a wall-clock ``datetime``, populated
# by ``MetadataStore``'s ``transition_*_state`` UPDATE. Monotonic time
# is not comparable to ``updated_at``; a second seam is the clean
# resolution. Tests inject a fake datetime here for the orphan-grace
# tests; production uses :func:`datetime.now` (UTC).
WallClockProvider = Callable[[], datetime]


# Default implementations.
default_time_provider: TimeProvider = time.monotonic


def default_wall_clock() -> datetime:
    """Production wall-clock — ``datetime.now(UTC)`` (`05` §3.1)."""
    return datetime.now(UTC)


__all__ = [
    "TimeProvider",
    "WallClockProvider",
    "default_time_provider",
    "default_wall_clock",
]
