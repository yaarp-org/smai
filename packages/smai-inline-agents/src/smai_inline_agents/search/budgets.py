"""Per-session call-count + per-provider rate-limit bookkeeping.

Per ``planner_refactor/design_notes/search_tool_surface.md`` §7 (cost
ceilings) and §9 (failure / retry policy).

This module ships layer-3 of the three-layer budget structure
(``architectural_decisions.md`` §11): forced-synthesis tool-budget
exhaustion. Layer 1 (PydanticAI ``UsageLimits``) is wired in Step 4
when the lit-review phase agent lands; Layer 2 (prompt-embedded
scaling rules) lives in the lit-review prompt and is iterated there.

The :class:`BudgetTracker` is constructed once per planner session and
the dispatcher checks + increments it before each provider call.
Hitting the per-session cap raises
:class:`~smai_inline_agents.search.errors.BudgetExhausted`; hitting a
per-provider rate-limit window raises
:class:`~smai_inline_agents.search.errors.RateLimitExceeded` (the
dispatcher backs off + retries per §9).
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from smai_inline_agents.search.config import BudgetsConfig
from smai_inline_agents.search.errors import BudgetExhausted, RateLimitExceeded


@dataclass
class RateLimit:
    """Leaky-bucket rate-limit spec for a single provider.

    ``max_calls`` per rolling ``window_seconds``. Multi-call windows
    (e.g. Semantic Scholar's 100 req / 5 min public pool) collapse
    to ``max_calls=100, window_seconds=300``.
    """

    max_calls: int
    window_seconds: float


# Conservative default rate-limits for the registered v1 providers.
# Mirrors ``search_tool_surface.md`` §6 (per-provider implementation
# notes) -- Semantic Scholar public pool runs at 1/sec to leave headroom
# for retries; arXiv's documented politeness floor is 3 seconds.
DEFAULT_PROVIDER_RATE_LIMITS: dict[str, RateLimit] = {
    "semantic_scholar": RateLimit(max_calls=1, window_seconds=1.0),
    "arxiv": RateLimit(max_calls=1, window_seconds=3.0),
}


class BudgetTracker:
    """Tracks per-tool call counts + per-provider rate-limit windows.

    One instance per planner session; constructed by
    :func:`~smai_inline_agents.search.dispatch.build_search_deps`.

    Layer-3 enforcement: after K calls of a capability, the session
    cap is hit and the dispatcher raises
    :class:`BudgetExhausted` for that capability. PydanticAI surfaces
    the error to the agent as a tool error, and the lit-review prompt's
    forced-synthesis instruction kicks in.

    Per-provider rate-limit windows use a per-provider asyncio.Semaphore
    plus a leaky-bucket deque of recent call timestamps. The check is
    non-blocking: if the window is full the tracker raises
    :class:`RateLimitExceeded` rather than sleeping inside
    ``check_and_increment``. The dispatcher decides whether to wait +
    retry (per §9 -- one retry with 500ms backoff for 5xx / network /
    rate-limit errors) or surface the error to the agent.
    """

    def __init__(
        self,
        *,
        web_search_cap_per_session: int = 10,
        academic_search_cap_per_session: int = 20,
        fetch_url_cap_per_session: int = 20,
        per_provider_rate_limit: dict[str, RateLimit] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._caps: dict[str, int] = {
            "web_search": web_search_cap_per_session,
            "academic_search": academic_search_cap_per_session,
            "fetch_url": fetch_url_cap_per_session,
        }
        self._counts: dict[str, int] = {k: 0 for k in self._caps}
        self._rate_limits: dict[str, RateLimit] = (
            dict(per_provider_rate_limit)
            if per_provider_rate_limit is not None
            else dict(DEFAULT_PROVIDER_RATE_LIMITS)
        )
        # Per-provider call-timestamp deques (leaky bucket).
        self._provider_windows: dict[str, deque[float]] = {}
        # Per-provider lock so concurrent fan-out calls serialize on the
        # increment + check_window operation.
        self._provider_locks: dict[str, asyncio.Lock] = {}
        # Clock seam for deterministic tests. Default is asyncio's loop time;
        # tests can inject a fake clock callable returning a float.
        self._clock = clock

    def _now(self) -> float:
        if self._clock is not None:
            return self._clock()
        # asyncio loop time -- monotonic, suitable for rate-limit math.
        return asyncio.get_event_loop().time()

    @classmethod
    def from_config(cls, budgets: BudgetsConfig) -> BudgetTracker:
        return cls(
            web_search_cap_per_session=budgets.web_search_cap_per_session,
            academic_search_cap_per_session=budgets.academic_search_cap_per_session,
            fetch_url_cap_per_session=budgets.fetch_url_cap_per_session,
        )

    def get_count(self, capability: str) -> int:
        """Read-only accessor for unit tests + egress-log enrichment."""
        return self._counts.get(capability, 0)

    def remaining(self, capability: str) -> int:
        """Calls remaining before :class:`BudgetExhausted` for this
        capability. Returns 0 once exhausted (negative values are
        clamped)."""
        cap = self._caps.get(capability, 0)
        return max(0, cap - self._counts.get(capability, 0))

    async def check_and_increment(self, capability: str, provider: str) -> None:
        """Raise :class:`BudgetExhausted` if the session cap is hit;
        raise :class:`RateLimitExceeded` if the per-provider window is
        hit.

        Increments the per-capability counter even on rate-limit-exceeded
        so a thrashing agent burns its session cap rather than looping
        on a 429.
        """
        if capability not in self._caps:
            raise ValueError(f"unknown capability: {capability!r}")
        cap = self._caps[capability]
        if self._counts[capability] >= cap:
            raise BudgetExhausted(
                f"session cap reached for {capability!r}: {self._counts[capability]} >= {cap}",
            )
        self._counts[capability] += 1

        rl = self._rate_limits.get(provider)
        if rl is None:
            return

        lock = self._provider_locks.setdefault(provider, asyncio.Lock())
        window = self._provider_windows.setdefault(provider, deque())
        async with lock:
            now = self._now()
            cutoff = now - rl.window_seconds
            while window and window[0] < cutoff:
                window.popleft()
            if len(window) >= rl.max_calls:
                raise RateLimitExceeded(
                    f"provider {provider!r} window full: {len(window)} >= "
                    f"{rl.max_calls} in last {rl.window_seconds}s",
                )
            window.append(now)


__all__ = [
    "DEFAULT_PROVIDER_RATE_LIMITS",
    "BudgetTracker",
    "RateLimit",
]
