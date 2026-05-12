"""Helpers wiring the agent loop's ``progress_sink`` to the
``agent_sessions`` MetadataStore telemetry row (DEC-033 #2).

The agent loop (:mod:`smai_agents.loop`) is deliberately
persistence-agnostic — it never imports a :class:`MetadataStore` type;
it just calls an opaque ``Callable[[AgentSession], Awaitable[None]]`` on
the same cadence as status writes. The orchestrator's dispatch handlers
(planner / harness builder / technique implementer) supply that closure;
this module is the one place that knows how to turn a session id + a
:class:`MetadataStore` into the closure, plus the open / close bookkeeping
around a dispatched session.

Everything here is best-effort: telemetry writes must never break a
dispatch or the loop. A store that doesn't implement the methods (an
older plugin) just gets no telemetry.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from smai_core.plugins import MetadataStore

if TYPE_CHECKING:
    from smai_agents.loop import AgentOutcome, AgentSession


async def open_agent_session(
    metadata_store: MetadataStore,
    *,
    parent_kind: str,
    parent_id: str,
    agent_role: str,
    llm: object,
) -> str | None:
    """Create the ``agent_sessions`` row at dispatch start; return its id.

    ``llm`` is the role's :class:`smai_core.plugins.LlmProvider`; the
    provider ``name`` and ``capabilities.model_id`` are read best-effort.
    Returns ``None`` if the store doesn't support the method (or any
    error) — callers treat ``None`` as "no telemetry, carry on".
    """
    provider_name = getattr(llm, "name", "(unknown)")
    model_id = getattr(getattr(llm, "capabilities", None), "model_id", "(unknown)")
    try:
        return await metadata_store.create_agent_session(
            parent_kind=parent_kind,
            parent_id=parent_id,
            agent_role=agent_role,
            llm_provider=str(provider_name),
            model_id=str(model_id),
        )
    except Exception:  # noqa: BLE001 — telemetry must not break dispatch
        return None


def make_progress_sink(
    metadata_store: MetadataStore,
    session_id: str | None,
) -> Callable[[AgentSession], Awaitable[None]] | None:
    """Build the loop's per-turn ``progress_sink`` for a tracked session.

    Returns ``None`` when ``session_id`` is ``None`` (no telemetry row);
    otherwise a closure that UPDATEs the running token / turn counts.
    All errors are swallowed — a flaky telemetry write must not abort the
    agent loop.
    """
    if session_id is None:
        return None
    captured_id = session_id

    async def _sink(session: AgentSession) -> None:
        try:
            await metadata_store.update_agent_session(
                captured_id,
                turn_count=session.turn_count,
                input_tokens=session.usage_total.input_tokens,
                cached_input_tokens=session.usage_total.cache_read_tokens,
                output_tokens=session.usage_total.output_tokens,
            )
        except Exception:  # noqa: BLE001 — telemetry must not break the loop
            pass

    return _sink


async def close_agent_session(
    metadata_store: MetadataStore,
    session_id: str | None,
    outcome: AgentOutcome | None,
) -> None:
    """Final UPDATE on loop exit — write ``ended_at`` (+ the realized
    totals when an :class:`AgentOutcome` is available).

    Per round-6 item D, callers wrap their ``run_*_session`` call in a
    ``try/finally`` and pass ``outcome=None`` from the ``finally`` so
    the ``agent_sessions`` row is closed even when the session raised
    (it would otherwise be left with ``ended_at=NULL`` forever)."""
    if session_id is None:
        return
    try:
        if outcome is not None:
            turn_count = outcome.turn_count
            input_tokens = outcome.usage_total.input_tokens
            cached_input_tokens = outcome.usage_total.cache_read_tokens
            output_tokens = outcome.usage_total.output_tokens
        else:
            # No outcome (the session raised) — preserve whatever totals
            # the progress_sink last wrote rather than clobbering them
            # with zeros; just stamp ``ended_at``.
            prior = await metadata_store.get_agent_session(session_id)
            turn_count = prior.turn_count if prior is not None else 0
            input_tokens = prior.input_tokens if prior is not None else 0
            cached_input_tokens = prior.cached_input_tokens if prior is not None else 0
            output_tokens = prior.output_tokens if prior is not None else 0
        await metadata_store.update_agent_session(
            session_id,
            turn_count=turn_count,
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
            ended_at=datetime.now(UTC),
        )
    except Exception:  # noqa: BLE001 — telemetry must not break dispatch
        pass


__all__ = ["close_agent_session", "make_progress_sink", "open_agent_session"]
