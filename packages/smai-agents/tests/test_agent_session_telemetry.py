"""Round-6 item D: ``close_agent_session`` must set ``ended_at`` on every
dispatch exit path — including the one where no :class:`AgentOutcome` is
available (the session raised). Pre-round-6 the row was left with
``ended_at=NULL`` forever on the not-finalized / raised branches."""

from __future__ import annotations

from smai_agents.agent_session_telemetry import (
    close_agent_session,
    open_agent_session,
)
from smai_store_sqlite import SqliteStore


async def test_close_agent_session_sets_ended_at_without_outcome() -> None:
    store = SqliteStore("sqlite+aiosqlite:///:memory:")
    await store.migrate()
    try:

        class _Llm:
            name = "stub"
            capabilities = None

        session_id = await open_agent_session(
            store,
            parent_kind="proposal",
            parent_id="prop-1",
            agent_role="planner",
            llm=_Llm(),
        )
        assert session_id is not None
        # The "session raised, no outcome to report" branch.
        await close_agent_session(store, session_id, None)
        rec = await store.get_agent_session(session_id)
        assert rec is not None
        assert rec.ended_at is not None
    finally:
        await store.dispose()
