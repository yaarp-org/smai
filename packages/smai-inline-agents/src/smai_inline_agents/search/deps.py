"""``SearchDeps`` -- the dependency payload for the planner search-tool
surface.

Per ``planner_refactor/design_notes/search_tool_surface.md`` §3.

Constructed once per planner session by
:func:`~smai_inline_agents.search.dispatch.build_search_deps` and
explicitly closed on session exit (via :meth:`SearchDeps.aclose`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from smai_inline_agents.search.budgets import BudgetTracker
from smai_inline_agents.search.egress_log import EgressLogger
from smai_inline_agents.search.protocols import (
    AcademicSearchProvider,
    UrlFetcher,
    WebSearchProvider,
)


@dataclass
class SearchDeps:
    """Dep payload for the literature-review PydanticAI agent.

    Not a Pydantic model: instances hold async-context resources
    (httpx.AsyncClient, file handles for the egress log) plus the
    chosen provider instances.

    Per D11: ``web_provider`` is ``None`` when no general-web provider
    is registered (the v1 default); the ``web_search`` dispatch raises
    :class:`~smai_inline_agents.search.errors.NotConfiguredError` in
    that case.
    """

    academic_providers: list[AcademicSearchProvider]
    url_fetcher: UrlFetcher
    egress_log: EgressLogger
    budgets: BudgetTracker
    session_id: str
    fetch_allowlist: tuple[str, ...] = ()
    web_provider: WebSearchProvider | None = None
    _closed: bool = field(default=False, init=False, repr=False)

    async def aclose(self) -> None:
        """Release async resources. Idempotent.

        Flushes the egress log to ArtifactStore. Production providers
        that hold httpx.AsyncClient instances are closed here too;
        currently each provider is responsible for its own client
        lifecycle.
        """
        if self._closed:
            return
        await self.egress_log.close()
        self._closed = True


__all__ = ["SearchDeps"]
