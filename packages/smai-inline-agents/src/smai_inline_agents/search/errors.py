"""Exception taxonomy for the planner search-tool surface.

Per ``planner_refactor/design_notes/search_tool_surface.md`` §9 (failure
policy) and §10 (D11 trust boundary / domain allowlist).
"""

from __future__ import annotations


class WebSearchProviderDown(Exception):
    """All retries exhausted; web_search returns ``[]`` to the agent (degraded).

    Per the failure-policy table in the design note §9: the lit-review
    phase tolerates web-search outage by degrading to empty results
    rather than hard-failing the planner.
    """


class AcademicSearchAllProvidersFailed(Exception):
    """All academic providers failed; academic_search raises to the agent.

    Failure aggregator: carries per-provider failure tuples for
    debugging and egress-log enrichment.
    """

    def __init__(self, failures: list[tuple[str, BaseException]]) -> None:
        self.failures = failures
        super().__init__(
            f"All {len(failures)} academic providers failed: "
            + ", ".join(f"{name}={exc!r}" for name, exc in failures),
        )


class UrlFetchFailed(Exception):
    """Direct + Jina both failed; fetch_url raises to the agent."""


class UrlNotAllowlisted(Exception):
    """The URL's host is not on the operator-configured fetch allowlist.

    Per D11 (``architectural_decisions.md §10``). Raised by the
    ``fetch_url`` dispatcher BEFORE issuing any HTTP request; the
    underlying httpx client must never see a non-allowlisted host.

    Carries the offending host and the allowlist that rejected it so
    the agent (and operator audit) can act on it.
    """

    def __init__(self, host: str, allowlist: tuple[str, ...]) -> None:
        self.host = host
        self.allowlist = allowlist
        super().__init__(
            f"host {host!r} is not on the fetch_url allowlist "
            f"({len(allowlist)} entries: {', '.join(allowlist)})"
        )


class BudgetExhausted(Exception):
    """Per-session cap hit; the tool raises this in lieu of calling a provider.

    Layer-3 of the three-layer budget structure
    (``architectural_decisions.md §11``): after K calls, the dispatcher
    raises this so PydanticAI surfaces a tool error and the agent must
    finalize via the forced-synthesis emit.
    """


class RateLimitExceeded(Exception):
    """Provider rate-limit window hit.

    Distinct from :class:`BudgetExhausted` (session cap): a transient
    backpressure signal from a specific provider. The dispatcher may
    back off + retry within the same call per §9.
    """


class ContentExtractionFailed(Exception):
    """Direct fetch returned a response we couldn't extract text from.

    Used by the ``DirectThenJinaFetcher`` to trigger the Jina fallback
    path internally; not raised to the agent.
    """


class NotConfiguredError(Exception):
    """The requested capability has no provider configured.

    Per D11: v1's ``web_search`` has no default provider; calls raise
    this until an operator opts in via
    ``planner.search.web_provider: <name>`` in ``smai.yaml``.
    """


__all__ = [
    "AcademicSearchAllProvidersFailed",
    "BudgetExhausted",
    "ContentExtractionFailed",
    "NotConfiguredError",
    "RateLimitExceeded",
    "UrlFetchFailed",
    "UrlNotAllowlisted",
    "WebSearchProviderDown",
]
