"""Jina Reader URL-fetch provider.

Per ``planner_refactor/design_notes/search_tool_surface.md`` §6.

Wraps ``https://r.jina.ai/{url}`` to fetch a cleaned Markdown rendering
of arbitrary web pages. Used as the fallback inside
:class:`~smai_inline_agents.search.providers.direct_fetch.DirectThenJinaFetcher`
and as the standalone fetcher when ``planner.search.url_fetch:
jina_only`` is configured.

* Auth: optional. Free tier hits the public endpoint; an API key
  (env ``JINA_API_KEY``) raises rate limits.
* Output: Markdown; we treat it as ``content_type="markdown"`` in
  :class:`~smai_inline_agents.search.types.FetchedDocument`.
* Allowlist enforcement: the ``fetch_url`` dispatcher in ``dispatch.py``
  enforces the D11 domain allowlist BEFORE this method is invoked; the
  provider here does not double-check.
"""

from __future__ import annotations

from urllib.parse import quote

import httpx

from smai_inline_agents.search.types import FetchedDocument

JINA_HOST = "r.jina.ai"
_JINA_BASE = f"https://{JINA_HOST}/"

# Retention cap for ``FetchedDocument.raw`` per design note §2.
_RAW_RETENTION_BYTES = 100 * 1024  # 100 KB


class JinaOnlyFetcher:
    """v1 standalone Jina Reader fetcher.

    Selectable via ``planner.search.url_fetch: jina_only``.
    """

    name: str = "jina_reader"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def fetch(self, url: str) -> FetchedDocument:
        text, raw = await jina_fetch(self._client, url, api_key=self._api_key)
        return FetchedDocument(
            url=url,  # type: ignore[arg-type] - HttpUrl coerces from str
            title=None,
            content_type="markdown",
            text=text,
            raw=raw,
            source="jina_reader",
        )


async def jina_fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    api_key: str | None,
) -> tuple[str, str | None]:
    """Run one Jina Reader fetch; return (markdown_text, raw).

    Helper exposed at module scope so
    :class:`~smai_inline_agents.search.providers.direct_fetch.DirectThenJinaFetcher`
    can reuse it as its fallback path without instantiating a second
    fetcher (and a second httpx client).

    ``raw`` is the original Markdown when it fits the retention cap;
    ``None`` when truncated.
    """
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    # Percent-encode the full target URL before concat: raw ``#`` / ``?``
    # in ``url`` would otherwise shift Jina's interpretation (e.g. ``#``
    # would be parsed as a fragment of the r.jina.ai URL rather than as
    # part of the path argument). ``safe=""`` encodes everything;
    # Jina's reader accepts the fully-encoded form.
    target = _JINA_BASE + quote(url, safe="")
    response = await client.get(target, headers=headers)
    response.raise_for_status()
    body = response.text
    raw = body if len(body.encode("utf-8")) <= _RAW_RETENTION_BYTES else None
    return body, raw


__all__ = [
    "JINA_HOST",
    "JinaOnlyFetcher",
    "jina_fetch",
]
