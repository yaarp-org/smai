"""Direct httpx + markdownify URL fetcher with optional Jina Reader fallback.

Per ``planner_refactor/design_notes/search_tool_surface.md`` §6.

* Direct path: httpx GET, content-type sniff, extract text via a small
  HTML-to-text pass for HTML responses; pass-through for Markdown /
  LaTeX / JSON.
* Fallback path: on extraction failure or HTTP error, fall back to
  :func:`smai_inline_agents.search.providers.jina.jina_fetch`.
* Allowlist enforcement is the dispatcher's responsibility (see
  :mod:`smai_inline_agents.search.dispatch`); the fetcher here trusts
  that the host has already been validated.

We don't depend on the ``markdownify`` package: the v1 extraction is a
minimal tag-stripper that preserves headers and links sufficient for
LLM consumption. If a future use case needs richer extraction, swap to
``markdownify`` behind the same FetchedDocument surface.
"""

from __future__ import annotations

import html as _html
import re
from typing import Literal

import httpx

from smai_inline_agents.search.errors import ContentExtractionFailed, UrlNotAllowlisted
from smai_inline_agents.search.providers.jina import JINA_HOST, jina_fetch
from smai_inline_agents.search.types import FetchedDocument

_RAW_RETENTION_BYTES = 100 * 1024  # 100 KB; matches jina.py.

_HTML_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_WHITESPACE_COLLAPSE_RE = re.compile(r"[\t\f\v ]+")
_BLANK_LINES_RE = re.compile(r"\n\s*\n+")


class DirectThenJinaFetcher:
    """v1 ``fetch_url`` implementation.

    Direct path first; Jina Reader fallback on failure when
    ``enable_jina_fallback=True``.
    """

    name: str = "direct"

    def __init__(
        self,
        *,
        enable_jina_fallback: bool = True,
        client: httpx.AsyncClient | None = None,
        jina_client: httpx.AsyncClient | None = None,
        jina_api_key: str | None = None,
        timeout_seconds: float = 30.0,
        fetch_allowlist: tuple[str, ...] | None = None,
    ) -> None:
        self._enable_jina = enable_jina_fallback
        self._fetch_allowlist = fetch_allowlist
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=True)
        # D11: install a request event hook that re-validates the host
        # of every outbound request -- including redirect targets httpx
        # is about to follow -- against the allowlist. Without this, a
        # 302 from an allowlisted host to a non-allowlisted one would
        # silently exfiltrate (the dispatcher only validates the
        # initial URL).
        if fetch_allowlist is not None:
            self._client.event_hooks.setdefault("request", []).append(
                self._validate_request_host,
            )
        # Share an httpx client for the Jina fallback so we don't open a
        # second connection pool per session.
        self._owns_jina_client = jina_client is None
        self._jina_client = jina_client if jina_client is not None else self._client
        self._jina_api_key = jina_api_key

    async def _validate_request_host(self, request: httpx.Request) -> None:
        """httpx request hook: re-validate ``request.url.host`` against
        the operator allowlist before httpx dispatches it.

        Fires on the initial GET (redundant with the dispatcher's check
        but cheap) AND on every redirect httpx is about to follow.
        Raises :class:`UrlNotAllowlisted` to abort the redirect chain
        before any data is sent.

        The Jina Reader host is allowed unconditionally: it's the
        fallback path's destination, never an attacker-supplied URL.
        The dispatcher never sees this host because the fetcher
        constructs the Jina URL internally.
        """
        if self._fetch_allowlist is None:
            return
        host = (request.url.host or "").lower()
        if host == JINA_HOST:
            return
        for entry in self._fetch_allowlist:
            entry_l = entry.lower()
            if host == entry_l or host.endswith("." + entry_l):
                return
        raise UrlNotAllowlisted(host, self._fetch_allowlist)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
        if self._owns_jina_client and not self._owns_client:
            # Only close if it was a distinct client we own (different
            # from the shared direct-fetch client).
            await self._jina_client.aclose()

    async def fetch(self, url: str) -> FetchedDocument:
        try:
            return await self._direct_fetch(url)
        except (httpx.HTTPError, ContentExtractionFailed):
            if not self._enable_jina:
                raise
            text, raw = await jina_fetch(self._jina_client, url, api_key=self._jina_api_key)
            return FetchedDocument(
                url=url,  # type: ignore[arg-type]
                title=None,
                content_type="markdown",
                text=text,
                raw=raw,
                source="jina_reader",
            )

    async def _direct_fetch(self, url: str) -> FetchedDocument:
        response = await self._client.get(url)
        response.raise_for_status()
        ctype_header = response.headers.get("content-type", "")
        content_type = _classify_content_type(ctype_header, url)
        body_bytes = response.content
        raw_text = body_bytes.decode(
            response.encoding or "utf-8",
            errors="replace",
        )
        title: str | None = None
        if content_type == "html":
            title = _extract_title(raw_text)
            text = _html_to_text(raw_text)
            if not text.strip():
                raise ContentExtractionFailed(
                    f"empty extracted text from {url!r} (content_type=html)"
                )
        elif content_type == "pdf":
            # PDFs route straight to Jina (real parser is out of scope
            # for v1 per the design note); raise ContentExtractionFailed
            # so the outer fetch() falls back.
            raise ContentExtractionFailed(
                f"PDF content at {url!r}; falling back to Jina Reader",
            )
        else:
            text = raw_text
        raw = raw_text if len(body_bytes) <= _RAW_RETENTION_BYTES else None
        return FetchedDocument(
            url=url,  # type: ignore[arg-type]
            title=title,
            content_type=content_type,
            text=text,
            raw=raw,
            source="direct",
        )


def _classify_content_type(
    header: str,
    url: str,
) -> Literal["html", "pdf", "markdown", "latex", "json", "other"]:
    """Map Content-Type + URL suffix to one of the
    :class:`~smai_inline_agents.search.types.FetchedDocument` literal
    values."""
    lower = header.lower()
    if "html" in lower:
        return "html"
    if "pdf" in lower or url.lower().endswith(".pdf"):
        return "pdf"
    if "markdown" in lower or url.lower().endswith((".md", ".markdown")):
        return "markdown"
    if "latex" in lower or url.lower().endswith((".tex", ".latex")):
        return "latex"
    if "json" in lower:
        return "json"
    return "other"


def _extract_title(body: str) -> str | None:
    match = _HTML_TITLE_RE.search(body)
    if match is None:
        return None
    raw = match.group(1).strip()
    return _html.unescape(raw) or None


def _html_to_text(body: str) -> str:
    """Minimal HTML-to-text extractor.

    Strips ``<script>`` / ``<style>`` blocks entirely, then strips
    remaining tags, unescapes entities, and collapses whitespace.
    Sufficient for LLM consumption of standard documentation /
    blog-post HTML; not a replacement for a real Markdown converter
    on JS-heavy SPAs (those route through the Jina fallback).
    """
    stripped = _HTML_SCRIPT_STYLE_RE.sub(" ", body)
    text = _HTML_TAG_RE.sub(" ", stripped)
    text = _html.unescape(text)
    text = _WHITESPACE_COLLAPSE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


__all__ = ["DirectThenJinaFetcher"]
