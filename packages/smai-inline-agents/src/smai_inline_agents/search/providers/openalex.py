"""OpenAlex academic-search provider.

Per ``planner_refactor/design_notes/search_tool_surface.md`` §6.

* Endpoint: ``https://api.openalex.org/works?search=...``.
* Auth: none. Polite-pool ``mailto=`` query parameter upgrades the
  rate-limit ceiling and is good-citizenship practice.
* Rate limit: 100,000 calls / day standard pool; the polite pool is
  effectively uncapped for any reasonable agent workload.
* Inverted abstract: OpenAlex returns abstracts as a positional inverted
  index (``{word: [positions]}``). We reconstruct the plain string.
* External ids: arXiv id may live under ``ids.arxiv`` or in the primary
  location's landing-page URL; DOI lives under ``ids.doi`` as a full URL.
"""

from __future__ import annotations

from typing import Any, cast

import httpx

from smai_inline_agents.search.types import AcademicResult


def _as_dict(value: Any) -> dict[str, Any]:
    """Coerce ``value`` to a ``dict[str, Any]`` or return an empty dict.

    Pyright-friendly idiom for traversing untyped JSON without a
    flood of ``cast`` boilerplate.
    """
    if isinstance(value, dict):
        return cast("dict[str, Any]", value)
    return {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return cast("list[Any]", value)
    return []


_OPENALEX_BASE = "https://api.openalex.org/works"


class OpenAlexProvider:
    """v1 OpenAlex academic provider.

    Constructed once per session; holds an :class:`httpx.AsyncClient`.
    Caller closes via :meth:`aclose` (or via the
    :func:`~smai_inline_agents.search.dispatch.build_search_deps`
    lifecycle).
    """

    name: str = "openalex"

    def __init__(
        self,
        *,
        polite_email: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._polite_email = polite_email
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def search(self, query: str, max_results: int) -> list[AcademicResult]:
        params: dict[str, str | int] = {
            "search": query,
            "per_page": max(1, min(max_results, 50)),
        }
        if self._polite_email:
            params["mailto"] = self._polite_email
        response = await self._client.get(_OPENALEX_BASE, params=params)
        response.raise_for_status()
        payload = cast("dict[str, Any]", response.json())
        works = _as_list(payload.get("results"))
        out: list[AcademicResult] = []
        for w in works[:max_results]:
            if isinstance(w, dict):
                out.append(self._map(cast("dict[str, Any]", w)))
        return out

    def _map(self, work: dict[str, Any]) -> AcademicResult:
        title_raw = work.get("title") or work.get("display_name") or ""
        title = str(title_raw)
        # ids block: {"openalex": "...", "doi": "https://doi.org/10.xxx", "mag": "..."}
        ids_block = _as_dict(work.get("ids"))
        doi_value = ids_block.get("doi")
        doi = _strip_doi_prefix(doi_value) if isinstance(doi_value, str) else None
        arxiv_id = _extract_arxiv_id(work)
        abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
        authorships = _as_list(work.get("authorships"))
        authors: list[str] = []
        for a in authorships:
            a_dict = _as_dict(a)
            inner = _as_dict(a_dict.get("author"))
            name = inner.get("display_name")
            if isinstance(name, str) and name:
                authors.append(name)
        year_raw = work.get("publication_year")
        year = int(year_raw) if isinstance(year_raw, int) else None
        cites_raw = work.get("cited_by_count")
        citation_count = int(cites_raw) if isinstance(cites_raw, int) else None
        pdf_url = _extract_pdf_url(work)
        return AcademicResult(
            title=title,
            authors=authors,
            year=year,
            arxiv_id=arxiv_id,
            doi=doi,
            abstract=abstract,
            citation_count=citation_count,
            pdf_url=pdf_url,  # type: ignore[arg-type] - HttpUrl coerces from str
            source=self.name,
        )


def _strip_doi_prefix(doi_url: str) -> str | None:
    """Strip ``https://doi.org/`` prefix and lowercase the result.

    Returns ``None`` for empty / malformed values.
    """
    raw = doi_url.strip()
    if not raw:
        return None
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if raw.lower().startswith(prefix):
            raw = raw[len(prefix) :]
            break
    return raw.lower() or None


def _extract_arxiv_id(work: dict[str, Any]) -> str | None:
    """Probe an OpenAlex Work record for an arXiv id.

    OpenAlex sometimes exposes arxiv ids under ``ids.arxiv`` or via the
    primary location's landing-page URL. We probe both.
    """
    ids_block = _as_dict(work.get("ids"))
    raw = ids_block.get("arxiv")
    if isinstance(raw, str) and raw.strip():
        candidate = raw.strip()
        for prefix in ("arxiv:", "arXiv:"):
            if candidate.startswith(prefix):
                candidate = candidate[len(prefix) :]
                break
        return _trim_arxiv_version(candidate) or None
    primary = _as_dict(work.get("primary_location"))
    landing = primary.get("landing_page_url")
    if isinstance(landing, str) and "arxiv.org/abs/" in landing:
        candidate = landing.rsplit("arxiv.org/abs/", 1)[-1].split("?", 1)[0]
        return _trim_arxiv_version(candidate) or None
    return None


def _trim_arxiv_version(raw: str) -> str:
    """Strip ``vN`` suffix from an arXiv id (e.g. ``2504.00255v1`` ->
    ``2504.00255``)."""
    candidate = raw.strip().strip("/")
    if "v" in candidate:
        head, tail = candidate.rsplit("v", 1)
        if tail.isdigit():
            return head
    return candidate


def _extract_pdf_url(work: dict[str, Any]) -> str | None:
    """Return a direct PDF URL when OpenAlex exposes one."""
    primary = _as_dict(work.get("primary_location"))
    pdf = primary.get("pdf_url")
    if isinstance(pdf, str) and pdf.startswith(("http://", "https://")):
        return pdf
    best = _as_dict(work.get("best_oa_location"))
    pdf = best.get("pdf_url")
    if isinstance(pdf, str) and pdf.startswith(("http://", "https://")):
        return pdf
    return None


def _reconstruct_abstract(inverted: object) -> str:
    """Rebuild the plain-text abstract from OpenAlex's inverted index.

    The inverted shape is ``{word: [positions]}``. We materialize an
    array indexed by position and join with single spaces.
    Returns ``""`` when no abstract is present.
    """
    if not isinstance(inverted, dict):
        return ""
    inverted_typed = cast("dict[Any, Any]", inverted)
    max_pos = -1
    positions: list[tuple[int, str]] = []
    for word_raw, pos_list_raw in inverted_typed.items():
        word = str(word_raw)
        if not isinstance(pos_list_raw, list):
            continue
        pos_list = cast("list[Any]", pos_list_raw)
        for pos_raw in pos_list:
            if not isinstance(pos_raw, int):
                continue
            positions.append((pos_raw, word))
            if pos_raw > max_pos:
                max_pos = pos_raw
    if max_pos < 0:
        return ""
    out: list[str] = [""] * (max_pos + 1)
    for pos, word in positions:
        out[pos] = word
    return " ".join(w for w in out if w)


__all__ = ["OpenAlexProvider"]
