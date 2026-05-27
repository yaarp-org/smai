"""v1 provider implementations for the planner search-tool surface.

Per ``planner_refactor/design_notes/search_tool_surface.md`` §6.

Modules:

* :mod:`.openalex` -- OpenAlex (academic, no auth, polite-pool email
  header upgrades rate limits).
* :mod:`.semantic_scholar` -- Semantic Scholar (academic, optional API
  key, rate-limit aware).
* :mod:`.arxiv_provider` -- arXiv via the ``arxiv`` Python package
  (academic, no auth).
* :mod:`.jina` -- Jina Reader (URL fetch fallback).
* :mod:`.direct_fetch` -- Direct httpx + markdownify URL fetcher with
  optional Jina fallback wrapped inside.
* :mod:`.tavily` -- Tavily provider stub. NOT registered in the v1
  default dispatch per D11; kept as reference for operator opt-in via
  ``planner.search.web_provider: tavily`` in ``smai.yaml``.
"""

from __future__ import annotations

__all__: list[str] = []
