"""Unit tests for the ingestion paper-fetch preprocessing (Step 3, Sub-PR A).

Covers the unit-testable surface: :func:`_parse_latex_corpus` (section
splitting + cite-key collection + bibliography arXiv-id recovery) and the
deferred S2ORC / Marker stubs returning ``None``. The network download
(arXiv source untar + concat) is exercised by Sub-PR B's real-LLM
dogfood, not here.
"""

from __future__ import annotations

from _ingestion_fakes import FIXTURE_LATEX  # type: ignore[import-not-found]
from smai_inline_agents.ingestion.fetch import (
    _fetch_from_s2orc,
    _fetch_via_marker,
    _parse_latex_corpus,
)


def test_parse_latex_corpus_extracts_title_and_abstract() -> None:
    corpus = _parse_latex_corpus(FIXTURE_LATEX, arxiv_id="1708.04552")
    assert corpus.arxiv_id == "1708.04552"
    assert "Cutout" in corpus.title
    assert "simple regularization technique" in corpus.abstract


def test_parse_latex_corpus_splits_sections_with_dotted_ids() -> None:
    corpus = _parse_latex_corpus(FIXTURE_LATEX, arxiv_id="1708.04552")
    by_id = {s.section_id: s for s in corpus.sections}
    # \section{Introduction} -> 1 ; \subsection{Background} -> 1.1 ;
    # \section{Method} -> 2
    assert by_id["1"].title == "Introduction"
    assert by_id["1"].depth == 1
    assert by_id["1.1"].title == "Background"
    assert by_id["1.1"].depth == 2
    assert by_id["2"].title == "Method"
    assert by_id["2"].depth == 1


def test_parse_latex_corpus_section_bodies() -> None:
    corpus = _parse_latex_corpus(FIXTURE_LATEX, arxiv_id="1708.04552")
    assert "found in Section 2" in corpus.sections_by_id["1.1"]
    assert "mask square regions" in corpus.sections_by_id["2"]


def test_parse_latex_corpus_collects_cite_keys() -> None:
    corpus = _parse_latex_corpus(FIXTURE_LATEX, arxiv_id="1708.04552")
    keys = set(corpus.cite_resolver.known_keys())
    # \cite{devries2017,wang2025} + \citep{kingma2014}
    assert {"devries2017", "wang2025", "kingma2014"} <= keys


def test_parse_latex_corpus_recovers_arxiv_ids_from_bibitems() -> None:
    corpus = _parse_latex_corpus(FIXTURE_LATEX, arxiv_id="1708.04552")
    wang = corpus.cite_resolver.resolve("wang2025")
    assert wang is not None
    assert wang.arxiv_id == "2501.00001"
    devries = corpus.cite_resolver.resolve("devries2017")
    assert devries is not None
    assert devries.arxiv_id == "1708.04552"


def test_parse_latex_corpus_cited_without_bibitem_has_no_arxiv_id() -> None:
    """kingma2014 is \\citep'd but has no \\bibitem -> resolvable key,
    arxiv_id None (search_literature treats it as un-fetchable)."""
    corpus = _parse_latex_corpus(FIXTURE_LATEX, arxiv_id="1708.04552")
    kingma = corpus.cite_resolver.resolve("kingma2014")
    assert kingma is not None
    assert kingma.arxiv_id is None


def test_parse_latex_corpus_unknown_cite_key_resolves_none() -> None:
    corpus = _parse_latex_corpus(FIXTURE_LATEX, arxiv_id="1708.04552")
    assert corpus.cite_resolver.resolve("not_a_key") is None


async def test_s2orc_stub_returns_none() -> None:
    assert await _fetch_from_s2orc("1708.04552") is None


async def test_marker_stub_returns_none() -> None:
    assert await _fetch_via_marker("1708.04552") is None
