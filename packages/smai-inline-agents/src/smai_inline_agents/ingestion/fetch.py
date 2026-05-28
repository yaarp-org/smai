"""Host-side paper-fetch preprocessing for the ingestion subagent.

The Paper Agent never reaches the network for paper content (design note
§6); this module produces the :class:`PaperCorpus` the agent reads. The
v1 fetch chain is LaTeX-source-first:

1. S2ORC pre-parsed cache -- STUBBED (deferred, returns ``None``).
2. arXiv LaTeX source via the ``arxiv`` package's
   ``Result.download_source()`` -- the real v1 path (>90% coverage per
   ``notes/03_paper_to_implementation_survey.md §C``).
3. Marker PDF fallback -- STUBBED (deferred, returns ``None``).

The long tail without LaTeX returns ``None`` and the Sub-PR B wrapper
routes the paper to its ``fetch_failed`` branch.

:func:`_parse_latex_corpus` is the unit-tested surface: it splits a
LaTeX string into :class:`SectionRef`s + ``sections_by_id`` + a
:class:`LatexCiteResolver`. The network download (untar + concat) is
exercised by Sub-PR B's real-LLM dogfood, not here.
"""

from __future__ import annotations

import asyncio
import gzip
import logging
import re
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from smai_inline_agents.ingestion.corpus import PaperCorpus
from smai_inline_agents.ingestion.schemas import CitedPaperRef, SectionRef

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public fetch chain.
# ---------------------------------------------------------------------------


async def fetch_paper_corpus(arxiv_id: str) -> PaperCorpus | None:
    """Fetch + parse a paper into a :class:`PaperCorpus`, or ``None``.

    Tries S2ORC (stub) -> arXiv LaTeX source -> Marker PDF (stub). A
    ``None`` return means every path failed; the caller routes the paper
    to the ``fetch_failed`` branch (design note §6).
    """
    s2orc = await _fetch_from_s2orc(arxiv_id)
    if s2orc is not None:
        return s2orc

    latex = await _fetch_arxiv_latex_source(arxiv_id)
    if latex is not None:
        return _parse_latex_corpus(latex, arxiv_id=arxiv_id)

    pdf_corpus = await _fetch_via_marker(arxiv_id)
    if pdf_corpus is not None:
        return pdf_corpus

    return None


async def _fetch_from_s2orc(arxiv_id: str) -> PaperCorpus | None:
    """S2ORC pre-parsed-cache fetch.

    # TODO(step-3-followup): wire the S2ORC pre-parsed cache so papers
    # with a cached structured parse skip LaTeX parsing entirely.
    # Deferred indefinitely -- LaTeX source covers >90% per notes/03 §C;
    # the long tail returns None and falls through to the next path.
    """
    del arxiv_id
    return None


async def _fetch_via_marker(arxiv_id: str) -> PaperCorpus | None:
    """Marker PDF-to-Markdown fallback for papers without LaTeX source.

    # TODO(step-3-followup): wire the Marker PDF fallback for the
    # long-tail papers arXiv hosts without LaTeX source. Deferred
    # indefinitely; returns None so such papers surface as fetch_failed.
    """
    del arxiv_id
    return None


async def _fetch_arxiv_latex_source(arxiv_id: str) -> str | None:
    """Download + concatenate the arXiv LaTeX source for ``arxiv_id``.

    Synchronous network + archive IO wrapped in ``asyncio.to_thread`` so
    the caller stays async. Any failure (paper not found, no source
    archive, unreadable archive) returns ``None`` -- the chain falls
    through rather than raising.
    """
    try:
        return await asyncio.to_thread(_download_latex_source_sync, arxiv_id)
    except Exception as exc:  # noqa: BLE001 - network/archive errors fall through to None
        logger.warning("arXiv LaTeX-source fetch failed for %s: %s", arxiv_id, exc)
        return None


def _load_arxiv() -> Any:
    """Lazy + type-erased import of the ``arxiv`` package (no type stubs).

    A seam tests can monkeypatch to avoid pulling the real package.
    """
    import arxiv as _arxiv  # type: ignore[import-untyped] # noqa: PLC0415

    return _arxiv


def _download_latex_source_sync(arxiv_id: str) -> str | None:
    """Blocking arXiv source download + LaTeX extraction."""
    arxiv_mod: Any = _load_arxiv()
    client: Any = arxiv_mod.Client(page_size=1, delay_seconds=3.0, num_retries=2)
    search: Any = arxiv_mod.Search(id_list=[arxiv_id])
    results: list[Any] = list(client.results(search))
    if not results:
        return None
    result = results[0]
    with tempfile.TemporaryDirectory() as tmp:
        archive_path = result.download_source(dirpath=tmp)
        return _read_latex_from_source_archive(Path(archive_path))


def _read_latex_from_source_archive(path: Path) -> str | None:
    """Concatenate the ``.tex`` files in an arXiv source archive.

    arXiv source is usually a ``.tar.gz`` of the LaTeX project; some
    single-file submissions are a gzipped ``.tex``. The main file (the
    one carrying ``\\documentclass`` / ``\\begin{document}``) is placed
    first so section parsing sees the document body.
    """
    tex_blobs: list[str] = []
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as tar:
            for member in tar.getmembers():
                if not member.isfile() or not member.name.endswith(".tex"):
                    continue
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                tex_blobs.append(handle.read().decode("utf-8", errors="replace"))
    else:
        try:
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
                tex_blobs.append(fh.read())
        except (OSError, gzip.BadGzipFile):
            return None

    if not tex_blobs:
        return None
    tex_blobs.sort(key=lambda blob: 0 if "\\begin{document}" in blob else 1)
    return "\n\n".join(tex_blobs)


# ---------------------------------------------------------------------------
# LaTeX parsing (the unit-tested surface).
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(r"\\(?:sub)*section\*?\s*\{")
_ENV_ABSTRACT_RE = re.compile(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", re.DOTALL)
_CITE_RE = re.compile(r"\\cite[a-zA-Z]*\*?\s*(?:\[[^\]]*\])*\s*\{([^}]*)\}")
_BIBITEM_RE = re.compile(
    r"\\bibitem(?:\[[^\]]*\])?\s*\{([^}]*)\}(.*?)"
    r"(?=\\bibitem|\\end\{thebibliography\}|$)",
    re.DOTALL,
)
_ARXIV_TAGGED_RE = re.compile(r"(?:arXiv:|abs/)\s*(\d{4}\.\d{4,5})", re.IGNORECASE)
_ARXIV_BARE_RE = re.compile(r"\b(\d{4}\.\d{4,5})\b")


def _parse_latex_corpus(latex: str, *, arxiv_id: str) -> PaperCorpus:
    """Split a LaTeX string into a structured :class:`PaperCorpus`."""
    title = _extract_braced_command(latex, "title") or ""
    abstract_match = _ENV_ABSTRACT_RE.search(latex)
    abstract = _normalize_ws(abstract_match.group(1)) if abstract_match else ""
    sections, sections_by_id = _split_sections(latex)
    cite_resolver = _build_cite_resolver(latex)
    return PaperCorpus(
        arxiv_id=arxiv_id,
        title=_normalize_ws(title),
        abstract=abstract,
        full_latex=latex,
        sections=sections,
        sections_by_id=sections_by_id,
        cite_resolver=cite_resolver,
    )


def _split_sections(latex: str) -> tuple[list[SectionRef], dict[str, str]]:
    """Build the ``SectionRef`` list + ``section_id -> raw text`` map."""
    matches = list(_SECTION_RE.finditer(latex))
    sections: list[SectionRef] = []
    sections_by_id: dict[str, str] = {}
    counters = [0] * 6
    for i, match in enumerate(matches):
        head = latex[match.start() : match.end()]
        depth = head.count("sub") + 1
        depth = min(depth, 6)
        title, after_title = _balanced(latex, match.end() - 1)
        title = _normalize_ws(title) or "(untitled)"
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(latex)
        body = latex[after_title:body_end].strip()
        section_id = _next_section_id(counters, depth)
        sections.append(SectionRef(section_id=section_id, title=title[:400], depth=depth))
        sections_by_id[section_id] = body
    return sections, sections_by_id


def _next_section_id(counters: list[int], depth: int) -> str:
    """Advance the per-depth counters and return the dotted section id."""
    counters[depth - 1] += 1
    for deeper in range(depth, len(counters)):
        counters[deeper] = 0
    return ".".join(str(counters[d]) for d in range(depth))


def _build_cite_resolver(latex: str) -> LatexCiteResolver:
    """Collect ``\\cite{}`` keys + best-effort bibliography arXiv ids."""
    refs: dict[str, CitedPaperRef] = {}
    for match in _CITE_RE.finditer(latex):
        for raw_key in match.group(1).split(","):
            key = raw_key.strip()
            if key and key not in refs:
                refs[key] = CitedPaperRef(cite_key=key)

    for match in _BIBITEM_RE.finditer(latex):
        key = match.group(1).strip()
        if not key:
            continue
        entry = match.group(2).strip()
        refs[key] = CitedPaperRef(
            cite_key=key,
            arxiv_id=_extract_arxiv_id(entry),
            raw_entry=_normalize_ws(entry)[:2000] or None,
        )
    return LatexCiteResolver(refs)


def _extract_arxiv_id(text: str) -> str | None:
    """Best-effort arXiv id from a bibliography entry."""
    tagged = _ARXIV_TAGGED_RE.search(text)
    if tagged is not None:
        return tagged.group(1)
    bare = _ARXIV_BARE_RE.search(text)
    return bare.group(1) if bare is not None else None


def _extract_braced_command(text: str, command: str) -> str | None:
    """Return the balanced-brace argument of ``\\command{...}``."""
    match = re.search(r"\\" + re.escape(command) + r"\s*\{", text)
    if match is None:
        return None
    content, _ = _balanced(text, match.end() - 1)
    return content


def _balanced(text: str, open_index: int) -> tuple[str, int]:
    """Return ``(content, index_after_close)`` for a ``{...}`` at ``open_index``.

    ``open_index`` points at the opening ``{``. Tracks nested braces so a
    title like ``\\title{Foo \\textbf{Bar}}`` is captured whole. If the
    brace never closes, captures to end of string.
    """
    depth = 0
    for i in range(open_index, len(text)):
        char = text[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_index + 1 : i], i + 1
    return text[open_index + 1 :], len(text)


def _normalize_ws(text: str) -> str:
    """Collapse runs of whitespace (incl. newlines) to single spaces."""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Cite resolver.
# ---------------------------------------------------------------------------


class LatexCiteResolver:
    """A :class:`CiteResolver` populated from a paper's LaTeX cite keys.

    Implements the ``CiteResolver`` Protocol structurally. Keys come from
    ``\\cite{}`` calls in the body; arXiv ids are recovered best-effort
    from ``\\bibitem`` entries (a key with no recoverable arXiv id
    resolves to a :class:`CitedPaperRef` with ``arxiv_id=None``, which
    ``search_literature`` treats as un-fetchable).
    """

    def __init__(self, refs: dict[str, CitedPaperRef]) -> None:
        self._refs = dict(refs)

    def resolve(self, cite_key: str) -> CitedPaperRef | None:
        return self._refs.get(cite_key.strip())

    def known_keys(self) -> list[str]:
        return sorted(self._refs)


__all__ = ["LatexCiteResolver", "fetch_paper_corpus"]
