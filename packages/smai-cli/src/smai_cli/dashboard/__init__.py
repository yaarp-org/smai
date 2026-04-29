"""Read-only HTTP dashboard for the ``smai serve`` verb.

Per ``designs/smai/09-cli.md`` §5.4: a minimal HTTP surface (FastAPI +
Jinja2) that reads from :class:`MetadataStore` + :class:`ArtifactStore`
through the existing :class:`Runtime` service surfaces. Pages cover the
four pipeline-tracking entity kinds (proposals, CGs, runs, papers) plus
per-entity detail views, mirroring `09` §1's verb surface from a
read-only perspective.

The surface is **read-only** by spec — mutations go through the
existing CLI verbs (``smai run`` / ``smai submit-proposal`` /
``smai approve-proposal`` / ``smai reject-proposal`` / ``smai ingest``).
The ``partial → submitted`` paper promotion (`08` §5.7) is a
synchronous CLI write through ``smai ingest --promote-partial``; the
paper-detail page surfaces the affordance text, no mutation endpoint.

Authentication / authorization: none (DEC-027 — multi-tenant auth is
the closed-side hosted-backend's concern; the OSS dashboard is
single-user local-only). The verb defaults to ``--host=127.0.0.1`` so
the surface is loopback-only unless the operator explicitly broadens
the bind.
"""

from __future__ import annotations

from smai_cli.dashboard.app import build_app

__all__ = ["build_app"]
