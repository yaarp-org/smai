"""SPA static-bundle mount for :func:`smai_api.app.make_api_app` (Task 4.N1).

Per ``designs/smai/12-ui-process.md`` §8.2 and ``13-frontend.md`` §12.3 /
§12.4: when the ``smai-ui`` package is installed and its bundle is
staged on disk, this module mounts the SPA at ``/`` and adds the
SPA-deep-link fallback so a browser refresh on
``/comparison-groups/cg_xyz`` re-serves ``index.html``. When ``smai-ui``
is missing or its bundle is not staged, the mount degrades silently —
the JSON API still works on its own.

Bearer-token bootstrap: when ``auth_config.enabled`` is true, the
``index.html`` handler interpolates a one-line ``<script>window.__SMAI_TOKEN__
= "...";</script>`` before ``</head>`` so the SPA's ``lib/api/client.ts``
can pick the token up on first load (per ``11-api.md`` §7.3 +
``13-frontend.md`` §12.4). The token is held in JS memory only; never
in localStorage; never in URLs.

The function :func:`maybe_mount_spa` is intentionally kept as the
single integration seam so :func:`smai_api.app.make_api_app` does not
grow conditional branches; tests for the SPA mount can call it directly
with a stub bundle path without constructing the full Runtime stack.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from smai_api.auth import _read_or_create_token_file

if TYPE_CHECKING:
    from smai_api.app import AuthConfig

_log = logging.getLogger(__name__)


def maybe_mount_spa(
    app: FastAPI,
    *,
    auth_config: AuthConfig | None,
    bundle_path: Path | None = None,
) -> Path | None:
    """Mount the SPA bundle on ``app`` when one is available.

    Resolution order for the bundle path:

    1. Explicit ``bundle_path`` kwarg (used by tests; pass a tmp_path
       with a stub ``index.html`` to exercise the mount end-to-end
       without installing ``smai-ui``).
    2. :func:`smai_ui.get_static_bundle_path` if the package is
       importable. ``ImportError`` (package missing — API-only install)
       and :class:`FileNotFoundError` (bundle staging skipped — pnpm
       was unavailable at install time) are both treated as "no bundle";
       the JSON API still serves and the SPA mount is skipped.

    Returns the resolved bundle path on success, or ``None`` if the
    mount was skipped. The return value is for caller logging /
    test-assertion convenience; the side-effect is the routes added to
    ``app``.

    Routes added (when a bundle is mounted):

    * ``GET /`` — serves ``index.html`` with optional bearer-token
      injection (per ``13-frontend.md`` §12.4).
    * Static-files mount at ``/_spa_static`` — for direct asset access
      via the fallback handler. The mount is internal; the SPA itself
      links to ``/assets/...`` so the 404 fallback (below) is what
      serves them.
    * 404 exception handler — when a non-``/api/`` path 404s, try to
      serve the requested file from the bundle (CSS/JS/font assets);
      otherwise re-serve ``index.html`` so client-side routing handles
      the deep link. ``/api/`` paths skip this fallback and surface the
      normal :class:`smai_api_spec.ErrorEnvelope` 404.
    """
    resolved = _resolve_bundle_path(bundle_path)
    if resolved is None:
        return None

    @app.get("/", include_in_schema=False)
    async def _spa_root() -> HTMLResponse:  # pyright: ignore[reportUnusedFunction]
        return _serve_index_html(resolved, auth_config)

    # Mount StaticFiles for the bundle assets. Mounting at `/_spa_static`
    # (rather than `/`) keeps the catch-all 404 handler in charge of
    # routing decisions; this submount exists only so direct-asset
    # requests to `/_spa_static/<file>` work for any future use case
    # that wants the canonical static-files semantics (range requests,
    # ETag, etc.).
    app.mount(
        "/_spa_static",
        StaticFiles(directory=str(resolved), html=False),
        name="spa_static",
    )

    @app.exception_handler(404)
    async def _spa_fallback(  # pyright: ignore[reportUnusedFunction]
        request: Request, exc: HTTPException
    ) -> Response:
        # `/api/*` 404s keep FastAPI's default wire format — the SPA
        # never serves those URLs, and the existing API contract has
        # not declared a generic `NOT_FOUND` ErrorCode (the catalog at
        # ``smai_api_spec.errors`` is closed and resource-specific:
        # CG_NOT_FOUND / RUN_NOT_FOUND / ARTIFACT_NOT_FOUND / etc.,
        # each raised by a typed exception). Adding a generic envelope
        # for unknown-route 404s would expand the API surface without
        # matching exception coverage; keep the existing default and
        # let the typed exception handlers continue to own envelope
        # responses for the resource-not-found family.
        if request.url.path.startswith("/api/"):
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
            )

        # Static asset request — try to serve it directly from the
        # bundle. Path is normalized via `Path.resolve()` and
        # constrained to the bundle root to defend against
        # `../../etc/passwd`-style traversal.
        rel = request.url.path.lstrip("/")
        if rel:
            candidate = (resolved / rel).resolve()
            try:
                candidate.relative_to(resolved.resolve())
            except ValueError:
                # Traversal attempt — fall through to index.html.
                pass
            else:
                if candidate.is_file():
                    return FileResponse(candidate)

        # SPA deep link — re-serve index.html with token injection so
        # client-side routing can pick up where the URL left off.
        return _serve_index_html(resolved, auth_config)

    _log.info("smai-api: mounted SPA bundle from %s", resolved)
    return resolved


def _resolve_bundle_path(bundle_path: Path | None) -> Path | None:
    if bundle_path is not None:
        if not bundle_path.is_dir() or not (bundle_path / "index.html").is_file():
            _log.warning(
                "smai-api: explicit bundle_path %s is missing or has no index.html; "
                "skipping SPA mount.",
                bundle_path,
            )
            return None
        return bundle_path
    try:
        # Lazy import: keeps `smai-api` installable + importable when
        # `smai-ui` is not in the environment (the API-only deployment
        # shape per `12` §8.2 and `smai-api`'s `[ui]` extra).
        from smai_ui import get_static_bundle_path  # noqa: PLC0415
    except ImportError:
        _log.info(
            "smai-api: smai-ui not installed; SPA mount skipped. Install "
            "`smai-api[ui]` to enable the dashboard at /."
        )
        return None
    try:
        return get_static_bundle_path()
    except FileNotFoundError as exc:
        _log.warning(
            "smai-api: smai-ui installed but bundle not staged (%s); SPA mount "
            "skipped. Run `pnpm build` from apps/ui/ and re-install smai-ui to "
            "stage the bundle.",
            exc,
        )
        return None


def _serve_index_html(bundle_path: Path, auth_config: AuthConfig | None) -> HTMLResponse:
    """Serve ``index.html`` with the bearer-token bootstrap when enabled."""
    html = (bundle_path / "index.html").read_text(encoding="utf-8")
    if auth_config is not None and auth_config.enabled:
        token_path = auth_config.token_path or Path.home() / ".smai" / "api-token"
        token = _read_or_create_token_file(token_path)
        # `json.dumps` quotes + escapes the string for safe JS embedding
        # (per `13-frontend.md` §12.4). Tokens come from
        # `secrets.token_urlsafe(32)` so they are URL-safe ASCII, but
        # we escape defensively.
        injection = f"<script>window.__SMAI_TOKEN__ = {json.dumps(token)};</script>"
        if "</head>" in html:
            html = html.replace("</head>", f"{injection}</head>", 1)
        else:
            # Vite always emits a `</head>`; if it ever stops, fall
            # back to prefixing the script.
            html = injection + html
    return HTMLResponse(html)


__all__ = ["maybe_mount_spa"]
