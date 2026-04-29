""":class:`FastAPI` app construction for the ``smai serve`` dashboard.

Per ``designs/smai/09-cli.md`` §5.4: takes a constructed
:class:`Runtime` and returns a ready-to-serve :class:`FastAPI`
application. Templates live alongside this module under ``templates/``
and the (minimal) CSS under ``static/`` per the recommended layout in
the Task 3.H1 brief.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from smai_cli.dashboard import pages
from smai_cli.runtime import Runtime

# Resolved at import time; the dashboard module ships its own
# ``templates/`` + ``static/`` subdirectories.
_DASHBOARD_DIR = Path(__file__).resolve().parent
_TEMPLATES_DIR = _DASHBOARD_DIR / "templates"
_STATIC_DIR = _DASHBOARD_DIR / "static"


def build_app(runtime: Runtime) -> FastAPI:
    """Construct a :class:`FastAPI` app bound to ``runtime``.

    The ``runtime`` argument is captured by-reference; route handlers
    read its sub-services on each request. The app is deliberately
    bare: no auth middleware (single-user local), no CORS (no API
    callers), no JSON-API endpoints (HTML to a browser only per
    `09-cli.md` §5.3).
    """
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    # Register custom Jinja filters. Jinja2's ``filters`` dict is
    # typed loosely upstream; cast through ``Any`` for the assignment
    # so pyright's strict mode doesn't flag the unknown value type.
    jinja_env = cast("Any", templates.env)
    env_filters = cast("dict[str, Any]", jinja_env.filters)
    env_filters["format_datetime"] = _format_datetime
    env_filters["short_state"] = _short_state

    app = FastAPI(
        title="smai dashboard",
        description=(
            "Read-only operational view of in-flight proposals, "
            "comparison groups, runs, and papers. Mutations go through "
            "the smai CLI verbs."
        ),
        version="0.1.0",
        docs_url=None,  # No OpenAPI/Swagger surface — HTML only.
        redoc_url=None,
        openapi_url=None,
    )
    app.state.runtime = runtime
    app.state.templates = templates

    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    pages.register_routes(app)
    return app


# === Jinja filters ===========================================================


def _format_datetime(value: Any) -> str:
    """Format a UTC ``datetime`` for the dashboard.

    The pipeline-tracking entities carry timezone-aware UTC timestamps;
    we format compactly without a seconds component for readability.
    """
    if not isinstance(value, datetime):
        return str(value)
    return value.strftime("%Y-%m-%d %H:%M UTC")


def _short_state(value: Any) -> str:
    """Render a state name as a CSS-class-friendly slug.

    Used to colour-code state cells (``state-running``, ``state-failed``).
    """
    if not isinstance(value, str):
        return ""
    return value.lower().replace("_", "-")


__all__ = ["build_app"]
