"""``make_api_app(runtime, *, auth_config=None)`` — the FastAPI app factory.

Per ``designs/smai/11-api.md`` and DEC-037: this function consumes a
constructed :class:`smai_cli.runtime.Runtime` and returns a ready-to-
serve :class:`fastapi.FastAPI` instance with:

* :class:`Runtime` bound to ``app.state.runtime`` so route handlers
  can read it via the :func:`smai_api._deps.get_runtime` dependency.
* Middleware mounted in the right order — Host validation BEFORE
  bearer-token check BEFORE route dispatch (per ``11`` §7).
* Central exception handlers registered for every Runtime / plugin
  exception type per ``11`` §6.2.
* Per-resource routers included.

The factory does not start a server — that is the consumer's
responsibility (``smai ui`` Task 4.L1 wires up uvicorn around it).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI
from smai_cli.runtime import Runtime

from smai_api.auth import (
    BearerTokenMiddleware,
    HostValidationMiddleware,
    _read_or_create_token_file,
)
from smai_api.errors import register_exception_handlers
from smai_api.routers import (
    comparison_groups,
    experiments,
    papers,
    proposals,
    runs,
    system,
)


@dataclass(frozen=True)
class AuthConfig:
    """Resolved auth settings for :func:`make_api_app`.

    ``enabled`` — flip on bearer-token mode per ``11`` §7.3.
    ``token_path`` — file holding the token; auto-generated on first
    construction if missing. ``None`` is treated as
    ``~/.smai/api-token`` per the doc default.
    ``allowed_hosts`` — override the default Host allowlist (loopback
    + the ``http://test`` ASGI placeholders). Useful for production
    deployments behind a reverse proxy that rewrites Host.
    """

    enabled: bool = False
    token_path: Path | None = None
    allowed_hosts: frozenset[str] | None = None


def make_api_app(
    runtime: Runtime,
    *,
    auth_config: AuthConfig | None = None,
) -> FastAPI:
    """Construct a configured :class:`FastAPI` app bound to ``runtime``.

    Routes use the URL constants from :mod:`smai_api_spec.paths`; route
    handlers call into the Runtime service surface (no business logic
    in this layer).

    Auth posture (per ``11`` §7):

    * Host validation is always-on; mismatches raise
      :class:`smai_api.errors.HostRejectedError` which the central
      handler renders as ``421 HOST_REJECTED``.
    * Bearer-token middleware is mounted only when ``auth_config.enabled``
      is ``True``; the token file is read once at construction (cached
      in middleware state) and re-validated per request.

    Auto-generated OpenAPI docs (``/docs``, ``/redoc``, ``/openapi.json``)
    are disabled per ``11`` §13 OQ10's lean-toward-disable guidance —
    the contract is the spec package + the design doc, not Swagger UI.
    """
    config = auth_config or AuthConfig()

    app = FastAPI(
        title="smai-api",
        description=(
            "FastAPI implementation of the SMAI v2 HTTP API contract per "
            "designs/smai/11-api.md and DEC-037."
        ),
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.runtime = runtime

    # Middleware ordering: Starlette mounts middleware in *reverse*
    # registration order (the last `add_middleware` call wraps the
    # innermost; the first call's middleware sees the outermost
    # request). We want Host validation outermost (cheapest reject),
    # bearer-token next, route dispatch innermost — so register
    # bearer-token FIRST and Host validation LAST.
    if config.enabled:
        token_path = config.token_path or Path.home() / ".smai" / "api-token"
        token = _read_or_create_token_file(token_path)
        app.add_middleware(BearerTokenMiddleware, token=token)
    app.add_middleware(
        HostValidationMiddleware,
        allowed_hosts=config.allowed_hosts,
    )

    register_exception_handlers(app)

    app.include_router(proposals.router)
    app.include_router(papers.router)
    app.include_router(experiments.router)
    app.include_router(comparison_groups.router)
    app.include_router(runs.router)
    app.include_router(system.router)

    return app


__all__ = ["AuthConfig", "make_api_app"]
