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
* Task 4.K3 lifespan: when the runtime's :class:`MetadataStore`
  reports :attr:`MetadataStoreCapabilities.supports_listen_notify`,
  spawns a dedicated asyncpg ``LISTEN smai_events`` task that feeds
  an in-process :class:`EventBroker` so the SSE handler in
  :mod:`smai_api.routers.events` sees cross-process state-change +
  worker-heartbeat events (per ``12-ui-process.md`` §6.3 / Case B).

The factory does not start a server — that is the consumer's
responsibility (``smai ui`` Task 4.L1 wires up uvicorn around it).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI
from smai_cli.runtime import Runtime
from smai_events import EventBroker

from smai_api._pg_listener import pg_listener_task, sqlalchemy_url_to_asyncpg_dsn
from smai_api.auth import (
    BearerTokenMiddleware,
    HostValidationMiddleware,
    _read_or_create_token_file,
)
from smai_api.errors import register_exception_handlers
from smai_api.routers import (
    comparison_groups,
    events,
    experiments,
    papers,
    proposals,
    runs,
    system,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

_log = logging.getLogger(__name__)


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
    event_broker: EventBroker | None = None,
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

    ``event_broker`` (Task 4.K2 — `12-ui-process.md` §6.2): the
    in-process :class:`EventBroker` backing the SSE
    ``GET /api/v1/events`` route. Resolution order:

    1. Explicit ``event_broker`` kwarg.
    2. ``runtime.event_broker`` (the in-band Runtime — ``smai dev``
       / ``smai ui --with-worker`` — always carries one).
    3. Auto-constructed broker, ONLY if the runtime's MetadataStore
       reports :attr:`supports_listen_notify` (Task 4.K3 / Case B —
       the listener task at startup feeds a freshly-constructed
       broker the SSE handler also drains).
    4. ``None`` — events route returns 503 + ``code: "EVENTS_DISABLED"``.

    Task 4.K3 lifespan (per ``12-ui-process.md`` §6.3):

    * On startup: if the resolved broker is non-``None`` AND the
      MetadataStore reports ``supports_listen_notify``, spawn the
      dedicated asyncpg ``LISTEN smai_events`` task feeding the
      broker. The DSN is extracted from the store's SQLAlchemy
      ``AsyncEngine`` URL.
    * On shutdown: signal the listener via its ``shutdown`` event;
      await with timeout; cancel on hang.

    Hybrid case (``smai ui --with-worker`` against Postgres — unusual
    but supported per the K3 brief): both an InProcessEventChannel
    AND the listener task feed the broker. The SSE handler will
    therefore deliver each transition twice — once from the in-band
    channel, once from the LISTEN side. Surfaced as a documentation
    note in :func:`smai_cli.main.ui` (Task 4.L1) so operators
    intentionally configuring this hybrid know to expect duplicates.
    """
    config = auth_config or AuthConfig()

    # Resolve the event broker per docstring: explicit override → the
    # runtime's broker → auto-construct (only when the Postgres LISTEN
    # path will populate it) → None.
    listen_notify_capable = _runtime_supports_listen_notify(runtime)
    resolved_broker: EventBroker | None = event_broker
    if resolved_broker is None:
        resolved_broker = getattr(runtime, "event_broker", None)
    if resolved_broker is None and listen_notify_capable:
        resolved_broker = EventBroker()

    lifespan = _build_lifespan(
        runtime=runtime,
        broker=resolved_broker,
        listen_notify_capable=listen_notify_capable,
    )

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
        lifespan=lifespan,
    )
    app.state.runtime = runtime
    app.state.event_broker = resolved_broker

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
    app.include_router(events.router)

    return app


# ---- Task 4.K3 lifespan helpers --------------------------------------------


def _runtime_supports_listen_notify(runtime: Runtime) -> bool:
    """Check whether the runtime's MetadataStore advertises
    :attr:`MetadataStoreCapabilities.supports_listen_notify` (Task 4.K3).

    Defensive on ``getattr`` so a runtime built around a hypothetical
    plugin without the new capability bit (Pydantic default
    ``False``) still gracefully degrades to "listener not spawned".
    """
    store = getattr(getattr(runtime, "plugins", None), "metadata_store", None)
    if store is None:
        return False
    capabilities = getattr(store, "capabilities", None)
    return bool(getattr(capabilities, "supports_listen_notify", False))


def _extract_listener_dsn(runtime: Runtime) -> str | None:
    """Extract a pure asyncpg DSN from the runtime's MetadataStore.

    Reaches into ``store._engine.url`` (a SQLAlchemy ``URL``) and
    drops the ``+asyncpg`` driver hint via
    :func:`sqlalchemy_url_to_asyncpg_dsn`. Returns ``None`` if the
    engine isn't introspectable (e.g., a hypothetical store without
    a SQLAlchemy ``_engine`` attribute) — the caller logs a warning
    and skips the listener.

    The ``_engine`` access is plugin-internal but the listener path
    is also Postgres-specific (gated by ``supports_listen_notify``),
    so the coupling is contained to that specific plugin pairing.
    """
    store = runtime.plugins.metadata_store
    engine: AsyncEngine | None = getattr(store, "_engine", None)
    if engine is None:
        return None
    url = engine.url
    rendered = url.render_as_string(hide_password=False)
    return sqlalchemy_url_to_asyncpg_dsn(rendered)


def _build_lifespan(
    *,
    runtime: Runtime,
    broker: EventBroker | None,
    listen_notify_capable: bool,
):  # noqa: ANN202 — FastAPI types lifespan loosely; closure return is acceptable
    """Build the FastAPI lifespan async-context-manager.

    No-op if there's no broker OR the store doesn't advertise
    ``supports_listen_notify``: the lifespan still mounts (FastAPI
    requires one), but it does no work — preserves the K2 behavior
    for the in-band Runtime path.

    When the listener IS spawned: holds the task for the lifetime of
    the app; signals shutdown via :class:`asyncio.Event`; awaits
    cleanup with a 5s timeout; cancels on hang. The listener task's
    own ``finally`` block removes the asyncpg listener and closes the
    connection.
    """

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        if broker is None or not listen_notify_capable:
            yield
            return

        dsn = _extract_listener_dsn(runtime)
        if dsn is None:
            _log.warning(
                "make_api_app: MetadataStore reports supports_listen_notify=True "
                "but no SQLAlchemy engine introspectable; cross-process events "
                "will not be delivered."
            )
            yield
            return

        shutdown_event = asyncio.Event()
        task = asyncio.create_task(
            pg_listener_task(dsn=dsn, broker=broker, shutdown=shutdown_event)
        )
        try:
            yield
        finally:
            shutdown_event.set()
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except TimeoutError:
                _log.warning("pg_listener_task did not exit within 5s; cancelling")
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass

    return lifespan


__all__ = ["AuthConfig", "make_api_app"]
