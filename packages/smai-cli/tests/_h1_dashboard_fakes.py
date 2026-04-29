"""Test helpers for the ``smai serve`` dashboard.

Per the workspace pytest ``--import-mode=importlib`` discovery layout,
every package's ``tests/`` is on ``sys.path`` once. Module name
``_h1_dashboard_fakes`` is unique within the workspace per the Task
3.H1 brief's filename-hygiene guidance (mirrors the cousin shape
``_e3_fakes`` / ``_g1_fakes``).

The dashboard's HTTP handlers pull a :class:`Runtime` off
``app.state.runtime``; the helpers in this module construct the
underlying service surfaces against a real
:class:`SqliteStore` (in-memory, migrated) so per-route tests exercise
the real :class:`MetadataStore` Protocol round-trip. We bypass the
full :class:`Runtime` context manager (no worker, no LLM providers)
because the dashboard is read-only — no agent dispatch fires from
any handler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smai_cli.runtime import (
    PapersService,
    ProposalsService,
    StatusService,
)
from smai_core.plugins import MetadataStore
from smai_store_sqlite import SqliteStore


@dataclass
class _PluginsBundle:
    """Minimal :class:`InstantiatedPlugins`-shaped bundle for the dashboard.

    The :class:`StatusService` / :class:`ProposalsService` /
    :class:`PapersService` constructors only read ``metadata_store``
    from the bundle — every other slot stays None for the read-only
    dashboard surface.
    """

    metadata_store: MetadataStore


class FakeRuntime:
    """Duck-typed :class:`smai_cli.runtime.Runtime` for dashboard tests.

    Exposes only the properties the dashboard handlers read:
    ``.status`` / ``.proposals`` / ``.papers``. Constructed once per
    test against an in-memory :class:`SqliteStore` and mounted onto
    ``app.state.runtime`` directly (bypassing
    :func:`smai_cli.dashboard.build_app`'s isinstance check via the
    ``cast`` in the page handlers).
    """

    def __init__(self, *, store: MetadataStore) -> None:
        plugins = _PluginsBundle(metadata_store=store)
        # The services only touch :attr:`metadata_store` on the
        # bundle, so the typed-dataclass shim is a structural fit even
        # though it isn't a full :class:`InstantiatedPlugins`.
        self._status = StatusService(plugins=plugins)  # type: ignore[arg-type]
        self._proposals = ProposalsService(plugins=plugins)  # type: ignore[arg-type]
        self._papers = PapersService(plugins=plugins)  # type: ignore[arg-type]
        self._store = store

    @property
    def status(self) -> StatusService:
        return self._status

    @property
    def proposals(self) -> ProposalsService:
        return self._proposals

    @property
    def papers(self) -> PapersService:
        return self._papers

    @property
    def metadata_store(self) -> MetadataStore:
        return self._store


async def make_in_memory_store() -> SqliteStore:
    """Construct a migrated in-memory :class:`SqliteStore`.

    Tests that need a populated store call this, then write their
    fixture rows via :meth:`MetadataStore.create_*` /
    :meth:`MetadataStore.transition_*_state`.
    """
    store = SqliteStore("sqlite+aiosqlite:///:memory:")
    await store.migrate()
    return store


def build_test_app(runtime: FakeRuntime) -> Any:
    """Construct a dashboard app pointed at ``runtime``.

    Wraps :func:`smai_cli.dashboard.build_app`. Returns the FastAPI
    app; tests pass it to :class:`fastapi.testclient.TestClient`.
    """
    from smai_cli.dashboard import build_app  # noqa: PLC0415

    # ``build_app`` types its argument as :class:`Runtime`; the
    # FakeRuntime is a duck-typed surface (same property names). The
    # cast keeps pyright quiet at the test site.
    return build_app(runtime)  # type: ignore[arg-type]


__all__ = [
    "FakeRuntime",
    "build_test_app",
    "make_in_memory_store",
]
