"""System request / response shapes per ``designs/smai/11-api.md`` §4.6.

Cross-cutting reads under ``/api/v1/system/``: version, config, plugin
discovery, plugin pre-flight, dashboard, migrate-status, health.

Endpoints covered (URL constants exported from :mod:`smai_api_spec.paths`):

* ``GET  .../system/version`` → :class:`SystemVersionResponse`
* ``GET  .../system/config`` → :class:`SystemConfigResponse`
* ``GET  .../system/plugins`` → :class:`SystemPluginsResponse`
* ``POST .../system/verify`` —
  :class:`SystemVerifyRequest` → :class:`SystemVerifyResponse`
* ``GET  .../system/dashboard`` → :class:`SystemDashboardResponse`
* ``GET  .../system/migrate-status`` → :class:`SystemMigrateStatusResponse`
* ``GET  .../system/health`` → :class:`SystemHealthResponse`
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from smai_api_spec._common import APIBaseModel, EntityKind

# === GET /api/v1/system/version =============================================


class SystemVersionResponse(APIBaseModel):
    """``smai-cli`` + ``smai-core`` versions plus the currently-loaded
    plugin packages. Mirrors the ``smai version`` CLI output.

    ``plugins`` is a mapping from PyPI distribution name to its installed
    version (e.g. ``{"smai-llm-bedrock": "0.1.0", ...}``).
    """

    smai_cli: str
    smai_core: str
    smai_api_spec: str
    plugins: dict[str, str]


# === GET /api/v1/system/config ==============================================


class SystemConfigResponse(APIBaseModel):
    """Read-only ``RuntimeConfig`` projection.

    ``config`` is the resolved ``smai.yaml`` after layering with secrets
    redacted (passwords / tokens / API keys masked as ``"***"``). The
    inner shape is ``RuntimeConfig``-owned (``09-cli.md`` §3); the API
    treats it as opaque ``dict[str, object]`` so additive shape changes
    upstream do not require a contract bump.
    """

    config: dict[str, object]


# === GET /api/v1/system/plugins =============================================


class PluginInfo(APIBaseModel):
    """One row in :class:`SystemPluginsResponse`.

    ``selected`` is ``True`` when this plugin is the currently-active
    selection for its interface namespace; ``False`` for other
    discovered plugins in the same namespace.
    """

    name: str
    distribution: str
    version: str
    selected: bool


class SystemPluginsResponse(APIBaseModel):
    """Discovered + selected plugins, grouped by Protocol namespace.

    Namespaces match the four entry-point groups per DEC-026:
    ``smai.llm_providers`` / ``smai.metadata_stores`` /
    ``smai.artifact_stores`` / ``smai.computes``.
    """

    llm_providers: list[PluginInfo]
    metadata_stores: list[PluginInfo]
    artifact_stores: list[PluginInfo]
    computes: list[PluginInfo]


# === POST /api/v1/system/verify =============================================


class SystemVerifyRequest(APIBaseModel):
    """Optional body for ``POST /api/v1/system/verify``.

    Per ``11`` §13 OQ5 (open): the ``plugins`` filter is a forward-
    compatibility hook. For v1, both implementations may ignore it and
    always run all four probes; if real workflows want partial verify,
    a minor-version bump tightens the behavior.
    """

    plugins: list[Literal["llm_provider", "metadata_store", "artifact_store", "compute"]] | None = (
        None
    )


class PluginVerifyResult(APIBaseModel):
    """Per-plugin probe outcome.

    Mirrors :class:`smai_cli.verify.VerifyResult`. ``latency_ms`` is
    ``None`` when the probe didn't issue an I/O round-trip (e.g. a
    construction error before the ping fired).
    """

    ok: bool
    reason: str
    latency_ms: float | None


class SystemVerifyResponse(APIBaseModel):
    """``200 OK`` body for ``POST /api/v1/system/verify``.

    Per ``11`` §5.2.5: 200 always (not 503 when one plugin fails) — the
    body carries the diagnostic and clients want to render per-plugin
    status, not retry on a single 503. ``overall_ok`` is the AND of the
    four ``.ok`` fields.
    """

    llm_provider: PluginVerifyResult
    metadata_store: PluginVerifyResult
    artifact_store: PluginVerifyResult
    compute: PluginVerifyResult
    overall_ok: bool


# === GET /api/v1/system/dashboard ===========================================


class SummaryCounts(APIBaseModel):
    """In-flight entity counts (the dashboard count pills).

    Membership in "in-flight" is per the ``_*_IN_FLIGHT_STATES``
    frozensets in :mod:`smai_cli.runtime` — proposals in
    {``proposal_submitted``, ``designing``, ``designed``}; CGs in
    {``draft``, ``implementing``, ``implemented``, ``running``,
    ``evaluating``}; runs in {``pending``, ``submitted``, ``running``};
    papers in {``submitted``, ``fetching``, ``screening``, ``planning``,
    ``partial``}.
    """

    proposals_in_flight: int
    cgs_in_flight: int
    runs_in_flight: int
    papers_in_flight: int


class RecentActivityItem(APIBaseModel):
    """One row in the dashboard's recent-activity feed.

    The feed is a flattened union of recent entity transitions across
    all kinds, sorted by ``ts`` descending.
    """

    kind: EntityKind
    id: str
    state: str
    ts: datetime


class SystemDashboardResponse(APIBaseModel):
    """``200 OK`` body for ``GET /api/v1/system/dashboard``.

    Composite read — the dashboard's index page renders the count pills
    + the recent-activity feed in one round-trip.
    """

    counts: SummaryCounts
    recent_activity: list[RecentActivityItem]


# === GET /api/v1/system/migrate-status ======================================


class SystemMigrateStatusResponse(APIBaseModel):
    """``smai migrate --check`` equivalent.

    ``current`` is ``None`` when the schema has never been stamped (a
    fresh database). ``head`` is the latest revision known to the
    ``smai-orchestrator`` migration package shipped in the running
    process. ``at_head`` is ``current == head``.
    """

    at_head: bool
    head_revision: str
    current: str | None


# === GET /api/v1/system/health ==============================================


class SystemHealthResponse(APIBaseModel):
    """Liveness probe — the only endpoint that does not touch any plugin.

    ``status`` is the literal string ``"ok"`` when the process is up.
    Process supervisors and load balancers should treat any non-200
    response as unhealthy.
    """

    status: Literal["ok"]


__all__ = [
    "PluginInfo",
    "PluginVerifyResult",
    "RecentActivityItem",
    "SummaryCounts",
    "SystemConfigResponse",
    "SystemDashboardResponse",
    "SystemHealthResponse",
    "SystemMigrateStatusResponse",
    "SystemPluginsResponse",
    "SystemVerifyRequest",
    "SystemVerifyResponse",
    "SystemVersionResponse",
]
