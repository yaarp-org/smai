"""``/api/v1/system`` router per ``designs/smai/11-api.md`` §4.6.

Cross-cutting reads + the verify probe. Keep handler bodies thin —
they delegate to existing CLI / Runtime helpers (verify probes,
plugin discovery, migration introspection).
"""

from __future__ import annotations

import importlib.metadata
from typing import Any, cast

from fastapi import APIRouter
from smai_api_spec import (
    PluginInfo,
    PluginVerifyResult,
    RecentActivityItem,
    SummaryCounts,
    SystemConfigResponse,
    SystemDashboardResponse,
    SystemHealthResponse,
    SystemMigrateStatusResponse,
    SystemPluginsResponse,
    SystemVerifyRequest,
    SystemVerifyResponse,
    SystemVersionResponse,
)
from smai_api_spec.paths import (
    SYSTEM_CONFIG,
    SYSTEM_DASHBOARD,
    SYSTEM_HEALTH,
    SYSTEM_MIGRATE_STATUS,
    SYSTEM_PLUGINS,
    SYSTEM_VERIFY,
    SYSTEM_VERSION,
)
from smai_cli.verify import (
    verify_artifact_store,
    verify_compute,
    verify_llm_provider,
    verify_metadata_store,
)

from smai_api._deps import RuntimeDep

router = APIRouter()

# Field-name suffixes that indicate a secret per ``11`` §4.6 — config
# values are redacted on the wire when their key matches any suffix.
_REDACT_SUFFIXES: tuple[str, ...] = (
    "_token",
    "_password",
    "_secret",
    "_key",
    "_api_key",
    "token",
    "password",
    "secret",
)
# Field names that legitimately end in one of the above suffixes but
# are NOT secrets and shouldn't be redacted (e.g. artifact-key paths,
# content-hash references).
_REDACT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "artifact_key",
        "technique_description_artifact_key",
        "experiment_plan_artifact_key",
        "harness_contract_artifact_key",
        "validation_config_artifact_key",
        "evaluation_result_artifact_key",
        "design_plan_artifact_key",
        "error_context_artifact_key",
        "raw_metrics_artifact_key",
        "experiment_plan_hash",
        "harness_contract_hash",
        "validation_config_hash",
        "code_review_result_hash",
        "technique_contract_hash",
        "harness_api_manifest_hash",
    }
)
_REDACTED_PLACEHOLDER = "<redacted>"


def _looks_like_secret_key(key: str) -> bool:
    """Return ``True`` if the key name suggests a secret value.

    Allowlists the common methodology-side ``*_artifact_key`` /
    ``*_hash`` field names that match the suffix heuristic but aren't
    secrets.
    """
    if key in _REDACT_ALLOWLIST:
        return False
    lowered = key.lower()
    return any(lowered.endswith(suffix) for suffix in _REDACT_SUFFIXES)


def _redact(value: object) -> object:
    """Recursively redact secrets in a config dict.

    ``dict`` values get key-by-key redaction; nested dicts / lists
    recurse. Other types pass through unchanged.
    """
    if isinstance(value, dict):
        out: dict[str, object] = {}
        for key, sub in cast("dict[str, object]", value).items():
            # ``model_dump`` always emits string keys; pyright knows
            # ``key`` is ``str`` here without an isinstance check.
            if _looks_like_secret_key(key) and sub not in (None, ""):
                out[key] = _REDACTED_PLACEHOLDER
            else:
                out[key] = _redact(sub)
        return out
    if isinstance(value, list):
        return [_redact(item) for item in cast("list[object]", value)]
    return value


# === GET /api/v1/system/version ============================================


@router.get(SYSTEM_VERSION, response_model=SystemVersionResponse)
async def get_system_version(
    runtime: RuntimeDep,
) -> SystemVersionResponse:
    """Versions of ``smai-cli`` + ``smai-core`` + ``smai-api-spec`` plus
    every currently-loaded plugin distribution per ``11`` §4.6 / 4.J1
    OQ6 RESOLVED 2026-05-03 (``smai_api_spec`` is a top-level field)."""
    del runtime  # plugin set is read from importlib.metadata, not the runtime
    return SystemVersionResponse(
        smai_cli=_safe_dist_version("smai-cli"),
        smai_core=_safe_dist_version("smai-core"),
        smai_api_spec=_safe_dist_version("smai-api-spec"),
        plugins=_loaded_plugin_versions(),
    )


# === GET /api/v1/system/config =============================================


@router.get(SYSTEM_CONFIG, response_model=SystemConfigResponse)
async def get_system_config(
    runtime: RuntimeDep,
) -> SystemConfigResponse:
    """Read-only ``RuntimeConfig`` projection with secrets redacted.

    Per ``11`` §4.6: every ``*_token`` / ``*_password`` / ``*_secret``
    / ``*_key`` becomes ``"<redacted>"``. The methodology-side
    ``*_artifact_key`` / ``*_hash`` field names are allowlisted (they
    look like secrets to the suffix heuristic but aren't).
    """
    # ``EngineConfig`` carries ``time_provider`` / ``wall_clock`` clock
    # seams typed as Callable — not JSON-serializable. The spec surface
    # treats these as implementation-internal so we exclude them from
    # the dumped projection per ``11`` §4.6 (``config`` is opaque
    # ``dict[str, object]`` — additive shape changes upstream don't
    # require a contract bump).
    raw = runtime.config.model_dump(
        mode="json",
        exclude={"engine": {"time_provider", "wall_clock"}},
    )
    redacted = cast("dict[str, object]", _redact(raw))
    return SystemConfigResponse(config=redacted)


# === GET /api/v1/system/plugins ============================================


@router.get(SYSTEM_PLUGINS, response_model=SystemPluginsResponse)
async def get_system_plugins(
    runtime: RuntimeDep,
) -> SystemPluginsResponse:
    """Discovered + selected plugins per Protocol namespace per ``11`` §4.6."""
    from smai_orchestrator import list_discovered_plugins

    discovered = list_discovered_plugins()
    plugins = runtime.config.plugins
    selected = {
        "smai.llm_providers": plugins.llm_provider,
        "smai.metadata_stores": plugins.metadata_store,
        "smai.artifact_stores": plugins.artifact_store,
        "smai.computes": plugins.compute,
    }

    def _bucket(group: str) -> list[PluginInfo]:
        names = discovered.get(group, [])
        active = selected[group]
        return [
            PluginInfo(
                name=name,
                distribution=name,  # entry-point name doubles as dist label
                version=_safe_entry_point_version(group, name),
                selected=(name == active),
            )
            for name in names
        ]

    return SystemPluginsResponse(
        llm_providers=_bucket("smai.llm_providers"),
        metadata_stores=_bucket("smai.metadata_stores"),
        artifact_stores=_bucket("smai.artifact_stores"),
        computes=_bucket("smai.computes"),
    )


# === POST /api/v1/system/verify ============================================


@router.post(SYSTEM_VERIFY, response_model=SystemVerifyResponse)
async def post_system_verify(
    runtime: RuntimeDep,
    body: SystemVerifyRequest | None = None,
) -> SystemVerifyResponse:
    """Run the four-plugin pre-flight (per ``11`` §5.2.5).

    **Costs LLM tokens** — the LLM probe issues a 1-token completion
    against the configured ``LlmProvider``. Returns 200 always; the
    body carries the per-plugin diagnostic.

    The ``body.plugins`` filter is a forward-compat hook per ``11`` §13
    OQ5 — for v1 we always run all four probes regardless and return
    the full envelope. Honoring the filter would surface "skipped"
    plugins in the response, which the Pydantic shape doesn't allow
    (each field is required).
    """
    del body
    # Pick any LlmProvider in the per-role map — they share the same
    # plugin instance in the v1 default; for per-role-divergent
    # deployments we probe one to surface the wiring without paying
    # N times the token cost.
    llm_providers = runtime.plugins.llm_providers
    llm = next(iter(llm_providers.values()))
    llm_result = await verify_llm_provider(llm)
    metadata_result = await verify_metadata_store(runtime.plugins.metadata_store)
    artifact_result = await verify_artifact_store(runtime.plugins.artifact_store)
    compute_result = await verify_compute(runtime.plugins.compute)
    overall_ok = llm_result.ok and metadata_result.ok and artifact_result.ok and compute_result.ok
    return SystemVerifyResponse(
        llm_provider=PluginVerifyResult(
            ok=llm_result.ok, reason=llm_result.reason, latency_ms=llm_result.latency_ms
        ),
        metadata_store=PluginVerifyResult(
            ok=metadata_result.ok,
            reason=metadata_result.reason,
            latency_ms=metadata_result.latency_ms,
        ),
        artifact_store=PluginVerifyResult(
            ok=artifact_result.ok,
            reason=artifact_result.reason,
            latency_ms=artifact_result.latency_ms,
        ),
        compute=PluginVerifyResult(
            ok=compute_result.ok,
            reason=compute_result.reason,
            latency_ms=compute_result.latency_ms,
        ),
        overall_ok=overall_ok,
    )


# === GET /api/v1/system/dashboard ==========================================


@router.get(SYSTEM_DASHBOARD, response_model=SystemDashboardResponse)
async def get_system_dashboard(
    runtime: RuntimeDep,
) -> SystemDashboardResponse:
    """Composite dashboard read — counts + recent-activity feed per ``11`` §4.6.

    The recent-activity feed is sourced from the active-entity
    aggregators on each service and sorted by ``updated_at`` descending.
    """
    counts_record = await runtime.status.summary_counts()
    counts = SummaryCounts(
        proposals_in_flight=counts_record.proposals_in_flight,
        cgs_in_flight=counts_record.cgs_in_flight,
        runs_in_flight=counts_record.runs_in_flight,
        papers_in_flight=counts_record.papers_in_flight,
    )
    recent: list[RecentActivityItem] = []
    proposals = await runtime.proposals.list_active()
    for p in proposals:
        recent.append(RecentActivityItem(kind="proposal", id=p.id, state=p.state, ts=p.updated_at))
    cgs = await runtime.status.list_active_cgs()
    for c in cgs:
        recent.append(
            RecentActivityItem(kind="comparison_group", id=c.id, state=c.state, ts=c.updated_at)
        )
    runs = await runtime.status.list_active_runs()
    for r in runs:
        recent.append(RecentActivityItem(kind="run", id=r.id, state=r.state, ts=r.updated_at))
    papers = await runtime.papers.list_active()
    for paper in papers:
        recent.append(
            RecentActivityItem(
                kind="paper", id=paper.arxiv_id, state=paper.state, ts=paper.updated_at
            )
        )
    recent.sort(key=lambda item: item.ts, reverse=True)
    # Cap the feed at 25 items — the SPA renders one screen of activity
    # per ``11`` §4.6; longer history is reachable via the per-resource
    # list pages.
    return SystemDashboardResponse(counts=counts, recent_activity=recent[:25])


# === GET /api/v1/system/migrate-status =====================================


@router.get(SYSTEM_MIGRATE_STATUS, response_model=SystemMigrateStatusResponse)
async def get_system_migrate_status(
    runtime: RuntimeDep,
) -> SystemMigrateStatusResponse:
    """``smai migrate --check`` equivalent per ``11`` §4.6.

    Best-effort: when the underlying ``MetadataStore`` doesn't expose
    a SQL engine handle (test stores, future non-SQL substrates), the
    response reports ``at_head=True`` against a synthetic ``"in-process"``
    revision. The real Alembic plugins (sqlite, postgres) plumb through
    cleanly via :func:`smai_orchestrator.migrations.get_head_revision` /
    :func:`get_current_revision`.
    """
    # Lazy import so tests with non-SQL stores don't pay the SQLAlchemy
    # import cost.
    try:
        from smai_orchestrator.migrations import (
            get_current_revision,
            get_head_revision,
        )
    except ImportError:
        return SystemMigrateStatusResponse(
            at_head=True, head_revision="in-process", current="in-process"
        )

    store = runtime.plugins.metadata_store
    engine = getattr(store, "_engine", None) or getattr(store, "engine", None)
    if engine is None:
        # Non-SQL store — report at-head against a synthetic revision.
        return SystemMigrateStatusResponse(
            at_head=True, head_revision="in-process", current="in-process"
        )
    try:
        current = await get_current_revision(engine)
        head = get_head_revision()
    except Exception:  # noqa: BLE001 — surface as best-effort at-head
        return SystemMigrateStatusResponse(
            at_head=True, head_revision="in-process", current="in-process"
        )
    return SystemMigrateStatusResponse(
        at_head=(current == head), head_revision=head, current=current
    )


# === GET /api/v1/system/health =============================================


@router.get(SYSTEM_HEALTH, response_model=SystemHealthResponse)
async def get_system_health() -> SystemHealthResponse:
    """Liveness probe — touches no plugins per ``11`` §4.6."""
    return SystemHealthResponse(status="ok")


# === Helpers ===============================================================


def _safe_dist_version(distribution: str) -> str:
    """Return the installed version of ``distribution``, or ``"0.0.0"``
    when the distribution metadata isn't available (workspace-mode
    install in CI, etc.)."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


def _loaded_plugin_versions() -> dict[str, str]:
    """Map distribution name → version for every installed plugin per
    the four entry-point groups."""
    out: dict[str, str] = {}
    for group in (
        "smai.llm_providers",
        "smai.metadata_stores",
        "smai.artifact_stores",
        "smai.computes",
    ):
        eps = importlib.metadata.entry_points(group=group)
        for ep in eps:
            dist = _entry_point_distribution(ep)
            if dist is None:
                continue
            out[dist] = _safe_dist_version(dist)
    return out


def _safe_entry_point_version(group: str, name: str) -> str:
    """Resolve the distribution version for entry point ``(group, name)``."""
    eps = importlib.metadata.entry_points(group=group, name=name)
    for ep in eps:
        dist = _entry_point_distribution(ep)
        if dist is not None:
            return _safe_dist_version(dist)
    return "0.0.0"


def _entry_point_distribution(ep: Any) -> str | None:
    """Best-effort extract the distribution name an entry point belongs to.

    ``importlib.metadata.EntryPoint`` exposes ``.dist.name`` only on
    Python 3.10+; the attribute itself is sometimes ``None``. We
    surface a ``str`` distribution name when available, ``None``
    otherwise.
    """
    dist = getattr(ep, "dist", None)
    if dist is None:
        return None
    name = getattr(dist, "name", None)
    if isinstance(name, str):
        return name
    return None


__all__ = ["router"]
