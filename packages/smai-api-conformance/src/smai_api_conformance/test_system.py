"""Conformance tests for the ``/api/v1/system/*`` endpoints.

Per ``designs/smai/11-api.md`` §4.6 / §5.2.5.

Per the carry-forward from 4.J1: the ``SystemConfigResponse.config``
field is an opaque ``dict[str, object]`` on the contract surface; the
test asserts the outer envelope only and does not deeply inspect
``config``. ``SystemVersionResponse`` includes the ``smai_api_spec``
field — verified via parse-into-model rather than a hand-rolled key
assertion.
"""

from __future__ import annotations

from httpx import AsyncClient
from smai_api_spec import (
    SystemConfigResponse,
    SystemDashboardResponse,
    SystemHealthResponse,
    SystemMigrateStatusResponse,
    SystemPluginsResponse,
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


class SystemConformanceTests:
    """Mixin: ``/api/v1/system/*`` endpoint conformance tests."""

    # ---- GET /api/v1/system/version --------------------------------------

    async def test_get_version(self, client: AsyncClient) -> None:
        """Version endpoint returns SystemVersionResponse.

        Per 4.J1 carry-forward: the response includes ``smai_api_spec``
        as a top-level field — verified by ``model_validate`` (the
        Pydantic ``extra="forbid"`` config rejects unknown fields, and
        a missing required field would also fail).
        """
        response = await client.get(SYSTEM_VERSION)
        assert response.status_code == 200, response.text
        body = SystemVersionResponse.model_validate(response.json())
        # Verify the smai_api_spec field is present (per 4.J1 OQ
        # adjudication — distinguishes spec-package version from
        # smai-cli/smai-core).
        assert body.smai_api_spec

    # ---- GET /api/v1/system/config ---------------------------------------

    async def test_get_config(self, client: AsyncClient) -> None:
        """Config endpoint returns SystemConfigResponse.

        Per 4.J1 carry-forward: ``config`` is opaque ``dict[str, object]``
        — outer-shape assertion only.
        """
        response = await client.get(SYSTEM_CONFIG)
        assert response.status_code == 200, response.text
        SystemConfigResponse.model_validate(response.json())

    # ---- GET /api/v1/system/plugins --------------------------------------

    async def test_get_plugins(self, client: AsyncClient) -> None:
        """Plugins endpoint returns SystemPluginsResponse with 4 namespaces."""
        response = await client.get(SYSTEM_PLUGINS)
        assert response.status_code == 200, response.text
        SystemPluginsResponse.model_validate(response.json())

    # ---- POST /api/v1/system/verify --------------------------------------

    async def test_post_verify(self, client: AsyncClient) -> None:
        """Verify endpoint returns 200 + SystemVerifyResponse.

        Per ``11-api.md`` §5.2.5: 200 always (not 503 when one plugin
        fails) — the body carries the diagnostic.
        """
        response = await client.post(SYSTEM_VERIFY, json={})
        assert response.status_code == 200, response.text
        body = SystemVerifyResponse.model_validate(response.json())
        # overall_ok must be the AND of the four .ok fields.
        expected_overall = (
            body.llm_provider.ok
            and body.metadata_store.ok
            and body.artifact_store.ok
            and body.compute.ok
        )
        assert body.overall_ok == expected_overall

    # ---- GET /api/v1/system/dashboard ------------------------------------

    async def test_get_dashboard(self, client: AsyncClient) -> None:
        """Dashboard endpoint returns SystemDashboardResponse."""
        response = await client.get(SYSTEM_DASHBOARD)
        assert response.status_code == 200, response.text
        SystemDashboardResponse.model_validate(response.json())

    # ---- GET /api/v1/system/migrate-status -------------------------------

    async def test_get_migrate_status(self, client: AsyncClient) -> None:
        """Migrate-status endpoint returns SystemMigrateStatusResponse."""
        response = await client.get(SYSTEM_MIGRATE_STATUS)
        assert response.status_code == 200, response.text
        body = SystemMigrateStatusResponse.model_validate(response.json())
        # at_head must equal current == head_revision.
        assert body.at_head == (body.current == body.head_revision)

    # ---- GET /api/v1/system/health ---------------------------------------

    async def test_get_health(self, client: AsyncClient) -> None:
        """Health endpoint returns SystemHealthResponse with status='ok'."""
        response = await client.get(SYSTEM_HEALTH)
        assert response.status_code == 200, response.text
        body = SystemHealthResponse.model_validate(response.json())
        assert body.status == "ok"


__all__ = ["SystemConformanceTests"]
