""":class:`MetadataStoreCapabilities` — static per-plugin capability flags.

Per ``designs/smai/07-plugin-interfaces.md`` §5.5 (multi-tenant fairness
ordering) and DEC-028.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class MetadataStoreCapabilities(BaseModel):
    """Static per-plugin capability flags (§5.5).

    ``is_tenant_aware``:
        ``True`` for tenant-aware plugins (e.g., the closed
        ``AuroraStore``); ``False`` for the single-tenant OSS references
        (``SqliteStore``, ``PostgresStore``). Conformance: when ``True``,
        the plugin's scheduling queries MUST honor the tenant-fairness
        contract (§5.8); when ``False``, the conformance suite skips the
        tenant-fairness group.

    ``supports_transactions``:
        Reserved for future single-row-only stores; v1 SQL plugins all
        ``True``.
    """

    model_config = ConfigDict(extra="forbid")

    is_tenant_aware: bool
    supports_transactions: bool = True
