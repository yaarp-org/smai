""":class:`MetadataStoreCapabilities` — static per-plugin capability flags.

Per ``designs/smai/07-plugin-interfaces.md`` §5.5 (multi-tenant fairness
ordering), §5.6.7 (lease semantics), §5.8 (capability surface), and
DEC-028 / DEC-035 #2.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class MetadataStoreCapabilities(BaseModel):
    """Static per-plugin capability flags (§5.5 / §5.6.7 / §5.8).

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

    ``supports_leasing``:
        ``True`` iff the plugin implements the §5.6.7 lease primitives
        (``acquire_lease`` / ``release_lease`` / ``extend_lease``) with
        durable, ABA-safe semantics — i.e., a worker that crashes mid-
        dispatch cannot block another worker indefinitely, and a stale
        token cannot release another worker's freshly-acquired lease
        (per DEC-035 #2). Multi-worker deployments REQUIRE this: per
        ``09-cli.md`` §6.2, ``smai start`` hard-exits when
        ``EngineConfig.worker_count > 1`` and the configured store
        reports ``supports_leasing=False``. The OSS references
        (``SqliteStore``, ``PostgresStore``) both implement the lease
        primitives and report ``True``; a hypothetical "ephemeral
        in-process" store could ship with ``False`` and operate in
        single-worker mode only. Defaults to ``True`` so new SQL-backed
        plugins inherit the safe value; new plugins that cannot satisfy
        the contract MUST opt out explicitly.
    """

    model_config = ConfigDict(extra="forbid")

    is_tenant_aware: bool
    supports_transactions: bool = True
    supports_leasing: bool = True
