"""Concurrency pools and slot accounting.

Per ``designs/smai/05-orchestrator.md`` §3.4 / DEC-034 #4. The pipeline-
spec declares **named concurrency pools**, each with a configurable
``limit`` and integer ``priority``. Each :class:`DispatchAction`
declares which pool it belongs to via :attr:`DispatchAction.pool`. At
dispatch time the worker computes available slots per pool
(``limit − in_flight_count``) and dispatches in priority order,
draining higher-priority pools first.

DEC-034 #4 settled the v2 default — four pools out of the box (``agent``,
``run``, ``gate``, ``inline``) — but C2 ships single-pool by default
(matching v1's two-pool shape minus the implicit gate / inline
collapse) and exercises the full four-pool path through
``EngineSpec.pools`` + tests. The four-pool default landing in the
SMAI CG-execution spec is Task 2.C4's scope.

The :class:`ConcurrencyPool` model itself lives in
:mod:`smai_orchestrator.engine.types` (it's a spec primitive, like
:class:`StateDef` and :class:`EdgeDef`); this module re-exports it so
callers that import from ``smai_orchestrator.worker`` see the same
namespace as the slot-computation helpers.
"""

from __future__ import annotations

from dataclasses import dataclass

from smai_core.plugins import EntityKind, MetadataStore

from smai_orchestrator.engine.config import EngineConfig
from smai_orchestrator.engine.types import ConcurrencyPool


@dataclass(frozen=True)
class PoolSlot:
    """Computed slot information for a pool at one phase-3 cycle.

    Returned by :func:`compute_pool_slots` so the worker has a single
    source for "what's the capacity right now?" queries — used both
    for dispatch-ordering decisions and for telemetry / logging.
    """

    name: str
    limit: int
    in_flight: int
    available: int
    priority: int


def _resolve_pool_limit(pool: ConcurrencyPool, config: EngineConfig) -> int:
    """Resolve the effective limit for ``pool`` (`05` §6).

    :attr:`EngineConfig.pool_overrides` lets a deployment override the
    spec-declared limit per pool — e.g., the spec declares
    ``agent.limit=10`` but the local-laptop deployment runs with 4. The
    override map keys by pool name; a pool name absent from the map
    falls back to the spec's declared :attr:`ConcurrencyPool.limit`.
    """
    return config.pool_overrides.get(pool.name, pool.limit)


def in_flight_states_for_pool(
    pool_name: str,
    state_to_pool: dict[str, str],
) -> list[str]:
    """Derive the set of in-progress state names belonging to ``pool_name``
    (`05` §3.4 / `07` §5.6.6).

    The :class:`MetadataStore`'s ``count_with_in_flight_jobs(entity_kind,
    in_flight_states)`` takes a caller-resolved state list per DEC-035
    #3 — the plugin stays spec-agnostic, the engine resolves "which
    states feed this pool" by walking the spec.

    Per `05` §1.4 / §3.4: a state's ``on_entry_dispatch.pool`` is the
    pool the dispatch action belongs to; the state itself is the
    in-progress state whose entities count against that pool's slot
    budget while their ``job_handle`` is non-null. The worker passes
    ``state_to_pool`` already derived from
    ``spec.states[*].on_entry_dispatch.pool``; this function filters
    by membership.
    """
    return sorted(s for s, p in state_to_pool.items() if p == pool_name)


async def compute_pool_slots(
    *,
    pools: list[ConcurrencyPool],
    state_to_pool: dict[str, str],
    entity_kind: EntityKind,
    metadata_store: MetadataStore,
    config: EngineConfig,
) -> list[PoolSlot]:
    """Compute available slots for every pool (`05` §3.4 / `07` §5.6.6).

    One :meth:`MetadataStore.count_with_in_flight_jobs` round-trip per
    pool. Returned in descending priority order so callers can iterate
    "drain highest-priority first" without a re-sort.

    Pools whose mapped in-flight states are empty (typically because
    the spec declares the pool but no state's :attr:`on_entry_dispatch`
    yet references it) report ``in_flight=0`` without invoking the
    plugin — keeps the count round-trip count tight for under-
    populated specs (e.g., the synthetic single-state engine fixture).
    """
    slots: list[PoolSlot] = []
    for pool in pools:
        limit = _resolve_pool_limit(pool, config)
        in_flight_states = in_flight_states_for_pool(pool.name, state_to_pool)
        if in_flight_states:
            in_flight = await metadata_store.count_with_in_flight_jobs(
                entity_kind, in_flight_states
            )
        else:
            in_flight = 0
        available = max(0, limit - in_flight)
        slots.append(
            PoolSlot(
                name=pool.name,
                limit=limit,
                in_flight=in_flight,
                available=available,
                priority=pool.priority,
            )
        )
    # Stable sort: descending priority, then ascending name (deterministic
    # tiebreak so telemetry / replay shapes don't drift across cycles).
    slots.sort(key=lambda s: (-s.priority, s.name))
    return slots


__all__ = [
    "ConcurrencyPool",
    "PoolSlot",
    "compute_pool_slots",
    "in_flight_states_for_pool",
]
