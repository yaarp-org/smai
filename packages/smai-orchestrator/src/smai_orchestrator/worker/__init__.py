"""Worker loop — three-phase poll cycle, concurrency pools, scheduling
queries.

Per ``designs/smai/05-orchestrator.md`` §3 + DEC-034 #4. Wraps the
engine substrate from :mod:`smai_orchestrator.engine` (Task 2.C1) in
the long-running worker that drives entities through their pipeline-
spec lifecycle. Single-worker default per `05` §6 / DEC-035 #2;
multi-worker leasing is Task 3.G1.

The package surface is split into two modules:

* :mod:`smai_orchestrator.worker.concurrency` — :class:`ConcurrencyPool`
  spec primitive plus slot-computation helpers. Imported at
  :mod:`smai_orchestrator.engine.types` model-build time so the
  pipeline-spec can reference pools.
* :mod:`smai_orchestrator.worker.loop` — three-phase poll cycle,
  scheduling-query references, top-level :func:`run_worker_loop`.

Only :mod:`worker.concurrency` is auto-imported here so the engine
can pull in :class:`ConcurrencyPool` mid-init without forcing
:mod:`worker.loop` (which itself depends on the full engine surface)
to load first. Callers that want the loop import explicitly from
:mod:`smai_orchestrator.worker.loop`.
"""

from smai_orchestrator.worker.concurrency import (
    ConcurrencyPool,
    PoolSlot,
    compute_pool_slots,
    in_flight_states_for_pool,
)

__all__ = [
    "ConcurrencyPool",
    "PoolSlot",
    "compute_pool_slots",
    "in_flight_states_for_pool",
]
