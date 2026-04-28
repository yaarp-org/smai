"""Full pipeline-spec — the canonical configuration the engine consumes.

Per ``designs/smai/05-orchestrator.md`` §5.1 (format) and §5.2 (what
goes into a spec). Composes the C1/C2 :class:`EngineSpec` substrate
into the user-facing :class:`PipelineSpec` with a top-level
:attr:`name`, :attr:`pools`, and a single collapsed
:attr:`scheduling_queries` map (resolving the C2 two-map split — see
the deviation note in :func:`PipelineSpec.engine_spec`).

A pipeline-spec is **frozen at worker startup** per `05` §5.1 — there
is no hot-reload in v2. Instances are constructed in user code (so
gate-rule and dispatch-handler callables are first-class Python
references) and then registered into the process-local
:mod:`smai_orchestrator.runtime.registry` for lookup by name.

This module owns the structural validators per `05` §5.2:

* All edge ``from_state`` / ``target_state`` references resolve in
  :attr:`states`.
* All :attr:`scheduling_queries` keys resolve in :attr:`states`.
* All :class:`DispatchAction.pool` references resolve in :attr:`pools`.
* :attr:`initial_state` resolves in :attr:`states`.
* No edges may originate from a terminal :class:`StateDef`
  (``is_terminal=True``).
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, model_validator
from smai_core.plugins import EntityKind

from smai_orchestrator.engine.types import (
    ConcurrencyPool,
    EdgeDef,
    EngineSpec,
    PhaseTrigger,
    SchedulingQueryRef,
    StateDef,
)


class PipelineSpec(BaseModel):
    """Full pipeline-spec — the configuration the engine consumes
    (`05` §5.1).

    Composes the C1/C2 :class:`EngineSpec` substrate (states, edges,
    initial state, entity kind, pools, scheduling queries) with a
    top-level :attr:`name` for the registry lookup. Single
    :attr:`scheduling_queries` map per `05` §5.1's canonical shape;
    the worker-loop's two-map shape (``phase1_queries`` /
    ``phase2_queries``) is derived in :meth:`engine_spec` by
    partitioning per-state on the dispatch-edge phase trigger.

    See the module docstring for the structural validators applied at
    construction time.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    """Spec identifier — registered in
    :mod:`smai_orchestrator.runtime.registry` and looked up by
    :attr:`smai_orchestrator.runtime.config.RuntimeConfig.pipelines`.
    Conventional values: ``"smai_cg_execution"``,
    ``"smai_proposal"``, ``"smai_paper_ingestion"`` per DEC-032."""

    entity_kind: EntityKind
    """Discriminator the engine's :mod:`_metadata_ops` adapter uses to
    dispatch per-kind :class:`MetadataStore` methods. One of
    ``"comparison_group"`` / ``"entry"`` / ``"run"`` / ``"proposal"``
    / ``"paper"`` per `01` §5."""

    initial_state: str
    """The state every freshly created entity occupies. Must resolve
    in :attr:`states`."""

    states: list[StateDef]
    """Every state the entity can occupy, including terminals. Names
    must be unique within the spec."""

    edges: list[EdgeDef]
    """Typed transitions between states. Multiple edges may leave the
    same state; evaluation order is declaration order — first
    ``advance=True`` wins (`05` §1.2)."""

    pools: list[ConcurrencyPool]
    """Concurrency-pool declarations referenced by
    :class:`DispatchAction.pool`. Per the v1 CG-execution shape (`05`
    §5.2) most specs declare two pools: ``agent`` (LLM-bound) and
    ``run`` (GPU-bound)."""

    scheduling_queries: dict[str, SchedulingQueryRef]
    """State name → :class:`MetadataStore` discovery-query reference.
    The worker calls each query per cycle to find candidate records.
    Per the canonical `05` §5.1 shape this is one map per state; the
    engine partitions queries into phase-1 (find in-flight) vs phase-2
    (find ready) by inspecting the spec's outgoing-edge ``fires_on``
    triggers — see :meth:`engine_spec` for the partition logic."""

    # === Structural validators (§5.2) =======================================

    @model_validator(mode="after")
    def _validate_initial_state(self) -> Self:
        state_names = {s.name for s in self.states}
        if self.initial_state not in state_names:
            raise ValueError(
                f"initial_state {self.initial_state!r} is not in states "
                f"({sorted(state_names)})"
            )
        return self

    @model_validator(mode="after")
    def _validate_unique_state_names(self) -> Self:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for state in self.states:
            if state.name in seen:
                duplicates.add(state.name)
            seen.add(state.name)
        if duplicates:
            raise ValueError(
                f"states declares duplicate names: {sorted(duplicates)}"
            )
        return self

    @model_validator(mode="after")
    def _validate_unique_pool_names(self) -> Self:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for pool in self.pools:
            if pool.name in seen:
                duplicates.add(pool.name)
            seen.add(pool.name)
        if duplicates:
            raise ValueError(
                f"pools declares duplicate names: {sorted(duplicates)}"
            )
        return self

    @model_validator(mode="after")
    def _validate_edge_state_refs(self) -> Self:
        state_names = {s.name for s in self.states}
        terminal_names = {s.name for s in self.states if s.is_terminal}
        for edge in self.edges:
            if edge.from_state not in state_names:
                raise ValueError(
                    f"edge {edge.name!r} references unknown from_state "
                    f"{edge.from_state!r}"
                )
            if edge.target_state not in state_names:
                raise ValueError(
                    f"edge {edge.name!r} references unknown target_state "
                    f"{edge.target_state!r}"
                )
            if edge.from_state in terminal_names:
                raise ValueError(
                    f"edge {edge.name!r} originates from terminal state "
                    f"{edge.from_state!r}; terminal states have no outgoing "
                    f"edges"
                )
        return self

    @model_validator(mode="after")
    def _validate_dispatch_pool_refs(self) -> Self:
        pool_names = {p.name for p in self.pools}
        for state in self.states:
            if state.on_entry_dispatch is None:
                continue
            pool = state.on_entry_dispatch.pool
            if pool not in pool_names:
                raise ValueError(
                    f"state {state.name!r} on_entry_dispatch references "
                    f"unknown pool {pool!r} (pools: {sorted(pool_names)})"
                )
        return self

    @model_validator(mode="after")
    def _validate_scheduling_query_state_refs(self) -> Self:
        state_names = {s.name for s in self.states}
        terminal_names = {s.name for s in self.states if s.is_terminal}
        for state_name in self.scheduling_queries:
            if state_name not in state_names:
                raise ValueError(
                    f"scheduling_queries declares state {state_name!r} but "
                    f"the spec has no matching StateDef"
                )
            if state_name in terminal_names:
                raise ValueError(
                    f"scheduling_queries[{state_name!r}] points at a "
                    f"terminal state; terminals never need discovery"
                )
        return self

    # === Engine substrate projection ========================================

    def engine_spec(self) -> EngineSpec:
        """Project this spec onto the worker-loop's :class:`EngineSpec`
        substrate.

        The worker loop (Task 2.C2) consumes a two-map scheduling-query
        shape (``phase1_queries`` / ``phase2_queries``) — phase 1
        discovers in-flight records to poll, phase 2 discovers ready
        candidates to dispatch. The canonical `05` §5.1 spec collapses
        these into one map keyed on state name, which we partition here
        by inspecting the state's outgoing edges:

        * If the state has any ``fires_on="job_succeeded"`` /
          ``"job_failed"`` outgoing edge → the query is a phase-1
          query (find in-flight entities at this state).
        * Else the query is a phase-2 query (find ready candidates at
          this state).

        Spec authors who need both phase-1 and phase-2 discovery on the
        same state author both — but the C2 worker keys them per state
        and per phase separately. C3 ships the simpler one-map external
        surface; C4's CG-execution spec validates the partition works
        for the SMAI shape. If a future spec surfaces a state with
        truly distinct phase-1 / phase-2 query callables, we revisit
        the collapse decision (one-line flip — re-expose the two-map
        shape on :class:`PipelineSpec`).
        """
        phase1: dict[str, SchedulingQueryRef] = {}
        phase2: dict[str, SchedulingQueryRef] = {}
        for state_name, query_ref in self.scheduling_queries.items():
            outgoing = self._outgoing_edges(state_name)
            has_phase1 = any(
                e.fires_on in {"job_succeeded", "job_failed"} for e in outgoing
            )
            if has_phase1:
                phase1[state_name] = query_ref
            else:
                phase2[state_name] = query_ref
        return EngineSpec(
            entity_kind=self.entity_kind,
            initial_state=self.initial_state,
            states=self.states,
            edges=self.edges,
            pools=self.pools,
            phase1_queries=phase1,
            phase2_queries=phase2,
        )

    def _outgoing_edges(self, state: str) -> list[EdgeDef]:
        return [e for e in self.edges if e.from_state == state]

    def state_def(self, state: str) -> StateDef:
        """Look up a :class:`StateDef` by name; raises :class:`KeyError`."""
        for s in self.states:
            if s.name == state:
                return s
        raise KeyError(f"unregistered state {state!r}")

    def edges_from(self, state: str, fires_on: PhaseTrigger) -> list[EdgeDef]:
        """Return the edges leaving ``state`` whose ``fires_on``
        matches ``fires_on``, in declaration order.
        """
        return [
            e for e in self.edges if e.from_state == state and e.fires_on == fires_on
        ]

    def model_dump_for_inspection(self) -> dict[str, Any]:
        """JSON-friendly dump for telemetry / replay.

        Callable references serialize as their qualified names per `05`
        §5.1 ("opaque but useful"); Pydantic's default ``model_dump``
        on ``Callable`` fields raises, so we route through a manual
        projection that stringifies the callables.
        """
        return {
            "name": self.name,
            "entity_kind": self.entity_kind,
            "initial_state": self.initial_state,
            "states": [
                {
                    "name": s.name,
                    "is_terminal": s.is_terminal,
                    "on_entry_dispatch": (
                        {
                            "name": s.on_entry_dispatch.name,
                            "pool": s.on_entry_dispatch.pool,
                            "handle_field": s.on_entry_dispatch.handle_field,
                            "handler": _qualname(s.on_entry_dispatch.handler),
                        }
                        if s.on_entry_dispatch is not None
                        else None
                    ),
                }
                for s in self.states
            ],
            "edges": [
                {
                    "name": e.name,
                    "from_state": e.from_state,
                    "target_state": e.target_state,
                    "fires_on": e.fires_on,
                    "gate_rule": _qualname(e.gate_rule),
                }
                for e in self.edges
            ],
            "pools": [p.model_dump() for p in self.pools],
            "scheduling_queries": {
                name: {
                    "name": ref.name,
                    "fair_scheduling": ref.fair_scheduling,
                    "fn": _qualname(ref.fn),
                }
                for name, ref in self.scheduling_queries.items()
            },
        }


def _qualname(callable_obj: object) -> str:
    """Best-effort qualified name for a callable, for inspection dumps."""
    module = getattr(callable_obj, "__module__", "?")
    qualname = getattr(callable_obj, "__qualname__", None) or getattr(
        callable_obj, "__name__", repr(callable_obj)
    )
    return f"{module}.{qualname}"


__all__ = ["PipelineSpec"]
