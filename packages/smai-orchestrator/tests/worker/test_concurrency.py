"""Tests for :mod:`smai_orchestrator.worker.concurrency` per
``05-orchestrator.md`` §3.4 / DEC-035 #3.

Covers:

* :func:`in_flight_states_for_pool` — the spec-author's source of
  truth for which states feed which pool.
* :func:`compute_pool_slots` — per-pool slot computation against
  :meth:`MetadataStore.count_with_in_flight_jobs`, including
  :attr:`EngineConfig.pool_overrides` substitution.
* Priority ordering: pools sorted descending priority, then ascending
  name (deterministic tiebreak).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from smai_orchestrator.engine import ConcurrencyPool, EngineConfig
from smai_orchestrator.worker import (
    PoolSlot,
    compute_pool_slots,
    in_flight_states_for_pool,
)


class _FakeMetadataStore:
    """Minimal :class:`MetadataStore` stub for slot-counting tests.

    Only exposes :meth:`count_with_in_flight_jobs`; the rest of the
    Protocol surface is untouched.
    """

    name: str = "fake-store"

    def __init__(
        self,
        counter: Callable[[str, list[str]], int] | None = None,
    ) -> None:
        self._counter = counter or (lambda kind, states: 0)
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    async def count_with_in_flight_jobs(
        self,
        entity_kind: str,
        in_flight_states: list[str],
    ) -> int:
        self.calls.append((entity_kind, tuple(in_flight_states)))
        return self._counter(entity_kind, in_flight_states)


def test_in_flight_states_for_pool_basic() -> None:
    state_to_pool = {
        "implementing": "agents",
        "running": "runs",
        "evaluating": "inline",
    }
    assert in_flight_states_for_pool("agents", state_to_pool) == ["implementing"]
    assert in_flight_states_for_pool("runs", state_to_pool) == ["running"]
    assert in_flight_states_for_pool("inline", state_to_pool) == ["evaluating"]


def test_in_flight_states_for_pool_multi_state() -> None:
    state_to_pool = {
        "running": "runs",
        "rerunning": "runs",
        "implementing": "agents",
    }
    # Sorted alphabetically for deterministic tiebreaks.
    assert in_flight_states_for_pool("runs", state_to_pool) == ["rerunning", "running"]


def test_in_flight_states_for_pool_empty_when_unmapped() -> None:
    assert in_flight_states_for_pool("nonexistent", {"x": "y"}) == []


async def test_compute_pool_slots_single_pool() -> None:
    store = _FakeMetadataStore(counter=lambda kind, states: 2)
    pools = [ConcurrencyPool(name="agents", limit=10)]
    state_to_pool = {"implementing": "agents"}
    config = EngineConfig()
    slots = await compute_pool_slots(
        pools=pools,
        state_to_pool=state_to_pool,
        entity_kind="cg",
        metadata_store=store,  # type: ignore[arg-type]
        config=config,
    )
    assert len(slots) == 1
    assert slots[0] == PoolSlot(
        name="agents", limit=10, in_flight=2, available=8, priority=0
    )


async def test_compute_pool_slots_pool_overrides_apply() -> None:
    store = _FakeMetadataStore(counter=lambda kind, states: 1)
    pools = [ConcurrencyPool(name="agents", limit=10)]
    state_to_pool = {"implementing": "agents"}
    config = EngineConfig(pool_overrides={"agents": 4})
    slots = await compute_pool_slots(
        pools=pools,
        state_to_pool=state_to_pool,
        entity_kind="cg",
        metadata_store=store,  # type: ignore[arg-type]
        config=config,
    )
    assert slots[0].limit == 4  # override applied
    assert slots[0].available == 3  # 4 - 1


async def test_compute_pool_slots_priority_ordering() -> None:
    """Higher priority drains first; ties broken alphabetically by name."""
    store = _FakeMetadataStore()
    pools = [
        ConcurrencyPool(name="low", limit=2, priority=1),
        ConcurrencyPool(name="high", limit=2, priority=10),
        ConcurrencyPool(name="mid", limit=2, priority=5),
    ]
    state_to_pool = {"x": "low", "y": "high", "z": "mid"}
    config = EngineConfig()
    slots = await compute_pool_slots(
        pools=pools,
        state_to_pool=state_to_pool,
        entity_kind="cg",
        metadata_store=store,  # type: ignore[arg-type]
        config=config,
    )
    assert [s.name for s in slots] == ["high", "mid", "low"]


async def test_compute_pool_slots_priority_tiebreak_alphabetical() -> None:
    store = _FakeMetadataStore()
    pools = [
        ConcurrencyPool(name="zebra", limit=1, priority=5),
        ConcurrencyPool(name="alpha", limit=1, priority=5),
    ]
    state_to_pool = {"a": "zebra", "b": "alpha"}
    slots = await compute_pool_slots(
        pools=pools,
        state_to_pool=state_to_pool,
        entity_kind="cg",
        metadata_store=store,  # type: ignore[arg-type]
        config=EngineConfig(),
    )
    assert [s.name for s in slots] == ["alpha", "zebra"]


async def test_compute_pool_slots_skips_unmapped_pool() -> None:
    """A spec may declare a pool with no states feeding it (e.g., the
    four-pool default per DEC-034 #4 with one empty); the worker
    should still surface it with ``in_flight=0`` without invoking
    the plugin (saves a round-trip)."""
    store = _FakeMetadataStore(
        counter=lambda kind, states: (_ for _ in ()).throw(
            AssertionError("plugin called with empty in_flight_states")
        )
    )
    pools = [ConcurrencyPool(name="empty_pool", limit=5)]
    state_to_pool: dict[str, str] = {}
    slots = await compute_pool_slots(
        pools=pools,
        state_to_pool=state_to_pool,
        entity_kind="cg",
        metadata_store=store,  # type: ignore[arg-type]
        config=EngineConfig(),
    )
    assert slots[0].in_flight == 0
    assert slots[0].available == 5
    assert store.calls == []  # plugin never invoked


async def test_compute_pool_slots_clamps_negative() -> None:
    """If ``in_flight > limit`` (e.g., a deployment shrunk the pool
    while jobs were running), :attr:`available` clamps to 0 rather
    than going negative."""
    store = _FakeMetadataStore(counter=lambda kind, states: 100)
    pools = [ConcurrencyPool(name="agents", limit=10)]
    state_to_pool = {"implementing": "agents"}
    slots = await compute_pool_slots(
        pools=pools,
        state_to_pool=state_to_pool,
        entity_kind="cg",
        metadata_store=store,  # type: ignore[arg-type]
        config=EngineConfig(),
    )
    assert slots[0].available == 0


async def test_compute_pool_slots_four_pool_default_shape() -> None:
    """Smoke test for the DEC-034 #4 four-pool default per `05` §3.4.

    The full SMAI CG-execution spec wires this up in Task 2.C4; here
    we just confirm the shape is supported.
    """
    store = _FakeMetadataStore(counter=lambda kind, states: 0)
    pools = [
        ConcurrencyPool(name="agent", limit=4, priority=20),
        ConcurrencyPool(name="run", limit=8, priority=30),
        ConcurrencyPool(name="gate", limit=16, priority=40),
        ConcurrencyPool(name="inline", limit=32, priority=10),
    ]
    state_to_pool = {
        "implementing": "agent",
        "running": "run",
        "reviewing": "gate",
        "evaluating": "inline",
    }
    slots = await compute_pool_slots(
        pools=pools,
        state_to_pool=state_to_pool,
        entity_kind="cg",
        metadata_store=store,  # type: ignore[arg-type]
        config=EngineConfig(),
    )
    assert [s.name for s in slots] == ["gate", "run", "agent", "inline"]
    # All have full availability (no in-flight).
    assert all(s.available == s.limit for s in slots)


# ------------------------------------------------------------ helpers ---


def _coro_zero(_: str, __: list[str]) -> Awaitable[int]:
    async def _ret() -> int:
        return 0

    return _ret()


__all__: list[str] = []
