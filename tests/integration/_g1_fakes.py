"""Cross-package fakes for Task 3.G1's multi-worker leasing tests.

Per the workspace's ``--import-mode=importlib`` rule + the integration
test tree's ``conftest.py``-driven ``sys.path`` mount: this module is
imported as ``_g1_fakes`` from ``test_multi_worker_leasing.py``. The
``_g1_`` prefix keeps it from colliding with sibling fixture modules
(``_e2_*`` / ``_e3_*``) under a shared ``sys.path``.

Provides a minimal :class:`Compute` / :class:`ArtifactStore` pair plus a
synthetic spec builder so the lease-correctness invariant can be
asserted against the engine's worker loop without the full plugin
matrix the smoke test wires up.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Literal

from smai_core.plugins import (
    ArtifactStoreCapabilities,
    ComputeCapabilities,
    JobHandle,
    JobStatus,
)
from smai_orchestrator.engine import (
    ConcurrencyPool,
    DispatchAction,
    DispatchContext,
    DispatchOutcome,
    EdgeDef,
    EngineSpec,
    GateContext,
    GateOutcome,
    StateDef,
)
from smai_orchestrator.engine.types import SchedulingQueryRef
from smai_orchestrator.entities.tracking import ComparisonGroupRecord


class FakeCompute:
    """Minimal :class:`Compute`-shaped stub.

    The synthetic spec's dispatch handler reports an inline outcome
    rather than submitting an external job, but the engine still
    requires a :class:`Compute` reference on :class:`DispatchContext` —
    this stub satisfies the Protocol surface without any real work.
    """

    name: str = "g1-fake-compute"
    capabilities: ComputeCapabilities = ComputeCapabilities(
        supports_gpu=False,
        max_timeout_seconds=3600,
    )

    async def submit(
        self,
        image: str,
        command: list[str],
        env: dict[str, str],
        gpu: bool = False,
        timeout_seconds: int = 3600,
        **plugin_options: object,
    ) -> JobHandle:
        del image, command, env, gpu, timeout_seconds, plugin_options
        raise AssertionError("g1 spec should never submit external jobs")

    async def status(self, handle: JobHandle) -> JobStatus:
        del handle
        raise AssertionError("g1 spec should never poll status")

    async def cancel(self, handle: JobHandle) -> None:
        del handle

    async def logs(self, handle: JobHandle) -> str:
        del handle
        return ""


class FakeArtifactStore:
    """Minimal in-memory :class:`ArtifactStore`-shaped stub."""

    name: str = "g1-fake-artifacts"
    capabilities: ArtifactStoreCapabilities = ArtifactStoreCapabilities(
        supports_presigned_urls=False,
        max_object_size_bytes=None,
    )

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    async def put(self, key: str, data: bytes, content_type: str | None = None) -> None:
        del content_type
        self._store[key] = data

    async def get(self, key: str) -> bytes:
        return self._store[key]

    async def exists(self, key: str) -> bool:
        return key in self._store

    async def list(self, prefix: str) -> AsyncIterator[str]:
        async def _gen() -> AsyncIterator[str]:
            for k in sorted(self._store):
                if k.startswith(prefix):
                    yield k

        return _gen()

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def url_for(
        self,
        key: str,
        expires_in: int = 3600,
        method: Literal["GET", "PUT"] = "GET",
    ) -> str:
        del expires_in, method
        return f"fake://{key}"


class CallTracker:
    """Records every dispatch-handler invocation across workers.

    The lease-correctness invariant is "no entity sees its dispatch
    handler fire twice"; ``calls`` enumerates per-entity firings so the
    test can assert exactly-once.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []  # (worker_id, entity_id)
        self._lock = asyncio.Lock()

    async def record(self, worker_id: str, entity_id: str, *, hold_seconds: float = 0.0) -> None:
        async with self._lock:
            self.calls.append((worker_id, entity_id))
        if hold_seconds > 0:
            # Holding the slot inside the lease window forces concurrent
            # workers' acquire_lease calls to actually meet contention.
            await asyncio.sleep(hold_seconds)


def make_g1_spec(tracker: CallTracker, *, hold_seconds: float = 0.0) -> EngineSpec:
    """Two-state synthetic spec: ``draft → implemented`` (terminal).

    The on-entry-dispatch handler is *inline* (``handle_field=None``)
    so the engine completes in one CAS state-transition + handler call;
    no external compute, no phase-1 polling. Each handler call is
    recorded into ``tracker``; ``hold_seconds`` lets the test choose
    how long the handler keeps its lease busy (used to force the peer
    worker's ``acquire_lease`` to observe contention).
    """

    async def _gate(_ctx: GateContext) -> GateOutcome:
        return GateOutcome(advance=True)

    async def _phase2_query(metadata_store) -> list[ComparisonGroupRecord]:
        # ``get_ready_for_harness_build`` is the canonical
        # :class:`SqliteStore` query for "CGs in ``draft`` with no
        # harness_job_handle" (`03` §3.5 / `07` §5.6.3). Our synthetic
        # spec's CGs are freshly created — handle is null — so this
        # returns the ``draft`` set we seed. Single-page is sufficient
        # for the contention fixture (< 50 entities).
        page = await metadata_store.get_ready_for_harness_build(limit=100, cursor=None)
        return list(page.items)

    async def _handler(ctx: DispatchContext) -> DispatchOutcome:
        # ``DispatchContext`` doesn't carry the worker_id; we reconstruct
        # it from the lease_holder_id by reading the CG row. The simpler
        # approach is to bind worker_id via closure when constructing
        # the spec — but the spec is shared across workers in this
        # test. Instead we rely on the entity's leased_by field (set
        # by acquire_lease just before this handler fires).
        cg = await ctx.metadata_store.get_cg(ctx.entity_id)
        assert cg is not None
        leased_by = cg.leased_by or "unknown"
        await tracker.record(leased_by, ctx.entity_id, hold_seconds=hold_seconds)
        return DispatchOutcome()

    action = DispatchAction(
        name="g1_inline_dispatch",
        handler=_handler,
        pool="agents",
        handle_field=None,  # inline — no external compute
    )

    return EngineSpec(
        entity_kind="cg",
        initial_state="draft",
        states=[
            StateDef(name="draft", on_entry_dispatch=None),
            StateDef(name="implemented", on_entry_dispatch=action, is_terminal=True),
        ],
        edges=[
            EdgeDef(
                name="advance",
                from_state="draft",
                target_state="implemented",
                gate_rule=_gate,
            ),
        ],
        # Pool with generous slot budget so the worker's pool-slot
        # accounting doesn't skip the dispatch (per `05` §3.4 / Task
        # 2.C2's slot-accounting logic in `_drive_phase3_for_records`).
        pools=[ConcurrencyPool(name="agents", limit=100)],
        phase2_queries={
            "draft": SchedulingQueryRef(name="g1_get_draft", fn=_phase2_query),
        },
    )


def make_seed_cg(cg_id: str) -> ComparisonGroupRecord:
    """Helper: construct a minimal CG record in ``draft``."""
    now = datetime.now(UTC)
    return ComparisonGroupRecord(
        id=cg_id,
        proposal_id=f"prop_{cg_id}",
        experiment_definition_id=f"exp_{cg_id}",
        state="draft",
        version=0,
        created_at=now,
        updated_at=now,
    )
