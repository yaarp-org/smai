"""Helper classes / builders shared across engine tests.

Imported directly by test modules (``from tests.engine._helpers import
...`` doesn't work because pytest's importmode/rootdir don't add the
test package to ``sys.path``; we use ``importlib``-mode collection but
import these helpers via a relative import). The fixtures themselves
live in :mod:`conftest.py` (auto-discovered by pytest).
"""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

from smai_core.plugins import (
    ArtifactStoreCapabilities,
    ComputeCapabilities,
    JobHandle,
    JobStatus,
)

# === FakeClock for wall-clock tests ==========================================


class FakeClock:
    """Datetime-based fake-clock for ``EngineConfig.wall_clock`` tests."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now: datetime = start or datetime(2026, 4, 28, 12, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now = self._now + timedelta(seconds=seconds)


class FakeMonotonic:
    """Float-based fake-clock for ``EngineConfig.time_provider`` tests."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds


# === FakeCompute =============================================================


@dataclass
class _PlannedSubmit:
    handle: JobHandle | None = None
    raises: BaseException | None = None


@dataclass
class _PlannedStatus:
    status: JobStatus | None = None
    raises: BaseException | None = None


class FakeCompute:
    """Protocol-conforming :class:`Compute` stub with deque-based
    outcome injection (mirrors ``FakeBedrockClient`` from Task 2.A1).
    """

    name: str = "fake-compute"
    capabilities: ComputeCapabilities = ComputeCapabilities(
        supports_gpu=False,
        max_timeout_seconds=3600,
    )

    def __init__(self) -> None:
        self._submits: deque[_PlannedSubmit] = deque()
        self._statuses: dict[str, deque[_PlannedStatus]] = {}
        self._sticky: dict[str, JobStatus] = {}
        self.submit_calls: list[dict[str, object]] = []
        self.cancel_calls: list[JobHandle] = []

    def enqueue_submit_handle(self, handle: JobHandle) -> None:
        self._submits.append(_PlannedSubmit(handle=handle))

    def enqueue_submit_raise(self, exc: BaseException) -> None:
        self._submits.append(_PlannedSubmit(raises=exc))

    def enqueue_status(self, handle_id: str, status: JobStatus) -> None:
        self._statuses.setdefault(handle_id, deque()).append(_PlannedStatus(status=status))

    def set_status(self, handle_id: str, status: JobStatus) -> None:
        self._sticky[handle_id] = status

    async def submit(  # noqa: PLR0913
        self,
        image: str,
        command: list[str],
        env: dict[str, str],
        gpu: bool = False,
        timeout_seconds: int = 3600,
        **plugin_options: object,
    ) -> JobHandle:
        self.submit_calls.append(
            {
                "image": image,
                "command": list(command),
                "env": dict(env),
                "gpu": gpu,
                "timeout_seconds": timeout_seconds,
            }
        )
        if not self._submits:
            raise RuntimeError("FakeCompute.submit: no outcome enqueued")
        outcome = self._submits.popleft()
        if outcome.raises is not None:
            raise outcome.raises
        assert outcome.handle is not None
        return outcome.handle

    async def status(self, handle: JobHandle) -> JobStatus:
        queue = self._statuses.get(handle.handle)
        if queue:
            planned = queue.popleft()
            if planned.raises is not None:
                raise planned.raises
            assert planned.status is not None
            return planned.status
        if handle.handle in self._sticky:
            return self._sticky[handle.handle]
        raise RuntimeError(f"FakeCompute.status: no outcome enqueued for handle {handle.handle!r}")

    async def logs(self, handle: JobHandle) -> str:
        return f"<no logs configured for {handle.handle!r}>"

    async def cancel(self, handle: JobHandle) -> None:
        self.cancel_calls.append(handle)


# === FakeArtifactStore =======================================================


class FakeArtifactStore:
    """Minimal in-memory :class:`ArtifactStore` stub for engine tests."""

    name: str = "fake-artifacts"
    capabilities: ArtifactStoreCapabilities = ArtifactStoreCapabilities(
        supports_presigned_urls=False,
        max_object_size_bytes=None,
    )

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    async def put(self, key: str, data: bytes, content_type: str | None = None) -> None:
        self._store[key] = data

    async def get(self, key: str) -> bytes:
        if key not in self._store:
            raise KeyError(f"FakeArtifactStore: no key {key!r}")
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
        return f"fake://{key}"


# === Builders ================================================================


def make_gate(
    *,
    advance: bool,
    reason: str | None = None,
    on_call: Callable[[], None] | None = None,
):
    from smai_orchestrator.engine import GateContext, GateOutcome

    async def _gate(ctx: GateContext) -> GateOutcome:
        if on_call is not None:
            on_call()
        return GateOutcome(advance=advance, reason=reason)

    return _gate


def make_dispatch(
    *,
    handle: JobHandle | None = None,
    error: str | None = None,
    on_call: Callable[[], None] | None = None,
    raises: BaseException | None = None,
):
    from smai_orchestrator.engine import DispatchContext, DispatchOutcome

    async def _handler(ctx: DispatchContext) -> DispatchOutcome:
        if on_call is not None:
            on_call()
        if raises is not None:
            raise raises
        outcome = DispatchOutcome()
        if handle is not None:
            outcome.submitted_handles.append(handle)
        if error is not None:
            outcome.error = error
        return outcome

    return _handler


def make_job_handle(handle: str = "job-1", plugin: str = "fake-compute") -> JobHandle:
    return JobHandle(plugin=plugin, handle=handle)


def make_job_status(
    state: Literal["submitted", "running", "succeeded", "failed", "cancelled", "timeout"],
    *,
    exit_code: int | None = None,
    failure_reason: str | None = None,
) -> JobStatus:
    return JobStatus(
        state=state,
        exit_code=exit_code,
        started_at=None,
        finished_at=None,
        failure_reason=failure_reason,
    )


__all__ = [
    "FakeArtifactStore",
    "FakeClock",
    "FakeCompute",
    "FakeMonotonic",
    "make_dispatch",
    "make_gate",
    "make_job_handle",
    "make_job_status",
]
