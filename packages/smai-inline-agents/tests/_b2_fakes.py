"""Fakes for 2.B2 tool tests.

Per the 2.B1 carry-forward: workspace ``pytest --import-mode=importlib``
puts every package's tests/ on sys.path, so unique module names matter.
``_b2_fakes`` avoids colliding with existing ``_agent_fakes`` (smai-agents
B1) / ``_fakes`` (bedrock plugin) / ``_helpers`` (orchestrator).
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from smai_core.plugins import (
    ComputeCapabilities,
    ComputeError,
    JobHandle,
    JobImageInvalid,
    JobStatus,
    WorkspaceHandle,
)


class FakeCompute:
    """Deterministic :class:`Compute` fake for tool tests.

    The ``submit`` / ``status`` / ``logs`` / ``cancel`` methods adhere
    to the runtime_checkable :class:`Compute` Protocol shape. Tests
    drive behavior by:

    * Pre-loading ``status_queue`` with :class:`JobStatus` values that
      ``status`` returns one-by-one (the last value sticks once the
      queue is drained, so a happy-path test seeds one ``running`` then
      one ``succeeded`` and the wait loop returns the latter).
    * Setting ``submit_should_raise`` to a :class:`ComputeError`
      subclass to force the submission to fail.
    * Optionally writing files into the workspace via the
      ``on_submit`` hook — used by the run_experiment harvest test
      to drop a fake ``metrics.json`` after submission "completes."
    """

    def __init__(
        self,
        *,
        status_queue: list[JobStatus] | None = None,
        submit_should_raise: ComputeError | None = None,
        on_submit: Callable[[Path], Awaitable[None]] | None = None,
        capabilities: ComputeCapabilities | None = None,
    ) -> None:
        self.name = "fake-compute"
        self.capabilities = capabilities or ComputeCapabilities(
            supports_gpu=True,
            max_timeout_seconds=3600,
        )
        self._status_queue: deque[JobStatus] = deque(status_queue or [])
        self._last_status: JobStatus | None = None
        self.submit_should_raise = submit_should_raise
        self.on_submit = on_submit
        self.submit_calls: list[dict[str, Any]] = []
        self.cancel_calls: list[JobHandle] = []
        self._workspace_for_submit: Path | None = None

    def set_workspace(self, workspace: Path) -> None:
        """Test-only seam — give the fake the workspace path so
        ``on_submit`` callbacks can drop files into it."""
        self._workspace_for_submit = workspace

    async def submit(
        self,
        image: str,
        command: list[str],
        env: dict[str, str],
        gpu: bool = False,
        timeout_seconds: int = 3600,
        **plugin_options: object,
    ) -> JobHandle:
        if self.submit_should_raise is not None:
            raise self.submit_should_raise
        self.submit_calls.append(
            {
                "image": image,
                "command": list(command),
                "env": dict(env),
                "gpu": gpu,
                "timeout_seconds": timeout_seconds,
                "plugin_options": dict(plugin_options),
            }
        )
        if self.on_submit is not None and self._workspace_for_submit is not None:
            await self.on_submit(self._workspace_for_submit)
        return JobHandle(plugin=self.name, handle="fake-job-1", metadata={})

    async def status(self, handle: JobHandle) -> JobStatus:
        del handle
        if self._status_queue:
            self._last_status = self._status_queue.popleft()
            return self._last_status
        if self._last_status is not None:
            return self._last_status
        raise AssertionError("FakeCompute.status: no status configured")

    async def logs(self, handle: JobHandle) -> str:
        del handle
        return "fake logs"

    async def cancel(self, handle: JobHandle) -> None:
        self.cancel_calls.append(handle)

    async def stage_workspace(self, local_path: Path) -> WorkspaceHandle:
        return WorkspaceHandle(plugin=self.name, handle=str(local_path))

    async def harvest_workspace(self, handle: WorkspaceHandle, local_path: Path) -> None:
        del handle, local_path


class FakeClock:
    """Deterministic monotonic clock for tool tests.

    Tests increment the clock manually (``clock.tick(s)``) and pass
    ``clock.now`` as the ``time_provider`` seam — the wait helper
    advances the clock by the poll interval so a ``max_wait_seconds``
    threshold deterministically trips on the right poll.
    """

    def __init__(self, *, start: float = 0.0) -> None:
        self._t = start

    def now(self) -> float:
        return self._t

    def tick(self, seconds: float) -> None:
        self._t += seconds


async def _no_sleep(seconds: float) -> None:
    """``sleep`` seam that never blocks — tests pass ``time.monotonic``-based
    progress through ``time_provider`` instead."""
    del seconds


# Re-export the no-sleep coroutine factory so tests reach for it under a
# stable name.
def make_no_sleep() -> Callable[[float], Awaitable[None]]:
    return _no_sleep


def make_clock_advancing_sleep(
    clock: FakeClock,
) -> Callable[[float], Awaitable[None]]:
    """``sleep`` seam that advances the fake clock instead of waiting."""

    async def _sleep(seconds: float) -> None:
        clock.tick(seconds)

    return _sleep


def real_time_provider() -> Callable[[], float]:
    """Convenience: returns the real ``time.monotonic`` for tests that
    don't care about clock determinism."""
    return time.monotonic


__all__ = [
    "FakeClock",
    "FakeCompute",
    "JobImageInvalid",
    "make_clock_advancing_sleep",
    "make_no_sleep",
    "real_time_provider",
]
