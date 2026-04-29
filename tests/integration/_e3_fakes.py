"""Cross-package fixtures for Task 3.E3 integration tests.

Per the brief's "test-fixture filename hygiene" guidance: ``_e3_*`` so
the fixtures don't collide with other tasks' test trees on
:mod:`importlib`-mode discovery.
"""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from typing import Literal

from smai_core.plugins import (
    ArtifactNotFound,
    ArtifactStoreCapabilities,
    ComputeCapabilities,
    JobHandle,
    JobStatus,
)


class E3FakeArtifactStore:
    """In-memory :class:`ArtifactStore` for run-sub-spec integration tests."""

    name: str = "e3-fake-artifacts"
    capabilities: ArtifactStoreCapabilities = ArtifactStoreCapabilities(
        supports_presigned_urls=False,
        max_object_size_bytes=None,
    )

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}

    async def put(self, key: str, data: bytes, content_type: str | None = None) -> None:
        del content_type
        self._data[key] = data

    async def get(self, key: str) -> bytes:
        if key not in self._data:
            raise ArtifactNotFound(key)
        return self._data[key]

    async def exists(self, key: str) -> bool:
        return key in self._data

    async def list(self, prefix: str) -> AsyncIterator[str]:
        async def _gen() -> AsyncIterator[str]:
            for key in sorted(self._data):
                if key.startswith(prefix):
                    yield key

        return _gen()

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def url_for(
        self,
        key: str,
        expires_in: int = 3600,
        method: Literal["GET", "PUT"] = "GET",
    ) -> str:
        del expires_in, method
        return f"e3-fake://{key}"


class E3FakeCompute:
    """:class:`Compute` that hands out fresh handles and accepts a
    per-handle status map.

    The integration test pre-loads ``status_map`` with the desired
    terminal :class:`JobStatus` per handle (mirroring how the smoke
    test's ``SmokeFakeCompute`` always returns ``"succeeded"``, but
    per-handle so we can drive a mixed-terminal scenario).
    """

    name: str = "e3-fake-compute"
    capabilities: ComputeCapabilities = ComputeCapabilities(
        supports_gpu=True,
        max_timeout_seconds=3600,
        supports_log_streaming=False,
    )

    def __init__(
        self,
        *,
        terminal_states: Sequence[Literal["succeeded", "failed", "inconclusive"]] | None = None,
    ) -> None:
        # ``terminal_states`` is consumed FIFO per ``submit`` call to
        # decide which :class:`JobStatus` ``status`` returns for that
        # handle. ``"inconclusive"`` is an integration-test alias —
        # :class:`JobState` doesn't have an ``inconclusive`` literal,
        # so we fake it by returning ``"succeeded"`` and *omitting*
        # the metrics artifact (the run sub-spec then routes the run
        # to ``RunState.inconclusive`` per `06` §1).
        self._next_terminals: deque[Literal["succeeded", "failed", "inconclusive"]] = deque(
            terminal_states or []
        )
        self._handle_to_terminal: dict[str, Literal["succeeded", "failed", "inconclusive"]] = {}
        self._counter = 0
        self.submit_calls: list[dict[str, object]] = []
        self.cancel_calls: list[JobHandle] = []

    async def submit(
        self,
        image: str,
        command: list[str],
        env: dict[str, str],
        gpu: bool = False,
        timeout_seconds: int = 3600,
        **plugin_options: object,
    ) -> JobHandle:
        del timeout_seconds, plugin_options
        self._counter += 1
        handle = JobHandle(plugin=self.name, handle=f"e3-{self._counter}")
        # Default to ``succeeded`` if the queue is exhausted (most tests
        # run all-success).
        terminal: Literal["succeeded", "failed", "inconclusive"] = (
            self._next_terminals.popleft() if self._next_terminals else "succeeded"
        )
        self._handle_to_terminal[handle.handle] = terminal
        self.submit_calls.append(
            {
                "image": image,
                "command": list(command),
                "env": dict(env),
                "gpu": gpu,
                "handle": handle.handle,
                "planned_terminal": terminal,
            }
        )
        return handle

    async def status(self, handle: JobHandle) -> JobStatus:
        terminal = self._handle_to_terminal.get(handle.handle, "succeeded")
        now = datetime.now(UTC).isoformat()
        if terminal == "failed":
            return JobStatus(
                state="failed",
                exit_code=1,
                started_at=now,
                finished_at=now,
                failure_reason="e3-fake-compute planned failure",
            )
        # Both ``succeeded`` and the integration-test ``inconclusive``
        # alias return :class:`JobState` ``succeeded``; the run sub-spec
        # distinguishes via metrics-artifact presence.
        return JobStatus(
            state="succeeded",
            exit_code=0,
            started_at=now,
            finished_at=now,
            failure_reason=None,
        )

    async def cancel(self, handle: JobHandle) -> None:
        self.cancel_calls.append(handle)

    async def logs(self, handle: JobHandle) -> str:
        del handle
        return ""

    def planned_terminal_for(self, handle_name: str) -> str | None:
        """Lookup the terminal we promised for a given handle.

        Useful for tests that need to know which (entry, seed) tuples
        should have metrics artifacts staged.
        """
        return self._handle_to_terminal.get(handle_name)


__all__ = [
    "E3FakeArtifactStore",
    "E3FakeCompute",
]
