"""Per-task fakes for ``make_compute_dispatcher`` unit tests.

Per-task fixture filename hygiene per CLAUDE.md "Testing patterns":
``_<task>_<purpose>.py`` to avoid sys-path collisions with sibling
plugin fakes under ``--import-mode=importlib``.

The :class:`RecordingCompute` mock is intentionally separate from the
shared ``FakeCompute`` in ``tests/engine/_helpers.py`` because the
factory unit tests assert the call ORDER (stage → submit) and capture
the ``workspace=`` plugin_option threaded through ``submit``, which
the shared fake does not record.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from _helpers import FakeArtifactStore  # type: ignore[import-not-found]
from smai_core.plugins import (
    Compute,
    ComputeCapabilities,
    JobHandle,
    JobStatus,
    WorkspaceHandle,
)
from smai_orchestrator.dispatch import CommandSpec, WorkspaceInputs, WorkspaceOutputs
from smai_orchestrator.engine.config import EngineConfig
from smai_orchestrator.engine.types import DispatchContext
from smai_store_sqlite import SqliteStore


@dataclass
class _RecordedCall:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


class RecordingCompute:
    """:class:`Compute`-conforming mock that records all calls in order.

    The factory's unit tests assert that ``stage_workspace`` fires
    BEFORE ``submit`` and that ``submit`` receives the staged handle
    via ``plugin_options['workspace']`` — recording every method call
    onto :attr:`calls` lets the tests pin both shape and order.
    """

    name: str = "recording-compute"
    capabilities: ComputeCapabilities = ComputeCapabilities(
        supports_gpu=False,
        max_timeout_seconds=3600,
        workspace_distribution="bind_mount",
    )

    def __init__(
        self,
        *,
        submit_handle: JobHandle | None = None,
        submit_raises: BaseException | None = None,
        logs_text: str = "",
    ) -> None:
        self._submit_handle = submit_handle or JobHandle(plugin=self.name, handle="job-default")
        self._submit_raises = submit_raises
        self._logs_text = logs_text
        self.calls: list[_RecordedCall] = []

    async def submit(  # noqa: PLR0913
        self,
        image: str,
        command: list[str],
        env: dict[str, str],
        gpu: bool = False,
        timeout_seconds: int = 3600,
        **plugin_options: object,
    ) -> JobHandle:
        self.calls.append(
            _RecordedCall(
                kind="submit",
                payload={
                    "image": image,
                    "command": list(command),
                    "env": dict(env),
                    "gpu": gpu,
                    "timeout_seconds": timeout_seconds,
                    "plugin_options": dict(plugin_options),
                },
            )
        )
        if self._submit_raises is not None:
            raise self._submit_raises
        return self._submit_handle

    async def status(self, handle: JobHandle) -> JobStatus:
        self.calls.append(_RecordedCall(kind="status", payload={"handle": handle}))
        return JobStatus(
            state="succeeded",
            exit_code=0,
            started_at=None,
            finished_at=None,
            failure_reason=None,
        )

    async def logs(self, handle: JobHandle) -> str:
        self.calls.append(_RecordedCall(kind="logs", payload={"handle": handle}))
        return self._logs_text

    async def cancel(self, handle: JobHandle) -> None:
        self.calls.append(_RecordedCall(kind="cancel", payload={"handle": handle}))

    async def stage_workspace(self, local_path: Path) -> WorkspaceHandle:
        self.calls.append(_RecordedCall(kind="stage_workspace", payload={"local_path": local_path}))
        return WorkspaceHandle(plugin=self.name, handle=str(local_path))

    async def harvest_workspace(self, handle: WorkspaceHandle, local_path: Path) -> None:
        self.calls.append(
            _RecordedCall(
                kind="harvest_workspace",
                payload={"handle": handle, "local_path": local_path},
            )
        )


async def build_metadata_store() -> SqliteStore:
    """In-memory :class:`SqliteStore` for ``DispatchContext`` construction.

    The factory's static resolvers don't consult the store; this exists
    only so the :class:`DispatchContext` Pydantic validator accepts the
    payload (the Protocol instance-check requires a real plugin).
    """
    store = SqliteStore("sqlite+aiosqlite:///:memory:")
    await store.migrate()
    return store


async def make_dispatch_context(
    *,
    compute: Compute,
    entity_id: str = "run-test",
) -> DispatchContext:
    """Construct a minimal :class:`DispatchContext` for factory tests.

    The factory does not consult ``artifact_store`` / ``metadata_store``
    / ``llm`` directly — only ``compute``. The resolvers passed in by
    the factory caller may consult them; the seed-run resolvers in
    ``run_record.py`` do. Real :class:`SqliteStore` and
    :class:`FakeArtifactStore` are wired in so the :class:`DispatchContext`
    Pydantic validator (which runtime-instance-checks against the
    plugin Protocols) accepts the payload.
    """
    metadata_store = await build_metadata_store()
    return DispatchContext(
        entity_kind="run",
        entity_id=entity_id,
        entity_state="submitted",
        entity_version=0,
        metadata_store=metadata_store,
        artifact_store=FakeArtifactStore(),  # type: ignore[arg-type]
        compute=compute,  # type: ignore[arg-type]
        llm=None,
        config=EngineConfig(),
        checkpointer=None,
    )


def static_image_resolver(image: str) -> Callable[[DispatchContext], Awaitable[str]]:
    """Image resolver that always returns ``image``."""

    async def _resolve(ctx: DispatchContext) -> str:
        del ctx
        return image

    return _resolve


def static_command_builder(
    spec: CommandSpec,
) -> Callable[[DispatchContext], Awaitable[CommandSpec]]:
    """Command builder that always returns ``spec``."""

    async def _build(ctx: DispatchContext) -> CommandSpec:
        del ctx
        return spec

    return _build


def static_workspace_inputs(path: Path | None) -> WorkspaceInputs:
    """``WorkspaceInputs`` whose resolver always returns ``path``."""

    async def _resolve(ctx: DispatchContext) -> Path | None:
        del ctx
        return path

    return WorkspaceInputs(resolver=_resolve)


__all__ = [
    "RecordingCompute",
    "WorkspaceOutputs",
    "make_dispatch_context",
    "static_command_builder",
    "static_image_resolver",
    "static_workspace_inputs",
]
