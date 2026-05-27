"""Per-task fixtures for the sub-PR B sandboxed harness-builder dispatch tests.

Per the SMAI test convention (``tests/_<task>_<purpose>.py``); the
per-task prefix keeps the module isolated from sibling-plugin fakes
under ``--import-mode=importlib``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smai_core.artifacts._envelope import ArtifactEnvelope
from smai_core.artifacts.harness_contract import HarnessContract, HarnessContractBody
from smai_core.artifacts.technique_contract import TechniqueContract, TechniqueContractBody
from smai_core.entities.factor import Factor
from smai_core.plugins import (
    ComputeCapabilities,
    JobHandle,
    JobStatus,
    LlmCapabilities,
    NormalizedMessage,
    WorkspaceHandle,
)


@dataclass
class _ComputeCall:
    """Record of one call against the recording fake Compute."""

    kind: str
    payload: dict[str, Any]


class RecordingCompute:
    """A minimal :class:`Compute` stand-in that records every call.

    Used by sub-PR B unit tests asserting :func:`make_dispatch_harness_build_sandboxed`
    threads the expected arg shape into the underlying compute. Mirrors
    the orchestrator-side ``_compute_dispatcher_fakes.RecordingCompute``
    pattern (per-task fixture-filename hygiene; cannot reuse cross-package).
    """

    name: str = "recording-compute"

    def __init__(
        self,
        *,
        submit_handle: JobHandle | None = None,
    ) -> None:
        self.calls: list[_ComputeCall] = []
        self._submit_handle = submit_handle or JobHandle(plugin=self.name, handle="job-fixture")

    @property
    def capabilities(self) -> ComputeCapabilities:
        return ComputeCapabilities(
            supports_gpu=False,
            max_timeout_seconds=3600,
            workspace_distribution="bind_mount",
        )

    async def stage_workspace(self, local_path: Path) -> WorkspaceHandle:
        self.calls.append(_ComputeCall(kind="stage_workspace", payload={"local_path": local_path}))
        return WorkspaceHandle(plugin=self.name, handle=str(local_path.resolve()))

    async def submit(  # noqa: PLR0913
        self,
        image: str,
        command: list[str],
        env: dict[str, str],
        gpu: bool = False,
        timeout_seconds: int = 3600,
        **plugin_options: Any,
    ) -> JobHandle:
        self.calls.append(
            _ComputeCall(
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
        return self._submit_handle

    async def status(self, handle: JobHandle) -> JobStatus:
        self.calls.append(_ComputeCall(kind="status", payload={"handle": handle}))
        return JobStatus(
            state="succeeded",
            exit_code=0,
            started_at=None,
            finished_at=None,
            failure_reason=None,
        )

    async def logs(self, handle: JobHandle) -> str:
        self.calls.append(_ComputeCall(kind="logs", payload={"handle": handle}))
        return ""

    async def cancel(self, handle: JobHandle) -> None:
        self.calls.append(_ComputeCall(kind="cancel", payload={"handle": handle}))

    async def harvest_workspace(self, handle: WorkspaceHandle, local_path: Path) -> None:
        self.calls.append(
            _ComputeCall(
                kind="harvest_workspace",
                payload={"handle": handle, "local_path": local_path},
            )
        )


class _RecordingArtifactStore:
    """In-memory :class:`ArtifactStore` for sub-PR B tests."""

    name: str = "recording-store"

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}

    async def put(
        self,
        key: str,
        body: bytes,
        *,
        content_type: str | None = None,
    ) -> None:
        del content_type
        self._data[key] = body

    async def get(self, key: str) -> bytes:
        if key not in self._data:
            from smai_core.plugins import ArtifactNotFound  # noqa: PLC0415

            raise ArtifactNotFound(key)
        return self._data[key]

    async def exists(self, key: str) -> bool:
        return key in self._data

    async def list(self, prefix: str) -> AsyncIterator[str]:
        # Match the Protocol's async-iterator return shape (the real
        # plugin implementations stream). Pre-realize the keys, then
        # return an async generator that yields them.
        matching = sorted(k for k in self._data if k.startswith(prefix))

        async def _iter() -> AsyncIterator[str]:
            for key in matching:
                yield key

        return _iter()

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def all_keys(self) -> list[str]:
        return sorted(self._data)


@dataclass
class _StubEntry:
    id: str
    is_baseline: bool


@dataclass
class _StubPage:
    items: list[Any]
    next_cursor: str | None = None


@dataclass
class _StubMetadataStore:
    """Minimal :class:`MetadataStore` covering what sub-PR B's
    sandboxed dispatcher touches (list_entries_for_cg +
    create_agent_session)."""

    baseline_entry_id: str = "entry-baseline"
    sessions: list[dict[str, Any]] = field(default_factory=list)

    async def list_entries_for_cg(
        self, cg_id: str, *, limit: int = 100, cursor: str | None = None
    ) -> _StubPage:
        del cg_id, limit, cursor
        return _StubPage(items=[_StubEntry(id=self.baseline_entry_id, is_baseline=True)])

    async def create_agent_session(self, **kwargs: Any) -> str:
        self.sessions.append(kwargs)
        return f"as-{len(self.sessions):08d}"


class _StubLlm:
    """:class:`LlmProvider` Protocol stand-in carrying name + capabilities
    so ``open_agent_session`` can read the provider / model identifiers.

    Sub-PR F (2026-05-26): expanded from the original single-attribute
    stub to a fuller Protocol implementor so pyright accepts it where
    ``LlmProvider`` is the declared type. ``call`` and
    ``credentials_for_subprocess`` raise / return empty stubs since the
    sandboxed-dispatch tests never exercise them on this fake.
    """

    name: str = "stub-llm"
    capabilities: LlmCapabilities = LlmCapabilities(
        model_id="stub-model",
        supports_caching=False,
        context_window=200_000,
        max_output_tokens=4096,
    )

    async def call(
        self,
        system: str,
        messages: list[NormalizedMessage],
        tools: object = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        cache_config: object = None,
    ) -> Any:
        del system, messages, tools, max_tokens, temperature, cache_config
        raise NotImplementedError("stub")

    async def credentials_for_subprocess(self) -> dict[str, str]:
        # Sub-PR F: second Protocol method.
        return {}

    async def stream(
        self,
        messages: Sequence[NormalizedMessage],
        **kwargs: Any,
    ) -> Any:
        del messages, kwargs
        raise NotImplementedError("stub")


@dataclass
class _StubEngineConfig:
    """Engine config stand-in. The sandboxed dispatcher doesn't actually
    read engine-config fields (they're a sub-PR C / sub-PR D concern);
    the field exists for DispatchContext shape parity."""

    pass


def make_contract() -> HarnessContract:
    """Minimal valid :class:`HarnessContract` for sandboxed dispatch tests."""
    envelope = ArtifactEnvelope(
        artifact_kind="harness_contract",
        artifact_id="hc-test",
        schema_version=1,
        compiler_version="0.0.0-test",
        content_hash="deadbeef" * 8,
        parent_experiment_id="exp-test",
    )
    body = HarnessContractBody(
        parent_experiment_hash="cafebabe" * 8,
        factor=Factor(name="test_factor", type="substitutive", description="test"),
        seeds=[42],
        fixed_variables=[],
        required_metrics=[],
        optional_telemetry=[],
        no_go_zones=[],
    )
    return HarnessContract(envelope=envelope, body=body)


def make_technique_contract() -> TechniqueContract:
    """Minimal valid baseline :class:`TechniqueContract`."""
    envelope = ArtifactEnvelope(
        artifact_kind="technique_contract",
        artifact_id="tc-test",
        schema_version=1,
        compiler_version="0.0.0-test",
        content_hash="abadcafe" * 8,
        parent_experiment_id="exp-test",
    )
    body = TechniqueContractBody(
        entry_id="entry-baseline",
        parent_experiment_id="exp-test",
        parent_experiment_hash="cafebabe" * 8,
        parent_harness_contract_hash="deadbeef" * 8,
        technique_id="baseline",
        is_baseline=True,
        level_value=None,
        technique_params=None,
        fidelity_anchor=None,
        standard=False,
    )
    return TechniqueContract(envelope=envelope, body=body)


@dataclass
class _StubDispatchContext:
    """:class:`DispatchContext`-shaped stand-in for sub-PR B tests."""

    entity_kind: str
    entity_id: str
    entity_state: str
    entity_version: int
    metadata_store: Any
    artifact_store: Any
    compute: Any
    llm: Any
    config: Any
    checkpointer: Any = None


__all__ = [
    "RecordingCompute",
    "_RecordingArtifactStore",
    "_StubDispatchContext",
    "_StubEngineConfig",
    "_StubLlm",
    "_StubMetadataStore",
    "make_contract",
    "make_technique_contract",
]
