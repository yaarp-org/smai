"""Shared fixture scaffolding for the Task 4.L1 ``smai ui`` integration tests.

Per the workspace's per-task fixture filename hygiene convention
(``tests/`` modules co-exist on ``sys.path`` once via
``--import-mode=importlib``), this module's ``_4_l1_*`` prefix is
unique workspace-wide.

Helpers shipped here:

* :func:`make_dev_runtime` — boots a real :class:`Runtime` against an
  in-memory :class:`SqliteStore` + :class:`LocalFsStore` + per-role
  stub LLM providers, with the worker loop disabled. Tests build the
  FastAPI app on top via :func:`smai_api.make_api_app` (mirroring
  what ``smai ui --no-worker`` would do at process scope).
* :func:`build_dev_smai_yaml` / :func:`build_postgres_smai_yaml` —
  string builders for the smoke / auto-detect tests; the yaml shapes
  exercise the layering pipeline and the auto-detect plugin-shape
  rule from ``12-ui-process.md`` §4.3 / §9.3.
"""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from smai_artifacts_localfs import LocalFsStore
from smai_cli.runtime import Runtime
from smai_core.plugins import (
    ArtifactNotFound,
    ArtifactStoreCapabilities,
    CacheConfig,
    ComputeCapabilities,
    JobHandle,
    JobStatus,
    LlmCapabilities,
    LlmProvider,
    ModelResponse,
    NormalizedMessage,
    TextContent,
    TokenUsage,
    ToolDefinition,
)
from smai_orchestrator import (
    DEFAULT_TASK_ROLES,
    PluginOverrides,
    PluginSelection,
    RuntimeConfig,
)
from smai_orchestrator.engine.config import EngineConfig


class _StubLlmProvider:
    """Deterministic :class:`LlmProvider` for the L1 tests."""

    def __init__(
        self,
        responses: Sequence[ModelResponse] | None = None,
        *,
        capabilities: LlmCapabilities | None = None,
    ) -> None:
        self.name = "stub-l1"
        self.capabilities = capabilities or LlmCapabilities(
            supports_caching=True,
            context_window=200_000,
            max_output_tokens=4_096,
            supports_tool_use=True,
            model_id="stub-l1:test",
        )
        self.model_id = "stub-l1:test"
        self._responses: deque[ModelResponse] = deque(responses or [])

    async def call(
        self,
        system: str,
        messages: list[NormalizedMessage],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        cache_config: CacheConfig | None = None,
    ) -> ModelResponse:
        del system, messages, tools, max_tokens, temperature, cache_config
        if self._responses:
            return self._responses.popleft()
        return ModelResponse(
            message=NormalizedMessage(role="assistant", content=[TextContent(text="stub")]),
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=1, output_tokens=1),
        )


class _FakeCompute:
    """No-op :class:`Compute` shim — the L1 tests never drive jobs."""

    name: str = "fake-l1"
    capabilities: ComputeCapabilities = ComputeCapabilities(
        supports_gpu=False,
        max_timeout_seconds=3600,
        supports_log_streaming=False,
    )

    def __init__(self) -> None:
        self._counter = 0

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
        self._counter += 1
        return JobHandle(plugin=self.name, handle=f"fake-l1-{self._counter}")

    async def status(self, handle: JobHandle) -> JobStatus:
        del handle
        return JobStatus(
            state="running",
            exit_code=None,
            started_at=None,
            finished_at=None,
            failure_reason=None,
        )

    async def cancel(self, handle: JobHandle) -> None:
        del handle

    async def logs(self, handle: JobHandle) -> str:
        del handle
        return ""


class _InMemoryArtifactStore:
    """In-memory :class:`ArtifactStore` for the smoke tests."""

    def __init__(self) -> None:
        self.name = "inmem-l1"
        self.capabilities = ArtifactStoreCapabilities(
            supports_presigned_urls=False,
            max_object_size_bytes=None,
        )
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
        keys = sorted(k for k in self._data if k.startswith(prefix))

        async def _gen() -> AsyncIterator[str]:
            for k in keys:
                yield k

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
        return f"inmem-l1://{key}"


def make_runtime_config() -> RuntimeConfig:
    """Dev-shaped :class:`RuntimeConfig` for the smoke tests."""
    return RuntimeConfig(
        engine=EngineConfig(poll_interval_seconds=10, supervisor_enabled=False),
        plugins=PluginSelection(
            llm_provider="bedrock",
            metadata_store="sqlite",
            artifact_store="localfs",
            compute="localgpu",
            metadata_store_config={"uri": "sqlite+aiosqlite:///:memory:"},
        ),
        pipelines=[
            "smai_cg_execution",
            "smai_cg_entries",
        ],
    )


@asynccontextmanager
async def make_dev_runtime(tmp_path: Path) -> AsyncIterator[Runtime]:
    """Boot a real :class:`Runtime` for the L1 smoke / bearer tests.

    Disables the worker loop (the L1 verb tests focus on the
    HTTP surface, not on agent dispatch). LLM providers are stubs;
    artifact store is the real :class:`LocalFsStore` so route
    handlers that read existing artifacts work; the metadata store
    is in-memory SQLite.
    """
    overrides = PluginOverrides(
        llm_providers={role: _StubLlmProvider() for role in DEFAULT_TASK_ROLES},
        artifact_store=LocalFsStore(tmp_path / "artifacts"),
        compute=_FakeCompute(),
    )
    config = make_runtime_config()
    async with Runtime.start_in_band(
        config,
        workspace_root=tmp_path / "workspaces",
        plugin_overrides=overrides,
        run_worker=False,
    ) as runtime:
        yield runtime


_DEV_YAML_TEMPLATE = """\
engine:
  poll_interval_seconds: 10
  worker_count: 1
plugins:
  llm_provider: bedrock
  metadata_store: sqlite
  artifact_store: localfs
  compute: localgpu
  llm_provider_config:
    region: us-east-1
    model_id: us.anthropic.claude-opus-4-6-v1
  metadata_store_config: {{ uri: "sqlite+aiosqlite:///:memory:" }}
  artifact_store_config: {{}}
  compute_config: {{}}
pipelines:
  - smai_cg_execution
  - smai_cg_entries
{api_block}
"""


_POSTGRES_YAML_TEMPLATE = """\
engine:
  poll_interval_seconds: 30
  worker_count: 1
plugins:
  llm_provider: bedrock
  metadata_store: postgres
  artifact_store: s3
  compute: localgpu
  llm_provider_config:
    region: us-east-1
    model_id: us.anthropic.claude-opus-4-6-v1
  metadata_store_config: {{ uri: "postgresql+asyncpg://smai:smai@localhost/test" }}
  artifact_store_config: {{ bucket: "test-bucket" }}
  compute_config: {{}}
pipelines:
  - smai_cg_execution
  - smai_cg_entries
{api_block}
"""


def build_dev_smai_yaml(*, api_block: str = "") -> str:
    """Render a dev-shaped (sqlite + localfs) smai.yaml string."""
    return _DEV_YAML_TEMPLATE.format(api_block=api_block)


def build_postgres_smai_yaml(*, api_block: str = "") -> str:
    """Render a postgres + s3 smai.yaml string (production-shape).

    Used by the auto-detect test to verify the rule flips off for
    any non-sqlite/non-localfs plugin combo.
    """
    return _POSTGRES_YAML_TEMPLATE.format(api_block=api_block)


__all__ = [
    "build_dev_smai_yaml",
    "build_postgres_smai_yaml",
    "make_dev_runtime",
    "make_runtime_config",
]


_ = LlmProvider  # silence unused-import for the LlmProvider Protocol re-export.
