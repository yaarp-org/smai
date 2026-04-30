"""Unit tests for ``smai_cli.verify`` helpers + ``smai verify`` CLI verb (Task 3.G3).

Exercises:

* :func:`verify_llm_provider` — pass on stub-OK, fail on raise.
* :func:`verify_metadata_store` — pass on stub-OK, fail on raise.
* :func:`verify_artifact_store` — pass on stub-OK, fail on raise.
* :func:`verify_compute` — pass on :class:`JobNotFound` (expected),
  fail on auth error.
* ``smai verify`` CLI verb against a stub PluginOverrides — exit 0 on
  all-pass; exit 1 on any-fail.
* ``smai verify`` refuses on incomplete plugin selection (defense-in-
  depth check; covered also by config-layering tests).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal

import pytest
from _cli_fakes import (  # type: ignore[import-not-found]
    FakeCompute,
    InMemoryArtifactStore,
    StubLlmProvider,
)
from smai_cli.verify import (
    verify_artifact_store,
    verify_compute,
    verify_llm_provider,
    verify_metadata_store,
)
from smai_core.plugins import (
    ArtifactStoreCapabilities,
    CacheConfig,
    JobHandle,
    JobStatus,
    LlmCapabilities,
    ModelResponse,
    NormalizedMessage,
    ToolDefinition,
)
from smai_core.plugins.compute import ComputeUnavailable, JobNotFound
from smai_core.plugins.metadata_store._capabilities import MetadataStoreCapabilities

# === Stubs scoped to this test module ========================================


class _RaisingLlmProvider(StubLlmProvider):
    """Raises on call — surfaces an LlmProvider FAIL."""

    def __init__(self, exc: Exception) -> None:
        super().__init__(responses=[])
        self._exc = exc

    async def call(  # type: ignore[override]
        self,
        system: str,
        messages: list[NormalizedMessage],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        cache_config: CacheConfig | None = None,
    ) -> ModelResponse:
        raise self._exc


class _OkMetadataStub:
    """:class:`MetadataStore` shape — only ``count_in_state`` is exercised
    by :func:`verify_metadata_store`. The rest of the Protocol is not
    invoked, so we return zero-shaped no-ops everywhere else; the
    structural shape comes from :class:`MetadataStoreCapabilities`.
    """

    name = "ok-meta-stub"
    capabilities = MetadataStoreCapabilities(is_tenant_aware=False, supports_transactions=True)

    async def count_in_state(self, entity_kind: str, states: list[str]) -> int:
        del entity_kind, states
        return 0


class _RaisingMetadataStub:
    name = "raising-meta-stub"
    capabilities = MetadataStoreCapabilities(is_tenant_aware=False, supports_transactions=True)

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def count_in_state(self, entity_kind: str, states: list[str]) -> int:
        del entity_kind, states
        raise self._exc


class _RaisingArtifactStore:
    name = "raising-artifact-stub"
    capabilities = ArtifactStoreCapabilities(
        supports_presigned_urls=False, max_object_size_bytes=None
    )

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def exists(self, key: str) -> bool:
        del key
        raise self._exc

    async def put(self, key: str, data: bytes, content_type: str | None = None) -> None:
        del key, data, content_type

    async def get(self, key: str) -> bytes:
        del key
        return b""

    async def list(self, prefix: str) -> AsyncIterator[str]:  # pragma: no cover
        del prefix

        async def _gen() -> AsyncIterator[str]:
            if False:
                yield ""

        return _gen()

    async def delete(self, key: str) -> None:
        del key

    async def url_for(
        self, key: str, expires_in: int = 3600, method: Literal["GET", "PUT"] = "GET"
    ) -> str:
        del key, expires_in, method
        return ""


class _JobNotFoundCompute(FakeCompute):
    """Compute whose ``status`` raises :class:`JobNotFound` (the expected
    success case for :func:`verify_compute`)."""

    async def status(self, handle: JobHandle) -> JobStatus:  # type: ignore[override]
        raise JobNotFound(handle)


class _AuthFailedCompute(FakeCompute):
    """Compute whose ``status`` raises a substrate-auth error — the
    failure case for :func:`verify_compute`."""

    async def status(self, handle: JobHandle) -> JobStatus:  # type: ignore[override]
        raise ComputeUnavailable("AWS Batch returned 403 Forbidden")


# === verify_llm_provider =====================================================


@pytest.mark.asyncio
async def test_verify_llm_provider_passes_on_clean_call() -> None:
    """A provider that returns cleanly produces ``ok=True``."""
    provider = StubLlmProvider(
        responses=None,  # uses fallback "stub response" path
        capabilities=LlmCapabilities(
            supports_caching=False,
            context_window=1000,
            max_output_tokens=10,
            supports_tool_use=False,
            model_id="stub-verify",
        ),
    )
    result = await verify_llm_provider(provider)
    assert result.ok is True
    assert "stub-cli" in result.reason or "responded" in result.reason
    assert result.latency_ms is not None and result.latency_ms >= 0


@pytest.mark.asyncio
async def test_verify_llm_provider_fails_on_raise() -> None:
    """An exception in ``call`` produces ``ok=False`` with the typed
    error class in the reason."""
    provider = _RaisingLlmProvider(RuntimeError("bedrock 403"))
    result = await verify_llm_provider(provider)
    assert result.ok is False
    assert "RuntimeError" in result.reason
    assert "bedrock 403" in result.reason


# === verify_metadata_store ===================================================


@pytest.mark.asyncio
async def test_verify_metadata_store_passes_on_clean_count() -> None:
    from typing import cast

    from smai_core.plugins import MetadataStore

    result = await verify_metadata_store(cast("MetadataStore", _OkMetadataStub()))
    assert result.ok is True
    assert "ok-meta-stub" in result.reason


@pytest.mark.asyncio
async def test_verify_metadata_store_fails_on_raise() -> None:
    from typing import cast

    from smai_core.plugins import MetadataStore

    result = await verify_metadata_store(
        cast("MetadataStore", _RaisingMetadataStub(RuntimeError("connection refused")))
    )
    assert result.ok is False
    assert "RuntimeError" in result.reason
    assert "connection refused" in result.reason


# === verify_artifact_store ===================================================


@pytest.mark.asyncio
async def test_verify_artifact_store_passes_on_clean_exists() -> None:
    result = await verify_artifact_store(InMemoryArtifactStore())
    assert result.ok is True
    assert "inmem-cli-artifacts" in result.reason


@pytest.mark.asyncio
async def test_verify_artifact_store_fails_on_raise() -> None:
    from typing import cast

    from smai_core.plugins import ArtifactStore

    result = await verify_artifact_store(
        cast("ArtifactStore", _RaisingArtifactStore(PermissionError("S3 AccessDenied")))
    )
    assert result.ok is False
    assert "PermissionError" in result.reason
    assert "AccessDenied" in result.reason


# === verify_compute ==========================================================


@pytest.mark.asyncio
async def test_verify_compute_passes_on_job_not_found() -> None:
    """The expected success case: probing a non-existent handle raises
    :class:`JobNotFound`, which the verifier treats as PASS."""
    result = await verify_compute(_JobNotFoundCompute())
    assert result.ok is True
    assert "JobNotFound" in result.reason


@pytest.mark.asyncio
async def test_verify_compute_fails_on_substrate_auth_error() -> None:
    """:class:`ComputeUnavailable` (the substrate is unreachable / auth
    failure) is the canonical fail case."""
    result = await verify_compute(_AuthFailedCompute())
    assert result.ok is False
    assert "ComputeUnavailable" in result.reason


@pytest.mark.asyncio
async def test_verify_compute_passes_on_synthetic_status() -> None:
    """Plugins that return a synthetic :class:`JobStatus` rather than
    raising :class:`JobNotFound` (e.g., :class:`LocalGpuCompute`'s
    no-PID-found branch returning ``state="failed"``) also count as
    PASS — the substrate accepted the call without an error."""
    result = await verify_compute(FakeCompute())
    assert result.ok is True
    # FakeCompute returns a "running" JobStatus stub — surface it as
    # PASS regardless of state.
    assert "responded" in result.reason


# === smai verify CLI verb ====================================================


def test_smai_verify_cli_text_output_all_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """``smai verify`` reports PASS for all four plugins when the stub
    overrides return clean."""
    from smai_cli.main import app

    # Inject overrides into instantiate_plugins so the verb uses our
    # stubs instead of discovering real plugins. The CLI verb calls
    # ``instantiate_plugins(runtime_config.plugins)`` (no overrides);
    # we patch it to return our pre-built fakes.
    from smai_orchestrator import (
        InstantiatedPlugins,
    )
    from typer.testing import CliRunner

    class _StubInstantiate:
        def __init__(self, plugins: InstantiatedPlugins) -> None:
            self._plugins = plugins

        async def __aenter__(self) -> InstantiatedPlugins:
            return self._plugins

        async def __aexit__(self, *_: object) -> None:
            del _

    plugins = InstantiatedPlugins(
        llm_providers={"planner": StubLlmProvider()},
        metadata_store=_OkMetadataStub(),  # type: ignore[arg-type]
        artifact_store=InMemoryArtifactStore(),  # type: ignore[arg-type]
        compute=_JobNotFoundCompute(),
    )

    def _fake_instantiate_plugins(_selection: object) -> _StubInstantiate:
        return _StubInstantiate(plugins)

    monkeypatch.setattr(
        "smai_orchestrator.instantiate_plugins",
        _fake_instantiate_plugins,
    )

    # Write a minimal smai.yaml so `load_runtime_config` succeeds.
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_dir:
        smai_yaml = Path(tmp_dir) / "smai.yaml"
        smai_yaml.write_text(
            """
engine: {}
plugins:
  llm_provider: bedrock
  metadata_store: sqlite
  artifact_store: localfs
  compute: localgpu
pipelines: ["smai_cg_execution", "smai_cg_entries"]
""",
            encoding="utf-8",
        )
        runner = CliRunner()
        result = runner.invoke(app, ["verify", "-c", str(smai_yaml)])
    assert result.exit_code == 0, result.output
    assert "PASS llm_provider" in result.output
    assert "PASS metadata_store" in result.output
    assert "PASS artifact_store" in result.output
    assert "PASS compute" in result.output


def test_smai_verify_cli_exits_one_on_any_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """``smai verify`` exits 1 if any plugin's probe fails."""
    from smai_cli.main import app
    from smai_orchestrator import InstantiatedPlugins
    from typer.testing import CliRunner

    class _StubInstantiate:
        def __init__(self, plugins: InstantiatedPlugins) -> None:
            self._plugins = plugins

        async def __aenter__(self) -> InstantiatedPlugins:
            return self._plugins

        async def __aexit__(self, *_: object) -> None:
            del _

    plugins = InstantiatedPlugins(
        llm_providers={"planner": StubLlmProvider()},
        metadata_store=_OkMetadataStub(),  # type: ignore[arg-type]
        artifact_store=InMemoryArtifactStore(),  # type: ignore[arg-type]
        compute=_AuthFailedCompute(),  # the failing plugin
    )

    def _fake_instantiate_plugins(_selection: object) -> _StubInstantiate:
        return _StubInstantiate(plugins)

    monkeypatch.setattr(
        "smai_orchestrator.instantiate_plugins",
        _fake_instantiate_plugins,
    )

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_dir:
        smai_yaml = Path(tmp_dir) / "smai.yaml"
        smai_yaml.write_text(
            """
engine: {}
plugins:
  llm_provider: bedrock
  metadata_store: sqlite
  artifact_store: localfs
  compute: localgpu
pipelines: ["smai_cg_execution", "smai_cg_entries"]
""",
            encoding="utf-8",
        )
        runner = CliRunner()
        result = runner.invoke(app, ["verify", "-c", str(smai_yaml)])
    assert result.exit_code == 1, result.output
    assert "FAIL compute" in result.output
    assert "ComputeUnavailable" in result.output


def test_smai_verify_cli_json_format(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--format=json`` emits a parseable JSON document with one entry
    per plugin interface."""
    import json

    from smai_cli.main import app
    from smai_orchestrator import InstantiatedPlugins
    from typer.testing import CliRunner

    class _StubInstantiate:
        def __init__(self, plugins: InstantiatedPlugins) -> None:
            self._plugins = plugins

        async def __aenter__(self) -> InstantiatedPlugins:
            return self._plugins

        async def __aexit__(self, *_: object) -> None:
            del _

    plugins = InstantiatedPlugins(
        llm_providers={"planner": StubLlmProvider()},
        metadata_store=_OkMetadataStub(),  # type: ignore[arg-type]
        artifact_store=InMemoryArtifactStore(),  # type: ignore[arg-type]
        compute=_JobNotFoundCompute(),
    )

    def _fake_instantiate_plugins(_selection: object) -> _StubInstantiate:
        return _StubInstantiate(plugins)

    monkeypatch.setattr(
        "smai_orchestrator.instantiate_plugins",
        _fake_instantiate_plugins,
    )

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_dir:
        smai_yaml = Path(tmp_dir) / "smai.yaml"
        smai_yaml.write_text(
            """
engine: {}
plugins:
  llm_provider: bedrock
  metadata_store: sqlite
  artifact_store: localfs
  compute: localgpu
pipelines: ["smai_cg_execution", "smai_cg_entries"]
""",
            encoding="utf-8",
        )
        runner = CliRunner()
        result = runner.invoke(app, ["verify", "-c", str(smai_yaml), "--format", "json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert set(payload.keys()) == {"llm_provider", "metadata_store", "artifact_store", "compute"}
    for entry in payload.values():
        assert entry["ok"] is True
        assert isinstance(entry["reason"], str)
