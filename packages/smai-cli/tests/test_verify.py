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
    verify_runtime_image_config,
    verify_runtime_image_probe,
)
from smai_core.plugins import (
    ArtifactStoreCapabilities,
    CacheConfig,
    ComputeCapabilities,
    JobHandle,
    JobStatus,
    LlmCapabilities,
    ModelResponse,
    NormalizedMessage,
    ToolDefinition,
)
from smai_core.plugins.compute import ComputeUnavailable, JobImageInvalid, JobNotFound
from smai_core.plugins.metadata_store._capabilities import MetadataStoreCapabilities
from smai_orchestrator.engine import (
    DEFAULT_RUNTIME_CPU_IMAGE,
    DEFAULT_RUNTIME_IMAGE,
)

# Published-reference image tags used across the round-11 image-config
# tests — any non-default string the registry-pull substrate would be
# able to pull.
_PUB_RUNTIME = "registry.example.com/org/smai-runtime:v2"
_PUB_RUNTIME_CPU = "registry.example.com/org/smai-runtime-cpu:v2"

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
    # Generic errors get no extra hint appended.
    assert "hint:" not in result.reason


@pytest.mark.asyncio
async def test_verify_llm_provider_hint_on_invalid_model_id() -> None:
    """A Bedrock ``ValidationException: The provided model identifier is
    invalid`` failure gets an actionable hint about inference-profile IDs."""
    provider = _RaisingLlmProvider(
        RuntimeError("ValidationException: The provided model identifier is invalid.")
    )
    result = await verify_llm_provider(provider)
    assert result.ok is False
    assert "list-inference-profiles" in result.reason


@pytest.mark.asyncio
async def test_verify_llm_provider_hint_on_model_not_granted() -> None:
    """A Bedrock ``AccessDeniedException: <model> is not available for this
    account`` failure gets a hint about granting model access in the console."""
    provider = _RaisingLlmProvider(
        RuntimeError(
            "AccessDeniedException: anthropic.claude-opus-4-7 is not available for this account."
        )
    )
    result = await verify_llm_provider(provider)
    assert result.ok is False
    assert "Model access" in result.reason


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

    def _fake_instantiate_plugins(_selection: object, **_kwargs: object) -> _StubInstantiate:
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

    def _fake_instantiate_plugins(_selection: object, **_kwargs: object) -> _StubInstantiate:
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


def test_smai_verify_passes_skip_migrate_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """``smai verify`` invokes ``instantiate_plugins`` with
    ``skip_migrate=True`` so the read-only probe does not mutate the
    configured store's schema (Task R4 fix #2).
    """
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

    captured_kwargs: dict[str, object] = {}

    def _capturing_instantiate_plugins(_selection: object, **kwargs: object) -> _StubInstantiate:
        captured_kwargs.update(kwargs)
        return _StubInstantiate(plugins)

    monkeypatch.setattr(
        "smai_orchestrator.instantiate_plugins",
        _capturing_instantiate_plugins,
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
    assert result.exit_code == 0, result.output
    assert captured_kwargs.get("skip_migrate") is True, (
        f"smai verify must call instantiate_plugins with skip_migrate=True; "
        f"observed kwargs={captured_kwargs}"
    )


def test_smai_verify_does_not_call_migrate_on_metadata_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: ``smai verify`` against a stub MetadataStore that
    exposes a ``migrate`` method must NOT trigger that method (Task R4
    fix #2 — verify is strictly read-only).
    """
    from smai_cli.main import app
    from smai_orchestrator import InstantiatedPlugins
    from typer.testing import CliRunner

    migrate_calls: list[str] = []

    class _MigrateRecordingStore(_OkMetadataStub):
        async def migrate(self) -> None:
            migrate_calls.append("called")

    class _StubInstantiate:
        def __init__(self, plugins: InstantiatedPlugins, *, skip_migrate: bool) -> None:
            self._plugins = plugins
            self._skip_migrate = skip_migrate

        async def __aenter__(self) -> InstantiatedPlugins:
            # Mirror the production helper's lifecycle hook gating: only
            # call migrate() when skip_migrate=False.
            if not self._skip_migrate:
                await self._plugins.metadata_store.migrate()  # type: ignore[attr-defined]
            return self._plugins

        async def __aexit__(self, *_: object) -> None:
            del _

    store = _MigrateRecordingStore()
    plugins = InstantiatedPlugins(
        llm_providers={"planner": StubLlmProvider()},
        metadata_store=store,  # type: ignore[arg-type]
        artifact_store=InMemoryArtifactStore(),  # type: ignore[arg-type]
        compute=_JobNotFoundCompute(),
    )

    def _fake_instantiate_plugins(
        _selection: object, *, skip_migrate: bool = False, **_kwargs: object
    ) -> _StubInstantiate:
        return _StubInstantiate(plugins, skip_migrate=skip_migrate)

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
    assert result.exit_code == 0, result.output
    assert migrate_calls == [], (
        f"smai verify must not trigger migrate(); observed {len(migrate_calls)} call(s)"
    )


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

    def _fake_instantiate_plugins(_selection: object, **_kwargs: object) -> _StubInstantiate:
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
    assert set(payload.keys()) == {
        "llm_provider",
        "metadata_store",
        "artifact_store",
        "compute",
        "runtime_image",
    }
    for entry in payload.values():
        assert entry["ok"] is True
        assert isinstance(entry["reason"], str)


# === verify_runtime_image_config (round 11) ==================================


class _RegistryPullCompute(FakeCompute):
    """Compute whose substrate pulls the job image from a registry — the
    operator must publish it. Mirrors Modal / RunPod's
    :attr:`ComputeCapabilities.requires_published_image`."""

    name: str = "fake-registry-pull"
    capabilities: ComputeCapabilities = ComputeCapabilities(
        supports_gpu=True,
        max_timeout_seconds=3600,
        supports_log_streaming=False,
        requires_published_image=True,
    )


class _ImageInvalidCompute(_RegistryPullCompute):
    """Registry-pull compute whose ``submit`` rejects the image — the
    probe's failure case."""

    async def submit(  # type: ignore[override]
        self,
        image: str,
        command: list[str],
        env: dict[str, str],
        gpu: bool = False,
        timeout_seconds: int = 3600,
        **plugin_options: object,
    ) -> JobHandle:
        del command, env, gpu, timeout_seconds, plugin_options
        raise JobImageInvalid(image, "manifest unknown")


def test_verify_runtime_image_config_local_build_passes() -> None:
    """A local-build substrate (``requires_published_image`` False, e.g.
    ``localgpu``) paired with the default image tags is the intended
    flow — PASS."""
    result = verify_runtime_image_config(
        FakeCompute(), DEFAULT_RUNTIME_IMAGE, DEFAULT_RUNTIME_CPU_IMAGE
    )
    assert result.ok is True
    assert result.latency_ms is None


def test_verify_runtime_image_config_registry_pull_default_image_fails() -> None:
    """A registry-pull substrate paired with the local-only built-in
    default runtime-image tags is the round-11 bug — FAIL with an
    actionable message naming every offending field and the runbook.
    Round 14: the agent image is no longer part of this check."""
    result = verify_runtime_image_config(
        _RegistryPullCompute(),
        DEFAULT_RUNTIME_IMAGE,
        DEFAULT_RUNTIME_CPU_IMAGE,
    )
    assert result.ok is False
    assert "engine.runtime_image" in result.reason
    assert "engine.runtime_cpu_image" in result.reason
    assert "engine.agent_image" not in result.reason
    assert "fake-registry-pull" in result.reason
    assert "OPERATIONS.md" in result.reason


def test_verify_runtime_image_config_registry_pull_overridden_passes() -> None:
    """A registry-pull substrate with both runtime images overridden to
    published references — PASS."""
    result = verify_runtime_image_config(_RegistryPullCompute(), _PUB_RUNTIME, _PUB_RUNTIME_CPU)
    assert result.ok is True


def test_verify_runtime_image_config_registry_pull_one_default_fails() -> None:
    """Only one image left at the default still fails — the check names
    just the offending field."""
    result = verify_runtime_image_config(
        _RegistryPullCompute(), _PUB_RUNTIME, DEFAULT_RUNTIME_CPU_IMAGE
    )
    assert result.ok is False
    assert "engine.runtime_cpu_image" in result.reason
    assert "engine.runtime_image is" not in result.reason


# === verify_runtime_image_probe (round 11, opt-in) ===========================


@pytest.mark.asyncio
async def test_verify_runtime_image_probe_passes_on_clean_submit() -> None:
    """The probe submits a no-op job per runtime image (GPU + CPU); a
    substrate that accepts the submit means the image is reachable —
    PASS."""
    result = await verify_runtime_image_probe(FakeCompute(), _PUB_RUNTIME, _PUB_RUNTIME_CPU)
    assert result.ok is True
    assert "smai-runtime:v2" in result.reason
    assert "smai-runtime-cpu:v2" in result.reason
    assert result.latency_ms is not None


@pytest.mark.asyncio
async def test_verify_runtime_image_probe_dedupes_equal_images() -> None:
    """When both runtime image fields point at the same tag the probe
    submits once."""
    compute = FakeCompute()
    result = await verify_runtime_image_probe(compute, "same:tag", "same:tag")
    assert result.ok is True
    # Only one probe job submitted (counter advanced once).
    assert compute._counter == 1


@pytest.mark.asyncio
async def test_verify_runtime_image_probe_probes_distinct_images() -> None:
    """Two distinct runtime image tags means two probe submits."""
    compute = FakeCompute()
    result = await verify_runtime_image_probe(compute, _PUB_RUNTIME, _PUB_RUNTIME_CPU)
    assert result.ok is True
    assert compute._counter == 2
    assert "runtime_cpu_image" in result.reason


@pytest.mark.asyncio
async def test_verify_runtime_image_probe_fails_on_image_invalid() -> None:
    """A substrate that rejects the image at ``submit`` time
    (:class:`JobImageInvalid`) fails the probe."""
    result = await verify_runtime_image_probe(
        _ImageInvalidCompute(),
        "registry.example.com/org/not-pushed:v2",
        "registry.example.com/org/not-pushed-cpu:v2",
    )
    assert result.ok is False
    assert "not reachable" in result.reason


# === smai verify CLI verb — round-11 runtime-image checks ====================


def _run_verify_cli(
    monkeypatch: pytest.MonkeyPatch,
    compute: object,
    *,
    extra_args: list[str] | None = None,
    engine_block: str = "engine: {}",
) -> object:
    """Invoke ``smai verify`` against a stub ``InstantiatedPlugins`` with
    the given ``compute``. Shared boilerplate for the round-11 / round-12
    CLI tests.

    ``engine_block`` is spliced into the generated ``smai.yaml`` as-is —
    the default ``engine: {}`` leaves every image field at its built-in
    default; a caller may override the runtime images to exercise the
    PASS path.
    """
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
        compute=compute,  # type: ignore[arg-type]
    )

    def _fake_instantiate_plugins(_selection: object, **_kwargs: object) -> _StubInstantiate:
        return _StubInstantiate(plugins)

    monkeypatch.setattr("smai_orchestrator.instantiate_plugins", _fake_instantiate_plugins)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_dir:
        smai_yaml = Path(tmp_dir) / "smai.yaml"
        smai_yaml.write_text(
            f"""
{engine_block}
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
        return runner.invoke(app, ["verify", "-c", str(smai_yaml), *(extra_args or [])])


def test_smai_verify_cli_registry_pull_default_image_exits_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``smai verify`` exits 1 when a registry-pull compute substrate is
    paired with the local-only default ``engine.runtime_image`` (the
    config under test leaves ``engine: {}`` so the defaults apply)."""
    result = _run_verify_cli(monkeypatch, _RegistryPullCompute())
    assert result.exit_code == 1, result.output  # type: ignore[attr-defined]
    assert "FAIL runtime_image" in result.output  # type: ignore[attr-defined]
    assert "OPERATIONS.md" in result.output  # type: ignore[attr-defined]


def test_smai_verify_cli_local_build_default_image_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local-build substrate + default image is the intended flow —
    the runtime_image check reports PASS and the verb exits 0."""
    result = _run_verify_cli(monkeypatch, _JobNotFoundCompute())
    assert result.exit_code == 0, result.output  # type: ignore[attr-defined]
    assert "PASS runtime_image" in result.output  # type: ignore[attr-defined]


def test_smai_verify_cli_registry_pull_runtime_images_overridden_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registry-pull substrate with both runtime images overridden to
    published references — the config check reports PASS (round 14: the
    agent image is no longer part of the check)."""
    engine_block = (
        f"engine:\n  runtime_image: {_PUB_RUNTIME}\n  runtime_cpu_image: {_PUB_RUNTIME_CPU}\n"
    )
    result = _run_verify_cli(monkeypatch, _RegistryPullCompute(), engine_block=engine_block)
    assert result.exit_code == 0, result.output  # type: ignore[attr-defined]
    assert "PASS runtime_image" in result.output  # type: ignore[attr-defined]


def test_smai_verify_cli_probe_image_flag_runs_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``smai verify --probe-image`` adds a ``runtime_image_probe`` check;
    without the flag the probe is skipped."""
    with_flag = _run_verify_cli(monkeypatch, _JobNotFoundCompute(), extra_args=["--probe-image"])
    assert with_flag.exit_code == 0, with_flag.output  # type: ignore[attr-defined]
    assert "runtime_image_probe" in with_flag.output  # type: ignore[attr-defined]

    without_flag = _run_verify_cli(monkeypatch, _JobNotFoundCompute())
    assert "runtime_image_probe" not in without_flag.output  # type: ignore[attr-defined]
