"""Unit tests for ``smai start`` CLI verb + :meth:`Runtime.start_worker` (Task 3.G3).

Exercises:

* ``smai start`` pre-flight: fails on stale schema (1), passes on
  schema-at-head + happy boot (with a stub configured to drain
  immediately).
* :meth:`Runtime.start_worker` smoke-test: boots against the SQLite +
  LocalFs + FakeCompute fixture matrix; the yielded :class:`Runtime`
  exposes :attr:`Runtime.worker_id` per the production-mode contract.
* ``_resolve_worker_id`` honors the resolution order: flag > env >
  fallback.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml
from _cli_fakes import (  # type: ignore[import-not-found]
    FakeCompute,
    StubLlmProvider,
)
from smai_artifacts_localfs import LocalFsStore
from smai_cli.main import _resolve_worker_id
from smai_cli.runtime import Runtime
from smai_orchestrator import (
    DEFAULT_TASK_ROLES,
    PluginOverrides,
    PluginSelection,
    RuntimeConfig,
)
from smai_orchestrator.engine.config import EngineConfig

# === _resolve_worker_id ======================================================


def test_resolve_worker_id_flag_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMAI_WORKER_ID", "env-worker")
    assert _resolve_worker_id(override="flag-worker") == "flag-worker"


def test_resolve_worker_id_env_used_when_flag_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMAI_WORKER_ID", "env-worker")
    assert _resolve_worker_id(override=None) == "env-worker"


def test_resolve_worker_id_falls_back_to_host_pid_uuid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SMAI_WORKER_ID", raising=False)
    resolved = _resolve_worker_id(override=None)
    # Fallback shape: ``<hostname>-<pid>-<8 hex>``. Be loose on
    # hostname (CI runners vary) but assert the structural shape.
    parts = resolved.rsplit("-", 2)
    assert len(parts) == 3
    pid = parts[1]
    suffix = parts[2]
    assert pid.isdigit()
    assert len(suffix) == 8
    assert all(c in "0123456789abcdef" for c in suffix)


# === Runtime.start_worker ====================================================


def _make_worker_runtime_config(*, sqlite_path: Path | None = None) -> RuntimeConfig:
    """Production-shaped :class:`RuntimeConfig` for the worker tests.

    Uses the same plugin selection as ``smai dev`` defaults but with
    a higher poll_interval and an explicit per-test SQLite URI when
    supplied — closer to production posture.
    """
    uri = (
        f"sqlite+aiosqlite:///{sqlite_path}"
        if sqlite_path is not None
        else "sqlite+aiosqlite:///:memory:"
    )
    return RuntimeConfig(
        engine=EngineConfig(poll_interval_seconds=30, worker_count=1),
        plugins=PluginSelection(
            llm_provider="bedrock",
            metadata_store="sqlite",
            artifact_store="localfs",
            compute="localgpu",
            metadata_store_config={"uri": uri},
        ),
        pipelines=["smai_cg_execution", "smai_cg_entries"],
    )


@pytest.mark.asyncio
async def test_start_worker_yields_runtime_with_pinned_worker_id(tmp_path: Path) -> None:
    """:meth:`Runtime.start_worker` requires ``worker_id`` and pins it
    onto :attr:`Runtime.worker_id`."""
    overrides = PluginOverrides(
        llm_providers={role: StubLlmProvider() for role in DEFAULT_TASK_ROLES},
        artifact_store=LocalFsStore(tmp_path / "artifacts"),
        compute=FakeCompute(),
    )
    config = _make_worker_runtime_config()
    pinned_id = "pinned-worker-g3-test"
    async with Runtime.start_worker(
        config,
        worker_id=pinned_id,
        workspace_root=tmp_path / "workspaces",
        plugin_overrides=overrides,
    ) as runtime:
        assert runtime.worker_id == pinned_id
        assert runtime.workspace_root == tmp_path / "workspaces"
        # The worker task is running in the background — it should NOT
        # block; we simply tear down via the context manager exit.


@pytest.mark.asyncio
async def test_start_worker_drains_on_context_exit(tmp_path: Path) -> None:
    """The background worker task is cancelled cleanly when the context
    manager exits."""
    overrides = PluginOverrides(
        llm_providers={role: StubLlmProvider() for role in DEFAULT_TASK_ROLES},
        artifact_store=LocalFsStore(tmp_path / "artifacts"),
        compute=FakeCompute(),
    )
    config = _make_worker_runtime_config()
    async with Runtime.start_worker(
        config,
        worker_id="drain-test",
        workspace_root=tmp_path / "workspaces",
        plugin_overrides=overrides,
    ) as runtime:
        # The runtime's ``_state.worker_task`` was started.
        assert runtime._state.worker_task is not None
        worker_task = runtime._state.worker_task
        assert not worker_task.done()
    # On context exit, shutdown_event was set and worker_task drained.
    assert worker_task.done()


# === smai start CLI verb pre-flight =========================================


def _write_smai_yaml_for_start(
    tmp_path: Path,
    *,
    sqlite_path: Path,
    metadata_plugin: str = "sqlite",
) -> Path:
    """Write a minimal smai.yaml pointing the metadata store at
    ``sqlite_path``."""
    cfg: dict[str, object] = {
        "engine": {
            "poll_interval_seconds": 30,
            "worker_count": 1,
            "fair_scheduling": "off",
        },
        "plugins": {
            "llm_provider": "bedrock",
            "metadata_store": metadata_plugin,
            "artifact_store": "localfs",
            "compute": "localgpu",
            "llm_provider_config": {
                "region": "us-east-1",
                "model_id": "us.anthropic.claude-opus-4-6-v1",
            },
            "metadata_store_config": {
                "uri": f"sqlite+aiosqlite:///{sqlite_path}",
            },
            "artifact_store_config": {},
            "compute_config": {},
        },
        "pipelines": ["smai_cg_execution", "smai_cg_entries"],
    }
    smai_yaml = tmp_path / "smai.yaml"
    smai_yaml.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return smai_yaml


def test_smai_start_refuses_on_stale_schema(tmp_path: Path) -> None:
    """``smai start`` exits 1 against an empty (un-stamped) DB."""
    from smai_cli.main import app
    from typer.testing import CliRunner

    sqlite_path = tmp_path / "state.db"
    smai_yaml = _write_smai_yaml_for_start(tmp_path, sqlite_path=sqlite_path)
    runner = CliRunner()
    result = runner.invoke(app, ["start", "-c", str(smai_yaml)])
    assert result.exit_code == 1, result.output
    assert "schema NOT at head" in result.output or "schema NOT at head" in (
        result.stderr if hasattr(result, "stderr") else result.output
    )
    # The error mentions running `smai migrate`.
    combined = result.output + (result.stderr if hasattr(result, "stderr") else "")
    assert "migrate" in combined


def test_smai_start_passes_pre_flight_after_migrate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After ``smai migrate``, ``smai start``'s pre-flight passes —
    boot reaches Runtime.start_worker (we cancel before it actually
    runs cycles by sending SIGINT immediately).

    This test exercises ONLY the pre-flight + boot kickoff. We don't
    drive cycles end-to-end here (the integration test does that).
    Instead we patch ``Runtime.start_worker`` to surface a sentinel +
    raise an immediate exit so the verb's body completes quickly.
    """
    from smai_cli.main import app
    from typer.testing import CliRunner

    sqlite_path = tmp_path / "state.db"
    smai_yaml = _write_smai_yaml_for_start(tmp_path, sqlite_path=sqlite_path)

    # Step 1 — migrate the DB to head so pre-flight passes.
    migrate_result = CliRunner().invoke(app, ["migrate", "-c", str(smai_yaml)])
    assert migrate_result.exit_code == 0, migrate_result.output
    assert sqlite_path.exists()

    # Step 2 — patch Runtime.start_worker so the verb body kicks the
    # boot lifecycle and immediately exits via a synthetic SystemExit.
    boot_observed: dict[str, object] = {}

    class _ImmediateBoot:
        def __init__(self, **kwargs: object) -> None:
            boot_observed.update(kwargs)

        async def __aenter__(self) -> object:
            raise SystemExit(0)

        async def __aexit__(self, *_: object) -> None:
            return None

    def _fake_start_worker(*_args: object, **kwargs: object) -> _ImmediateBoot:
        return _ImmediateBoot(**kwargs)

    monkeypatch.setattr("smai_cli.main.Runtime.start_worker", _fake_start_worker)
    monkeypatch.setenv("SMAI_HOME", str(tmp_path / "smai_home"))

    # Step 3 — invoke. Pre-flight + plugin-completeness + boot kickoff
    # all must succeed for SystemExit(0) to surface as exit_code==0.
    result = CliRunner().invoke(app, ["start", "-c", str(smai_yaml), "--worker-id", "test-w-1"])
    # Exit code 0 from the synthetic SystemExit raised inside the
    # context manager.
    assert result.exit_code == 0, (result.output, getattr(result, "exception", None))
    assert boot_observed["worker_id"] == "test-w-1"


def test_smai_start_resolves_worker_id_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``--worker-id``, ``smai start`` honors the
    ``SMAI_WORKER_ID`` env."""
    from smai_cli.main import app
    from typer.testing import CliRunner

    sqlite_path = tmp_path / "state.db"
    smai_yaml = _write_smai_yaml_for_start(tmp_path, sqlite_path=sqlite_path)
    CliRunner().invoke(app, ["migrate", "-c", str(smai_yaml)])

    boot_observed: dict[str, object] = {}

    class _ImmediateBoot:
        def __init__(self, **kwargs: object) -> None:
            boot_observed.update(kwargs)

        async def __aenter__(self) -> object:
            raise SystemExit(0)

        async def __aexit__(self, *_: object) -> None:
            return None

    def _fake_start_worker(*_args: object, **kwargs: object) -> _ImmediateBoot:
        return _ImmediateBoot(**kwargs)

    monkeypatch.setattr("smai_cli.main.Runtime.start_worker", _fake_start_worker)
    monkeypatch.setenv("SMAI_WORKER_ID", "env-supplied-worker")
    monkeypatch.setenv("SMAI_HOME", str(tmp_path / "smai_home"))

    result = CliRunner().invoke(app, ["start", "-c", str(smai_yaml)])
    assert result.exit_code == 0, (result.output, getattr(result, "exception", None))
    assert boot_observed["worker_id"] == "env-supplied-worker"


def test_smai_start_refuses_on_incomplete_plugin_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A smai.yaml that omits a plugin slot fails Pydantic validation
    (``PluginSelection`` requires non-empty strings) — surfaces as a
    clear ConfigValidationError before the schema check fires.

    This test verifies the load-time guard; the
    ``_validate_plugin_completeness`` defense-in-depth check is
    structurally unreachable from a well-formed smai.yaml because
    Pydantic's ``str`` field type rejects ``None``. We exercise that
    helper directly via :func:`smai_cli.main._validate_plugin_completeness`
    in a sibling unit test below.
    """
    from smai_cli.main import app
    from typer.testing import CliRunner

    bad_yaml = tmp_path / "smai.yaml"
    bad_yaml.write_text(
        # `plugins.compute` missing — Pydantic rejects.
        """
engine: {}
plugins:
  llm_provider: bedrock
  metadata_store: sqlite
  artifact_store: localfs
pipelines: ["smai_cg_execution"]
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("SMAI_HOME", str(tmp_path / "smai_home"))
    result = CliRunner().invoke(app, ["start", "-c", str(bad_yaml)])
    assert result.exit_code == 1


def test_validate_plugin_completeness_rejects_empty_strings() -> None:
    """The defense-in-depth check rejects a hand-rolled
    :class:`PluginSelection` whose fields contain only whitespace."""
    import typer
    from smai_cli.main import _validate_plugin_completeness

    # Build an object that pydantic would have validated, but whose
    # fields are empty after the fact (defense-in-depth).
    class _Sel:
        llm_provider = ""
        metadata_store = "sqlite"
        artifact_store = "localfs"
        compute = "localgpu"

    class _Cfg:
        plugins = _Sel()

    with pytest.raises(typer.Exit):
        _validate_plugin_completeness(_Cfg())


# === Lease-capability enforcement (Task R4 fix #1) ===========================


def test_enforce_lease_capability_rejects_multi_worker_against_non_leasing_store() -> None:
    """``_enforce_lease_capability`` exits 1 when ``worker_count > 1`` but
    the configured store reports ``supports_leasing=False`` (per `09` §6.2
    / DEC-035 #2)."""
    import typer
    from smai_cli.main import _enforce_lease_capability
    from smai_core.plugins.metadata_store._capabilities import MetadataStoreCapabilities

    class _Store:
        capabilities = MetadataStoreCapabilities(
            is_tenant_aware=False,
            supports_transactions=True,
            supports_leasing=False,
        )

    class _Plugins:
        metadata_store = _Store()

    class _RuntimePlugins:
        metadata_store = "fakestore"

    class _EngineCfg:
        worker_count = 4

    class _Config:
        engine = _EngineCfg()
        plugins = _RuntimePlugins()

    class _Runtime:
        config = _Config()
        plugins = _Plugins()

    with pytest.raises(typer.Exit):
        _enforce_lease_capability(_Runtime())


def test_enforce_lease_capability_allows_single_worker_against_non_leasing_store() -> None:
    """Single-worker deployments are allowed against non-lease-capable
    stores — the lease contention only materializes across workers."""
    from smai_cli.main import _enforce_lease_capability
    from smai_core.plugins.metadata_store._capabilities import MetadataStoreCapabilities

    class _Store:
        capabilities = MetadataStoreCapabilities(
            is_tenant_aware=False,
            supports_transactions=True,
            supports_leasing=False,
        )

    class _Plugins:
        metadata_store = _Store()

    class _RuntimePlugins:
        metadata_store = "fakestore"

    class _EngineCfg:
        worker_count = 1

    class _Config:
        engine = _EngineCfg()
        plugins = _RuntimePlugins()

    class _Runtime:
        config = _Config()
        plugins = _Plugins()

    # No raise — single-worker is allowed.
    _enforce_lease_capability(_Runtime())


def test_enforce_lease_capability_allows_multi_worker_against_leasing_store() -> None:
    """Multi-worker is allowed when the store reports ``supports_leasing=True``."""
    from smai_cli.main import _enforce_lease_capability
    from smai_core.plugins.metadata_store._capabilities import MetadataStoreCapabilities

    class _Store:
        capabilities = MetadataStoreCapabilities(
            is_tenant_aware=False,
            supports_transactions=True,
            supports_leasing=True,
        )

    class _Plugins:
        metadata_store = _Store()

    class _RuntimePlugins:
        metadata_store = "postgres"

    class _EngineCfg:
        worker_count = 8

    class _Config:
        engine = _EngineCfg()
        plugins = _RuntimePlugins()

    class _Runtime:
        config = _Config()
        plugins = _Plugins()

    _enforce_lease_capability(_Runtime())


# === Worker pre-flight: container-image config check (round 11 / 12) =========

_PUB_RUNTIME = "registry.example.com/org/smai-runtime:v2"
_PUB_RUNTIME_CPU = "registry.example.com/org/smai-runtime-cpu:v2"
_PUB_AGENT = "registry.example.com/org/smai-agent:v2"


def _make_image_check_runtime(
    *,
    requires_published_image: bool,
    runtime_image: str,
    runtime_cpu_image: str,
    agent_image: str = _PUB_AGENT,
) -> object:
    """Build a minimal duck-typed runtime for the ``_enforce_container_images_published``
    tests — exposes only ``plugins.compute.capabilities`` + ``config.engine``.

    ``agent_image`` defaults to a published reference so a caller
    exercising only the runtime images is not tripped by the round-12
    agent-image check; pass the default ``smai-agent:dev`` explicitly to
    exercise it."""
    from smai_core.plugins import ComputeCapabilities

    class _Compute:
        name = "modal" if requires_published_image else "localgpu"
        capabilities = ComputeCapabilities(
            supports_gpu=True,
            max_timeout_seconds=3600,
            requires_published_image=requires_published_image,
        )

    class _Plugins:
        compute = _Compute()

    class _EngineCfg:
        pass

    _EngineCfg.runtime_image = runtime_image  # type: ignore[attr-defined]
    _EngineCfg.runtime_cpu_image = runtime_cpu_image  # type: ignore[attr-defined]
    _EngineCfg.agent_image = agent_image  # type: ignore[attr-defined]

    class _Config:
        engine = _EngineCfg()

    class _Runtime:
        config = _Config()
        plugins = _Plugins()

    return _Runtime()


def test_enforce_container_images_published_rejects_registry_pull_default_image() -> None:
    """``_enforce_container_images_published`` exits 1 when a registry-pull
    compute substrate is paired with the local-only default runtime
    image (round 11)."""
    import typer
    from smai_cli.main import _enforce_container_images_published
    from smai_orchestrator.engine import DEFAULT_RUNTIME_CPU_IMAGE, DEFAULT_RUNTIME_IMAGE

    runtime = _make_image_check_runtime(
        requires_published_image=True,
        runtime_image=DEFAULT_RUNTIME_IMAGE,
        runtime_cpu_image=DEFAULT_RUNTIME_CPU_IMAGE,
    )
    with pytest.raises(typer.Exit):
        _enforce_container_images_published(runtime)


def test_enforce_container_images_published_allows_local_build_default_image() -> None:
    """A local-build substrate with the default image tags is the
    intended flow — no raise."""
    from smai_cli.main import _enforce_container_images_published
    from smai_orchestrator.engine import (
        DEFAULT_AGENT_IMAGE,
        DEFAULT_RUNTIME_CPU_IMAGE,
        DEFAULT_RUNTIME_IMAGE,
    )

    runtime = _make_image_check_runtime(
        requires_published_image=False,
        runtime_image=DEFAULT_RUNTIME_IMAGE,
        runtime_cpu_image=DEFAULT_RUNTIME_CPU_IMAGE,
        agent_image=DEFAULT_AGENT_IMAGE,
    )
    _enforce_container_images_published(runtime)


def test_enforce_container_images_published_allows_registry_pull_overridden_image() -> None:
    """A registry-pull substrate with all three images overridden to
    published references — no raise."""
    from smai_cli.main import _enforce_container_images_published

    runtime = _make_image_check_runtime(
        requires_published_image=True,
        runtime_image=_PUB_RUNTIME,
        runtime_cpu_image=_PUB_RUNTIME_CPU,
        agent_image=_PUB_AGENT,
    )
    _enforce_container_images_published(runtime)


def test_enforce_container_images_published_rejects_registry_pull_default_agent_image() -> None:
    """Round 12: runtime images published but ``engine.agent_image``
    still the local-only default — the worker pre-flight exits 1."""
    import typer
    from smai_cli.main import _enforce_container_images_published
    from smai_orchestrator.engine import DEFAULT_AGENT_IMAGE

    runtime = _make_image_check_runtime(
        requires_published_image=True,
        runtime_image=_PUB_RUNTIME,
        runtime_cpu_image=_PUB_RUNTIME_CPU,
        agent_image=DEFAULT_AGENT_IMAGE,
    )
    with pytest.raises(typer.Exit):
        _enforce_container_images_published(runtime)


def test_enforce_container_images_published_allows_local_build_default_agent_image() -> None:
    """Round 12: a local-build substrate building ``smai-agent:dev``
    locally is the intended flow — no raise even at the default."""
    from smai_cli.main import _enforce_container_images_published
    from smai_orchestrator.engine import DEFAULT_AGENT_IMAGE

    runtime = _make_image_check_runtime(
        requires_published_image=False,
        runtime_image=_PUB_RUNTIME,
        runtime_cpu_image=_PUB_RUNTIME_CPU,
        agent_image=DEFAULT_AGENT_IMAGE,
    )
    _enforce_container_images_published(runtime)


# === SIGINT/SIGTERM signal handling ==========================================


@pytest.mark.asyncio
async def test_start_worker_drains_on_simulated_signal(tmp_path: Path) -> None:
    """End-to-end (in-process) drive: boot ``Runtime.start_worker``, wait
    until the worker is running, simulate SIGTERM by setting the
    shutdown_event directly, assert the runtime drains cleanly.

    We don't invoke the CLI verb here (subprocess + signal delivery is
    flaky in pytest); the verb is the thin asyncio-event wrapper, and
    ``Runtime.start_worker``'s drain semantics are the load-bearing
    contract — exercised via the same context-manager exit path.
    """
    overrides = PluginOverrides(
        llm_providers={role: StubLlmProvider() for role in DEFAULT_TASK_ROLES},
        artifact_store=LocalFsStore(tmp_path / "artifacts"),
        compute=FakeCompute(),
    )
    config = _make_worker_runtime_config()

    async with Runtime.start_worker(
        config,
        worker_id="signal-test",
        workspace_root=tmp_path / "workspaces",
        plugin_overrides=overrides,
    ) as runtime:
        # Simulate a SIGTERM arriving 0ms in by setting the shutdown
        # event directly. The worker task drains.
        runtime._state.shutdown_event.set()
        # Give the worker task a moment to observe the event.
        await asyncio.sleep(0.05)

    # After context exit, the worker_task is done.
    assert runtime._state.worker_task is not None
    assert runtime._state.worker_task.done()
