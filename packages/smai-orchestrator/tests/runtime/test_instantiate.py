"""Tests for the plugin instantiation flow.

Per ``09-cli.md`` §4. Covers:

* Entry-point discovery against the four wave-1 plugins (sqlite,
  localfs, localgpu, bedrock).
* Plugin-specific options pass through as constructor kwargs.
* :class:`SqliteStore` lifecycle hooks (``migrate`` post-construct,
  ``dispose`` on context exit).
* Teardown order: reverse construction order on context exit.
* Error paths: :class:`PluginNotFound` on a bogus name;
  :class:`PluginInstantiationError` on a constructor that raises;
  :class:`PluginConformanceError` on a class missing a Protocol attr.
* :class:`PluginOverrides` short-circuits discovery for the
  corresponding interface.

LocalGpuCompute construction triggers a ``docker info`` preflight
that fails on machines without Docker; we pass ``skip_preflight=True``
through the config dict so the test runs anywhere. BedrockProvider
construction does NOT make AWS calls; we pass a fake client through
the config dict to avoid even constructing a botocore session.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Any

import pytest
from smai_artifacts_localfs import LocalFsStore
from smai_compute_localgpu import LocalGpuCompute
from smai_core.plugins import ArtifactStore, MetadataStore
from smai_llm_bedrock import BedrockProvider
from smai_orchestrator.runtime import (
    DEFAULT_TASK_ROLES,
    PluginConformanceError,
    PluginInstantiationError,
    PluginNotFound,
    PluginOverrides,
    PluginSelection,
    instantiate_plugins,
    list_discovered_plugins,
)
from smai_store_sqlite import SqliteStore

# Re-mount engine helpers (FakeCompute, FakeArtifactStore).
_ENGINE_TESTS_DIR = Path(__file__).resolve().parent.parent / "engine"
sys.path.insert(0, str(_ENGINE_TESTS_DIR))

from _helpers import FakeArtifactStore, FakeCompute  # type: ignore[import-not-found] # noqa: E402

# Re-mount the runtime fakes (FakeLlmProvider).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _runtime_fakes import FakeLlmProvider  # type: ignore[import-not-found] # noqa: E402

# A test-only fake bedrock client; matches the duck shape
# ``BedrockProvider`` expects (just needs a ``converse`` method that
# the conformance suite can drive — we never call it here).


class _FakeBedrockClient:
    def __init__(self) -> None:
        self._conformance_queue: deque = deque()

    def converse(self, **kwargs: Any) -> Any:  # pragma: no cover - unused
        raise RuntimeError("FakeBedrockClient.converse not configured for this test")


def _bedrock_safe_config() -> dict[str, Any]:
    """Construct BedrockProvider config that doesn't reach AWS.

    Passes a fake ``bedrock_client`` so the constructor doesn't try to
    build a real botocore session.
    """
    return {
        "region": "us-east-1",
        "model_id": "us.anthropic.claude-opus-4-6-v1",
        "bedrock_client": _FakeBedrockClient(),
        "sleep": _no_sleep,
    }


async def _no_sleep(_seconds: float) -> None:  # pragma: no cover - unused in tests
    return


def _localgpu_safe_config() -> dict[str, Any]:
    """LocalGpuCompute config with ``skip_preflight=True`` so the
    ``docker info`` call is not made (CI runners may not have Docker).
    """
    return {"skip_preflight": True}


def _all_real_selection() -> PluginSelection:
    return PluginSelection(
        llm_provider="bedrock",
        metadata_store="sqlite",
        artifact_store="localfs",
        compute="localgpu",
        llm_provider_config=_bedrock_safe_config(),
        metadata_store_config={"uri": "sqlite+aiosqlite:///:memory:"},
        artifact_store_config={},
        compute_config=_localgpu_safe_config(),
    )


# === list_discovered_plugins ================================================


def test_list_discovered_plugins_finds_all_four_groups() -> None:
    discovered = list_discovered_plugins()
    assert "smai.llm_providers" in discovered
    assert "smai.metadata_stores" in discovered
    assert "smai.artifact_stores" in discovered
    assert "smai.computes" in discovered
    # Wave-1 plugin names show up.
    assert "bedrock" in discovered["smai.llm_providers"]
    assert "sqlite" in discovered["smai.metadata_stores"]
    assert "localfs" in discovered["smai.artifact_stores"]
    assert "localgpu" in discovered["smai.computes"]


# === Discovery + construction (real plugins) ================================


async def test_instantiate_real_plugins_e2e(tmp_path: Path) -> None:
    """All four wave-1 plugins instantiate via the discovery flow."""
    selection = _all_real_selection()
    selection.artifact_store_config = {"root": str(tmp_path / "artifacts")}

    async with instantiate_plugins(selection) as plugins:
        assert isinstance(plugins.metadata_store, SqliteStore)
        assert isinstance(plugins.artifact_store, LocalFsStore)
        assert isinstance(plugins.compute, LocalGpuCompute)
        # All eight TaskRoles map to the same shared BedrockProvider
        # instance (single PluginSelection.llm_provider, no per-role
        # SMAI_MODEL_<ROLE> env overrides set for this test).
        assert set(plugins.llm_providers.keys()) == set(DEFAULT_TASK_ROLES)
        unique_providers = {id(p) for p in plugins.llm_providers.values()}
        assert len(unique_providers) == 1
        assert isinstance(next(iter(plugins.llm_providers.values())), BedrockProvider)

        # Protocol smoke-checks pass.
        assert isinstance(plugins.metadata_store, MetadataStore)
        assert isinstance(plugins.artifact_store, ArtifactStore)


async def test_sqlite_store_migrate_runs_post_construction() -> None:
    """SqliteStore exposes ``migrate()``; after instantiate_plugins the
    schema is laid down (a CRUD round-trip works).
    """
    selection = PluginSelection(
        llm_provider="bedrock",
        metadata_store="sqlite",
        artifact_store="localfs",
        compute="localgpu",
        llm_provider_config=_bedrock_safe_config(),
        metadata_store_config={"uri": "sqlite+aiosqlite:///:memory:"},
        artifact_store_config={},
        compute_config=_localgpu_safe_config(),
    )
    async with instantiate_plugins(selection) as plugins:
        # If migrate() didn't run, this query would fail with a "no
        # such table" error. The empty page result confirms the schema
        # is in place.
        page = await plugins.metadata_store.list_entries_for_cg("nonexistent")
        assert page.items == []


async def test_skip_migrate_suppresses_schema_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``skip_migrate=True`` keeps :meth:`SqliteStore.migrate` from being
    called during plugin construction (Task R4 fix #2).

    Read-only probes (``smai verify`` per `09-cli.md` §1) flip this kwarg
    so the verify pings are strictly read-only — surfacing a stale
    schema as a probe failure rather than silently mutating the store
    as a boot side-effect.
    """
    migrate_calls: list[str] = []
    real_migrate = SqliteStore.migrate

    async def recording_migrate(self: SqliteStore) -> None:
        migrate_calls.append("called")
        await real_migrate(self)

    monkeypatch.setattr(SqliteStore, "migrate", recording_migrate)

    selection = _all_real_selection()
    async with instantiate_plugins(selection, skip_migrate=True):
        pass
    assert migrate_calls == [], (
        "SqliteStore.migrate must NOT be called when skip_migrate=True; "
        f"observed {len(migrate_calls)} call(s)"
    )

    # Sanity check: with skip_migrate=False (the default), migrate IS
    # called — establishes that the kwarg is the load-bearing toggle.
    migrate_calls.clear()
    async with instantiate_plugins(selection):
        pass
    assert migrate_calls == ["called"]


async def test_dispose_runs_on_context_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """SqliteStore's ``dispose`` is called as a teardown callback on
    ``AsyncExitStack`` exit. Wrap the method to record the call rather
    than rely on post-dispose engine behavior (which varies by
    SQLAlchemy version).
    """
    dispose_calls: list[str] = []
    real_dispose = SqliteStore.dispose

    async def recording_dispose(self: SqliteStore) -> None:
        dispose_calls.append("called")
        await real_dispose(self)

    monkeypatch.setattr(SqliteStore, "dispose", recording_dispose)

    selection = _all_real_selection()
    async with instantiate_plugins(selection):
        assert dispose_calls == []  # not yet
    assert dispose_calls == ["called"]


async def test_reverse_teardown_order() -> None:
    """Teardown happens in reverse construction order — verified by
    capturing dispose calls on plugins that record them.
    """
    teardown_log: list[str] = []

    class RecordingMeta:
        name = "rec-meta"
        capabilities = None  # type: ignore[assignment]

        async def migrate(self) -> None:
            return

        async def dispose(self) -> None:
            teardown_log.append("metadata")

    class RecordingArtifacts:
        name = "rec-arts"
        capabilities = None  # type: ignore[assignment]

        async def dispose(self) -> None:
            teardown_log.append("artifacts")

    class RecordingCompute:
        name = "rec-compute"
        capabilities = None  # type: ignore[assignment]

        async def dispose(self) -> None:
            teardown_log.append("compute")

    class RecordingLlm:
        name = "rec-llm"
        capabilities = None  # type: ignore[assignment]

        async def dispose(self) -> None:
            teardown_log.append("llm")

    # Drive instantiate_plugins with overrides — but we still want the
    # AsyncExitStack to fire dispose. Overrides bypass the discovery /
    # dispose-registration; we use ``extra_teardown`` to register
    # dispose explicitly so the teardown-order assertion is meaningful.
    rm = RecordingMeta()
    ra = RecordingArtifacts()
    rc = RecordingCompute()
    rl = RecordingLlm()
    selection = _all_real_selection()
    overrides = PluginOverrides(
        metadata_store=rm,  # type: ignore[arg-type]
        artifact_store=ra,  # type: ignore[arg-type]
        compute=rc,  # type: ignore[arg-type]
        llm_providers=dict.fromkeys(DEFAULT_TASK_ROLES, rl),  # type: ignore[arg-type]
        # Register dispose callbacks in construction order; AsyncExitStack
        # calls them in reverse order on exit.
        extra_teardown=[rm.dispose, ra.dispose, rc.dispose, rl.dispose],
    )
    async with instantiate_plugins(selection, overrides=overrides):
        pass

    # Teardown order: extra_teardown is pushed in order [meta, arts,
    # compute, llm], AsyncExitStack pops in reverse, so we observe
    # llm → compute → artifacts → metadata.
    assert teardown_log == ["llm", "compute", "artifacts", "metadata"]


# === Error paths ============================================================


async def test_plugin_not_found_raises_with_available() -> None:
    selection = PluginSelection(
        llm_provider="bedrock",
        metadata_store="ghost-store",  # not registered
        artifact_store="localfs",
        compute="localgpu",
    )
    with pytest.raises(PluginNotFound) as excinfo:
        async with instantiate_plugins(selection):
            pass
    assert excinfo.value.name == "ghost-store"
    assert excinfo.value.group == "smai.metadata_stores"
    assert "sqlite" in excinfo.value.available


async def test_plugin_constructor_failure_wrapped() -> None:
    """A constructor that raises surfaces as
    :class:`PluginInstantiationError` with the original cause attached.
    """
    selection = PluginSelection(
        llm_provider="bedrock",
        metadata_store="sqlite",
        artifact_store="localfs",
        compute="localgpu",
        # SqliteStore takes ``uri`` — pass a kwarg it won't accept.
        metadata_store_config={"uri": "ok", "bogus_kwarg": 42},
    )
    with pytest.raises(PluginInstantiationError) as excinfo:
        async with instantiate_plugins(selection):
            pass
    assert excinfo.value.name == "sqlite"
    assert isinstance(excinfo.value.__cause__, TypeError)


async def test_plugin_conformance_failure_raised() -> None:
    """Classes missing a Protocol attribute fail the isinstance smoke
    check — raised as :class:`PluginConformanceError`.

    We can't easily inject a bogus class into the entry-point set
    without modifying ``pyproject.toml``, so we exercise the
    PluginOverrides path with a non-conforming object — wait, overrides
    skip the isinstance check. Instead we monkeypatch the loader to
    return a bogus class.
    """
    from smai_orchestrator.runtime import instantiate as inst_module

    class NotAStore:
        # Missing the entire MetadataStore Protocol.
        name = "not-a-store"

        def __init__(self, **_kwargs: Any) -> None:
            return

    real_loader = inst_module._load_entry_point  # noqa: SLF001

    def fake_loader(group: str, name: str) -> type:
        if group == "smai.metadata_stores":
            return NotAStore
        return real_loader(group, name)

    inst_module._load_entry_point = fake_loader  # noqa: SLF001
    try:
        selection = _all_real_selection()
        with pytest.raises(PluginConformanceError) as excinfo:
            async with instantiate_plugins(selection):
                pass
        assert excinfo.value.name == "sqlite"
        assert excinfo.value.interface is MetadataStore
    finally:
        inst_module._load_entry_point = real_loader  # noqa: SLF001


# === PluginOverrides short-circuits discovery ===============================


async def test_overrides_short_circuit_discovery() -> None:
    """Override-supplied plugins skip entry-point discovery entirely;
    the corresponding ``selection.<interface>`` is ignored.
    """
    selection = PluginSelection(
        llm_provider="bedrock",
        metadata_store="ghost-name",  # would fail discovery
        artifact_store="ghost-artifacts",  # would fail discovery
        compute="ghost-compute",  # would fail discovery
        llm_provider_config=_bedrock_safe_config(),
    )
    overrides = PluginOverrides(
        metadata_store=_make_in_memory_store(),  # type: ignore[arg-type]
        artifact_store=FakeArtifactStore(),  # type: ignore[arg-type]
        compute=FakeCompute(),  # type: ignore[arg-type]
    )
    async with instantiate_plugins(selection, overrides=overrides) as plugins:
        # The bogus selection.metadata_store / .artifact_store /
        # .compute names did not trip PluginNotFound — overrides won.
        assert plugins.metadata_store is overrides.metadata_store
        assert plugins.artifact_store is overrides.artifact_store
        assert plugins.compute is overrides.compute
        # llm_providers came from discovery (no override for it).
        assert isinstance(next(iter(plugins.llm_providers.values())), BedrockProvider)


async def test_overrides_with_llm_providers_short_circuits() -> None:
    """All four interfaces overridden — discovery never runs (the
    ``selection`` field values can be entirely bogus)."""
    fake_llm = FakeLlmProvider()
    overrides = PluginOverrides(
        llm_providers=dict.fromkeys(DEFAULT_TASK_ROLES, fake_llm),  # type: ignore[arg-type]
        metadata_store=_make_in_memory_store(),  # type: ignore[arg-type]
        artifact_store=FakeArtifactStore(),  # type: ignore[arg-type]
        compute=FakeCompute(),  # type: ignore[arg-type]
    )
    selection = PluginSelection(
        llm_provider="ghost-llm",
        metadata_store="ghost-store",
        artifact_store="ghost-arts",
        compute="ghost-compute",
    )
    async with instantiate_plugins(selection, overrides=overrides) as plugins:
        for role in DEFAULT_TASK_ROLES:
            assert plugins.llm_providers[role] is fake_llm


def _make_in_memory_store() -> SqliteStore:
    """Construct a SqliteStore directly (bypassing instantiate_plugins)
    for use in the override-supplied path.

    The override-supplied store's lifecycle is the test's
    responsibility — in practice the test process exits before any
    consequence of skipping ``dispose`` is observable.
    """
    store = SqliteStore("sqlite+aiosqlite:///:memory:")
    return store


# === Default TaskRole set ===================================================


def test_default_task_roles_match_smai_agents() -> None:
    """DEFAULT_TASK_ROLES is a hard-coded mirror of
    :data:`smai_agents.model_selection.TaskRole`. If the literal in
    smai-agents drifts, this test fails — surfacing the reconciliation
    point so a future code change has to update both.

    We import smai_agents lazily in the test (orchestrator's
    pyproject does NOT depend on smai-agents; the import works here
    only because uv-workspace puts every package on ``sys.path`` for
    test runs).
    """
    from smai_agents.model_selection import TaskRole

    # ``TaskRole`` is a typing.Literal; introspect via __args__.
    expected = tuple(TaskRole.__args__)  # type: ignore[attr-defined]
    assert DEFAULT_TASK_ROLES == expected
