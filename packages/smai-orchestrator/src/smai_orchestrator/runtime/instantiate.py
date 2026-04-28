"""Plugin discovery + instantiation flow.

Per ``designs/smai/09-cli.md`` §4 / ``07-plugin-interfaces.md`` §3.2.
At startup the runtime walks the resolved :class:`PluginSelection`
and constructs the four plugin instances:

1. Look up the configured plugin name (e.g.,
   ``plugins.metadata_store = "sqlite"``).
2. Enumerate entry points in the corresponding namespace (e.g.,
   ``importlib.metadata.entry_points(group="smai.metadata_stores")``).
3. Find the entry whose name matches; load the class.
4. Instantiate with the corresponding ``*_config`` dict as kwargs.
5. ``isinstance(plugin, Interface)`` smoke-check per `07` §3.2.
6. Bind into a typed :class:`InstantiatedPlugins` named-tuple.

The four plugins are wrapped in an :class:`AsyncExitStack` so they
tear down in **reverse construction order** on context exit. Plugins
that need lifecycle hooks (e.g., :class:`SqliteStore` needs
``await migrate()`` after construction and ``await dispose()`` on
teardown) are detected by attribute presence; we duck-type rather
than introduce a separate "Lifecycle" Protocol because the four
shipped plugins have asymmetric lifecycle needs (only SqliteStore has
both hooks).

Failure modes are surfaced as typed exceptions before any worker work
begins:

* :class:`PluginNotFound` — entry-point name doesn't resolve in the
  group.
* :class:`PluginInstantiationError` — constructor raised.
* :class:`PluginConformanceError` — runtime ``isinstance`` smoke
  check failed (the loaded class is missing a Protocol attribute).

The CLI is responsible for the user-facing error message; this
module's exceptions carry enough context (plugin name, group, root
cause) for the CLI to format helpfully.

Test injection: the :class:`PluginOverrides` parameter accepts pre-
instantiated plugin instances that bypass entry-point discovery for
the corresponding interface. Production callers pass
``overrides=None``; tests use this hook to inject fakes without
needing to register a fake entry point at the package boundary.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple

from smai_core.plugins import (
    ArtifactStore,
    Compute,
    LlmProvider,
    MetadataStore,
)

from smai_orchestrator.runtime.config import PluginSelection

# Single shape for the duck-typed lifecycle hooks we recognize on
# plugin instances (``async migrate()``, ``async dispose()``).
_AsyncNoArg = Callable[[], Awaitable[None]]

# Entry-point group name → human-readable interface label, used in
# error messages. Per `07` §3.2 the four namespaces are fixed.
ENTRY_POINT_GROUPS: dict[str, str] = {
    "smai.llm_providers": "LlmProvider",
    "smai.metadata_stores": "MetadataStore",
    "smai.artifact_stores": "ArtifactStore",
    "smai.computes": "Compute",
}


PluginInterface = Literal["llm_provider", "metadata_store", "artifact_store", "compute"]


_INTERFACE_TO_GROUP: dict[PluginInterface, str] = {
    "llm_provider": "smai.llm_providers",
    "metadata_store": "smai.metadata_stores",
    "artifact_store": "smai.artifact_stores",
    "compute": "smai.computes",
}


# === Errors =================================================================


class PluginNotFound(LookupError):
    """The configured plugin name is not registered in the entry-point
    group (`09` §4: "no matching entry point").

    Carries :attr:`available` (the names that ARE registered) so the
    CLI can suggest near-misses.
    """

    def __init__(self, *, group: str, name: str, available: list[str]) -> None:
        self.group = group
        self.name = name
        self.available = available
        super().__init__(
            f"no plugin {name!r} registered in entry-point group {group!r}; "
            f"available: {sorted(available)}"
        )


class PluginInstantiationError(RuntimeError):
    """The plugin constructor raised (`09` §4: "constructor failure").

    Wraps the underlying exception so the CLI surface can preserve the
    full traceback while the runtime carries the plugin identity.
    """

    def __init__(self, *, group: str, name: str, cause: BaseException) -> None:
        self.group = group
        self.name = name
        self.__cause__ = cause
        super().__init__(
            f"plugin {name!r} (group {group!r}) failed to instantiate: "
            f"{type(cause).__name__}: {cause}"
        )


class PluginConformanceError(TypeError):
    """The instantiated plugin failed the runtime ``isinstance``
    smoke-check (`09` §4: "conformance smoke-check failure").

    This means the loaded class is missing one or more Protocol
    attributes — a packaging bug in the plugin. Full conformance is
    verified at plugin development time via the conformance test
    suite; this check is fail-fast only.
    """

    def __init__(self, *, group: str, name: str, interface: type) -> None:
        self.group = group
        self.name = name
        self.interface = interface
        super().__init__(
            f"plugin {name!r} (group {group!r}) does not satisfy the "
            f"{interface.__name__} Protocol — full conformance is verified "
            f"by the plugin's conformance test suite"
        )


# === Output shape ===========================================================


class InstantiatedPlugins(NamedTuple):
    """The four constructed plugin instances, ready to pass to
    :func:`smai_orchestrator.worker.loop.run_worker_loop`.

    :attr:`llm_providers` is a per-role map: keys are conventionally
    :data:`smai_agents.model_selection.TaskRole` literals (typed as
    ``str`` here to keep the orchestrator independent of
    ``smai-agents``). The dict has one shared instance value when
    every role resolves to the same ``(provider, model_id)`` pair —
    which is the v1 default (Opus across the heavy roles, Sonnet
    across the bounded ones, both via ``bedrock``); per-role
    instances diverge only when the user overrides via
    ``SMAI_MODEL_<ROLE>`` env vars.

    The named-tuple shape is structural rather than the design's "PluginSet"
    Pydantic class because the runtime never serializes this object —
    it's a single-use handoff from :func:`instantiate_plugins` to the
    worker loop.
    """

    llm_providers: dict[str, LlmProvider]
    metadata_store: MetadataStore
    artifact_store: ArtifactStore
    compute: Compute


@dataclass(frozen=True)
class PluginOverrides:
    """Test-only injection of pre-instantiated plugins, bypassing
    entry-point discovery for the corresponding interface.

    The integration test that drives ``run_worker_loop`` end-to-end
    needs fake :class:`Compute` and fake :class:`LlmProvider`
    instances (real Docker / real Bedrock calls aren't acceptable in
    unit tests); this parameter lets the test inject them without
    needing to register a fake entry point inside ``pyproject.toml``
    at the package boundary. Per-interface override is independent —
    pass only the interfaces you want to override; the rest go
    through the discovery flow.

    Production callers pass :data:`None` to
    :func:`instantiate_plugins`. The helper does not migrate /
    initialize override-supplied plugins (the test that constructs
    them owns their lifecycle); only discovery-supplied plugins get
    the lifecycle hooks.
    """

    llm_providers: dict[str, LlmProvider] | None = None
    metadata_store: MetadataStore | None = None
    artifact_store: ArtifactStore | None = None
    compute: Compute | None = None
    extra_teardown: list[_AsyncNoArg] = field(default_factory=list[_AsyncNoArg])


# === Discovery + construction ================================================


def _load_entry_point(group: str, name: str) -> type:
    """Resolve ``(group, name)`` to a plugin class via
    :mod:`importlib.metadata`.

    Raises :class:`PluginNotFound` if the name doesn't exist in the
    group. Raises :class:`PluginInstantiationError` if the entry-point
    target fails to import (this is the "package installed but module
    broken" case).
    """
    eps = importlib.metadata.entry_points(group=group)
    matching = [ep for ep in eps if ep.name == name]
    if not matching:
        available = sorted({ep.name for ep in eps})
        raise PluginNotFound(group=group, name=name, available=available)
    ep = matching[0]
    try:
        cls = ep.load()
    except Exception as exc:  # noqa: BLE001 — surface as typed wrapper
        raise PluginInstantiationError(group=group, name=name, cause=exc) from exc
    if not isinstance(cls, type):
        raise PluginInstantiationError(
            group=group,
            name=name,
            cause=TypeError(f"entry point {ep.name!r} resolved to {cls!r}, expected a class"),
        )
    return cls


def list_discovered_plugins() -> dict[str, list[str]]:
    """Walk all four entry-point groups and return the discovered map.

    Used by the ``smai plugins`` verb (Task 2.D2) to surface "what does
    pip have installed" without reading every plugin's
    ``pyproject.toml``.
    """
    out: dict[str, list[str]] = {}
    for group in ENTRY_POINT_GROUPS:
        eps = importlib.metadata.entry_points(group=group)
        out[group] = sorted({ep.name for ep in eps})
    return out


def _construct_with_kwargs(cls: type, *, group: str, name: str, kwargs: dict[str, Any]) -> Any:
    """Call ``cls(**kwargs)`` — wrapping any exception in
    :class:`PluginInstantiationError`.
    """
    try:
        return cls(**kwargs)
    except Exception as exc:  # noqa: BLE001 — typed wrapper
        raise PluginInstantiationError(group=group, name=name, cause=exc) from exc


def _check_protocol(instance: object, *, group: str, name: str, interface: type) -> None:
    """Runtime ``isinstance`` smoke-check (per `07` §3.2 — attribute
    presence only).
    """
    if not isinstance(instance, interface):
        raise PluginConformanceError(group=group, name=name, interface=interface)


# === Lifecycle hooks (duck-typed) ===========================================
#
# The four shipped plugins have asymmetric lifecycle needs:
#
# * :class:`SqliteStore`: needs ``await migrate()`` after construction
#   and ``await dispose()`` on teardown.
# * :class:`LocalFsStore`: no async lifecycle.
# * :class:`LocalGpuCompute`: no async lifecycle.
# * :class:`BedrockProvider`: no async lifecycle.
#
# Rather than introduce a fifth Protocol ("PluginLifecycle") to cover
# this, we duck-type by attribute presence. The two recognized hooks
# are ``async migrate()`` (post-construction; called once) and ``async
# dispose()`` (called on context exit). When / if a future plugin
# needs different hooks, this layer adapts; the Protocol set in
# ``smai-core`` stays narrow.


async def _maybe_migrate(plugin: object) -> None:
    migrate: object = getattr(plugin, "migrate", None)
    if migrate is not None and callable(migrate):
        result = migrate()
        if isinstance(result, Awaitable):
            await result


async def _maybe_dispose(plugin: object) -> None:
    dispose: object = getattr(plugin, "dispose", None)
    if dispose is not None and callable(dispose):
        result = dispose()
        if isinstance(result, Awaitable):
            await result


# === Construct the four plugins =============================================


async def _construct_metadata_store(
    selection: PluginSelection, stack: AsyncExitStack
) -> MetadataStore:
    group = "smai.metadata_stores"
    cls = _load_entry_point(group, selection.metadata_store)
    instance = _construct_with_kwargs(
        cls,
        group=group,
        name=selection.metadata_store,
        kwargs=selection.metadata_store_config,
    )
    _check_protocol(instance, group=group, name=selection.metadata_store, interface=MetadataStore)
    await _maybe_migrate(instance)
    stack.push_async_callback(_maybe_dispose, instance)
    return instance


async def _construct_artifact_store(
    selection: PluginSelection, stack: AsyncExitStack
) -> ArtifactStore:
    group = "smai.artifact_stores"
    cls = _load_entry_point(group, selection.artifact_store)
    instance = _construct_with_kwargs(
        cls,
        group=group,
        name=selection.artifact_store,
        kwargs=selection.artifact_store_config,
    )
    _check_protocol(instance, group=group, name=selection.artifact_store, interface=ArtifactStore)
    await _maybe_migrate(instance)
    stack.push_async_callback(_maybe_dispose, instance)
    return instance


async def _construct_compute(selection: PluginSelection, stack: AsyncExitStack) -> Compute:
    group = "smai.computes"
    cls = _load_entry_point(group, selection.compute)
    instance = _construct_with_kwargs(
        cls,
        group=group,
        name=selection.compute,
        kwargs=selection.compute_config,
    )
    _check_protocol(instance, group=group, name=selection.compute, interface=Compute)
    await _maybe_migrate(instance)
    stack.push_async_callback(_maybe_dispose, instance)
    return instance


# Default :data:`smai_agents.model_selection.TaskRole` set used when
# the orchestrator builds a per-role :class:`LlmProvider` map.
# Hard-coded here (rather than imported from ``smai_agents``) to keep
# ``smai-orchestrator`` independent of ``smai-agents`` per the C2
# pyproject. The TaskRole literal in ``smai_agents`` is the
# authoritative source — a future spec change is a one-line update
# here. Surfaced in the status note as a reconciliation point.
DEFAULT_TASK_ROLES: tuple[str, ...] = (
    "planner",
    "harness_builder",
    "technique_implementer",
    "code_reviewer",
    "contextual_evaluator",
    "supervisor",
    "screener",
    "enricher",
)


async def _construct_llm_providers(
    selection: PluginSelection, stack: AsyncExitStack
) -> dict[str, LlmProvider]:
    """Construct one :class:`LlmProvider` per :data:`DEFAULT_TASK_ROLES`,
    sharing instances when multiple roles resolve to the same
    ``(provider, model_id)`` pair.

    Per the v1 default and `04` §4, all eight roles resolve to a
    single :data:`PluginSelection.llm_provider` plugin (only the
    ``model_id`` varies across roles); deviations come from the
    ``SMAI_MODEL_<ROLE>`` env-var overrides handled by
    :func:`smai_agents.model_selection.get_model_for_task`. We keep the
    per-role resolution out of this layer (it pulls in
    ``smai_agents``); v1 ships every role pointed at the same single
    instance and lets Task 2.D2's CLI surface (the consumer of
    :func:`get_model_for_task`) build the per-role override map by
    re-instantiating the plugin per unique resolved ``model_id`` and
    shipping the result via :class:`PluginOverrides`.

    This is the simpler-than-recommended shape — the brief proposed
    enumerating ``TaskRole.__args__`` and resolving per role here, but
    that requires an ``smai-orchestrator → smai-agents`` runtime
    dependency that the package's ``pyproject.toml`` does not declare
    today and that DEC-029's "minimum-coupling-by-default" stance
    argues against adding speculatively. The integration acceptance
    is satisfied by the simpler shape; Task 2.D2 (CLI) is the
    natural place for the per-role override pathway when it lands
    multi-model deployments.
    """
    group = "smai.llm_providers"
    cls = _load_entry_point(group, selection.llm_provider)
    instance = _construct_with_kwargs(
        cls,
        group=group,
        name=selection.llm_provider,
        kwargs=selection.llm_provider_config,
    )
    _check_protocol(instance, group=group, name=selection.llm_provider, interface=LlmProvider)
    await _maybe_migrate(instance)
    stack.push_async_callback(_maybe_dispose, instance)
    return {role: instance for role in DEFAULT_TASK_ROLES}


# === Top-level entry point ==================================================


@asynccontextmanager
async def instantiate_plugins(
    selection: PluginSelection,
    *,
    overrides: PluginOverrides | None = None,
) -> AsyncGenerator[InstantiatedPlugins, None]:
    """Discover and instantiate the four plugins per ``selection``.

    The returned object is an async context manager — on entry it
    constructs (or accepts the override-supplied) plugin instances; on
    exit it tears them down in reverse construction order. Plugins
    that expose ``async def dispose()`` get it called; the rest are
    no-ops on teardown.

    Construction order:

    1. :class:`MetadataStore` (the storage backbone — every other
       plugin's data eventually round-trips through it).
    2. :class:`ArtifactStore` (large-blob persistence).
    3. :class:`Compute` (job lifecycle — independent of storage but
       constructed last so its preflight check runs after the storage
       layer is up).
    4. :class:`LlmProvider` (per-role map — see
       :func:`_construct_llm_providers`).

    Override-supplied plugins skip the discovery / construction /
    isinstance / migrate steps. Their teardown is the test's
    responsibility (we don't ``dispose`` what we didn't construct);
    use :attr:`PluginOverrides.extra_teardown` to register additional
    callbacks if needed.
    """
    overrides = overrides or PluginOverrides()
    async with AsyncExitStack() as stack:
        metadata_store = (
            overrides.metadata_store
            if overrides.metadata_store is not None
            else await _construct_metadata_store(selection, stack)
        )
        artifact_store = (
            overrides.artifact_store
            if overrides.artifact_store is not None
            else await _construct_artifact_store(selection, stack)
        )
        compute = (
            overrides.compute
            if overrides.compute is not None
            else await _construct_compute(selection, stack)
        )
        llm_providers = (
            overrides.llm_providers
            if overrides.llm_providers is not None
            else await _construct_llm_providers(selection, stack)
        )
        for callback in overrides.extra_teardown:
            stack.push_async_callback(callback)
        yield InstantiatedPlugins(
            llm_providers=llm_providers,
            metadata_store=metadata_store,
            artifact_store=artifact_store,
            compute=compute,
        )


__all__ = [
    "DEFAULT_TASK_ROLES",
    "ENTRY_POINT_GROUPS",
    "InstantiatedPlugins",
    "PluginConformanceError",
    "PluginInstantiationError",
    "PluginInterface",
    "PluginNotFound",
    "PluginOverrides",
    "instantiate_plugins",
    "list_discovered_plugins",
]
