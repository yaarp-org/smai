"""Runtime configuration umbrella, pipeline-spec format, and plugin
instantiation flow.

Per ``designs/smai/05-orchestrator.md`` §5.1 / §6 and
``designs/smai/09-cli.md`` §3 / §4. This sub-package composes the C1/C2
engine substrate (:class:`EngineSpec`, :class:`EngineConfig`,
:class:`Checkpointer`) into the user-facing surface the CLI consumes:

* :class:`PipelineSpec` — full pipeline-spec per `05` §5.1; collapses
  the worker-loop's two-map scheduling-query shape into one.
* :class:`RuntimeConfig` — top-level umbrella; co-loaded by the CLI
  per `09` §3.
* :class:`PluginSelection` — the four plugin names + opaque per-plugin
  config.
* :func:`instantiate_plugins` — entry-point discovery + async-context-
  managed lifecycle per `09` §4.
* Pipeline-spec registry — :func:`register_pipeline_spec` /
  :func:`get_pipeline_spec` / :func:`list_registered_specs`.

Out of scope, deferred:

* CLI verb surface (``smai dev`` / ``smai run`` / etc.) — Task 2.D2.
* Concrete CG-execution / proposal / paper-ingestion specs — Tasks
  2.C4 / 3.E1 / 3.E2.
* Multi-worker leasing — Task 3.G1.
* CLI config layering (env → file → flags) — Task 2.D2.
"""

from smai_orchestrator.runtime.config import (
    ApiAuthConfig,
    ApiConfig,
    ApiSseConfig,
    PluginSelection,
    RuntimeConfig,
)
from smai_orchestrator.runtime.instantiate import (
    DEFAULT_TASK_ROLES,
    ENTRY_POINT_GROUPS,
    InstantiatedPlugins,
    PluginConformanceError,
    PluginInstantiationError,
    PluginInterface,
    PluginNotFound,
    PluginOverrides,
    instantiate_plugins,
    list_discovered_plugins,
)
from smai_orchestrator.runtime.registry import (
    DuplicateSpecError,
    SpecNotRegisteredError,
    get_pipeline_spec,
    list_registered_specs,
    register_pipeline_spec,
    reset_registry,
)
from smai_orchestrator.runtime.spec import PipelineSpec

__all__ = [
    "ApiAuthConfig",
    "ApiConfig",
    "ApiSseConfig",
    "DEFAULT_TASK_ROLES",
    "ENTRY_POINT_GROUPS",
    "DuplicateSpecError",
    "InstantiatedPlugins",
    "PipelineSpec",
    "PluginConformanceError",
    "PluginInstantiationError",
    "PluginInterface",
    "PluginNotFound",
    "PluginOverrides",
    "PluginSelection",
    "RuntimeConfig",
    "SpecNotRegisteredError",
    "get_pipeline_spec",
    "instantiate_plugins",
    "list_discovered_plugins",
    "list_registered_specs",
    "register_pipeline_spec",
    "reset_registry",
]
