"""Engine, pipeline-spec format, worker loop, checkpointer; SMAI PipelineSpec instances.

Public surface:

* Pipeline-tracking record types (Task 1.10): runtime instances of the
  methodology entities in ``smai_core.entities``, persisted via the
  ``MetadataStore`` plugin and driven through the orchestrator's state
  machines per ``designs/smai/01-data-model.md`` §5.
* Runtime configuration umbrella (Task 2.C3): :class:`PipelineSpec`,
  :class:`RuntimeConfig`, :class:`PluginSelection`,
  :func:`instantiate_plugins`, and the spec registry — see
  :mod:`smai_orchestrator.runtime`.
"""

from smai_orchestrator.entities import (
    CGState,
    ComparisonGroupRecord,
    EntryRecord,
    EntryState,
    FactorModelRecord,
    PaperRecord,
    PaperState,
    ProposalRecord,
    ProposalState,
    RunRecord,
    RunState,
)
from smai_orchestrator.runtime import (
    DEFAULT_TASK_ROLES,
    DuplicateSpecError,
    InstantiatedPlugins,
    PipelineSpec,
    PluginConformanceError,
    PluginInstantiationError,
    PluginNotFound,
    PluginOverrides,
    PluginSelection,
    RuntimeConfig,
    SpecNotRegisteredError,
    get_pipeline_spec,
    instantiate_plugins,
    list_discovered_plugins,
    list_registered_specs,
    register_pipeline_spec,
    reset_registry,
)

__all__ = [
    "CGState",
    "ComparisonGroupRecord",
    "DEFAULT_TASK_ROLES",
    "DuplicateSpecError",
    "EntryRecord",
    "EntryState",
    "FactorModelRecord",
    "InstantiatedPlugins",
    "PaperRecord",
    "PaperState",
    "PipelineSpec",
    "PluginConformanceError",
    "PluginInstantiationError",
    "PluginNotFound",
    "PluginOverrides",
    "PluginSelection",
    "ProposalRecord",
    "ProposalState",
    "RunRecord",
    "RunState",
    "RuntimeConfig",
    "SpecNotRegisteredError",
    "get_pipeline_spec",
    "instantiate_plugins",
    "list_discovered_plugins",
    "list_registered_specs",
    "register_pipeline_spec",
    "reset_registry",
]
