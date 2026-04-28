"""Harness/technique Python runtime, fixed templates, ``HarnessAPIManifest``.

Implements ``designs/smai/10-runtime-and-templates.md``: the runtime substrate
that hosts harness pipelines and technique modules, the fixed-template
integration layer, the post-build ``HarnessAPIManifest`` artifact, the
manifest-driven type check, the no-go-zone hash check, and the
metric-emission contract that produces ``RawMetrics``-shaped output.

Per DEC-026 / DEC-029, ``smai-runtime`` is a pipeline-layer package; it
depends on ``smai-core`` (methodology layer) and may bring in ML-stack
runtime deps (``torch``, ``numpy``, etc.) — those are not subject to
``smai-core``'s allowlist (``tools/check_deps.py`` rule 2 / 3 cover the
methodology and plugin packages, not this one).
"""

from smai_runtime.components import (
    ADMISSIBLE_PATTERNS_FOR_KEY,
    COMPONENT_FIELD_FOR_KEY,
    HarnessComponents,
)
from smai_runtime.errors import (
    CONTRACT_ERROR_FILENAME,
    NO_GO_ZONE_ERROR_FILENAME,
    MetricsContractError,
    NoGoZoneErrorCode,
    NoGoZoneHashError,
    TechniqueOutputContractError,
    TechniqueOutputErrorCode,
    write_contract_error,
    write_no_go_zone_error,
)
from smai_runtime.factor_aware import (
    assert_additive_baseline_has_no_required_extension_points,
    assert_substitutive_entry_provides_value,
    is_additive_baseline,
)
from smai_runtime.integrator import integrate_technique_output
from smai_runtime.manifest import (
    MANIFEST_SCHEMA_VERSION,
    HarnessAPIManifest,
    HarnessExtensionPoint,
    IntegrationPattern,
    compute_harness_version_hash,
    compute_manifest_hash,
    freeze_manifest,
    manifest_canonical_form,
)
from smai_runtime.metrics import (
    METRICS_FILENAME,
    VALIDATION_RESULTS_FILENAME,
    build_seed_run_outcome,
    filter_to_known_keys,
    optional_runtime_keys,
    required_runtime_keys,
    validate_metrics_dict,
    write_metrics,
)
from smai_runtime.no_go_zone import (
    RUNTIME_TEMPLATE_VERSION,
    TEMPLATE_WORKSPACE_PATHS,
    check_no_go_zones,
    compute_expected_hashes,
    read_template_bytes,
)
from smai_runtime.runner import (
    EXIT_OK,
    EXIT_RUNTIME_FAILURE,
    EXIT_USAGE_ERROR,
    RunMode,
    RunnerArgs,
    parse_args,
    run,
)
from smai_runtime.seed import seed_everything
from smai_runtime.type_check import check_technique_output
from smai_runtime.workspace import (
    HARNESS_API_MANIFEST_FILENAME,
    HARNESS_CONTRACT_FILENAME,
    TECHNIQUE_CONTRACT_FILENAME,
    WORKSPACE_SUBDIRS,
    build_runtime_config,
    create_workspace_skeleton,
    load_contracts,
    materialize_workspace,
    write_contracts,
    write_template_files,
    write_validation_results,
)

__all__ = [
    "ADMISSIBLE_PATTERNS_FOR_KEY",
    "COMPONENT_FIELD_FOR_KEY",
    "CONTRACT_ERROR_FILENAME",
    "EXIT_OK",
    "EXIT_RUNTIME_FAILURE",
    "EXIT_USAGE_ERROR",
    "HARNESS_API_MANIFEST_FILENAME",
    "HARNESS_CONTRACT_FILENAME",
    "HarnessAPIManifest",
    "HarnessComponents",
    "HarnessExtensionPoint",
    "IntegrationPattern",
    "MANIFEST_SCHEMA_VERSION",
    "METRICS_FILENAME",
    "MetricsContractError",
    "NO_GO_ZONE_ERROR_FILENAME",
    "NoGoZoneErrorCode",
    "NoGoZoneHashError",
    "RUNTIME_TEMPLATE_VERSION",
    "RunMode",
    "RunnerArgs",
    "TECHNIQUE_CONTRACT_FILENAME",
    "TEMPLATE_WORKSPACE_PATHS",
    "TechniqueOutputContractError",
    "TechniqueOutputErrorCode",
    "VALIDATION_RESULTS_FILENAME",
    "WORKSPACE_SUBDIRS",
    "assert_additive_baseline_has_no_required_extension_points",
    "assert_substitutive_entry_provides_value",
    "build_runtime_config",
    "build_seed_run_outcome",
    "check_no_go_zones",
    "check_technique_output",
    "compute_expected_hashes",
    "compute_harness_version_hash",
    "compute_manifest_hash",
    "create_workspace_skeleton",
    "filter_to_known_keys",
    "freeze_manifest",
    "integrate_technique_output",
    "is_additive_baseline",
    "load_contracts",
    "manifest_canonical_form",
    "materialize_workspace",
    "optional_runtime_keys",
    "parse_args",
    "read_template_bytes",
    "required_runtime_keys",
    "run",
    "seed_everything",
    "validate_metrics_dict",
    "write_contract_error",
    "write_contracts",
    "write_metrics",
    "write_no_go_zone_error",
    "write_template_files",
    "write_validation_results",
]
