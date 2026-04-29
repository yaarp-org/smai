"""CLI verbs, config layering, :class:`RuntimeConfig` loading, in-band Runtime.

Per ``designs/smai/09-cli.md`` §9 / §1.2 — the CLI is one consumer of
the in-process service surface; programmatic Tier-A consumers import
:class:`Runtime` and the sub-services directly. Task 2.D3's smoke test
also wraps :meth:`Runtime.start_in_band` to drive the end-to-end test.

Public surface (Phase 2):

* :class:`Runtime` — async-context-managed in-band Runtime
  (:meth:`Runtime.start_in_band`).
* :class:`ExperimentsService` / :class:`StatusService` — verb-to-service
  surfaces (`09` §1.2).
* :func:`load_runtime_config` — defaults → file → env → flags layering
  (`09` §2).
* :func:`build_per_role_llm_providers` — per-role :class:`LlmProvider`
  fan-out (`04` §4).
"""

from smai_cli.config import (
    PHASE_2_DEFAULT_PIPELINES,
    ConfigFileError,
    ConfigValidationError,
    base_defaults,
    dev_defaults,
    load_runtime_config,
)
from smai_cli.per_role_llm import (
    UnsupportedLlmProvider,
    build_per_role_llm_providers,
)
from smai_cli.runtime import (
    CGNotFoundError,
    CGStatus,
    ExperimentsService,
    Runtime,
    StatusService,
    WaitTimeoutError,
)

__all__ = [
    "CGNotFoundError",
    "CGStatus",
    "ConfigFileError",
    "ConfigValidationError",
    "ExperimentsService",
    "PHASE_2_DEFAULT_PIPELINES",
    "Runtime",
    "StatusService",
    "UnsupportedLlmProvider",
    "WaitTimeoutError",
    "base_defaults",
    "build_per_role_llm_providers",
    "dev_defaults",
    "load_runtime_config",
]
