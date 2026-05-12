"""Runtime configuration umbrella — :class:`RuntimeConfig` +
:class:`PluginSelection`.

Per ``designs/smai/09-cli.md`` §3 — the typed model produced by the
CLI's config-layering pipeline (env → ``smai.yaml`` → flags). The
layering itself is the CLI's territory (Task 2.D2); this module ships
the typed shape and a couple of small construction helpers
(:meth:`RuntimeConfig.with_overrides`).

The two sub-models are **orthogonal axes** per DEC-028 / `09` §3:
changing :attr:`engine.fair_scheduling` does not change
:attr:`plugins.metadata_store`, and vice-versa. The CLI keeps them on
the same :class:`RuntimeConfig` so they travel together but does not
couple them.

The single-or-list :attr:`pipelines` shape resolves a doc-internal
inconsistency between `05` §6 (``pipeline: str``) and `09` §3
(``pipelines: list[str]``); we go with the `09` §3 plural because
``smai dev`` per `05` §7.1 / DEC-032 registers the three SMAI
pipeline-specs concurrently — single-spec deployments pass a
one-element list. Surfaced as a one-line spec reconciliation in the
status note.

Per Task 4.L1 / ``12-ui-process.md`` §9.1: the :attr:`api` block holds
the ``smai ui`` verb's settings (host / port / auto-detect knob, bearer
auth, SSE keepalive). It is a third orthogonal axis — independent of
:attr:`engine` and :attr:`plugins` — and ships a sensible default so
``smai dev`` / ``smai start`` callers (which never read ``api``) are
unaffected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from smai_orchestrator.engine.config import EngineConfig


class PluginSelection(BaseModel):
    """Per-interface plugin name + plugin-specific config block (`09`
    §3).

    Each :attr:`*_config` is a free-form dict at this layer; the
    resolved plugin constructor parses it into its own typed config
    inside the plugin package. The CLI does not type-check plugin
    configs — that's the plugin's responsibility (it raises a clear
    error if its config is malformed).
    """

    model_config = ConfigDict(extra="forbid")

    llm_provider: str
    """Entry-point name (``"bedrock"`` / ``"anthropic"`` / ``"openai"``
    in the shipped plugins). Resolves against the ``smai.llm_providers``
    entry-point group at instantiation."""

    metadata_store: str
    """Entry-point name (``"sqlite"`` / ``"postgres"``). Resolves
    against ``smai.metadata_stores``."""

    artifact_store: str
    """Entry-point name (``"localfs"`` / ``"s3"``). Resolves against
    ``smai.artifact_stores``."""

    compute: str
    """Entry-point name (``"localgpu"`` / ``"modal"`` / ``"runpod"``).
    Resolves against ``smai.computes``."""

    llm_provider_config: dict[str, Any] = Field(default_factory=dict)
    """Plugin-specific kwargs passed to the LlmProvider constructor.
    BedrockProvider's required keys: ``region``, ``model_id``."""

    metadata_store_config: dict[str, Any] = Field(default_factory=dict)
    """Plugin-specific kwargs. SqliteStore: ``uri`` (defaults to
    in-memory)."""

    artifact_store_config: dict[str, Any] = Field(default_factory=dict)
    """Plugin-specific kwargs. LocalFsStore: ``root`` (defaults to
    ``$SMAI_ARTIFACTS_ROOT`` or ``~/.smai/artifacts``)."""

    compute_config: dict[str, Any] = Field(default_factory=dict)
    """Plugin-specific kwargs. LocalGpuCompute: ``agent_image``,
    ``runtime_image``, ``runtime_cpu_image``, ``workspace``,
    ``skip_preflight`` (these only tune the "image not found" build
    hint — the experiment-run image is chosen by the orchestrator from
    :attr:`EngineConfig.runtime_image` / :attr:`runtime_cpu_image` per
    the CG's ``compute.gpu`` flag)."""


class ApiAuthConfig(BaseModel):
    """Bearer-token auth posture for ``smai ui`` (per ``11`` §7.3 / `12`
    §9.1).

    ``enabled=False`` is the default — Host-header validation alone is
    the canonical local posture. Flipping ``enabled=True`` opts in to
    the belt-and-suspenders bearer-token mode; the token file at
    ``token_path`` is auto-generated on first launch (``0o600``,
    :func:`secrets.token_urlsafe(32)`) and preserved across restarts so
    browser tabs keep working.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    token_path: Path = Field(
        default_factory=lambda: Path("~/.smai/api-token").expanduser(),
    )


class ApiSseConfig(BaseModel):
    """SSE channel knobs the API surfaces (per ``11`` §8 / `12` §9.1).

    The verb itself does not consume these (``smai_api.make_api_app``
    reads them via the runtime / event broker plumbing); they live here
    so :func:`smai_cli.config.load_runtime_config` can expose the same
    layered ``smai.yaml`` → env → flags pipeline for them.
    """

    model_config = ConfigDict(extra="forbid")

    heartbeat_seconds: int = 15
    ring_buffer_size: int = 100
    refetch_all_on_overflow: bool = True


class ApiConfig(BaseModel):
    """``smai ui`` verb settings per ``12-ui-process.md`` §9.1.

    Independent of :attr:`RuntimeConfig.engine` and
    :attr:`RuntimeConfig.plugins`. Default values are tuned for the
    laptop case (Case A in `12` §5.1): loopback bind, no auth, in-
    process worker auto-enabled by the auto-detect rule (`12` §4.3).

    The ``with_worker`` field accepts ``"auto"`` (the default — defer
    to the resolution rule in `12` §4.3 / §9.3), ``True`` (force
    in-process worker), or ``False`` (force no in-process worker). The
    CLI flag (``--with-worker`` / ``--no-worker``) overrides this per
    the standard config-layering precedence chain.
    """

    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    with_worker: Literal["auto", True, False] = "auto"
    reload: bool = False
    auth: ApiAuthConfig = Field(default_factory=ApiAuthConfig)
    sse: ApiSseConfig = Field(default_factory=ApiSseConfig)


class RuntimeConfig(BaseModel):
    """Top-level config umbrella co-loaded by the CLI (`09` §3).

    Constructed once at CLI entry and frozen for the duration of the
    process. There is no hot-reload; re-reading config means restarting
    the worker.
    """

    model_config = ConfigDict(extra="forbid")

    engine: EngineConfig
    """Engine-behavior knobs — separable from plugin selection per
    DEC-028 / `05` §6."""

    plugins: PluginSelection
    """Per-interface plugin selection + opaque per-plugin config block
    per `09` §3 / `07` §3.2."""

    pipelines: list[str]
    """Names of registered :class:`PipelineSpec` instances the worker
    should drive. Per DEC-032 / `05` §7.1, ``smai dev`` registers the
    three SMAI specs (``smai_cg_execution`` / ``smai_proposal`` /
    ``smai_paper_ingestion``); single-spec deployments pass a
    one-element list. Names are looked up against
    :mod:`smai_orchestrator.runtime.registry`."""

    api: ApiConfig = Field(default_factory=ApiConfig)
    """``smai ui`` verb settings (per Task 4.L1 / `12-ui-process.md`
    §9.1). Optional — every field defaults; ``smai dev`` / ``smai
    start`` ignore the block entirely. Layered identically to the
    other axes through :func:`smai_cli.config.load_runtime_config`."""

    def with_overrides(self, **overrides: Any) -> RuntimeConfig:
        """Return a new :class:`RuntimeConfig` with the given top-level
        fields replaced.

        Convenience helper for callers that want to layer per-call
        overrides on top of a loaded config without re-parsing the
        full layering pipeline. Production CLI flag-override is owned
        by Task 2.D2; this helper is for programmatic / test use.
        """
        return self.model_copy(update=overrides)


__all__ = [
    "ApiAuthConfig",
    "ApiConfig",
    "ApiSseConfig",
    "PluginSelection",
    "RuntimeConfig",
]
