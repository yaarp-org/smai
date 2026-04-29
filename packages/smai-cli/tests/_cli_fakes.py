"""Shared fixtures + builders for the smai-cli tests.

Per the workspace's pytest ``--import-mode=importlib`` discovery
layout, every package's ``tests/`` is on ``sys.path`` once. The
module name ``_cli_fakes`` is unique within the workspace (matches
the ``_specs_fakes`` / ``_runtime_fakes`` / ``_b3_fakes`` /
``_agent_fakes`` cousin shape).

Builders shipped here:

* :class:`StubLlmProvider` — minimal :class:`LlmProvider` clone
  (avoids the smai-orchestrator test tree dependency). Drives no
  real LLM calls; spec tests in this package don't drive the agent
  loop.
* :class:`FakeCompute` — minimal :class:`Compute` with no submit-
  side behavior; required only because the worker loop's
  ``run_worker_cycle`` accepts a Compute instance.
* :func:`make_experiment_yaml` — programmatic builder for a
  compilable :class:`ExperimentDocument` YAML, with the technique
  pre-registered.
* :func:`make_registries_with_technique` — Registries factory the
  ExperimentsService can use in place of :func:`load_default_registries`
  so the smoke test compiles without external technique-pool seeding.
"""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal

import yaml
from smai_core import (
    DslDocument,
    DslDocumentAdapter,
    Registries,
    TechniqueRef,
    load_default_registries,
)
from smai_core.plugins import (
    ArtifactNotFound,
    ArtifactStoreCapabilities,
    CacheConfig,
    ComputeCapabilities,
    JobHandle,
    JobStatus,
    LlmCapabilities,
    ModelResponse,
    NormalizedMessage,
    TextContent,
    TokenUsage,
    ToolDefinition,
)

# === StubLlmProvider =========================================================


class StubLlmProvider:
    """Deterministic :class:`LlmProvider` for CLI tests."""

    def __init__(
        self,
        responses: Sequence[ModelResponse] | None = None,
        *,
        capabilities: LlmCapabilities | None = None,
    ) -> None:
        self.name = "stub-cli"
        self.capabilities = capabilities or LlmCapabilities(
            supports_caching=True,
            context_window=200_000,
            max_output_tokens=4_096,
            supports_tool_use=True,
            model_id="stub-cli:test",
        )
        self.model_id = "stub-cli:test"
        self._responses: deque[ModelResponse] = deque(responses or [])
        self.calls: list[dict[str, Any]] = []

    async def call(
        self,
        system: str,
        messages: list[NormalizedMessage],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        cache_config: CacheConfig | None = None,
    ) -> ModelResponse:
        self.calls.append({"system": system, "messages_n": len(messages)})
        if not self._responses:
            # Fall through to a safe default for tests that don't drive
            # the agent loop (CLI tests that just need a constructable
            # provider).
            return ModelResponse(
                message=NormalizedMessage(
                    role="assistant", content=[TextContent(text="stub response")]
                ),
                stop_reason="end_turn" if False else "tool_use",
                usage=TokenUsage(input_tokens=1, output_tokens=1),
            )
        return self._responses.popleft()


# === FakeCompute =============================================================


class FakeCompute:
    """No-op :class:`Compute` adequate for CLI tests that don't drive
    the worker loop.

    Submitted jobs are accepted into a queue but never reported as
    completed; the CLI's submit / status round-trip doesn't poll
    Compute, so this stub never has its ``status`` queried in tests.
    """

    name: str = "fake-cli"
    capabilities: ComputeCapabilities = ComputeCapabilities(
        supports_gpu=False,
        max_timeout_seconds=3600,
        supports_log_streaming=False,
    )

    def __init__(self) -> None:
        self._handles: dict[str, JobHandle] = {}
        self._counter = 0

    async def submit(
        self,
        image: str,
        command: list[str],
        env: dict[str, str],
        gpu: bool = False,
        timeout_seconds: int = 3600,
        **plugin_options: object,
    ) -> JobHandle:
        del image, command, env, gpu, timeout_seconds, plugin_options
        self._counter += 1
        handle = JobHandle(plugin=self.name, handle=f"fake-{self._counter}")
        self._handles[handle.handle] = handle
        return handle

    async def status(self, handle: JobHandle) -> JobStatus:
        del handle
        return JobStatus(
            state="running",
            exit_code=None,
            started_at=None,
            finished_at=None,
            failure_reason=None,
        )

    async def cancel(self, handle: JobHandle) -> None:
        self._handles.pop(handle.handle, None)

    async def logs(self, handle: JobHandle) -> str:
        del handle
        return ""


# === InMemoryArtifactStore ===================================================


class InMemoryArtifactStore:
    """Drop-in :class:`ArtifactStore` with an in-process dict backend."""

    def __init__(self) -> None:
        self.name = "inmem-cli-artifacts"
        self.capabilities = ArtifactStoreCapabilities(
            supports_presigned_urls=False,
            max_object_size_bytes=None,
        )
        self._data: dict[str, bytes] = {}

    async def put(self, key: str, data: bytes, content_type: str | None = None) -> None:
        del content_type
        self._data[key] = data

    async def get(self, key: str) -> bytes:
        if key not in self._data:
            raise ArtifactNotFound(key)
        return self._data[key]

    async def exists(self, key: str) -> bool:
        return key in self._data

    async def list(self, prefix: str) -> AsyncIterator[str]:
        keys = sorted(k for k in self._data if k.startswith(prefix))

        async def _gen() -> AsyncIterator[str]:
            for k in keys:
                yield k

        return _gen()

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def url_for(
        self,
        key: str,
        expires_in: int = 3600,
        method: Literal["GET", "PUT"] = "GET",
    ) -> str:
        del expires_in, method
        return f"inmem://{key}"


# === Registries with starter techniques =====================================


def make_registries_with_technique(technique_id: str = "tech_cutout") -> Registries:
    """Construct a :class:`Registries` with a single registered
    technique so the starter experiment compiles cleanly.
    """
    technique = TechniqueRef(
        id=technique_id,
        name="Cutout",
        description="Cutout regularization technique.",
        category="augmentation",
        compatible_factor_types=["additive"],
        standard=True,
        affects_extension_points=["train_transforms"],
    )
    base = load_default_registries(technique_registry={technique.id: technique})
    return base


# === Experiment YAML fixture ================================================

EXPERIMENT_YAML = """\
kind: experiment
experiment:
  id: cg_example
  hypothesis: "Cutout improves accuracy on CIFAR-10."
  factors:
    - name: augmentation
      type: additive
      description: "cutout on/off"
  controlled_conditions:
    dataset:
      name: cifar10
      split: train
      version: v1
    optimization:
      optimizer: sgd
      lr: 0.1
    seeds: [1, 2, 3]
  entries:
    - id: entry_baseline
      is_baseline: true
      level:
        factor: augmentation
        name: absent
    - id: entry_cutout
      is_baseline: false
      level:
        factor: augmentation
        name: cutout
        technique_id: tech_cutout
  validation:
    metric: { kind: atomic, ref: accuracy }
    direction: higher_is_better
    aggregation: { method: mean }
    comparison:
      rule: compare_to_baseline
      threshold: 0.01
    seed_count_required: 3
"""


def parse_experiment_yaml() -> DslDocument:
    """Convenience: parse :data:`EXPERIMENT_YAML` into a typed :class:`DslDocument`."""
    payload = yaml.safe_load(EXPERIMENT_YAML)
    return DslDocumentAdapter.validate_python(payload, context={"smai_mode": "dsl"})


__all__ = [
    "EXPERIMENT_YAML",
    "FakeCompute",
    "InMemoryArtifactStore",
    "StubLlmProvider",
    "make_registries_with_technique",
    "parse_experiment_yaml",
]
