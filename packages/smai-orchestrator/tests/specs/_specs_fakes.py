"""Shared fixtures + builders for Task 2.C4 spec tests.

Conventions per the workspace's pytest ``--import-mode=importlib`` +
``conftest.py`` discovery layout: every package's ``tests/`` is on
``sys.path`` once, so module names must be unique. ``_specs_fakes``
avoids collision with ``_helpers`` (engine), ``_b3_fakes`` /
``_agent_fakes`` (smai-agents), and ``_runtime_fakes`` (orchestrator
runtime tests).

Builders shipped here:

* :class:`StubLlmProvider` — duplicate of the smai-agents stub, kept
  here so spec tests don't pull the smai-agents test-tree onto path.
* :func:`make_harness_contract` / :func:`make_technique_contract` —
  smai-core artifact constructors with frozen content_hash.
* :func:`make_validation_config` / :func:`make_evaluation_metrics_payload` —
  ValidationConfig + RawMetrics-shaped JSON for the evaluator.
* :func:`make_review_response` / :func:`make_contextual_response` —
  canned :class:`ModelResponse` objects emitting structured tool_use
  bodies the gate trio + evaluation dispatch consume.
* :func:`stage_harness_artifacts` / :func:`stage_per_entry_artifacts` —
  side-effect helpers that pre-write harness/manifest/validation
  artifacts to an :class:`ArtifactStore` so tests can start CGs in
  ``implementing`` state without driving a real harness builder agent.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from smai_core import (
    ComputeRequirements,
    Factor,
    HarnessContract,
    HarnessContractBody,
    PaperFidelityAnchor,
    ProposalFidelityAnchor,
    TechniqueContract,
    TechniqueContractBody,
    freeze_with_hash,
)
from smai_core.artifacts._envelope import ArtifactEnvelope
from smai_core.artifacts.validation_config import ValidationConfig, ValidationConfigBody
from smai_core.entities.metric import AtomicMetricRef
from smai_core.entities.validation import (
    AggregationRule,
    ComparisonRule,
)
from smai_core.plugins import (
    ArtifactNotFound,
    ArtifactStoreCapabilities,
    CacheConfig,
    LlmCapabilities,
    ModelResponse,
    NormalizedMessage,
    StopReason,
    TextContent,
    TokenUsage,
    ToolDefinition,
    ToolUseContent,
)
from smai_orchestrator.entities.tracking import ComparisonGroupRecord, EntryRecord
from smai_runtime import (
    MANIFEST_SCHEMA_VERSION,
    RUNTIME_TEMPLATE_VERSION,
    HarnessAPIManifest,
    HarnessExtensionPoint,
    compute_harness_version_hash,
    freeze_manifest,
)

# === Stub LlmProvider ========================================================


class StubLlmProvider:
    """Deterministic :class:`LlmProvider` for spec tests.

    Mirrors smai-agents' ``StubLlmProvider`` (kept local so the smai-
    agents test-tree isn't required on ``sys.path``).
    """

    def __init__(
        self,
        responses: Sequence[ModelResponse],
        *,
        capabilities: LlmCapabilities | None = None,
    ) -> None:
        self.name = "stub"
        self.capabilities = capabilities or LlmCapabilities(
            supports_caching=True,
            context_window=200_000,
            max_output_tokens=4_096,
            supports_tool_use=True,
            model_id="stub:test",
        )
        self._responses: deque[ModelResponse] = deque(responses)
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
        if not self._responses:
            raise AssertionError("StubLlmProvider: ran out of canned responses")
        self.calls.append(
            {
                "system": system,
                "messages": [m.model_dump() for m in messages],
                "tools": [t.model_dump() for t in tools] if tools else None,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "cache_config": cache_config.model_dump() if cache_config else None,
            }
        )
        return self._responses.popleft()


# === In-memory ArtifactStore (when SqliteStore + LocalFsStore are unavailable) ===


class InMemoryArtifactStore:
    """Drop-in :class:`ArtifactStore` with an in-process dict backend.

    Used when tests don't need real :class:`LocalFsStore` semantics and
    can avoid disk I/O.
    """

    def __init__(self) -> None:
        self.name = "inmem-artifacts"
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
            for key in keys:
                yield key

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


# === Builders for Pydantic artifacts ========================================


def make_harness_contract(
    *,
    factor_type: Literal["additive", "substitutive"] = "additive",
    factor_name: str = "augmentation",
    parent_experiment_id: str = "exp-1",
    parent_experiment_hash: str = "exp-hash",
    gpu: bool = True,
) -> HarnessContract:
    """Build a frozen :class:`HarnessContract` with populated ``content_hash``.

    ``gpu`` sets ``body.compute.gpu`` (defaults to the pre-Phase-1
    ``True`` shape); pass ``False`` to exercise CPU-only dispatch.
    """
    envelope = ArtifactEnvelope(
        artifact_kind="harness_contract",
        artifact_id="hc-1",
        schema_version=1,
        compiler_version="0.0.0-test",
        parent_experiment_id=parent_experiment_id,
        registry_hashes={},
        surface_map={},
    )
    body = HarnessContractBody(
        parent_experiment_hash=parent_experiment_hash,
        factor=Factor(
            name=factor_name,
            type=factor_type,
            description="test factor",
        ),
        seeds=[42, 1337],
        fixed_variables=[],
        required_metrics=[
            AtomicMetricRef(ref="accuracy"),
        ],
        optional_telemetry=[],
        no_go_zones=["experiment.py", "techniques/__init__.py"],
        compute=ComputeRequirements(gpu=gpu),
    )
    contract = HarnessContract(envelope=envelope, body=body)
    return freeze_with_hash(contract)


def make_technique_contract(
    *,
    parent_harness_contract_hash: str,
    entry_id: str = "entry-1",
    technique_id: str | None = "tq-1",
    is_baseline: bool = False,
    standard: bool = False,
    fidelity_anchor_kind: str | None = "proposal",
) -> TechniqueContract:
    envelope = ArtifactEnvelope(
        artifact_kind="technique_contract",
        artifact_id=f"tc-{entry_id}",
        schema_version=1,
        compiler_version="0.0.0-test",
        parent_experiment_id="exp-1",
        registry_hashes={},
        surface_map={},
    )
    anchor: PaperFidelityAnchor | ProposalFidelityAnchor | None = None
    if fidelity_anchor_kind == "proposal":
        anchor = ProposalFidelityAnchor(proposal_id="prop-1")
    elif fidelity_anchor_kind == "paper":
        anchor = PaperFidelityAnchor(doi="10.1234/test", arxiv_id="2401.12345")
    body = TechniqueContractBody(
        entry_id=entry_id,
        parent_experiment_id="exp-1",
        parent_experiment_hash="exp-hash",
        parent_harness_contract_hash=parent_harness_contract_hash,
        technique_id=technique_id,
        technique_params=None,
        level_value=None,
        is_baseline=is_baseline,
        fidelity_anchor=anchor,
        standard=standard,
    )
    contract = TechniqueContract(envelope=envelope, body=body)
    return freeze_with_hash(contract)


def make_validation_config(
    *,
    parent_experiment_hash: str = "exp-hash",
    baseline_entry_id: str = "entry-baseline",
    seed_count_required: int = 1,
) -> ValidationConfig:
    envelope = ArtifactEnvelope(
        artifact_kind="validation_config",
        artifact_id="vc-1",
        schema_version=1,
        compiler_version="0.0.0-test",
        parent_experiment_id="exp-1",
        registry_hashes={},
        surface_map={},
    )
    body = ValidationConfigBody(
        parent_experiment_hash=parent_experiment_hash,
        metric=AtomicMetricRef(ref="accuracy"),
        direction="higher_is_better",
        aggregation=AggregationRule(method="mean"),
        comparison=ComparisonRule(
            rule="compare_to_baseline",
            baseline_entry_id=baseline_entry_id,
            threshold=0.0,
        ),
        seed_count_required=seed_count_required,
    )
    return freeze_with_hash(ValidationConfig(envelope=envelope, body=body))


def make_minimal_manifest(
    *,
    parent_harness_contract_hash: str,
    factor_type: Literal["additive", "substitutive"] = "additive",
) -> HarnessAPIManifest:
    """Build a frozen :class:`HarnessAPIManifest` with a known harness file set."""
    files: dict[str, bytes] = {
        "__init__.py": b"# placeholder harness package\n",
        "trainer.py": b"def train_one_epoch():\n    pass\n",
    }
    harness_version_hash = compute_harness_version_hash(files)
    if factor_type == "additive":
        ext_points = [
            HarnessExtensionPoint(
                key="train_transforms",
                type_signature="list[Callable[[Tensor], Tensor]]",
                purpose="optional training-set transforms",
                optional=True,
                integration_pattern="append",
            ),
        ]
    else:
        ext_points = [
            HarnessExtensionPoint(
                key="model_wrapper",
                type_signature="Callable[[nn.Module], nn.Module]",
                purpose="mandatory architecture slot",
                optional=False,
                integration_pattern="wrap",
            ),
        ]
    manifest = HarnessAPIManifest(
        extension_points=ext_points,
        integration_pattern_summary="test fixture",
        harness_version_hash=harness_version_hash,
        parent_harness_contract_hash=parent_harness_contract_hash,
        manifest_schema_version=MANIFEST_SCHEMA_VERSION,
        runtime_template_version=RUNTIME_TEMPLATE_VERSION,
    )
    return freeze_manifest(manifest)


# === ModelResponse builders =================================================


def model_response(
    *,
    text: str | None = None,
    tool_uses: list[tuple[str, str, dict[str, Any]]] | None = None,
    stop_reason: StopReason = "tool_use",
    input_tokens: int = 1,
    output_tokens: int = 1,
) -> ModelResponse:
    content: list[TextContent | ToolUseContent] = []
    if text is not None:
        content.append(TextContent(text=text))
    if tool_uses is not None:
        for tu_id, tu_name, tu_input in tool_uses:
            content.append(
                ToolUseContent(
                    id=tu_id,
                    name=tu_name,
                    input=dict(tu_input),
                )
            )
    return ModelResponse(
        message=NormalizedMessage(role="assistant", content=list(content)),
        stop_reason=stop_reason,
        usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def make_review_response(
    *,
    overall_pass: bool,
    findings: list[dict[str, Any]] | None = None,
    tool_name: str = "submit_review",
) -> ModelResponse:
    """Build a ModelResponse emitting a :class:`CodeReviewResult` tool_use.

    :class:`CodeReviewResult` enforces an invariant: ``overall_pass``
    equals ``not any(severity=='critical')`` — see the schema's model
    validator. Tests that pass ``overall_pass=False`` must include at
    least one critical finding; tests that pass ``overall_pass=True``
    must have zero critical findings.
    """
    payload: dict[str, Any] = {
        "overall_pass": overall_pass,
        "findings": findings or [],
    }
    return model_response(tool_uses=[("call_1", tool_name, payload)])


def make_contextual_response(
    *,
    overall_verdict: Literal["promising", "inconclusive", "not_promising"] = "promising",
    tool_name: str = "submit_evaluation",
) -> ModelResponse:
    """Build a ModelResponse emitting a :class:`ContextualVerdict` tool_use.

    The schema is rule-agnostic per DEC-031 #8; a minimal payload
    satisfies the validator.
    """
    payload: dict[str, Any] = {
        "overall_verdict": overall_verdict,
        "summary": "test contextual verdict",
        "rankings": [],
        "insights": ["test insight"],
        "limitations": [],
        "suggested_follow_ups": [],
    }
    return model_response(tool_uses=[("call_1", tool_name, payload)])


# === Record builders =========================================================


def make_cg(
    cg_id: str = "cg-1",
    *,
    state: str = "draft",
    code_review_attempt: int = 0,
) -> ComparisonGroupRecord:
    now = datetime(2026, 4, 28, tzinfo=UTC)
    return ComparisonGroupRecord(
        id=cg_id,
        proposal_id=f"prop_{cg_id}",
        experiment_definition_id=f"exp_{cg_id}",
        state=state,  # type: ignore[arg-type]
        version=0,
        code_review_attempt=code_review_attempt,
        created_at=now,
        updated_at=now,
    )


def make_entry(
    entry_id: str = "entry-1",
    *,
    cg_id: str = "cg-1",
    state: str = "pending",
    technique_id: str | None = "tq-1",
    is_baseline: bool = False,
    implementation_attempt: int = 0,
) -> EntryRecord:
    now = datetime(2026, 4, 28, tzinfo=UTC)
    return EntryRecord(
        id=entry_id,
        cg_id=cg_id,
        technique_id=technique_id,
        is_baseline=is_baseline,
        entry_id=entry_id,
        state=state,  # type: ignore[arg-type]
        implementation_attempt=implementation_attempt,
        version=0,
        created_at=now,
        updated_at=now,
    )


# === Stage helpers — pre-write artifacts to ArtifactStore ===================


@dataclass
class StagedHarnessArtifacts:
    """Bundle of artifacts the harness builder would produce; for tests
    that bypass the agent loop and start CGs in ``implementing``."""

    harness_contract: HarnessContract
    manifest: HarnessAPIManifest


async def stage_harness_artifacts(
    *,
    artifact_store: Any,
    cg_id: str,
    factor_type: Literal["additive", "substitutive"] = "additive",
    validation_passed: bool = True,
) -> StagedHarnessArtifacts:
    """Write harness/contract.json, harness/manifest.json,
    harness/validation_results.json, harness/__init__.py + harness/trainer.py.
    """
    contract = make_harness_contract(factor_type=factor_type)
    manifest = make_minimal_manifest(
        parent_harness_contract_hash=contract.envelope.content_hash,
        factor_type=factor_type,
    )
    await artifact_store.put(
        f"comparison-groups/{cg_id}/harness/contract.json",
        contract.model_dump_json().encode("utf-8"),
    )
    await artifact_store.put(
        f"comparison-groups/{cg_id}/harness/manifest.json",
        manifest.model_dump_json().encode("utf-8"),
    )
    await artifact_store.put(
        f"comparison-groups/{cg_id}/harness/validation_results.json",
        json.dumps({"passed": validation_passed}).encode("utf-8"),
    )
    await artifact_store.put(
        f"comparison-groups/{cg_id}/harness/__init__.py",
        b"# placeholder harness package\n",
    )
    await artifact_store.put(
        f"comparison-groups/{cg_id}/harness/trainer.py",
        b"def train_one_epoch():\n    pass\n",
    )
    return StagedHarnessArtifacts(harness_contract=contract, manifest=manifest)


async def stage_per_entry_artifacts(
    *,
    artifact_store: Any,
    cg_id: str,
    entry_id: str,
    technique_id: str | None,
    is_baseline: bool,
    parent_harness_contract_hash: str,
    validation_passed: bool = True,
    code: str | None = None,
) -> TechniqueContract:
    """Write technique_contract.json, code/validation_results.json, optional code."""
    contract = make_technique_contract(
        parent_harness_contract_hash=parent_harness_contract_hash,
        entry_id=entry_id,
        technique_id=technique_id,
        is_baseline=is_baseline,
        fidelity_anchor_kind=None if is_baseline else "proposal",
    )
    await artifact_store.put(
        f"comparison-groups/{cg_id}/entries/{entry_id}/technique_contract.json",
        contract.model_dump_json().encode("utf-8"),
    )
    await artifact_store.put(
        f"comparison-groups/{cg_id}/entries/{entry_id}/code/validation_results.json",
        json.dumps({"passed": validation_passed}).encode("utf-8"),
    )
    if technique_id is not None and code is not None:
        await artifact_store.put(
            f"comparison-groups/{cg_id}/entries/{entry_id}/code/techniques/{technique_id}.py",
            code.encode("utf-8"),
        )
    return contract


async def stage_validation_config(
    *,
    artifact_store: Any,
    cg_id: str,
    baseline_entry_id: str,
) -> ValidationConfig:
    cfg = make_validation_config(baseline_entry_id=baseline_entry_id)
    await artifact_store.put(
        f"comparison-groups/{cg_id}/validation_config.json",
        cfg.model_dump_json().encode("utf-8"),
    )
    return cfg


async def stage_run_metrics(
    *,
    artifact_store: Any,
    cg_id: str,
    entry_id: str,
    seed: int,
    accuracy: float = 0.85,
) -> str:
    """Write a metrics.json artifact for a (entry, seed). Returns the key."""
    key = f"comparison-groups/{cg_id}/runs/{entry_id}/{seed}/metrics.json"
    payload = {"accuracy": accuracy}
    await artifact_store.put(key, json.dumps(payload).encode("utf-8"))
    return key


__all__ = [
    "InMemoryArtifactStore",
    "StagedHarnessArtifacts",
    "StubLlmProvider",
    "make_cg",
    "make_contextual_response",
    "make_entry",
    "make_harness_contract",
    "make_minimal_manifest",
    "make_review_response",
    "make_technique_contract",
    "make_validation_config",
    "model_response",
    "stage_harness_artifacts",
    "stage_per_entry_artifacts",
    "stage_run_metrics",
    "stage_validation_config",
]
