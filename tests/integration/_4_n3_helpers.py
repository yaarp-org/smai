"""Helpers for the Task 4.N3 end-to-end integration test.

Per the workspace's per-task fixture filename hygiene convention
(``--import-mode=importlib`` puts every package's tests/ on sys.path —
cross-package fixture filenames must not collide). ``_4_n3_*`` is the
Task 4.N3 prefix, distinct from ``_e2_*`` / ``_e3_*`` /
``_4_l1_*`` siblings.

Builders shipped here:

* :func:`wait_for_state_change` — drives an
  :class:`smai_events.EventBroker` subscriber forward until it receives
  a :class:`StateChangeEvent` matching the requested ``kind`` /
  ``id`` / ``target_state`` triple. Bounded by a timeout (default 30s)
  and surfaces the entity's last-known
  :class:`MetadataStore`-side state on TimeoutError so debugging
  "why didn't the entity reach X" is straightforward.
* :func:`make_n3_runtime_config` — :class:`RuntimeConfig` with a
  file-based SQLite URI (``run_worker=True`` requires shared-state
  across the worker + API connections — see the function's
  docstring), the dev-default plugin slot names, and a fast poll
  interval (1 second) so the canonical journey fits inside the
  ~30s wall-clock acceptance budget.
* :class:`StubLlmProvider` / :class:`FakeAlwaysSucceededCompute` —
  deterministic plugin shims; the LLM stub queues a response per role,
  the compute stub reports succeeded immediately so the worker's
  phase-1 advances on the next cycle.
* :func:`make_planner_responses_novel_technique` — canned planner
  ``ModelResponse`` sequence driving the novel-technique-variant planner
  through ``set_classification → draft_create_technique →
  draft_comparison → set_conditions → draft_assertion → finalize_plan
  → finish``. Mirrors the proposal-spec test pattern + reproduce-paper
  workflow test pattern.
* :func:`make_canned_review_pass` /
  :func:`make_canned_contextual_promising` — canned LLM responses
  consumed by the code-reviewer + contextual-evaluator agent dispatches.
* :func:`pre_stage_cg_artifacts` — writes the manifest, harness file
  set, per-entry validation_results, treatment technique source, and
  per-(entry, seed=0) metrics under the deterministic CG namespace
  ahead of the worker's harness-build cycle. The manifest's
  ``parent_harness_contract_hash`` is a placeholder string — the
  orchestrator's manifest-fanout gate body only requires
  ``manifest.content_hash`` to be non-empty (which
  :func:`smai_runtime.freeze_manifest` populates) and does NOT
  cross-validate against the persisted harness contract.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from smai_core.plugins import (
    CacheConfig,
    ComputeCapabilities,
    JobHandle,
    JobStatus,
    LlmCapabilities,
    ModelResponse,
    NormalizedContent,
    NormalizedMessage,
    StopReason,
    TextContent,
    TokenUsage,
    ToolDefinition,
    ToolUseContent,
)
from smai_events import EnvelopedEvent
from smai_orchestrator import (
    PluginSelection,
    RuntimeConfig,
)
from smai_orchestrator.engine.config import EngineConfig
from smai_orchestrator.specs import (
    HARNESS_MANIFEST_KEY_TEMPLATE,
    HARNESS_VALIDATION_KEY_TEMPLATE,
    RUN_METRICS_KEY_TEMPLATE,
    TECHNIQUE_VALIDATION_KEY_TEMPLATE,
)
from smai_runtime import (
    MANIFEST_SCHEMA_VERSION,
    RUNTIME_TEMPLATE_VERSION,
    HarnessAPIManifest,
    HarnessExtensionPoint,
    compute_harness_version_hash,
    freeze_manifest,
)

# === Test-config shape =======================================================


def make_n3_runtime_config(
    *,
    poll_interval_seconds: int = 1,
    sqlite_path: str | None = None,
) -> RuntimeConfig:
    """Build a :class:`RuntimeConfig` for the N3 E2E test.

    Mirrors :func:`tests.integration._e2_integration_fakes.build_smoke_runtime_config_for_papers`
    with three adjustments:

    * ``poll_interval_seconds`` is a parameter (default 1s) so the
      ``run_worker=True`` driving completes the full canonical journey
      within the brief's ~30s wall-clock budget. Dev defaults' 10s
      poll would push the journey toward 50s.
    * ``supervisor_enabled=False`` keeps the worker loop deterministic
      across cycles (no supervisor tick interleaving with the entity
      dispatches we're driving forward).
    * ``sqlite_path`` is required (no in-memory default) — the
      ``run_worker=True`` mode runs the worker async tasks concurrently
      with the API request handlers, which acquire connections in
      parallel. SQLAlchemy's default ``AsyncAdaptedQueuePool`` for
      ``:memory:`` SQLite gives each connection its own per-connection
      database, and writes from one connection are invisible to
      another. A file-based SQLite path forces a single shared
      database file across the connection pool — the writes the
      ``ProposalsService.submit`` connection commits are visible to
      the next connection's ``ProposalsService.get`` read.

    All five SMAI pipeline-spec names appear in ``pipelines`` so
    :meth:`Runtime.start_in_band` registers proposal + paper +
    cg-execution + entry + run-record specs.
    """
    if sqlite_path is None:
        raise ValueError(
            "make_n3_runtime_config requires sqlite_path (file-based SQLite); "
            "the in-memory default does not share state across connections "
            "with run_worker=True. See the docstring."
        )
    return RuntimeConfig(
        engine=EngineConfig(
            poll_interval_seconds=poll_interval_seconds,
            supervisor_enabled=False,
        ),
        plugins=PluginSelection(
            llm_provider="bedrock",
            metadata_store="sqlite",
            artifact_store="localfs",
            compute="localgpu",
            metadata_store_config={"uri": f"sqlite+aiosqlite:///{sqlite_path}"},
        ),
        pipelines=[
            "smai_cg_execution",
            "smai_cg_entries",
            "smai_proposal_pipeline",
            "smai_paper_ingestion",
            "smai_run_record",
        ],
    )


# === StubLlmProvider =========================================================


class StubLlmProvider:
    """Deterministic :class:`LlmProvider` for the N3 E2E test.

    Queue-driven canned-response provider. Workspace-local copy of the
    pattern shared across test trees (``packages/smai-cli/tests/_cli_fakes.py``,
    ``packages/smai-orchestrator/tests/specs/_specs_fakes.py``,
    ``tests/integration/_e2_integration_fakes.py``); kept here so the
    integration tree doesn't pull a sibling tests/ directory onto
    ``sys.path``.
    """

    def __init__(
        self,
        responses: Sequence[ModelResponse] | None = None,
        *,
        capabilities: LlmCapabilities | None = None,
        name: str = "stub-n3",
    ) -> None:
        self.name = name
        self.capabilities = capabilities or LlmCapabilities(
            supports_caching=True,
            context_window=200_000,
            max_output_tokens=4_096,
            supports_tool_use=True,
            model_id=f"{name}:test",
        )
        self.model_id = f"{name}:test"
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
        del tools, max_tokens, temperature, cache_config
        self.calls.append({"system": system, "messages_n": len(messages)})
        if not self._responses:
            raise AssertionError(f"StubLlmProvider {self.name!r}: ran out of canned responses")
        return self._responses.popleft()


# === FakeCompute (always-succeeded variant) ==================================


class FakeAlwaysSucceededCompute:
    """:class:`Compute` shim whose :meth:`status` always reports ``succeeded``.

    Mirrors :class:`tests.integration.test_smoke_e2e.SmokeFakeCompute` so
    the worker's phase-1 path observes the success-edge predicate the
    cycle after a dispatch handler submits. The artifact side effects
    that a real harness-builder / technique-implementer container
    would write to ArtifactStore are pre-staged via
    :func:`pre_stage_cg_artifacts` ahead of the registration cycle.
    """

    name: str = "n3-fake-compute"
    capabilities: ComputeCapabilities = ComputeCapabilities(
        supports_gpu=False,
        max_timeout_seconds=3600,
        supports_log_streaming=False,
    )

    def __init__(self) -> None:
        self.submitted: list[dict[str, Any]] = []
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
        del timeout_seconds, plugin_options
        self._counter += 1
        handle = JobHandle(plugin=self.name, handle=f"n3-{self._counter}")
        self.submitted.append(
            {
                "image": image,
                "command": list(command),
                "env": dict(env),
                "gpu": gpu,
                "handle": handle.handle,
            }
        )
        return handle

    async def status(self, handle: JobHandle) -> JobStatus:
        del handle
        now = datetime.now(UTC).isoformat()
        return JobStatus(
            state="succeeded",
            exit_code=0,
            started_at=now,
            finished_at=now,
            failure_reason=None,
        )

    async def cancel(self, handle: JobHandle) -> None:
        del handle

    async def logs(self, handle: JobHandle) -> str:
        del handle
        return ""


# === Canned LLM-response builders ============================================


def _model_response(
    *,
    tool_uses: list[tuple[str, str, dict[str, Any]]] | None = None,
    text: str | None = None,
    stop_reason: StopReason = "tool_use",
) -> ModelResponse:
    content: list[NormalizedContent] = []
    if text is not None:
        content.append(TextContent(text=text))
    if tool_uses is not None:
        for tu_id, tu_name, tu_input in tool_uses:
            content.append(ToolUseContent(id=tu_id, name=tu_name, input=dict(tu_input)))
    return ModelResponse(
        message=NormalizedMessage(role="assistant", content=content),
        stop_reason=stop_reason,
        usage=TokenUsage(input_tokens=1, output_tokens=1),
    )


@dataclass(frozen=True)
class N3ProposalShape:
    """Deterministic id triple defining the N3 proposal + CG shape.

    The planner canned-response sequence emits these literally; the
    pre-staging helper writes artifacts under the resulting CG
    namespace. Stored as a frozen dataclass so the test reads each
    field by name (``shape.cg_id``, ``shape.treatment_entry_id``)
    without mistyping.

    The final CG id follows the proposal-spec's default
    ``cg_id_for`` resolver: ``f"{proposal_id}--{draft_cg_id}"``.
    """

    proposal_id: str
    draft_cg_id: str
    baseline_entry_id: str
    treatment_entry_id: str
    technique_symbolic_name: str

    @property
    def cg_id(self) -> str:
        return f"{self.proposal_id}--{self.draft_cg_id}"


def default_n3_shape() -> N3ProposalShape:
    return N3ProposalShape(
        proposal_id="prop-n3-test-001",
        draft_cg_id="cg-n3-test",
        baseline_entry_id="entry-n3-baseline",
        treatment_entry_id="entry-n3-treatment",
        technique_symbolic_name="tech_n3_treatment",
    )


def make_planner_responses_novel_technique(*, shape: N3ProposalShape) -> list[ModelResponse]:
    """Canned response sequence for the novel-technique-variant planner.

    Drives the planner agent through one CG: classification (additive
    augmentation factor), one technique definition, one comparison
    with baseline + treatment, controlled conditions, validation
    assertion, finalize, finish. Same shape as the reproduce-paper
    workflow test's ``_make_reproduce_paper_proposal_responses``;
    re-authored locally so the N3 test doesn't reach across into
    ``test_reproduce_paper_workflow.py``'s private helper.
    """
    factor_dim = "augmentation"
    factor_type = "additive"
    return [
        _model_response(
            tool_uses=[
                (
                    "tu-classify",
                    "set_classification",
                    {
                        "factor_dimension": factor_dim,
                        "factor_type": factor_type,
                        "rationale": "novel-technique E2E test classification",
                    },
                )
            ],
        ),
        _model_response(
            tool_uses=[
                (
                    "tu-create",
                    "draft_create_technique",
                    {
                        "symbolic_name": shape.technique_symbolic_name,
                        "name": "N3 E2E Treatment",
                        "description": "Novel technique placeholder for the N3 E2E test.",
                        "category": factor_dim,
                        "compatible_factor_types": [factor_type],
                        "standard": False,
                        "fidelity_anchor": {
                            "kind": "proposal",
                            "proposal_id": shape.proposal_id,
                        },
                        "affects_extension_points": ["train_transforms"],
                    },
                )
            ],
        ),
        _model_response(
            tool_uses=[
                (
                    "tu-comparison",
                    "draft_comparison",
                    {
                        "id": shape.draft_cg_id,
                        "hypothesis": "N3 E2E hypothesis",
                        "factor_dimension": factor_dim,
                        "factor_type": factor_type,
                        "factor_description": "N3 E2E factor description",
                        "entries": [
                            {
                                "id": shape.baseline_entry_id,
                                "is_baseline": True,
                                "level": {
                                    "factor": factor_dim,
                                    "name": "baseline",
                                    "technique_symbolic_name": None,
                                },
                            },
                            {
                                "id": shape.treatment_entry_id,
                                "is_baseline": False,
                                "level": {
                                    "factor": factor_dim,
                                    "name": "treatment",
                                    "technique_symbolic_name": shape.technique_symbolic_name,
                                },
                            },
                        ],
                    },
                )
            ],
        ),
        _model_response(
            tool_uses=[
                (
                    "tu-conditions",
                    "set_conditions",
                    {
                        "cg_id": shape.draft_cg_id,
                        "conditions": {
                            "dataset": {"name": "cifar10", "split": "train", "version": "v1"},
                            "optimization": {"optimizer": "sgd", "lr": 0.1},
                            "seeds": [1],
                        },
                    },
                )
            ],
        ),
        _model_response(
            tool_uses=[
                (
                    "tu-assertion",
                    "draft_assertion",
                    {
                        "cg_id": shape.draft_cg_id,
                        "validation": {
                            "metric": {"ref": "accuracy", "kind": "atomic"},
                            "direction": "higher_is_better",
                            "aggregation_method": "mean",
                            "comparison_rule": "compare_to_baseline",
                            "threshold": 0.01,
                            "seed_count_required": 1,
                        },
                    },
                )
            ],
        ),
        _model_response(tool_uses=[("tu-finalize", "finalize_plan", {})]),
        _model_response(
            tool_uses=[
                (
                    "tu-finish",
                    "finish",
                    {"success": True, "summary": "n3 E2E planner complete"},
                )
            ],
        ),
    ]


def make_canned_review_pass() -> ModelResponse:
    """``submit_review`` tool-use with ``overall_pass=True`` (no findings).

    Consumed by the code-reviewer dispatch on the
    ``implemented → running`` gate path.
    """
    return _model_response(
        tool_uses=[
            (
                "review_call_1",
                "submit_review",
                {"overall_pass": True, "findings": []},
            )
        ],
    )


def make_canned_contextual_promising() -> ModelResponse:
    """``submit_evaluation`` tool-use with ``overall_verdict='promising'``.

    Consumed by the contextual-evaluator dispatch on the
    ``evaluating → complete`` gate path. Same shape as the smoke test's
    canned response.
    """
    return _model_response(
        tool_uses=[
            (
                "ctx_call_1",
                "submit_evaluation",
                {
                    "overall_verdict": "promising",
                    "summary": "n3 E2E contextual verdict",
                    "rankings": [],
                    "insights": ["n3 E2E insight"],
                    "limitations": [],
                    "suggested_follow_ups": [],
                },
            )
        ],
    )


# === Pre-staging =============================================================


HARNESS_INIT_PY = b"# placeholder harness package (Task 4.N3 E2E test).\n"
HARNESS_TRAINER_PY = b"def train_one_epoch():\n    pass\n"
TECHNIQUE_CODE_TEMPLATE = (
    "def apply(config):\n"
    "    # placeholder technique for the N3 E2E test.\n"
    "    return {{'train_transforms': []}}\n"
)


def _build_n3_manifest() -> HarnessAPIManifest:
    """Hand-rolled manifest matching the pre-staged harness file set.

    Mirrors :func:`tests.integration.test_smoke_e2e._build_smoke_manifest`.
    The ``parent_harness_contract_hash`` is a placeholder — the
    orchestrator's manifest-fanout gate body only checks
    ``manifest.content_hash`` is non-empty (set by
    :func:`freeze_manifest` from the manifest's own JSON).

    The placeholder is sound for an in-process E2E test because the
    only check that would catch a mismatch lives inside
    :func:`smai_runtime.runner` (which runs INSIDE a real Compute
    container — bypassed by :class:`FakeAlwaysSucceededCompute`).
    """
    files: dict[str, bytes] = {
        "__init__.py": HARNESS_INIT_PY,
        "trainer.py": HARNESS_TRAINER_PY,
    }
    manifest = HarnessAPIManifest(
        extension_points=[
            HarnessExtensionPoint(
                key="train_transforms",
                type_signature="list[Callable[[Tensor], Tensor]]",
                purpose="optional training-set transforms (n3 e2e fixture)",
                optional=True,
                integration_pattern="append",
            ),
        ],
        integration_pattern_summary="n3 e2e fixture",
        harness_version_hash=compute_harness_version_hash(files),
        # Placeholder string — see method docstring. Validated as
        # non-empty by :class:`HarnessAPIManifest`'s Pydantic shape;
        # not cross-validated against the contract by the gates.
        parent_harness_contract_hash="placeholder-n3-contract-hash",
        manifest_schema_version=MANIFEST_SCHEMA_VERSION,
        runtime_template_version=RUNTIME_TEMPLATE_VERSION,
    )
    return freeze_manifest(manifest)


async def pre_stage_cg_artifacts(
    *,
    artifact_store: Any,
    shape: N3ProposalShape,
    baseline_accuracy: float = 0.50,
    treatment_accuracy: float = 0.85,
) -> None:
    """Pre-stage the artifact side effects normally written by the
    harness-builder, technique-implementer, and runtime containers.

    Writes:

    * ``harness/__init__.py``, ``harness/trainer.py`` — the file set
      whose hash matches the manifest's ``harness_version_hash``.
    * ``harness/manifest.json`` — frozen manifest with non-empty
      ``content_hash`` so the manifest-fanout gate body advances
      ``implementing → implemented``.
    * ``harness/validation_results.json`` — ``{"passed": true}`` —
      the harness-validation gate predicate.
    * ``entries/{baseline_entry_id}/code/validation_results.json`` —
      ``{"passed": true}``. The additive baseline starts in
      ``implemented`` per ``_register_buffer``'s F2 lift but the
      orchestrator's per-entry validation gate still reads the file
      defensively.
    * ``entries/{treatment_entry_id}/code/validation_results.json`` —
      ``{"passed": true}`` — the per-entry validation gate predicate.
    * ``entries/{treatment_entry_id}/code/techniques/{technique_id}.py``
      — the technique source the code-reviewer reads.
    * ``runs/{baseline_entry_id}/0/metrics.json`` +
      ``runs/{treatment_entry_id}/0/metrics.json`` — the runs
      dispatch's ``seeds=(0,)`` default produces one run per entry;
      the run-record spec advances each to ``completed`` once
      ``metrics.json`` parses, then the CG-spec advances
      ``running → evaluating``.

    Pre-stage timing: this helper can be called BEFORE submission.
    The CG record does not yet exist, so artifacts are orphaned in
    the store; once the proposal is approved + registered, the CG
    record materializes with the deterministic ``cg_id`` and the
    pre-staged artifacts are visible to the next worker cycle's
    gate bodies. Pre-staging upfront avoids the race between the
    registration cycle and the CG-spec's first dispatch.
    """
    cg_id = shape.cg_id
    manifest = _build_n3_manifest()

    # Harness file set.
    await artifact_store.put(
        f"comparison-groups/{cg_id}/harness/__init__.py",
        HARNESS_INIT_PY,
    )
    await artifact_store.put(
        f"comparison-groups/{cg_id}/harness/trainer.py",
        HARNESS_TRAINER_PY,
    )

    # Harness manifest + validation_results.
    await artifact_store.put(
        HARNESS_MANIFEST_KEY_TEMPLATE.format(cg_id=cg_id),
        manifest.model_dump_json().encode("utf-8"),
    )
    await artifact_store.put(
        HARNESS_VALIDATION_KEY_TEMPLATE.format(cg_id=cg_id),
        json.dumps({"passed": True}).encode("utf-8"),
    )

    # Per-entry validation_results + treatment technique source.
    for entry_id in (shape.baseline_entry_id, shape.treatment_entry_id):
        await artifact_store.put(
            TECHNIQUE_VALIDATION_KEY_TEMPLATE.format(cg_id=cg_id, entry_id=entry_id),
            json.dumps({"passed": True}).encode("utf-8"),
        )
    await artifact_store.put(
        (
            f"comparison-groups/{cg_id}/entries/{shape.treatment_entry_id}"
            f"/code/techniques/{shape.technique_symbolic_name}.py"
        ),
        TECHNIQUE_CODE_TEMPLATE.encode("utf-8"),
    )

    # Per-(entry, seed=0) metrics. Treatment intentionally above
    # baseline so the verdict comes out positive — though the test
    # only asserts journey traversal + artifact retrieval.
    for entry_id, accuracy in (
        (shape.baseline_entry_id, baseline_accuracy),
        (shape.treatment_entry_id, treatment_accuracy),
    ):
        await artifact_store.put(
            RUN_METRICS_KEY_TEMPLATE.format(cg_id=cg_id, entry_id=entry_id, seed=0),
            json.dumps({"accuracy": accuracy}).encode("utf-8"),
        )


# === Event-stream waiter =====================================================


async def wait_for_state_change(
    events: AsyncIterator[Any],
    *,
    kind: str,
    id: str,
    target_state: str,
    timeout: float = 30.0,
) -> None:
    """Drain an :class:`EventBroker` subscriber until a matching event arrives.

    Awaits ``async for event in events`` with an :func:`asyncio.wait_for`
    deadline. Each iteration rechecks the deadline; on timeout, raises
    :class:`TimeoutError` so debugging "why didn't the entity reach X"
    is bounded — the caller's outer ``try``/``except`` block reads the
    current :class:`MetadataStore`-side state for diagnostic context
    (the broker iterator is single-consumer, so we cannot peek without
    interfering with subsequent waits).

    Matching predicate: the event is an :class:`EnvelopedEvent` carrying
    a :class:`StateChangeEvent` (per the broker's payload union) whose
    ``kind`` + ``id`` + ``to_state`` match the requested triple. The
    ``WorkerHeartbeatEvent`` payloads are skipped silently — heartbeats
    are noise here. Receipt of the overflow sentinel raises
    :class:`AssertionError` so a buffer-sizing regression fails loudly
    rather than hanging.
    """
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"timed out waiting for state_change(kind={kind!r}, id={id!r}, to={target_state!r})"
            )
        try:
            event = await asyncio.wait_for(_anext(events), timeout=remaining)
        except TimeoutError as exc:
            raise TimeoutError(
                f"timed out waiting for state_change(kind={kind!r}, id={id!r}, to={target_state!r})"
            ) from exc

        if not isinstance(event, EnvelopedEvent):
            # Overflow sentinel — fail loudly. The N3 test does not
            # produce enough event volume to overflow the per-subscriber
            # buffer (default 100); receipt indicates a buffer-sizing
            # bug or an unexpectedly slow consumer.
            raise AssertionError(
                "broker delivered overflow sentinel; the N3 test should "
                "not exceed the per-subscriber buffer."
            )
        payload = event.event
        # Heartbeats carry no kind / id; only state-change payloads can match.
        kind_attr = getattr(payload, "kind", None)
        id_attr = getattr(payload, "id", None)
        to_attr = getattr(payload, "to_state", None)
        if kind_attr == kind and id_attr == id and to_attr == target_state:
            return


async def _anext(it: AsyncIterator[Any]) -> Any:
    """Local ``anext`` helper.

    Typing as ``AsyncIterator[Any] → Any`` keeps the helper's signature
    independent of caller-side typing.
    """
    return await it.__anext__()


__all__ = [
    "FakeAlwaysSucceededCompute",
    "HARNESS_INIT_PY",
    "HARNESS_TRAINER_PY",
    "N3ProposalShape",
    "StubLlmProvider",
    "TECHNIQUE_CODE_TEMPLATE",
    "default_n3_shape",
    "make_canned_contextual_promising",
    "make_canned_review_pass",
    "make_n3_runtime_config",
    "make_planner_responses_novel_technique",
    "pre_stage_cg_artifacts",
    "wait_for_state_change",
]
