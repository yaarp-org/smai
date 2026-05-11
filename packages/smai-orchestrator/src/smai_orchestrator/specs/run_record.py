""":class:`RunRecord` sub-state-machine pipeline-spec — Task 3.E3.

Per ``designs/smai/03-state-machine.md`` §3.9 / ``designs/smai/01-data-model.md``
§5.5 / DEC-034 #3. Lifts the per-(entry × seed) run lifecycle from the
Phase-2 inline approximation in :mod:`.cg_execution` into a peer
:class:`PipelineSpec` registered alongside the CG-execution spec at
worker startup.

Cross-spec coordination (per `05-orchestrator.md` §1.3 / §1.5):

* The CG-execution spec's ``running`` on-entry dispatch handler creates
  one :class:`RunRecord` per ``(entry × seed)`` tuple in ``pending``
  state and returns. It no longer submits :class:`Compute` jobs nor
  polls inline.
* This sub-spec's phase-2 discovery picks up ``pending`` runs via
  :meth:`MetadataStore.get_ready_to_run` and dispatches each through
  ``pending → submitted`` (write-first :meth:`Compute.submit`).
* Phase-1 polling against :meth:`Compute.status` fires the terminal
  edges from ``submitted`` (the engine's phase-1 mechanics — see
  ``05`` §3.1 / :func:`smai_orchestrator.engine.phase1_step` — only
  observe :class:`JobState`-terminal Compute states; the
  ``submitted → running`` edge in the §3.9 stub is a brief erratum
  flagged below and surfaced as an open item).
* Once every child :class:`RunRecord` reaches a terminal state, the
  CG-execution spec's ``running → evaluating`` gate (sibling-read
  pattern at :func:`._make_gate_runs_all_terminal_with_success` /
  :func:`._make_gate_runs_all_terminal_zero_success`) fires on the
  next CG-spec poll cycle, advancing the parent CG.

Spec ambiguities resolved
-------------------------

* **``submitted → running`` is unreachable under the current engine.**
  The §3.9 stub enumerates ``submitted → running`` ``fires_on=job_succeeded``
  with a gate that "reads :class:`Compute.status` for ``JobState.running``."
  Phase-1 (per `05` §3.1 / :func:`phase1_step`) only fires edges when
  :class:`Compute` reports a *terminal* state — when status returns
  ``"submitted"`` or ``"running"`` the engine early-returns
  ``Phase1Outcome(status="running")`` without evaluating any edges. The
  ``submitted → running`` transition therefore cannot be driven by the
  engine today. **Resolution.** The spec ships terminal edges directly
  from ``submitted`` (the in-progress state where phase-1 actually
  observes terminal :class:`JobStatus`) — ``submitted → succeeded``,
  ``submitted → inconclusive``, ``submitted → failed``. The ``running``
  state is declared (RunState literal includes it; future engine
  extensions may make it reachable via a ``fires_on=job_running``
  trigger) and carries mirror terminal edges so the spec is forward-
  compatible. Surfaced for supervisor adjudication; if the engine grows
  a ``job_running`` trigger, the ``submitted → running`` edge slots in
  cleanly without further spec changes.

* **`raw_metrics_artifact_key` not persisted.** The Phase-2 inline
  approximation wrote ``raw_metrics_artifact_key`` onto the
  :class:`RunRecord` after the run terminated. Phase-1's CAS
  transition (per :func:`phase1_step`) only zeros the handle field —
  the engine doesn't surface a "pass arbitrary fields from the gate
  body" seam, and §1.3 forbids gates from writing entity state. The
  metrics-artifact key is fully derived from
  ``(cg_id, entry_id, seed)`` via :data:`RUN_METRICS_KEY_TEMPLATE`,
  so the post-lift evaluator path computes it directly rather than
  reading it back off the run record. The
  :attr:`RunRecord.raw_metrics_artifact_key` field is left in place
  for historical compatibility but is now effectively unused — a
  follow-up cleanup can drop it once the CG-execution evaluator path
  has fully migrated to the derived shape (see
  :func:`._build_raw_metrics_for_cg`).

* **Lease semantics (DEC-035 #2)** are an engine-wide concern (`05`
  §3.5) that Task 3.G1 wires uniformly across every spec. The run sub-
  spec doesn't add its own lease wrapping — it inherits whatever the
  engine's dispatch wrapper applies once 3.G1 lands. The ``runs`` pool
  declaration matches the CG-execution spec's pool declaration so the
  worker's slot accounting aggregates correctly per `05` §3.4.

* **Run-level retry semantics** (`03` §3.9 open item) deferred. v1 has
  no auto-retry on runs; the brief doesn't introduce one either. A
  ``failed → pending`` edge with budget gating is left as a follow-up.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any, cast

from smai_core.artifacts import HarnessContract
from smai_core.plugins import (
    ArtifactNotFound,
    ArtifactStore,
    JobHandle,
)

from smai_orchestrator.engine.types import (
    ConcurrencyPool,
    DispatchAction,
    DispatchContext,
    DispatchOutcome,
    EdgeDef,
    GateContext,
    GateOutcome,
    SchedulingQueryRef,
    StateDef,
)
from smai_orchestrator.entities.tracking import RunRecord
from smai_orchestrator.runtime.registry import register_pipeline_spec
from smai_orchestrator.runtime.spec import PipelineSpec
from smai_orchestrator.specs.cg_execution import (
    HARNESS_CONTRACT_KEY_TEMPLATE,
    POOL_RUNS,
    RUN_METRICS_KEY_TEMPLATE,
    drain_to_list,
)

__all__ = [
    "RUN_RECORD_SPEC_NAME",
    "build_run_record_spec",
    "register_run_record_spec",
]


# === Constants ==============================================================

RUN_RECORD_SPEC_NAME = "smai_run_record"


# === Helpers ================================================================


def _read_metrics_payload(payload: bytes) -> dict[str, Any] | None:
    """Best-effort parse of a per-run ``metrics.json`` body.

    Returns the parsed dict on success or ``None`` on any parse
    failure / non-dict body. The caller treats ``None`` as
    "metrics unparseable" → :class:`RunState` ``inconclusive`` per
    `06-mechanical-evaluation.md` §1's "insufficient completed seeds"
    semantic (the failure reason here being "completed but emitted
    no usable metric").
    """
    try:
        body: object = json.loads(payload)
    except (ValueError, TypeError):
        return None
    if not isinstance(body, dict):
        return None
    return cast(dict[str, Any], body)


async def _metrics_artifact_present_and_parseable(
    *,
    artifact_store: ArtifactStore,
    metrics_key: str,
) -> bool:
    """Predicate for the ``submitted → succeeded`` gate.

    The job exited with :class:`JobState` ``succeeded``; we additionally
    require that the runtime emitted a parseable metrics body at the
    expected key. Anything else routes to :class:`RunState`
    ``inconclusive`` (`06-mechanical-evaluation.md` §1 — "the run
    completed but no usable metric was emitted").
    """
    try:
        payload = await artifact_store.get(metrics_key)
    except ArtifactNotFound:
        return False
    return _read_metrics_payload(payload) is not None


def _metrics_key_for_run(run: RunRecord) -> str:
    """Derive the canonical metrics-artifact key for a run record.

    Per the v1 path layout (`CLAUDE.md`'s S3 artifact layout) and
    :data:`RUN_METRICS_KEY_TEMPLATE` from :mod:`.cg_execution`. The key
    is fully derived from ``(cg_id, entry_id, seed)`` — no lookup on
    the run record is required, which lets gate bodies be read-only
    against entity state per `05` §1.3.
    """
    return RUN_METRICS_KEY_TEMPLATE.format(cg_id=run.cg_id, entry_id=run.entry_id, seed=run.seed)


# === Gate-rule callable factories ===========================================


def _make_gate_pending_ready() -> Callable[[GateContext], Awaitable[GateOutcome]]:
    """``pending → submitted`` dispatch_time gate.

    The phase-2 query :meth:`MetadataStore.get_ready_to_run` already
    filters to runs in ``pending`` state with a null
    ``compute_job_handle`` per `07` §5.6.3, so the gate body simply
    advances. The on-entry dispatch on ``submitted`` does the real work
    (Compute.submit + handle write).
    """

    async def _gate(ctx: GateContext) -> GateOutcome:
        del ctx  # the scheduling query already materialized the predicate
        return GateOutcome(advance=True, reason="run pending → submitted always-fire")

    return _gate


def _make_gate_run_succeeded_with_metrics() -> Callable[[GateContext], Awaitable[GateOutcome]]:
    """``submitted → succeeded`` (or ``running → succeeded``) job_succeeded gate.

    Phase-1 fires this when :class:`Compute.status` returns
    ``"succeeded"``. We additionally require the per-run metrics file
    to be present and parseable at the canonical key per `01` §5.5 /
    `06-mechanical-evaluation.md` §1 — a job that exited 0 but emitted
    no usable metric routes to :class:`RunState` ``inconclusive``
    instead, via the next edge in declaration order.
    """

    async def _gate(ctx: GateContext) -> GateOutcome:
        if ctx.job_outcome is None or ctx.job_outcome.state != "succeeded":
            return GateOutcome(advance=False, reason="run job not succeeded")
        run = await ctx.metadata_store.get_run(ctx.entity_id)
        if run is None:
            return GateOutcome(advance=False, reason=f"RunRecord {ctx.entity_id!r} not found")
        metrics_key = _metrics_key_for_run(run)
        if not await _metrics_artifact_present_and_parseable(
            artifact_store=ctx.artifact_store, metrics_key=metrics_key
        ):
            return GateOutcome(
                advance=False,
                reason=f"metrics artifact missing or unparseable at {metrics_key!r}",
            )
        return GateOutcome(advance=True, reason="run succeeded; metrics present and parseable")

    return _gate


def _make_gate_run_succeeded_no_metrics() -> Callable[[GateContext], Awaitable[GateOutcome]]:
    """``submitted → inconclusive`` (or ``running → inconclusive``) gate.

    Declared after the ``succeeded`` edge so a parseable metrics file
    short-circuits to ``succeeded``. Reaching this gate means
    :class:`Compute.status` was ``"succeeded"`` but the per-run
    metrics file was missing or unparseable — the
    :class:`RunState`-``inconclusive`` semantic per `01` §5.5 (terminal
    but distinct from ``failed``, which is a job-level error).
    """

    async def _gate(ctx: GateContext) -> GateOutcome:
        if ctx.job_outcome is None or ctx.job_outcome.state != "succeeded":
            return GateOutcome(advance=False, reason="run job not succeeded")
        run = await ctx.metadata_store.get_run(ctx.entity_id)
        if run is None:
            return GateOutcome(advance=False, reason=f"RunRecord {ctx.entity_id!r} not found")
        metrics_key = _metrics_key_for_run(run)
        if await _metrics_artifact_present_and_parseable(
            artifact_store=ctx.artifact_store, metrics_key=metrics_key
        ):
            return GateOutcome(advance=False, reason="metrics present; route to succeeded")
        return GateOutcome(
            advance=True,
            reason="run completed but emitted no usable metric (inconclusive)",
        )

    return _gate


def _make_gate_run_failed_terminal() -> Callable[[GateContext], Awaitable[GateOutcome]]:
    """``submitted → failed`` (or ``running → failed``) job_failed gate.

    Phase-1 fires this when :class:`Compute.status` returns ``failed``,
    ``cancelled``, or ``timeout`` (per
    :func:`smai_orchestrator.engine.phase1_step._failure_outcomes_to_trigger`).
    No retry budget at the run level (v1 carry-forward — `03` §3.9
    open item). Always advance to the terminal.
    """

    async def _gate(ctx: GateContext) -> GateOutcome:
        del ctx
        return GateOutcome(advance=True, reason="run job did not succeed; terminal")

    return _gate


# === Dispatch-handler factory ===============================================


async def _load_compute_requirements_gpu(artifact_store: ArtifactStore, cg_id: str) -> bool:
    """Read ``body.compute.gpu`` off the CG's persisted :class:`HarnessContract`.

    The contract artifact at
    ``comparison-groups/{cg_id}/harness/contract.json`` is the locked
    source of truth for the CG's compute requirements (`02` §7.4 / round-3
    friction (C)). Pre-Phase-1 contracts (no ``compute`` block in their
    JSON body) deserialize with :class:`ComputeRequirements` defaults
    (``gpu=True``), preserving historic dispatch behavior.

    If the artifact is missing — which would be a contract-flow bug, since
    the dispatcher only runs after the CG transitioned past ``draft`` —
    we return ``True`` rather than crash the dispatcher, so a stray run
    job still submits with the historical default while the bug is
    diagnosed.
    """
    key = HARNESS_CONTRACT_KEY_TEMPLATE.format(cg_id=cg_id)
    try:
        contract_bytes = await artifact_store.get(key)
    except ArtifactNotFound:
        return True
    contract = HarnessContract.model_validate_json(contract_bytes)
    return contract.body.compute.gpu


def _make_dispatch_run_compute_submit(
    *,
    image: str,
) -> Callable[[DispatchContext], Awaitable[DispatchOutcome]]:
    """``submitted`` on-entry dispatch — submits the per-(entry × seed)
    training job to :class:`Compute`.

    Engine surrounds this with write-first ordering (per
    :func:`smai_orchestrator.engine.run_dispatch`):

    1. CAS the run from ``pending`` → ``submitted``.
    2. Call this handler. The handler reads the run record from the
       store (the engine's :class:`DispatchContext` doesn't carry the
       loaded record, just the entity-id / version / state), constructs
       the runtime command, calls :meth:`Compute.submit`, and returns
       the handle in :attr:`DispatchOutcome.submitted_handles`.
    3. CAS the handle onto :attr:`RunRecord.compute_job_handle` via the
       engine's ``handle_field`` write.

    On :class:`Compute.submit` failure the handler raises; the engine
    forward-rolls-back the run to ``pending``. The next cycle's
    phase-2 discovery picks it up again. v1 has no run-level retry
    budget (`03` §3.9 open item) — re-discovery is unconditional;
    operators with budget concerns can add a deployment-specific gate.

    The ``gpu`` argument to :meth:`Compute.submit` is read from the
    CG's :class:`HarnessContract` artifact (see
    :func:`_load_compute_requirements_gpu`) so CPU-only experiments
    (e.g. methodology smoke runs, kNN comparisons, anything not GPU-
    bound, and the macOS-LocalGpu path which refuses ``gpu=True``)
    dispatch correctly. The contract is the locked source of truth;
    one artifact-store read per submit.
    """

    async def _dispatch(ctx: DispatchContext) -> DispatchOutcome:
        run = await ctx.metadata_store.get_run(ctx.entity_id)
        if run is None:
            return DispatchOutcome(error=f"RunRecord {ctx.entity_id!r} not found")

        gpu = await _load_compute_requirements_gpu(ctx.artifact_store, run.cg_id)
        metrics_key = _metrics_key_for_run(run)
        command = [
            "python",
            "-m",
            "smai_runtime.runner",
            "--cg-id",
            run.cg_id,
            "--entry-id",
            run.entry_id,
            "--seed",
            str(run.seed),
            "--metrics-key",
            metrics_key,
        ]
        env = {
            "SMAI_CG_ID": run.cg_id,
            "SMAI_ENTRY_ID": run.entry_id,
            "SMAI_SEED": str(run.seed),
            "SMAI_METRICS_KEY": metrics_key,
        }
        handle: JobHandle = await ctx.compute.submit(
            image=image,
            command=command,
            env=env,
            gpu=gpu,
        )
        return DispatchOutcome(submitted_handles=[handle])

    return _dispatch


# === Public factories ========================================================


def build_run_record_spec(
    *,
    runtime_image: str = "smai-runtime:dev",
) -> PipelineSpec:
    """Build the SMAI :class:`RunRecord` sub-state-machine
    :class:`PipelineSpec` (entity_kind=``"run"``).

    Args:
        runtime_image: Container image the per-(entry × seed) training
            jobs run inside. Defaults to ``smai-runtime:dev`` (Task 2.D1
            convention; matches :func:`build_cg_execution_spec`).

    Per `01` §5.5 the canonical :class:`RunState` literal is
    ``pending → submitted → running → succeeded | failed | inconclusive``;
    the spec declares all six states. The ``running`` state is currently
    unreachable under the present engine (phase-1 doesn't fire on
    non-terminal :class:`Compute` reports — see module docstring's
    "spec ambiguities resolved" note); its terminal mirror edges are
    declared so a future engine extension supporting a
    ``fires_on=job_running`` trigger slots in cleanly without further
    spec changes. Today the engine drives every run through
    ``pending → submitted → {succeeded | failed | inconclusive}``.
    """
    submit_dispatch = _make_dispatch_run_compute_submit(image=runtime_image)

    states: list[StateDef] = [
        StateDef(name="pending"),
        StateDef(
            name="submitted",
            on_entry_dispatch=DispatchAction(
                name="run.dispatch_compute_submit",
                handler=submit_dispatch,
                pool=POOL_RUNS,
                handle_field="compute_job_handle",
            ),
        ),
        StateDef(name="running"),
        StateDef(name="succeeded", is_terminal=True),
        StateDef(name="failed", is_terminal=True),
        StateDef(name="inconclusive", is_terminal=True),
    ]

    edges: list[EdgeDef] = [
        EdgeDef(
            name="run.pending → submitted",
            from_state="pending",
            target_state="submitted",
            gate_rule=_make_gate_pending_ready(),
            fires_on="dispatch_time",
        ),
        # Phase-1 success path — declared first so ``succeeded`` short-
        # circuits before ``inconclusive`` evaluates.
        EdgeDef(
            name="run.submitted → succeeded",
            from_state="submitted",
            target_state="succeeded",
            gate_rule=_make_gate_run_succeeded_with_metrics(),
            fires_on="job_succeeded",
        ),
        EdgeDef(
            name="run.submitted → inconclusive",
            from_state="submitted",
            target_state="inconclusive",
            gate_rule=_make_gate_run_succeeded_no_metrics(),
            fires_on="job_succeeded",
        ),
        EdgeDef(
            name="run.submitted → failed",
            from_state="submitted",
            target_state="failed",
            gate_rule=_make_gate_run_failed_terminal(),
            fires_on="job_failed",
        ),
        # Mirror edges from ``running`` for the future-engine-extension
        # path (see module docstring). Declaration-order parallels
        # ``submitted``: success, inconclusive, failed.
        EdgeDef(
            name="run.running → succeeded",
            from_state="running",
            target_state="succeeded",
            gate_rule=_make_gate_run_succeeded_with_metrics(),
            fires_on="job_succeeded",
        ),
        EdgeDef(
            name="run.running → inconclusive",
            from_state="running",
            target_state="inconclusive",
            gate_rule=_make_gate_run_succeeded_no_metrics(),
            fires_on="job_succeeded",
        ),
        EdgeDef(
            name="run.running → failed",
            from_state="running",
            target_state="failed",
            gate_rule=_make_gate_run_failed_terminal(),
            fires_on="job_failed",
        ),
    ]

    pools: list[ConcurrencyPool] = [
        # Mirror the CG-execution spec's pool set so worker slot
        # accounting aggregates across both per `05` §3.4. The run
        # sub-spec only consumes ``runs``; the others are declared for
        # consistency with the CG spec.
        ConcurrencyPool(name=POOL_RUNS, limit=4, priority=100),
    ]

    scheduling_queries: dict[str, SchedulingQueryRef] = {
        "pending": SchedulingQueryRef(
            name="get_ready_to_run",
            fn=drain_to_list("get_ready_to_run"),
        ),
        "submitted": SchedulingQueryRef(
            name="get_in_flight_runs",
            fn=drain_to_list("get_in_flight_runs"),
        ),
        # ``running`` is currently unreachable but declared for forward
        # compatibility — once the engine grows a ``job_running``
        # trigger this slot starts firing.
        "running": SchedulingQueryRef(
            name="get_in_flight_runs",
            fn=drain_to_list("get_in_flight_runs"),
        ),
    }

    return PipelineSpec(
        name=RUN_RECORD_SPEC_NAME,
        entity_kind="run",
        initial_state="pending",
        states=states,
        edges=edges,
        pools=pools,
        scheduling_queries=scheduling_queries,
    )


def register_run_record_spec(
    *,
    runtime_image: str = "smai-runtime:dev",
) -> PipelineSpec:
    """Construct and register the :class:`RunRecord` sub-spec.

    Mirrors the separation Task 3.E1 uses for the proposal-pipeline
    spec: a per-spec ``register_*`` helper so the supervisor can merge
    Phase-3 spec registrations into :func:`register_smai_specs` at
    commit time without touching the per-task brief.

    Returns the constructed spec for callers that want a direct
    reference (e.g., to project to :class:`EngineSpec` for the worker
    loop or to drive a single cycle in tests).
    """
    spec = build_run_record_spec(runtime_image=runtime_image)
    register_pipeline_spec(spec)
    return spec
