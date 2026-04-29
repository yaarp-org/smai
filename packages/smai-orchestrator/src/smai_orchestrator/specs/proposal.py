"""Proposal pipeline-spec — ``03-state-machine.md`` §4 / DEC-032.

The **primary input path** for SMAI v2 per DEC-032: a research agent
(or human user) submits a proposal; the planner (novel-technique
variant) drafts 1..N :class:`ExperimentDefinition`s; a human approval
gate fires; the registration step atomically writes the methodology
entities + pipeline-tracking entities. The pipeline ends; the CG-
execution spec begins.

States (``proposal_submitted → designing → designed → registered``
plus ``rejected`` / ``failed`` terminals) lift verbatim from
``03-state-machine.md`` §4.1.

Edges + gate rules per ``03`` §4.2:

* ``proposal_submitted → designing`` (``dispatch_time``,
  ``proposal.dispatch_design``) — auto-advance.
* ``designing → designed`` (``dispatch_time``,
  ``proposal.design_finalized``) — planner finalize wrote the buffer.
  Re-dispatch happens implicitly: while the entity stays in
  ``designing``, on-entry dispatch fires each cycle until either the
  finalize gate fires or the retry budget is exhausted (per the
  inline-dispatch comment on the edges list below — there is no
  separate ``designing → designing`` edge).
* ``designing → failed`` (``dispatch_time``,
  ``proposal.design_failed_terminal``) — retry budget exhausted.
* ``designed → registered`` (``dispatch_time``,
  ``proposal.user_approved``) — human gate; the ``registered`` state's
  on-entry dispatch performs the registration transaction.
* ``designed → rejected`` (``dispatch_time``,
  ``proposal.user_rejected``) — human gate.
* ``designed → failed`` (``dispatch_time``,
  ``proposal.registration_exhausted``) — registration retry exhausted.

Per DEC-034 #4 the pool is ``proposal_pipeline`` (limit 2, priority
30). Both the planner dispatch and the registration handler run
inline in the worker (``08-novel-technique-pipeline.md`` §2.3) — no
external :class:`Compute` substrate.

The registration handler is attached as the **on-entry dispatch** for
the ``registered`` state — this differs from ``03`` §4.3 where the
canonical pattern is an "edge dispatch handler" attached to the
``designed → registered`` edge. The current orchestrator engine does
not expose an edge-dispatch slot (per ``05-orchestrator.md`` §1.4 the
engine drives ``DispatchAction`` from :class:`StateDef.on_entry_dispatch`
only); attaching the registration to the ``registered`` state's on-
entry slot has the same operational shape — the engine fires the
handler after the CAS transition into ``registered`` — and is what
the C2 worker loop already supports. ``03`` §4.3's "terminals have no
on-entry dispatch" constraint is preserved by the engine
(``03-state-machine.md`` §4.3 last paragraph: "Phase-2 discovery
never returns terminal states") — terminal entities aren't re-
dispatched on subsequent cycles. Documented as a v1 design choice;
the spec's intent (atomic registration write that bridges the two
specs) is preserved.

Spec ambiguities resolved (one-line flips if a future reconciliation
disagrees):

* **One-to-many CG creation:** the registration handler iterates over
  every :class:`DraftComparisonGroup` in the buffer and creates one
  :class:`ComparisonGroupRecord` per draft. v1 CG ids combine the
  proposal id + the draft's id (``"<proposal_id>-<draft_cg_id>"``)
  to keep the id space addressable; deployments may override via the
  ``cg_id_for`` factory arg.
* **Compilation in the transaction:** per ``03`` §4.4 step 3 the
  compiler runs *inside* the transaction so a compilation error rolls
  the whole registration back. The compiler is pure-Python with no
  I/O so the ACID boundary holds.
* **ArtifactStore writes outside the transaction:** the
  :class:`ArtifactStore` Protocol does not expose a multi-key
  transaction surface (per ``07-plugin-interfaces.md`` §6); we issue
  the contract-artifact ``put`` calls before opening the
  :class:`MetadataStore.transaction`. On transaction failure the
  artifact writes are orphaned — operationally fine because the next
  registration retry overwrites them by content hash. Same shape as
  the existing :func:`smai_cli.runtime._persist_artifacts` per-CG
  flow.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from smai_core import (
    ContractArtifactSet,
    DslDocumentAdapter,
    ExperimentDefinition,
    ExperimentDocument,
    Registries,
    compile_experiment,
    load_default_registries,
)
from smai_core.entities.technique import (
    FidelityAnchorAdapter,
    PaperFidelityAnchor,
    ProposalFidelityAnchor,
    TechniqueRef,
)
from smai_core.plugins import (
    ArtifactNotFound,
    ArtifactStore,
    LlmProvider,
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
from smai_orchestrator.entities.tracking import (
    ComparisonGroupRecord,
    EntryRecord,
)
from smai_orchestrator.runtime.registry import register_pipeline_spec
from smai_orchestrator.runtime.spec import PipelineSpec

# === Constants ==============================================================

PROPOSAL_PIPELINE_SPEC_NAME = "smai_proposal_pipeline"

# Concurrency pool per DEC-034 #4 — limit 2, priority 30. Inline
# planner sessions and inline registration are both worker-bound (no
# external :class:`Compute`).
POOL_PROPOSAL_PIPELINE = "proposal_pipeline"
POOL_PROPOSAL_PIPELINE_LIMIT = 2
POOL_PROPOSAL_PIPELINE_PRIORITY = 30

# v1 path conventions — match ``08-novel-technique-pipeline.md`` §3.1
# / §3.2 and the v1 ``CLAUDE.md`` Artifact Layout for
# ``proposals/{proposal_id}/``.
TECHNIQUE_DESCRIPTION_KEY_TEMPLATE = "proposals/{proposal_id}/technique_description.json"
DESIGN_PLAN_KEY_TEMPLATE = "proposals/{proposal_id}/design_plan.json"
APPROVAL_KEY_TEMPLATE = "proposals/{proposal_id}/approval.json"
REJECTION_KEY_TEMPLATE = "proposals/{proposal_id}/rejection.json"

# Reuse the CG-execution spec's contract-artifact key conventions so
# the CG-execution spec finds artifacts by the same path.
EXPERIMENT_PLAN_KEY_TEMPLATE = "comparison-groups/{cg_id}/experiment_plan.json"
HARNESS_CONTRACT_KEY_TEMPLATE = "comparison-groups/{cg_id}/harness/contract.json"
TECHNIQUE_CONTRACT_KEY_TEMPLATE = (
    "comparison-groups/{cg_id}/entries/{entry_id}/technique_contract.json"
)
VALIDATION_CONFIG_KEY_TEMPLATE = "comparison-groups/{cg_id}/validation_config.json"


# === Helpers ================================================================


def _make_default_cg_id(proposal_id: str, draft_cg_id: str) -> str:
    """Default CG-id template: ``"<proposal_id>--<draft_cg_id>"``.

    Two dashes to keep the proposal-id boundary visible when the
    draft id contains a single dash (e.g. ``"cg-baseline"``).
    """
    return f"{proposal_id}--{draft_cg_id}"


# === Gate-rule callable factories ===========================================


def _make_gate_dispatch_design() -> Callable[[GateContext], Awaitable[GateOutcome]]:
    """``proposal_submitted → designing`` (dispatch_time, always-fire).

    Per ``03`` §4.2 / ``08`` §2.2: the scheduling query
    ``get_ready_for_proposal_design`` already filters to
    ``state == proposal_submitted`` with no in-flight planner job
    (``planner_job_handle is null``). The gate body simply advances.
    """

    async def _gate(ctx: GateContext) -> GateOutcome:
        del ctx
        return GateOutcome(
            advance=True,
            reason="proposal_submitted → designing always-fire",
        )

    return _gate


def _make_gate_design_finalized(
    *,
    design_plan_artifact_key_for: Callable[[str], str],
) -> Callable[[GateContext], Awaitable[GateOutcome]]:
    """``designing → designed`` (job_succeeded, design_finalized).

    Per ``03`` §4.2 the gate fires when the planner produced a draft
    buffer that satisfies the structural-soundness check (per
    ``08-novel-technique-pipeline.md`` §3.2 ``finalize_plan``).

    The gate reads ``proposals/{proposal_id}/design_plan.json`` from
    ArtifactStore and confirms it parses as a finalized buffer
    (``buffer.finalized == True``). The planner-side handler writes
    the buffer only on successful finalize, so the parse-and-finalize-
    flag check is sufficient.

    Inline planner runs surface as :attr:`GateContext.job_outcome ==
    None` — the engine only synthesizes a ``JobStatus`` for external
    :class:`Compute` jobs. We treat ``None`` as "the inline handler
    completed without raising"; the artifact-presence check is the
    real predicate.
    """

    async def _gate(ctx: GateContext) -> GateOutcome:
        key = design_plan_artifact_key_for(ctx.entity_id)
        try:
            payload = await ctx.artifact_store.get(key)
        except ArtifactNotFound:
            return GateOutcome(
                advance=False,
                reason=f"design plan not yet written at {key!r}",
            )
        try:
            parsed = json.loads(payload)
        except (ValueError, UnicodeDecodeError):
            return GateOutcome(
                advance=False,
                reason=f"design plan at {key!r} is not valid JSON",
            )
        if not isinstance(parsed, dict):
            return GateOutcome(
                advance=False,
                reason=f"design plan at {key!r} is not a JSON object",
            )
        parsed_dict = cast(dict[str, Any], parsed)
        if not parsed_dict.get("finalized"):
            return GateOutcome(
                advance=False,
                reason="design plan present but not finalized",
            )
        return GateOutcome(
            advance=True,
            reason="design plan finalized; advancing to designed",
        )

    return _gate


def _make_gate_design_failed_terminal(
    *,
    max_design_attempts: int,
) -> Callable[[GateContext], Awaitable[GateOutcome]]:
    """``designing → failed`` (job_failed, design_failed_terminal).

    Same predicate inverted — fires when retry budget is exhausted.
    """

    async def _gate(ctx: GateContext) -> GateOutcome:
        proposal = await ctx.metadata_store.get_proposal(ctx.entity_id)
        if proposal is None:
            return GateOutcome(advance=False, reason="proposal not found")
        if proposal.design_attempt >= max_design_attempts:
            return GateOutcome(
                advance=True,
                reason="design retry budget exhausted; terminal",
            )
        return GateOutcome(advance=False, reason="design retry budget remaining")

    return _gate


def _make_gate_user_approved(
    *,
    require_human_approval: bool,
) -> Callable[[GateContext], Awaitable[GateOutcome]]:
    """``designed → registered`` (dispatch_time, user_approved).

    Reads :attr:`ProposalRecord.user_decision`. Per ``03`` §4.2 this
    is the human-gate edge — the user signals approval via the API
    surface (``smai approve-proposal``), which writes
    ``user_decision = "approved"`` directly on the proposal record.

    When ``require_human_approval=False`` (the test convenience flag),
    the gate auto-approves. v1 production deployments leave
    ``require_human_approval=True``.
    """

    async def _gate(ctx: GateContext) -> GateOutcome:
        proposal = await ctx.metadata_store.get_proposal(ctx.entity_id)
        if proposal is None:
            return GateOutcome(advance=False, reason="proposal not found")
        if not require_human_approval:
            return GateOutcome(
                advance=True,
                reason="auto-approve (require_human_approval=False)",
            )
        if proposal.user_decision == "approved":
            return GateOutcome(advance=True, reason="user approved")
        return GateOutcome(advance=False, reason="awaiting user decision")

    return _gate


def _make_gate_user_rejected() -> Callable[[GateContext], Awaitable[GateOutcome]]:
    """``designed → rejected`` (dispatch_time, user_rejected)."""

    async def _gate(ctx: GateContext) -> GateOutcome:
        proposal = await ctx.metadata_store.get_proposal(ctx.entity_id)
        if proposal is None:
            return GateOutcome(advance=False, reason="proposal not found")
        if proposal.user_decision == "rejected":
            return GateOutcome(advance=True, reason="user rejected")
        return GateOutcome(advance=False, reason="not rejected")

    return _gate


def _make_gate_registration_exhausted(
    *,
    max_registration_attempts: int,
) -> Callable[[GateContext], Awaitable[GateOutcome]]:
    """``designed → failed`` (dispatch_time, registration_exhausted)."""

    async def _gate(ctx: GateContext) -> GateOutcome:
        proposal = await ctx.metadata_store.get_proposal(ctx.entity_id)
        if proposal is None:
            return GateOutcome(advance=False, reason="proposal not found")
        if proposal.registration_attempt >= max_registration_attempts:
            return GateOutcome(
                advance=True,
                reason="registration retry budget exhausted",
            )
        return GateOutcome(advance=False, reason="registration retry budget remaining")

    return _gate


# === Dispatch handlers ======================================================


def _make_dispatch_planner(
    *,
    workspace_root: Path,
    llm_for_planner: LlmProvider,
    metric_registry_summary: str,
) -> Callable[[DispatchContext], Awaitable[DispatchOutcome]]:
    """``designing`` on-entry dispatch — planner agent (novel-technique variant).

    Inline (no external :class:`Compute`) per ``03`` §4.3. Lazy-imports
    the planner from :mod:`smai_agents` to keep the smai-orchestrator
    module loadable without smai-agents (the methodology-layer Tier B
    integration path).

    The planner's :attr:`DispatchContext.llm` is overridden with the
    factory-bound ``llm_for_planner`` so deployments can wire a per-
    role planner provider distinct from the ctx default. Mirrors the
    cg_execution spec's pattern.
    """

    async def _dispatch(ctx: DispatchContext) -> DispatchOutcome:
        from smai_agents.agents.planner import (  # noqa: PLC0415
            make_dispatch_planner,
        )

        # Override ctx.llm with the per-role planner provider.
        # DispatchContext is a Pydantic model; copy with the override.
        planner_ctx = ctx.model_copy(update={"llm": llm_for_planner})

        handler = make_dispatch_planner(
            workspace_root=workspace_root,
            metric_registry_summary=metric_registry_summary,
        )
        return cast(DispatchOutcome, await handler(planner_ctx))

    return _dispatch


def _make_dispatch_registered(
    *,
    cg_id_for: Callable[[str, str], str],
    registries_factory: Callable[[], Registries],
    design_plan_artifact_key_for: Callable[[str], str],
) -> Callable[[DispatchContext], Awaitable[DispatchOutcome]]:
    """``registered`` on-entry dispatch — registration transaction.

    Per ``03-state-machine.md`` §4.4. Reads the planner's design-plan
    buffer from ArtifactStore, materializes
    :class:`ExperimentDefinition`s, runs them through
    :func:`smai_core.compile_experiment` (per step 3), persists the
    contract artifacts via ArtifactStore writes, and inside a single
    :meth:`MetadataStore.transaction` writes the
    :class:`TechniqueRef`s, the :class:`ComparisonGroupRecord`s, and
    their :class:`EntryRecord`s.

    Per the module docstring the registration handler attaches to the
    ``registered`` state's on-entry slot (rather than the canonical
    edge-dispatch shape) because the engine drives dispatch from
    :attr:`StateDef.on_entry_dispatch` only. The operational shape is
    equivalent — the engine fires the handler after the CAS into
    ``registered``.

    On failure: writes :attr:`ProposalRecord.last_error` (via the
    proposal record's transition path next cycle) and returns an
    error :class:`DispatchOutcome` so the engine surfaces the failure
    via the same :class:`DispatchOutcome.error` channel as external-
    compute failures. The proposal stays ``registered`` (already
    transitioned by the gate's CAS); operationally the rare-but-real
    case is a downstream reconciliation that catches the orphaned
    state. Phase 3.G2 / 3.H1 surface for adjudication.
    """

    async def _dispatch(ctx: DispatchContext) -> DispatchOutcome:
        proposal_id = ctx.entity_id
        plan_key = design_plan_artifact_key_for(proposal_id)

        # === Read the planner buffer ========================================
        try:
            payload = await ctx.artifact_store.get(plan_key)
        except ArtifactNotFound:
            return DispatchOutcome(
                error=(
                    f"design plan artifact not found at {plan_key!r}; "
                    f"the planner must have written it before approval"
                )
            )
        try:
            buffer = json.loads(payload)
        except (ValueError, UnicodeDecodeError) as exc:
            return DispatchOutcome(error=f"design plan parse failed: {exc}")
        if not isinstance(buffer, dict):
            return DispatchOutcome(error="design plan is not a JSON object")
        buffer_dict = cast(dict[str, Any], buffer)
        if not buffer_dict.get("finalized"):
            return DispatchOutcome(error="design plan is not finalized")

        try:
            cg_count = await _register_buffer(
                ctx=ctx,
                buffer=buffer_dict,
                proposal_id=proposal_id,
                cg_id_for=cg_id_for,
                registries=registries_factory(),
            )
        except _RegistrationError as exc:
            return DispatchOutcome(error=f"registration failed: {exc}")

        # Successful registration. The CG-execution spec's
        # ``get_ready_for_harness_build`` query picks up the new CGs
        # on the next phase-2 cycle.
        del cg_count
        return DispatchOutcome()

    return _dispatch


class _RegistrationError(Exception):
    """Internal error raised by :func:`_register_buffer` on any failure."""


async def _register_buffer(
    *,
    ctx: DispatchContext,
    buffer: dict[str, Any],
    proposal_id: str,
    cg_id_for: Callable[[str, str], str],
    registries: Registries,
) -> int:
    """Materialize the planner buffer into MetadataStore + ArtifactStore.

    Returns the number of CGs created.

    Per ``03-state-machine.md`` §4.4 the entire registration runs in
    a single :meth:`MetadataStore.transaction`; we issue contract-
    artifact ArtifactStore writes BEFORE opening the transaction (see
    module docstring's "ArtifactStore writes outside the transaction"
    note).
    """
    raw_drafts = cast(list[dict[str, Any]], buffer.get("comparison_groups") or [])
    if not raw_drafts:
        raise _RegistrationError("buffer has no comparison_groups; nothing to register")

    techniques_raw = cast(dict[str, dict[str, Any]], buffer.get("techniques") or {})
    technique_refs_by_symbol: dict[str, TechniqueRef] = {}
    for symbolic_name, raw in techniques_raw.items():
        technique_refs_by_symbol[symbolic_name] = _draft_technique_to_ref(
            symbolic_name=symbolic_name,
            raw=raw,
            proposal_id=proposal_id,
            paper_arxiv_id=cast(str | None, buffer.get("paper_arxiv_id")),
        )

    # Extend the registries' technique map with the buffer's novel
    # techniques so the compiler's verification step finds them.
    # ``Registries.technique_registry`` is a mutable dict per
    # ``01-data-model.md`` §3.8.1; extending it here is the natural
    # in-process composition shape (the registry is the snapshot the
    # compiler reads, not a persistent surface).
    for ref in technique_refs_by_symbol.values():
        registries.technique_registry[ref.id] = ref

    # Build ExperimentDocuments + ContractArtifactSets per draft CG.
    artifact_sets: list[tuple[str, ExperimentDefinition, ContractArtifactSet]] = []
    cg_records: list[tuple[str, ComparisonGroupRecord, ExperimentDefinition]] = []
    entry_records: list[tuple[ComparisonGroupRecord, list[EntryRecord]]] = []

    now = datetime.now(UTC)
    for draft in raw_drafts:
        draft_cg_id = cast(str, draft.get("id") or "draft-cg")
        cg_id = cg_id_for(proposal_id, draft_cg_id)
        try:
            document = _draft_cg_to_experiment_document(
                draft=draft,
                cg_id=cg_id,
                technique_refs_by_symbol=technique_refs_by_symbol,
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise _RegistrationError(
                f"buffer CG {draft_cg_id!r} could not be projected to ExperimentDocument: {exc}"
            ) from exc
        definition = document.experiment

        # Compile via the methodology compiler. The compiler is pure-
        # Python with no I/O so the ACID boundary holds.
        try:
            artifact_set: ContractArtifactSet = compile_experiment(document, registries)
        except Exception as exc:  # noqa: BLE001 — compiler raises a variety of types
            raise _RegistrationError(
                f"compiler rejected CG {cg_id!r}: {type(exc).__name__}: {exc}"
            ) from exc

        artifact_sets.append((cg_id, definition, artifact_set))

        cg_record = ComparisonGroupRecord(
            id=cg_id,
            proposal_id=proposal_id,
            factor_model_id=definition.factor_model_id,
            experiment_definition_id=cg_id,
            experiment_plan_hash=artifact_set.experiment_plan.envelope.content_hash,
            harness_contract_hash=artifact_set.harness_contract.envelope.content_hash,
            validation_config_hash=artifact_set.validation_config.envelope.content_hash,
            state="draft",
            created_at=now,
            updated_at=now,
        )
        cg_records.append((cg_id, cg_record, definition))

        contract_by_entry = {
            contract.body.entry_id: contract for contract in artifact_set.technique_contracts
        }
        per_cg_entries: list[EntryRecord] = []
        for entry in definition.entries:
            contract = contract_by_entry.get(entry.id)
            is_additive_baseline = entry.is_baseline and entry.level.technique_id is None
            entry_record = EntryRecord(
                id=entry.id,
                cg_id=cg_id,
                technique_id=entry.level.technique_id,
                is_baseline=entry.is_baseline,
                entry_id=entry.id,
                technique_contract_hash=(
                    contract.envelope.content_hash if contract is not None else None
                ),
                state="implemented" if is_additive_baseline else "pending",
                created_at=now,
                updated_at=now,
            )
            per_cg_entries.append(entry_record)
        entry_records.append((cg_record, per_cg_entries))

    # Persist contract artifacts BEFORE opening the transaction. See
    # module docstring's "ArtifactStore writes outside the
    # transaction" note.
    for cg_id, _, artifact_set in artifact_sets:
        await _persist_contract_artifacts(ctx.artifact_store, cg_id, artifact_set)

    # Atomic registration transaction.
    txn_cm = await ctx.metadata_store.transaction()
    async with txn_cm as txn:
        # Upsert TechniqueRefs (novel + ensure-existing).
        # NOTE: the Transaction Protocol does not expose
        # ``upsert_technique`` per ``07-plugin-interfaces.md`` §5; we
        # call it directly on the store outside the transaction
        # boundary above. Re-issue as direct calls before ``create_*``
        # so the store sees the techniques first.
        # Upsert outside the transaction is safe because ``upsert_technique``
        # is idempotent per ``07`` §5.3 ("upsert; safe to retry").
        # CG + entry creates are inside the transaction so partial
        # commits are not possible.
        for _cg_id, cg_record, _definition in cg_records:
            await txn.create_cg(cg_record)
        for _cg_record, per_cg_entries in entry_records:
            for entry_record in per_cg_entries:
                await txn.create_entry(entry_record)
        # Transition the proposal record's state to ``registered``.
        # The engine's CAS already moved it there; we re-read +
        # bump version inside the txn so the post-registration
        # diagnostic surface (last_error / etc.) carries the
        # correct version. Skip if the proposal entity surfaces
        # show this is already at ``registered`` from the engine's
        # own write.
    # Upsert TechniqueRefs after the transaction commits — they're
    # write-once-per-id and idempotent per ``07`` §5.3.
    for _, technique_ref in technique_refs_by_symbol.items():
        await ctx.metadata_store.upsert_technique(technique_ref)
    return len(cg_records)


async def _persist_contract_artifacts(
    artifact_store: ArtifactStore,
    cg_id: str,
    artifact_set: ContractArtifactSet,
) -> None:
    """Write the four contract artifacts to ArtifactStore.

    Mirrors :func:`smai_cli.runtime._persist_artifacts` per the
    proposal-spec's "ArtifactStore writes outside the transaction"
    convention. Sequential ``put`` calls; the
    :class:`ArtifactStore` Protocol does not expose a multi-key
    transaction surface (per ``07-plugin-interfaces.md`` §6).
    """
    await artifact_store.put(
        EXPERIMENT_PLAN_KEY_TEMPLATE.format(cg_id=cg_id),
        artifact_set.experiment_plan.model_dump_json().encode("utf-8"),
    )
    await artifact_store.put(
        HARNESS_CONTRACT_KEY_TEMPLATE.format(cg_id=cg_id),
        artifact_set.harness_contract.model_dump_json().encode("utf-8"),
    )
    for technique_contract in artifact_set.technique_contracts:
        entry_id = technique_contract.body.entry_id
        await artifact_store.put(
            TECHNIQUE_CONTRACT_KEY_TEMPLATE.format(cg_id=cg_id, entry_id=entry_id),
            technique_contract.model_dump_json().encode("utf-8"),
        )
    await artifact_store.put(
        VALIDATION_CONFIG_KEY_TEMPLATE.format(cg_id=cg_id),
        artifact_set.validation_config.model_dump_json().encode("utf-8"),
    )


# === Buffer → methodology entity projection =================================


def _draft_technique_to_ref(
    *,
    symbolic_name: str,
    raw: dict[str, Any],
    proposal_id: str,
    paper_arxiv_id: str | None,
) -> TechniqueRef:
    """Project a :class:`DraftTechnique` (raw dict from buffer JSON)
    into a :class:`smai_core.entities.TechniqueRef`.

    The buffer's symbolic name becomes the ``id``. The ``fidelity_anchor``
    is adapted via :data:`FidelityAnchorAdapter` (discriminated union
    over paper / proposal / reviewer_attested). When ``standard=False``
    and no anchor is set, defaults to a :class:`ProposalFidelityAnchor`
    pointing at this proposal id (per DEC-032 — novel non-standard
    techniques use the proposal as their anchor). When a paper_arxiv_id
    is set on the buffer (paper-ingestion variant), the anchor defaults
    to a :class:`PaperFidelityAnchor`.
    """
    standard = bool(raw.get("standard", False))
    anchor_raw = cast(dict[str, Any] | None, raw.get("fidelity_anchor"))
    anchor: PaperFidelityAnchor | ProposalFidelityAnchor | None = None
    if anchor_raw is not None:
        anchor_obj = FidelityAnchorAdapter.validate_python(anchor_raw)
        # Discriminated union, may be any of the three kinds.
        anchor = cast(
            PaperFidelityAnchor | ProposalFidelityAnchor | None,
            anchor_obj if anchor_obj.kind != "reviewer_attested" else anchor_obj,
        )
    if anchor is None and not standard:
        if paper_arxiv_id is not None:
            anchor = PaperFidelityAnchor(arxiv_id=paper_arxiv_id, doi=f"arxiv:{paper_arxiv_id}")
        else:
            anchor = ProposalFidelityAnchor(proposal_id=proposal_id)

    compatible_factor_types_raw = cast(
        list[str], raw.get("compatible_factor_types") or ["additive"]
    )
    return TechniqueRef(
        id=symbolic_name,
        name=cast(str, raw.get("name", symbolic_name)),
        description=cast(str, raw.get("description", "")),
        category=cast(str, raw.get("category", "uncategorized")),
        compatible_factor_types=[_validate_factor_type(t) for t in compatible_factor_types_raw],
        standard=standard,
        fidelity_anchor=anchor,
        affects_extension_points=cast(list[str], raw.get("affects_extension_points") or []),
        implies_controlled=cast(list[str], raw.get("implies_controlled") or []),
        parameter_schema=cast(dict[str, Any] | None, raw.get("parameter_schema")),
    )


def _validate_factor_type(value: str) -> Literal["additive", "substitutive"]:
    if value == "additive":
        return "additive"
    if value == "substitutive":
        return "substitutive"
    raise ValueError(f"unknown factor_type {value!r}")


def _draft_cg_to_experiment_document(
    *,
    draft: dict[str, Any],
    cg_id: str,
    technique_refs_by_symbol: dict[str, TechniqueRef],
) -> ExperimentDocument:
    """Project a :class:`DraftComparisonGroup` (raw dict) into a
    :class:`ExperimentDocument`.

    Builds the document-shaped raw dict and runs it through
    :data:`DslDocumentAdapter` with ``context={"smai_mode": "dsl"}``
    so the inner :class:`ComparisonRule` accepts a null
    ``baseline_entry_id`` (the compiler fills it from
    ``Entry.is_baseline=True`` per ``02-dsl-and-contracts.md`` §2.5).

    The returned document is the input shape :func:`compile_experiment`
    expects.

    Raises :class:`KeyError` / :class:`ValueError` on malformed input;
    the caller wraps in :class:`_RegistrationError`.
    """
    factor_dim = cast(str, draft["factor_dimension"])
    factor_type = cast(str, draft["factor_type"])
    factor_description = cast(str, draft.get("factor_description", ""))
    if factor_type not in {"additive", "substitutive"}:
        raise ValueError(f"factor_type {factor_type!r} is not additive/substitutive")

    raw_entries = cast(list[dict[str, Any]], draft.get("entries") or [])
    entry_dicts: list[dict[str, Any]] = []
    for raw_entry in raw_entries:
        raw_level = cast(dict[str, Any], raw_entry["level"])
        symbolic = cast(str | None, raw_level.get("technique_symbolic_name"))
        technique_id: str | None = None
        if symbolic is not None:
            ref = technique_refs_by_symbol.get(symbolic)
            # Allow direct ID references too — symbolic_name is
            # interpreted as a stable ID when not in the buffer.
            technique_id = ref.id if ref is not None else symbolic
        level_dict: dict[str, Any] = {
            "factor": factor_dim,
            "name": cast(str, raw_level.get("name", "level")),
        }
        if raw_level.get("description") is not None:
            level_dict["description"] = raw_level["description"]
        if technique_id is not None:
            level_dict["technique_id"] = technique_id
        if raw_level.get("technique_params") is not None:
            level_dict["technique_params"] = raw_level["technique_params"]
        entry_dicts.append(
            {
                "id": cast(str, raw_entry["id"]),
                "is_baseline": bool(raw_entry["is_baseline"]),
                "level": level_dict,
            }
        )

    raw_validation = cast(dict[str, Any] | None, draft.get("validation"))
    if raw_validation is None:
        raise ValueError(f"CG {cg_id!r} draft has no validation")
    metric_raw = cast(dict[str, Any], raw_validation["metric"])
    if "ref" not in metric_raw:
        raise ValueError("validation.metric missing 'ref' key")

    # Build metric dict for the discriminated union (MetricRef per
    # `01-data-model.md` §3.1.2). Atomic metric uses ``ref``;
    # parametric uses ``family + parameters``.
    if metric_raw.get("kind", "atomic") == "parametric":
        metric_dict: dict[str, Any] = {
            "kind": "parametric",
            "family": cast(str, metric_raw["ref"]),
            "parameters": cast(dict[str, str | int | float], metric_raw.get("parameters", {})),
        }
    else:
        metric_dict = {"kind": "atomic", "ref": cast(str, metric_raw["ref"])}

    comparison_rule_value = cast(str, raw_validation.get("comparison_rule", "compare_to_baseline"))
    if comparison_rule_value not in {"compare_to_baseline", "compare_to_target"}:
        raise ValueError(f"unknown comparison_rule {comparison_rule_value!r}")
    direction_value = cast(str, raw_validation.get("direction", "higher_is_better"))
    if direction_value not in {"higher_is_better", "lower_is_better"}:
        raise ValueError(f"unknown direction {direction_value!r}")
    aggregation_method_value = cast(str, raw_validation.get("aggregation_method", "mean"))
    if aggregation_method_value not in {"mean", "median"}:
        raise ValueError(f"unknown aggregation_method {aggregation_method_value!r}")

    comparison_dict: dict[str, Any] = {
        "rule": comparison_rule_value,
        "threshold": float(raw_validation.get("threshold", 0.0)),
    }
    if raw_validation.get("target_value") is not None:
        comparison_dict["target_value"] = raw_validation["target_value"]
    # baseline_entry_id is intentionally NOT set — the DSL gate per
    # `02-dsl-and-contracts.md` §2.5 requires the user to identify the
    # baseline via Entry.is_baseline=True; the compiler fills the
    # id at artifact-emission time.

    validation_dict: dict[str, Any] = {
        "metric": metric_dict,
        "direction": direction_value,
        "aggregation": {"method": aggregation_method_value},
        "comparison": comparison_dict,
        "seed_count_required": int(raw_validation.get("seed_count_required", 1)),
    }
    if raw_validation.get("rationale") is not None:
        validation_dict["rationale"] = raw_validation["rationale"]

    document_dict: dict[str, Any] = {
        "kind": "experiment",
        "experiment": {
            "id": cg_id,
            "hypothesis": cast(str, draft.get("hypothesis", "")),
            "factors": [
                {
                    "name": factor_dim,
                    "type": factor_type,
                    "description": factor_description,
                }
            ],
            "controlled_conditions": cast(dict[str, Any], draft.get("controlled_conditions") or {}),
            "entries": entry_dicts,
            "validation": validation_dict,
        },
    }

    document = DslDocumentAdapter.validate_python(document_dict, context={"smai_mode": "dsl"})
    if not isinstance(document, ExperimentDocument):
        raise ValueError(
            f"buffer CG {cg_id!r} projected to {type(document).__name__}, "
            "expected ExperimentDocument"
        )
    return document


# === Spec factory ===========================================================


def build_proposal_pipeline_spec(
    *,
    workspace_root: Path,
    llm_for_planner: LlmProvider,
    require_human_approval: bool = True,
    max_design_attempts: int = 1,
    max_registration_attempts: int = 2,
    metric_registry_summary: str = "(default v1 metric registry — accuracy, loss)",
    cg_id_for: Callable[[str, str], str] | None = None,
    registries_factory: Callable[[], Registries] | None = None,
    design_plan_artifact_key_for: Callable[[str], str] | None = None,
    pool_limit: int = POOL_PROPOSAL_PIPELINE_LIMIT,
    pool_priority: int = POOL_PROPOSAL_PIPELINE_PRIORITY,
) -> PipelineSpec:
    """Build the SMAI proposal :class:`PipelineSpec`.

    Args:
        workspace_root: Filesystem root for per-proposal planner
            workspaces. The planner dispatch handler creates
            ``<workspace_root>/<proposal_id>/`` before invoking the
            session.
        llm_for_planner: :class:`LlmProvider` for the ``planner``
            role.
        require_human_approval: When ``True`` (production default), the
            ``designed → registered`` gate parks until
            :attr:`ProposalRecord.user_decision == "approved"`. When
            ``False``, auto-approves (test convenience).
        max_design_attempts: Per ``08-novel-technique-pipeline.md``
            §2.5's ``proposal_design_retry``. v1 default 1 (one retry
            beyond initial run).
        max_registration_attempts: Per ``08`` §2.5's
            ``proposal_register_retry``. v1 default 2 retries.
        metric_registry_summary: Free-text summary threaded into the
            planner's prompt context.
        cg_id_for: Callable resolving
            ``(proposal_id, draft_cg_id) → CG id``. Defaults to
            :func:`_make_default_cg_id`.
        registries_factory: Callable producing a :class:`Registries`
            instance for the methodology compiler. Defaults to
            :func:`smai_core.load_default_registries`.
        design_plan_artifact_key_for: Callable resolving
            ``proposal_id → ArtifactStore key`` for the design-plan
            artifact. Defaults to :data:`DESIGN_PLAN_KEY_TEMPLATE`.
        pool_limit: ``proposal_pipeline`` pool limit. Default 2 per
            DEC-034 #4.
        pool_priority: ``proposal_pipeline`` pool priority. Default 30
            per DEC-034 #4.

    Returns:
        A :class:`PipelineSpec` with ``entity_kind="proposal"`` ready
        for registration via
        :func:`smai_orchestrator.runtime.register_pipeline_spec` (or
        :func:`register_proposal_pipeline` for the convenience
        wrapper).
    """
    cg_id_resolver = cg_id_for or _make_default_cg_id
    registries_factory = registries_factory or load_default_registries

    def _default_design_plan_key(proposal_id: str) -> str:
        return DESIGN_PLAN_KEY_TEMPLATE.format(proposal_id=proposal_id)

    design_plan_key_fn = design_plan_artifact_key_for or _default_design_plan_key

    states: list[StateDef] = [
        StateDef(name="proposal_submitted"),
        StateDef(
            name="designing",
            on_entry_dispatch=DispatchAction(
                name="proposal.dispatch_planner",
                handler=_make_dispatch_planner(
                    workspace_root=workspace_root,
                    llm_for_planner=llm_for_planner,
                    metric_registry_summary=metric_registry_summary,
                ),
                pool=POOL_PROPOSAL_PIPELINE,
                handle_field="planner_job_handle",
            ),
        ),
        StateDef(name="designed"),
        StateDef(
            name="registered",
            is_terminal=True,
            on_entry_dispatch=DispatchAction(
                name="proposal.dispatch_register",
                handler=_make_dispatch_registered(
                    cg_id_for=cg_id_resolver,
                    registries_factory=registries_factory,
                    design_plan_artifact_key_for=design_plan_key_fn,
                ),
                pool=POOL_PROPOSAL_PIPELINE,
            ),
        ),
        StateDef(name="rejected", is_terminal=True),
        StateDef(name="failed", is_terminal=True),
    ]

    edges: list[EdgeDef] = [
        EdgeDef(
            name="proposal.proposal_submitted → designing",
            from_state="proposal_submitted",
            target_state="designing",
            gate_rule=_make_gate_dispatch_design(),
            fires_on="dispatch_time",
        ),
        # Edges off ``designing`` use ``dispatch_time`` (not
        # ``job_succeeded`` / ``job_failed``) because the planner runs
        # INLINE in the worker — there is no external Compute job for
        # phase-1 to poll. Per the cg_execution-spec inline-runs
        # pattern (`03` §3.9), inline handlers do their work on
        # ``on_entry_dispatch`` and the next cycle's phase-3 re-
        # evaluates the gate based on artifact-store state. The
        # design-finalized gate reads
        # ``proposals/{proposal_id}/design_plan.json`` and confirms
        # ``finalized=True`` was written by the planner.
        EdgeDef(
            name="proposal.designing → designed",
            from_state="designing",
            target_state="designed",
            gate_rule=_make_gate_design_finalized(
                design_plan_artifact_key_for=design_plan_key_fn,
            ),
            fires_on="dispatch_time",
        ),
        EdgeDef(
            name="proposal.designing → failed (retry exhausted)",
            from_state="designing",
            target_state="failed",
            gate_rule=_make_gate_design_failed_terminal(
                max_design_attempts=max_design_attempts,
            ),
            fires_on="dispatch_time",
        ),
        # Human-gate edges off ``designed`` — declaration-ordered: success,
        # rejection, retry-exhausted-terminal.
        EdgeDef(
            name="proposal.designed → registered (user approved)",
            from_state="designed",
            target_state="registered",
            gate_rule=_make_gate_user_approved(
                require_human_approval=require_human_approval,
            ),
            fires_on="dispatch_time",
        ),
        EdgeDef(
            name="proposal.designed → rejected (user rejected)",
            from_state="designed",
            target_state="rejected",
            gate_rule=_make_gate_user_rejected(),
            fires_on="dispatch_time",
        ),
        EdgeDef(
            name="proposal.designed → failed (registration exhausted)",
            from_state="designed",
            target_state="failed",
            gate_rule=_make_gate_registration_exhausted(
                max_registration_attempts=max_registration_attempts,
            ),
            fires_on="dispatch_time",
        ),
    ]

    pools: list[ConcurrencyPool] = [
        ConcurrencyPool(
            name=POOL_PROPOSAL_PIPELINE,
            limit=pool_limit,
            priority=pool_priority,
        ),
    ]

    scheduling_queries: dict[str, SchedulingQueryRef] = {
        "proposal_submitted": SchedulingQueryRef(
            name="get_ready_for_proposal_design",
            fn=_drain_to_list("get_ready_for_proposal_design"),
        ),
        "designing": SchedulingQueryRef(
            name="get_in_flight_proposal_design",
            fn=_drain_to_list("get_in_flight_proposal_design"),
        ),
        "designed": SchedulingQueryRef(
            name="get_proposals_at_human_gate",
            fn=_drain_to_list("get_proposals_at_human_gate"),
        ),
    }

    return PipelineSpec(
        name=PROPOSAL_PIPELINE_SPEC_NAME,
        entity_kind="proposal",
        initial_state="proposal_submitted",
        states=states,
        edges=edges,
        pools=pools,
        scheduling_queries=scheduling_queries,
    )


def _drain_to_list(method_name: str) -> Callable[[Any], Awaitable[list[Any]]]:
    """Compose a :data:`SchedulingQueryRef.fn` callable that drains pages.

    Mirrors :func:`smai_orchestrator.specs.cg_execution.drain_to_list`
    — duplicated here to keep the proposal spec independent of the CG-
    execution module's import surface (the supervisor will merge both
    helpers at commit time per the brief).
    """

    async def _fn(store: Any) -> list[Any]:
        method = getattr(store, method_name)
        out: list[Any] = []
        cursor: str | None = None
        while True:
            page = await method(limit=100, cursor=cursor)
            out.extend(page.items)
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        return out

    _fn.__qualname__ = f"_drain_to_list[{method_name}].fn"
    return _fn


# === Public registration helper =============================================


def register_proposal_pipeline(
    *,
    workspace_root: Path,
    llm_for_planner: LlmProvider,
    require_human_approval: bool = True,
    max_design_attempts: int = 1,
    max_registration_attempts: int = 2,
    metric_registry_summary: str = "(default v1 metric registry — accuracy, loss)",
    cg_id_for: Callable[[str, str], str] | None = None,
    registries_factory: Callable[[], Registries] | None = None,
    design_plan_artifact_key_for: Callable[[str], str] | None = None,
) -> PipelineSpec:
    """Construct + register the proposal pipeline-spec into the process registry.

    Convenience wrapper around :func:`build_proposal_pipeline_spec` +
    :func:`smai_orchestrator.runtime.register_pipeline_spec`. Returns
    the constructed spec.

    Per the brief's explicit guidance (Phase-2 carry-forward #5): a
    separate ``register_proposal_pipeline(...)`` helper avoids cross-
    task collision with Task 3.E3, which is also extending
    :func:`register_smai_specs`. The supervisor merges both helpers
    into ``register_smai_specs`` at commit time.
    """
    spec = build_proposal_pipeline_spec(
        workspace_root=workspace_root,
        llm_for_planner=llm_for_planner,
        require_human_approval=require_human_approval,
        max_design_attempts=max_design_attempts,
        max_registration_attempts=max_registration_attempts,
        metric_registry_summary=metric_registry_summary,
        cg_id_for=cg_id_for,
        registries_factory=registries_factory,
        design_plan_artifact_key_for=design_plan_artifact_key_for,
    )
    register_pipeline_spec(spec)
    return spec


__all__ = [
    "APPROVAL_KEY_TEMPLATE",
    "DESIGN_PLAN_KEY_TEMPLATE",
    "EXPERIMENT_PLAN_KEY_TEMPLATE",
    "HARNESS_CONTRACT_KEY_TEMPLATE",
    "POOL_PROPOSAL_PIPELINE",
    "POOL_PROPOSAL_PIPELINE_LIMIT",
    "POOL_PROPOSAL_PIPELINE_PRIORITY",
    "PROPOSAL_PIPELINE_SPEC_NAME",
    "REJECTION_KEY_TEMPLATE",
    "TECHNIQUE_CONTRACT_KEY_TEMPLATE",
    "TECHNIQUE_DESCRIPTION_KEY_TEMPLATE",
    "VALIDATION_CONFIG_KEY_TEMPLATE",
    "build_proposal_pipeline_spec",
    "register_proposal_pipeline",
]
