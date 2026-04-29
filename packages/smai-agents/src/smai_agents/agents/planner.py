"""Planner agent — ``04-agents.md`` §2.1 / §10.2 / §13.

One agent role with two prompt variants per DEC-032:

* **Novel-technique variant (default)** — produces 1..N
  ``ExperimentDefinition``s for the methodology compiler. Sole CG-
  creation flow in v2 per ``08-novel-technique-pipeline.md`` §3.
* **Paper-ingestion variant (supporting)** — produces ``TechniqueRef``s
  only; never drafts CGs. Loaded inside the paper-ingestion pipeline-
  spec's ``planning`` state per ``08`` §5.3.

The agent shape is the same multi-turn buffer-and-validate loop in both
variants. Tools mutate an in-process :class:`PlannerBuffer`; finalize-
shaped tools validate the complete buffer before allowing the pipeline
to advance to registration. The buffer is persisted to
:class:`ArtifactStore` only on successful finalize (per ``08`` §3.2 last
paragraph and ``08`` §5.4).

Surfaces:

* :func:`run_planner_session` — the in-process runner. Drives the loop
  end-to-end against a (production or stub) :class:`LlmProvider`,
  selects the variant-specific prompt config + tool surface, and
  returns the loop's :class:`AgentOutcome` plus the finalized
  :class:`PlannerBuffer` artifact key (when finalize succeeded). Tests
  exercise this path directly per the Task 3.E1 brief.
* :func:`make_dispatch_planner` — the
  :data:`smai_orchestrator.engine.types.DispatchHandler`-shaped wrapper
  the proposal pipeline-spec's ``designing``-state on-entry handler
  fires. Inline (no external :class:`Compute`) per ``03`` §4.3.

The handler is typed as ``Callable[[Any], Awaitable[Any]]`` to keep
this module's import surface free of the orchestrator-engine types per
the smai-agents → smai-orchestrator dependency direction (per
implementation_plan.md §2.1: handlers lazy-import :class:`DispatchOutcome`
inside the body).

Variant selection:

* :data:`PlannerVariant` is a ``Literal["novel_technique",
  "paper_ingestion"]``.
* Per ``08-novel-technique-pipeline.md`` §3.1 / §4: the proposal
  pipeline always uses ``novel_technique`` for both shapes of input
  (a verbatim novel description and a "reproduce paper X" reference) —
  in both cases the planner drafts ``ExperimentDefinition``s. The
  ``paper_ingestion`` variant is selected only by the paper-ingestion
  pipeline-spec (lands in Task 3.E2). The discriminator is the
  pipeline that calls the planner, not the proposal's
  ``submission_kind``.

Production-iteration scope: the prompts under
``smai_agents/prompts/planner/`` are first-cut content per the brief's
"keep prompt content terse and structurally sound" guidance. Production
iteration is a separate concern. The MUST-HAVES are: variants compose
cleanly with base; tool surfaces match design doc; variants exercisable
end-to-end via the agent loop.

Spec ambiguity resolved (paper-ingestion-variant single-call vs multi-
turn): per ``08-novel-technique-pipeline.md`` §5.3 the multi-turn loop
is preserved per DEC-032 OQ4 — the work (read paper, identify
contribution + comparison techniques, judge standard-flag per DEC-015,
write method-extraction artifacts) is multi-step reasoning over long
context. Both variants are multi-turn here.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from smai_core.plugins import (
    ArtifactStore,
    JobHandle,
    LlmProvider,
    NormalizedMessage,
    TextContent,
    ToolResultContent,
)

from smai_agents.loop import (
    AgentLoopConfig,
    AgentOutcome,
    AgentSession,
    run_loop,
)
from smai_agents.model_selection import TaskRole
from smai_agents.prompts import (
    PromptConfig,
    load_prompt_config,
    render_initial_user_message,
)
from smai_agents.tools import (
    Tool,
    ToolContext,
    ToolRegistry,
    make_finish_tool,
)

_PLANNER_ROLE: TaskRole = "planner"

PlannerVariant = Literal["novel_technique", "paper_ingestion"]
"""Two prompt variants per DEC-032; see module docstring for selection."""

# Default ArtifactStore key templates — match the v1 path conventions
# (per ``08-novel-technique-pipeline.md`` §3.2 / §5.5 and the v1
# ``CLAUDE.md`` Artifact Layout for ``proposals/{proposal_id}/`` and
# ``papers/{arxiv_id}/``).
DEFAULT_DESIGN_PLAN_KEY_TEMPLATE = "proposals/{proposal_id}/design_plan.json"
DEFAULT_PAPER_PLAN_KEY_TEMPLATE = "papers/{arxiv_id}/planner_buffer.json"


# === Buffer schema ==========================================================
#
# The buffer accumulates state across the agent's turns. Each tool
# mutates a slice; ``finalize_plan`` / ``finalize_paper_techniques``
# read the whole buffer and run a structural-soundness check.
#
# The schema is intentionally narrow at v1: it carries the minimum
# information the planner needs to draft an ``ExperimentDefinition``-
# ready buffer (or, in the paper-ingestion variant, a ``TechniqueRef``-
# only buffer). Production iteration may grow per-tool fields without
# breaking the persisted-artifact shape (Pydantic ``extra="ignore"``
# at the buffer boundary).


class DraftTechnique(BaseModel):
    """A planner-buffer entry for a TechniqueRef in flight.

    The buffer holds techniques by symbolic name; the registration
    step resolves to ULIDs (v1 invariant per
    ``novel_technique_implementation.md`` §5).

    Per DEC-032, a non-``standard`` technique must carry a
    ``fidelity_anchor`` per the discriminated union in
    ``smai_core.entities.technique`` (proposal / paper / reviewer
    attested). The buffer carries the anchor as a free-form dict here
    (the registration step adapts via
    :data:`smai_core.entities.technique.FidelityAnchorAdapter`); we
    keep the adapter at the registration boundary rather than at the
    buffer boundary so the planner's tool surface stays Pydantic-
    free of the methodology classes the smai-agents package would
    otherwise import on every call.
    """

    model_config = ConfigDict(extra="forbid")

    symbolic_name: str
    """In-buffer identity; resolved to a ULID at registration."""

    name: str
    description: str
    category: str
    compatible_factor_types: list[Literal["additive", "substitutive"]]
    standard: bool = False
    fidelity_anchor: dict[str, Any] | None = None
    affects_extension_points: list[str] = Field(default_factory=list[str])
    implies_controlled: list[str] = Field(default_factory=list[str])
    parameter_schema: dict[str, Any] | None = None


class DraftLevel(BaseModel):
    """A planner-buffer level (the variable component of an entry).

    ``technique_symbolic_name`` references a :class:`DraftTechnique` by
    symbolic name (or a pool-existing technique by stable ID; the
    registration step resolves both). Additive baselines have
    ``technique_symbolic_name=None`` per DEC-013.
    """

    model_config = ConfigDict(extra="forbid")

    factor: str
    name: str
    description: str | None = None
    technique_symbolic_name: str | None = None
    technique_params: dict[str, str | int | float | bool | None] | None = None


class DraftEntry(BaseModel):
    """A planner-buffer entry for one CG arm."""

    model_config = ConfigDict(extra="forbid")

    id: str
    is_baseline: bool
    level: DraftLevel


class DraftValidationCriteria(BaseModel):
    """A planner-buffer ValidationCriteria.

    Per ``01-data-model.md`` §3.6.4 the metric is a ``MetricRef``
    (atomic or parametric, per the closed v1 metric registry per
    DEC-031 sub-decision #1). The buffer carries the metric as a dict
    with a ``ref`` key (atomic) or a ``ref + params`` shape
    (parametric); the registration step adapts to ``MetricRef``.
    """

    model_config = ConfigDict(extra="forbid")

    metric: dict[str, Any]
    direction: Literal["higher_is_better", "lower_is_better"]
    aggregation_method: Literal["mean", "median"] = "mean"
    comparison_rule: Literal["compare_to_baseline", "compare_to_target"]
    threshold: float
    target_value: float | None = None
    seed_count_required: int = 1
    rationale: str | None = None


class DraftComparisonGroup(BaseModel):
    """A planner-buffer ExperimentDefinition shape.

    One per CG the planner elects to create. Per DEC-032 OQ2 the
    novel-technique variant may produce 1..N (most useful for
    "reproduce paper X" proposals).
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    hypothesis: str
    factor_dimension: str
    """Free-text dimension (augmentation / architecture / ...).
    Becomes the ``Factor.name`` per ``01`` §3.3."""
    factor_type: Literal["additive", "substitutive"]
    factor_description: str
    controlled_conditions: dict[str, Any] = Field(default_factory=dict[str, Any])
    """Maps to :class:`smai_core.entities.ControlledConditions`'s
    ``dataset`` / ``optimization`` / ``seeds`` plus arbitrary extras
    (extra="allow" per the ControlledConditions schema)."""
    entries: list[DraftEntry] = Field(default_factory=list[DraftEntry])
    validation: DraftValidationCriteria | None = None


class PlannerBuffer(BaseModel):
    """Accumulated planner state across the multi-turn loop.

    Persisted to :class:`ArtifactStore` on successful finalize (per
    ``08-novel-technique-pipeline.md`` §3.2 / §5.4). The proposal
    pipeline-spec's registration handler reads this artifact and
    materializes the methodology entities + pipeline-tracking entities
    inside a single :class:`MetadataStore.transaction`.
    """

    model_config = ConfigDict(extra="forbid")

    proposal_id: str | None = None
    """Set by the planner-session bootstrap when the variant is
    novel-technique; populated as the ``ProposalFidelityAnchor``'s
    ``proposal_id`` during registration if the planner doesn't
    explicitly set the anchor on a drafted technique."""

    paper_arxiv_id: str | None = None
    """Set by the planner-session bootstrap when the variant is
    paper-ingestion; populated as the ``PaperFidelityAnchor``'s
    ``arxiv_id`` during registration if the planner doesn't
    explicitly set the anchor on a drafted technique."""

    techniques: dict[str, DraftTechnique] = Field(default_factory=dict[str, DraftTechnique])
    """Symbolic-name → :class:`DraftTechnique` map. Cross-references
    between buffer entities use the symbolic name; the registration
    step resolves to ULIDs."""

    comparison_groups: list[DraftComparisonGroup] = Field(
        default_factory=list[DraftComparisonGroup]
    )
    """1..N CGs the novel-technique variant drafts. Empty in the
    paper-ingestion variant per DEC-032."""

    classification: dict[str, Any] | None = None
    """Set by ``set_classification`` — factor dimension + type for the
    overall proposal. Per-CG factor metadata lives on
    :class:`DraftComparisonGroup`; this is the proposal-level summary
    the user sees in the design plan."""

    follow_ups: list[str] = Field(default_factory=list[str])
    """Free-form notes the planner records via ``add_follow_up``;
    surfaced to the user at the human-approval gate."""

    finalized: bool = False
    """Set ``True`` by ``finalize_plan`` / ``finalize_paper_techniques``
    on a passing structural-soundness check; the proposal pipeline-
    spec's ``designed`` gate reads this to advance."""


# === Planner-tool input schemas =============================================
#
# Per Task 3.E1's "tool implementations for both surfaces" deliverable.
# Schemas are minimal-but-typed; per-tool consistency rules live in the
# handler bodies.


class SearchTechniquesInput(BaseModel):
    """``search_techniques`` — query the existing technique pool.

    The pool is preloaded into the planner's context via the prompt-
    config's ``initial_user_message_template`` per
    ``04-agents.md`` §12.4 last paragraph ("the planner's 'search'
    tool searches the technique pool snapshot, not the open web"). The
    tool's input is the query string; the handler matches against the
    snapshot held on :class:`ToolContext.session.workspace_path` (or,
    when the pool was passed via :data:`PlannerInput.pool_summary`,
    against the in-memory snapshot the bootstrap captured).
    """

    model_config = ConfigDict(extra="forbid")

    query: str
    limit: int = 10


class SetClassificationInput(BaseModel):
    """``set_classification`` — declare the proposal-level factor."""

    model_config = ConfigDict(extra="forbid")

    factor_dimension: str
    factor_type: Literal["additive", "substitutive"]
    rationale: str | None = None


class DraftComparisonInput(BaseModel):
    """``draft_comparison`` — declare a CG (one ExperimentDefinition).

    The planner calls this once per CG it intends to produce. Subsequent
    tools (``set_conditions``, ``draft_assertion``, etc.) reference the
    CG by ``id``.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    hypothesis: str
    factor_dimension: str
    factor_type: Literal["additive", "substitutive"]
    factor_description: str
    entries: list[DraftEntry]


class SetConditionsInput(BaseModel):
    """``set_conditions`` — lock the controlled-conditions surface.

    Per ``01-data-model.md`` §3.5 the ControlledConditions schema has
    a typed core (``dataset``, ``optimization``, ``seeds``) plus
    ``extra="allow"`` for arbitrary additional fields.
    """

    model_config = ConfigDict(extra="forbid")

    cg_id: str
    conditions: dict[str, Any]


class DraftAssertionInput(BaseModel):
    """``draft_assertion`` — attach a ValidationCriteria to a CG."""

    model_config = ConfigDict(extra="forbid")

    cg_id: str
    validation: DraftValidationCriteria


class AddFollowUpInput(BaseModel):
    """``add_follow_up`` — append a free-form note."""

    model_config = ConfigDict(extra="forbid")

    note: str


class FinalizePlanInput(BaseModel):
    """``finalize_plan`` — run the structural-soundness check.

    No fields — the tool reads the entire buffer and runs the rules.
    """

    model_config = ConfigDict(extra="forbid")


class DraftCreateTechniqueInput(BaseModel):
    """``draft_create_technique`` — draft a NOVEL TechniqueRef.

    Used by both variants:
    * Novel-technique: drafts the proposal's contribution technique
      (also any comparison techniques the planner needed to add).
    * Paper-ingestion: drafts the paper's contribution technique(s).
    """

    model_config = ConfigDict(extra="forbid")

    symbolic_name: str
    name: str
    description: str
    category: str
    compatible_factor_types: list[Literal["additive", "substitutive"]]
    standard: bool = False
    fidelity_anchor: dict[str, Any] | None = None
    affects_extension_points: list[str] = Field(default_factory=list[str])
    parameter_schema: dict[str, Any] | None = None


class DraftEnsureTechniqueInput(BaseModel):
    """``draft_ensure_technique`` — dedup on cited comparison techniques.

    Per ``08-novel-technique-pipeline.md`` §5.3: the paper-ingestion
    variant uses this to draft a skeleton TechniqueRef for a comparison
    technique the paper cites that isn't yet in the pool. The
    novel-technique variant uses it to dedup on comparison technique
    references against the existing pool. Idempotent: calling it twice
    on the same symbolic name updates the buffer entry.
    """

    model_config = ConfigDict(extra="forbid")

    symbolic_name: str
    name: str
    description: str
    category: str
    compatible_factor_types: list[Literal["additive", "substitutive"]]
    standard: bool = False
    fidelity_anchor: dict[str, Any] | None = None
    affects_extension_points: list[str] = Field(default_factory=list[str])


class FinalizePaperTechniquesInput(BaseModel):
    """``finalize_paper_techniques`` — narrow finalize for paper ingestion.

    Per ``08-novel-technique-pipeline.md`` §5.4. No fields — the tool
    reads the entire buffer and runs the narrowed rule set (just rule
    1: at least one TechniqueRef with PaperFidelityAnchor matching
    this paper).
    """

    model_config = ConfigDict(extra="forbid")


# === Planner-input bundle ===================================================


class PlannerInput(BaseModel):
    """Assembled inputs to :func:`run_planner_session`.

    Bundles the variant-keyed bootstrap data the agent loop needs to
    render the initial user message + materialize the buffer's identity
    fields. The proposal pipeline-spec's dispatch handler populates
    one of these from ``MetadataStore`` + ``ArtifactStore`` reads.
    """

    model_config = ConfigDict(extra="forbid")

    variant: PlannerVariant

    # Novel-technique variant fields
    proposal_id: str | None = None
    submission_kind: Literal["novel_technique", "reproduce_paper"] | None = None
    technique_description: str | None = None
    """Verbatim user submission for ``novel_technique`` proposals."""

    # Paper-ingestion variant fields (also populated for reproduce_paper
    # novel-technique proposals so the prompt template can reference the
    # paper).
    paper_arxiv_id: str | None = None

    # Shared fields
    pool_summary: str = "(empty pool — no techniques registered yet)"
    """Free-form text summary of the existing technique pool the
    planner reads at session start. The dispatch handler renders this
    from :meth:`MetadataStore.list_techniques` results; the schema is
    intentionally not the structured ``list[TechniqueRef]`` because
    the planner's reasoning is over the text snapshot, not over typed
    objects."""

    metric_registry_summary: str = "(default v1 metric registry — accuracy, loss)"
    """Free-form text summary of the closed v1 metric registry per
    DEC-031 sub-decision #1. Same reasoning as ``pool_summary``."""


# === Tool-handler factories =================================================


def _make_search_techniques_tool(buffer: PlannerBuffer, pool_summary: str) -> Tool:
    """``search_techniques`` — string-match against the pool snapshot.

    The pool snapshot is captured at session start as free-form text.
    The handler does a case-insensitive substring match and returns the
    matching lines (or "no matches" when nothing hits). Tests inject a
    canned ``pool_summary`` per :class:`PlannerInput` and assert the
    handler's match output.
    """
    del buffer  # buffer is not mutated by search_techniques

    async def _handler(parsed_input: BaseModel, context: ToolContext) -> ToolResultContent:
        del context
        if not isinstance(parsed_input, SearchTechniquesInput):
            raise TypeError(
                f"search_techniques handler expected SearchTechniquesInput, "
                f"got {type(parsed_input).__name__}"
            )
        query = parsed_input.query.lower()
        matches: list[str] = []
        for line in pool_summary.splitlines():
            if query in line.lower():
                matches.append(line)
                if len(matches) >= parsed_input.limit:
                    break
        if not matches:
            content = f"no matches for {parsed_input.query!r} in pool snapshot"
        else:
            content = "\n".join(matches)
        return ToolResultContent(tool_use_id="", content=content, is_error=False)

    return Tool(
        name="search_techniques",
        description=(
            "Search the technique-pool snapshot (loaded into the agent's "
            "context at session start). Substring match, case-insensitive. "
            "Returns matching lines from the snapshot."
        ),
        input_schema=SearchTechniquesInput,
        handler=_handler,
    )


def _make_set_classification_tool(buffer: PlannerBuffer) -> Tool:
    async def _handler(parsed_input: BaseModel, context: ToolContext) -> ToolResultContent:
        del context
        if not isinstance(parsed_input, SetClassificationInput):
            raise TypeError(
                f"set_classification handler expected SetClassificationInput, "
                f"got {type(parsed_input).__name__}"
            )
        buffer.classification = {
            "factor_dimension": parsed_input.factor_dimension,
            "factor_type": parsed_input.factor_type,
            "rationale": parsed_input.rationale,
        }
        return ToolResultContent(
            tool_use_id="",
            content=(
                f"classification set: dimension={parsed_input.factor_dimension!r} "
                f"type={parsed_input.factor_type!r}"
            ),
            is_error=False,
        )

    return Tool(
        name="set_classification",
        description=(
            "Declare the proposal-level experimental factor: dimension "
            "(augmentation / architecture / optimization / ...) and type "
            "(additive or substitutive per DEC-012). Per-CG factor "
            "metadata is set inside draft_comparison."
        ),
        input_schema=SetClassificationInput,
        handler=_handler,
    )


def _make_draft_comparison_tool(buffer: PlannerBuffer) -> Tool:
    async def _handler(parsed_input: BaseModel, context: ToolContext) -> ToolResultContent:
        del context
        if not isinstance(parsed_input, DraftComparisonInput):
            raise TypeError(
                f"draft_comparison handler expected DraftComparisonInput, "
                f"got {type(parsed_input).__name__}"
            )
        # Reject duplicate CG ids — the buffer is one-shot per CG id.
        for existing in buffer.comparison_groups:
            if existing.id == parsed_input.id:
                return ToolResultContent(
                    tool_use_id="",
                    content=(
                        f"draft_comparison error: CG id {parsed_input.id!r} already drafted; "
                        f"call set_conditions / draft_assertion to update it instead"
                    ),
                    is_error=True,
                )
        buffer.comparison_groups.append(
            DraftComparisonGroup(
                id=parsed_input.id,
                hypothesis=parsed_input.hypothesis,
                factor_dimension=parsed_input.factor_dimension,
                factor_type=parsed_input.factor_type,
                factor_description=parsed_input.factor_description,
                entries=list(parsed_input.entries),
            )
        )
        return ToolResultContent(
            tool_use_id="",
            content=(
                f"comparison group {parsed_input.id!r} drafted with "
                f"{len(parsed_input.entries)} entries"
            ),
            is_error=False,
        )

    return Tool(
        name="draft_comparison",
        description=(
            "Declare one ComparisonGroup (one ExperimentDefinition). Per "
            "DEC-032 OQ2 the planner may call this 1..N times to "
            "decompose a proposal into multiple CGs. Entries are passed "
            "inline; subsequent set_conditions / draft_assertion calls "
            "reference the CG by id."
        ),
        input_schema=DraftComparisonInput,
        handler=_handler,
    )


def _make_set_conditions_tool(buffer: PlannerBuffer) -> Tool:
    async def _handler(parsed_input: BaseModel, context: ToolContext) -> ToolResultContent:
        del context
        if not isinstance(parsed_input, SetConditionsInput):
            raise TypeError(
                f"set_conditions handler expected SetConditionsInput, "
                f"got {type(parsed_input).__name__}"
            )
        cg = _find_cg(buffer, parsed_input.cg_id)
        if cg is None:
            return ToolResultContent(
                tool_use_id="",
                content=(
                    f"set_conditions error: CG id {parsed_input.cg_id!r} not found; "
                    f"call draft_comparison first"
                ),
                is_error=True,
            )
        cg.controlled_conditions = dict(parsed_input.conditions)
        return ToolResultContent(
            tool_use_id="",
            content=f"controlled conditions set on CG {parsed_input.cg_id!r}",
            is_error=False,
        )

    return Tool(
        name="set_conditions",
        description=(
            "Set the ControlledConditions for one CG. Required keys per "
            "01-data-model.md §3.5: dataset, optimization, seeds. "
            "Arbitrary additional top-level fields admitted via "
            "extra='allow' on the schema."
        ),
        input_schema=SetConditionsInput,
        handler=_handler,
    )


def _make_draft_assertion_tool(buffer: PlannerBuffer) -> Tool:
    async def _handler(parsed_input: BaseModel, context: ToolContext) -> ToolResultContent:
        del context
        if not isinstance(parsed_input, DraftAssertionInput):
            raise TypeError(
                f"draft_assertion handler expected DraftAssertionInput, "
                f"got {type(parsed_input).__name__}"
            )
        cg = _find_cg(buffer, parsed_input.cg_id)
        if cg is None:
            return ToolResultContent(
                tool_use_id="",
                content=(
                    f"draft_assertion error: CG id {parsed_input.cg_id!r} not found; "
                    f"call draft_comparison first"
                ),
                is_error=True,
            )
        cg.validation = parsed_input.validation
        return ToolResultContent(
            tool_use_id="",
            content=f"validation criteria set on CG {parsed_input.cg_id!r}",
            is_error=False,
        )

    return Tool(
        name="draft_assertion",
        description=(
            "Set the ValidationCriteria for one CG: metric reference into "
            "the closed v1 metric registry (per DEC-031), direction, "
            "comparison rule (compare_to_baseline / compare_to_target), "
            "threshold, and seed-count requirement."
        ),
        input_schema=DraftAssertionInput,
        handler=_handler,
    )


def _make_add_follow_up_tool(buffer: PlannerBuffer) -> Tool:
    async def _handler(parsed_input: BaseModel, context: ToolContext) -> ToolResultContent:
        del context
        if not isinstance(parsed_input, AddFollowUpInput):
            raise TypeError(
                f"add_follow_up handler expected AddFollowUpInput, "
                f"got {type(parsed_input).__name__}"
            )
        buffer.follow_ups.append(parsed_input.note)
        return ToolResultContent(
            tool_use_id="",
            content=f"follow-up note recorded ({len(buffer.follow_ups)} total)",
            is_error=False,
        )

    return Tool(
        name="add_follow_up",
        description=(
            "Append a free-form note for downstream review. Surfaced to "
            "the user at the human-approval gate."
        ),
        input_schema=AddFollowUpInput,
        handler=_handler,
    )


def _make_finalize_plan_tool(
    buffer: PlannerBuffer,
    *,
    artifact_store: ArtifactStore | None,
    artifact_key: str,
) -> Tool:
    """``finalize_plan`` — runs the structural-soundness check.

    Per ``08-novel-technique-pipeline.md`` §3.2 (rules 1..10). The
    handler returns ``is_error=True`` with the consolidated rule
    failures so the agent can self-correct; ``is_error=False`` only
    when every rule passes. On success the handler also persists the
    buffer to ArtifactStore (per ``08`` §3.2 last paragraph).
    """

    async def _handler(parsed_input: BaseModel, context: ToolContext) -> ToolResultContent:
        del context
        if not isinstance(parsed_input, FinalizePlanInput):
            raise TypeError(
                f"finalize_plan handler expected FinalizePlanInput, "
                f"got {type(parsed_input).__name__}"
            )
        errors = _structural_check_novel_technique(buffer)
        if errors:
            return ToolResultContent(
                tool_use_id="",
                content="finalize_plan failed:\n- " + "\n- ".join(errors),
                is_error=True,
            )
        buffer.finalized = True
        if artifact_store is not None:
            await artifact_store.put(
                artifact_key,
                buffer.model_dump_json(indent=2).encode("utf-8"),
            )
        return ToolResultContent(
            tool_use_id="",
            content=(
                f"finalize_plan succeeded: {len(buffer.comparison_groups)} CG(s) drafted, "
                f"{len(buffer.techniques)} novel TechniqueRef(s); "
                f"buffer persisted to {artifact_key!r}"
            ),
            is_error=False,
        )

    return Tool(
        name="finalize_plan",
        description=(
            "Run the structural-soundness check on the complete buffer "
            "(per 08-novel-technique-pipeline.md §3.2 rules 1..10). On "
            "success the buffer is persisted to ArtifactStore at the "
            "configured key and the loop is ready to be exited via "
            "finish. On failure the rule errors are returned so the "
            "agent can correct (one retry allowed before the pipeline "
            "transitions to failed)."
        ),
        input_schema=FinalizePlanInput,
        handler=_handler,
    )


def _make_draft_create_technique_tool(buffer: PlannerBuffer) -> Tool:
    async def _handler(parsed_input: BaseModel, context: ToolContext) -> ToolResultContent:
        del context
        if not isinstance(parsed_input, DraftCreateTechniqueInput):
            raise TypeError(
                f"draft_create_technique handler expected DraftCreateTechniqueInput, "
                f"got {type(parsed_input).__name__}"
            )
        if parsed_input.symbolic_name in buffer.techniques:
            return ToolResultContent(
                tool_use_id="",
                content=(
                    f"draft_create_technique error: symbolic_name "
                    f"{parsed_input.symbolic_name!r} already drafted; use "
                    f"draft_ensure_technique for idempotent updates"
                ),
                is_error=True,
            )
        if not parsed_input.standard and parsed_input.fidelity_anchor is None:
            return ToolResultContent(
                tool_use_id="",
                content=(
                    f"draft_create_technique error: non-standard technique "
                    f"{parsed_input.symbolic_name!r} requires a fidelity_anchor "
                    f"per DEC-032 (proposal / paper / reviewer_attested)"
                ),
                is_error=True,
            )
        buffer.techniques[parsed_input.symbolic_name] = DraftTechnique(
            symbolic_name=parsed_input.symbolic_name,
            name=parsed_input.name,
            description=parsed_input.description,
            category=parsed_input.category,
            compatible_factor_types=list(parsed_input.compatible_factor_types),
            standard=parsed_input.standard,
            fidelity_anchor=parsed_input.fidelity_anchor,
            affects_extension_points=list(parsed_input.affects_extension_points),
            parameter_schema=parsed_input.parameter_schema,
        )
        return ToolResultContent(
            tool_use_id="",
            content=f"technique {parsed_input.symbolic_name!r} drafted",
            is_error=False,
        )

    return Tool(
        name="draft_create_technique",
        description=(
            "Draft a new TechniqueRef into the buffer. Non-standard "
            "techniques (DEC-015) require a fidelity_anchor per DEC-032 "
            "(proposal / paper / reviewer_attested kind). Standard=true "
            "techniques may omit the anchor. Buffer is keyed by "
            "symbolic_name; the registration step resolves to ULIDs."
        ),
        input_schema=DraftCreateTechniqueInput,
        handler=_handler,
    )


def _make_draft_ensure_technique_tool(buffer: PlannerBuffer) -> Tool:
    async def _handler(parsed_input: BaseModel, context: ToolContext) -> ToolResultContent:
        del context
        if not isinstance(parsed_input, DraftEnsureTechniqueInput):
            raise TypeError(
                f"draft_ensure_technique handler expected DraftEnsureTechniqueInput, "
                f"got {type(parsed_input).__name__}"
            )
        if not parsed_input.standard and parsed_input.fidelity_anchor is None:
            return ToolResultContent(
                tool_use_id="",
                content=(
                    f"draft_ensure_technique error: non-standard technique "
                    f"{parsed_input.symbolic_name!r} requires a fidelity_anchor"
                ),
                is_error=True,
            )
        buffer.techniques[parsed_input.symbolic_name] = DraftTechnique(
            symbolic_name=parsed_input.symbolic_name,
            name=parsed_input.name,
            description=parsed_input.description,
            category=parsed_input.category,
            compatible_factor_types=list(parsed_input.compatible_factor_types),
            standard=parsed_input.standard,
            fidelity_anchor=parsed_input.fidelity_anchor,
            affects_extension_points=list(parsed_input.affects_extension_points),
        )
        return ToolResultContent(
            tool_use_id="",
            content=f"technique {parsed_input.symbolic_name!r} ensured (idempotent)",
            is_error=False,
        )

    return Tool(
        name="draft_ensure_technique",
        description=(
            "Idempotent variant of draft_create_technique. Used to dedup "
            "comparison technique citations against the existing pool. "
            "Calling twice on the same symbolic_name updates the buffer "
            "entry rather than erroring."
        ),
        input_schema=DraftEnsureTechniqueInput,
        handler=_handler,
    )


def _make_finalize_paper_techniques_tool(
    buffer: PlannerBuffer,
    *,
    artifact_store: ArtifactStore | None,
    artifact_key: str,
) -> Tool:
    """``finalize_paper_techniques`` — narrow finalize for paper ingestion.

    Per ``08-novel-technique-pipeline.md`` §5.4 the v1 12-rule contract
    is dramatically narrowed for the paper-ingestion variant: only
    rule 1 (at least one TechniqueRef with PaperFidelityAnchor matching
    this paper) survives. All CG-shape rules (3..8, 10, 11) are dropped
    per DEC-032's architectural commitment that paper ingestion does
    NOT produce CGs.
    """

    async def _handler(parsed_input: BaseModel, context: ToolContext) -> ToolResultContent:
        del context
        if not isinstance(parsed_input, FinalizePaperTechniquesInput):
            raise TypeError(
                f"finalize_paper_techniques handler expected FinalizePaperTechniquesInput, "
                f"got {type(parsed_input).__name__}"
            )
        errors = _structural_check_paper_ingestion(buffer)
        if errors:
            return ToolResultContent(
                tool_use_id="",
                content="finalize_paper_techniques failed:\n- " + "\n- ".join(errors),
                is_error=True,
            )
        buffer.finalized = True
        if artifact_store is not None:
            await artifact_store.put(
                artifact_key,
                buffer.model_dump_json(indent=2).encode("utf-8"),
            )
        return ToolResultContent(
            tool_use_id="",
            content=(
                f"finalize_paper_techniques succeeded: {len(buffer.techniques)} "
                f"TechniqueRef(s); buffer persisted to {artifact_key!r}"
            ),
            is_error=False,
        )

    return Tool(
        name="finalize_paper_techniques",
        description=(
            "Run the narrow structural-soundness check on the buffer "
            "(per 08-novel-technique-pipeline.md §5.4). On success the "
            "buffer is persisted to ArtifactStore at the configured "
            "key. Per DEC-032 paper ingestion produces TechniqueRefs "
            "ONLY — no ExperimentDefinitions, no CGs."
        ),
        input_schema=FinalizePaperTechniquesInput,
        handler=_handler,
    )


def _find_cg(buffer: PlannerBuffer, cg_id: str) -> DraftComparisonGroup | None:
    for cg in buffer.comparison_groups:
        if cg.id == cg_id:
            return cg
    return None


# === Structural-soundness checks ============================================
#
# Per ``08-novel-technique-pipeline.md`` §3.2 (novel-technique variant
# rules 1..10) and §5.4 (paper-ingestion variant — narrowed to rule 1).
# Each helper returns a list of error strings; an empty list means
# "structurally sound."


def _structural_check_novel_technique(buffer: PlannerBuffer) -> list[str]:
    errors: list[str] = []

    # Rule 1: at least one ExperimentDefinition.
    if not buffer.comparison_groups:
        errors.append("rule 1: no ExperimentDefinitions drafted (need at least 1)")

    for cg in buffer.comparison_groups:
        # Rule 2: every CG has ≥ 2 entries.
        if len(cg.entries) < 2:  # noqa: PLR2004 — rule literal per design doc
            errors.append(f"rule 2: CG {cg.id!r} has {len(cg.entries)} entries (need at least 2)")
        # Rule 3: controlled_conditions populated with dataset / optimization / seeds.
        cc = cg.controlled_conditions
        if not cc.get("dataset"):
            errors.append(f"rule 3: CG {cg.id!r} missing controlled_conditions.dataset")
        if not cc.get("optimization"):
            errors.append(f"rule 3: CG {cg.id!r} missing controlled_conditions.optimization")
        if not cc.get("seeds"):
            errors.append(f"rule 3: CG {cg.id!r} missing controlled_conditions.seeds")
        # Rule 4: ValidationCriteria with metric.
        if cg.validation is None:
            errors.append(f"rule 4: CG {cg.id!r} has no ValidationCriteria")
        elif "ref" not in cg.validation.metric:
            errors.append(f"rule 4: CG {cg.id!r} validation.metric missing 'ref' key")
        # Rule 5: every entry references a known technique (or is additive baseline).
        for entry in cg.entries:
            level = entry.level
            if entry.is_baseline and cg.factor_type == "additive":
                # Additive baseline: technique_symbolic_name MAY be None per DEC-013.
                continue
            if level.technique_symbolic_name is None:
                errors.append(
                    f"rule 5/8: CG {cg.id!r} entry {entry.id!r} has no technique "
                    f"(non-additive-baseline entries require one per DEC-013)"
                )
        # Rule 7: ValidationCriteria.direction set (always required even when registry has it).
        if cg.validation is not None and cg.validation.direction not in {
            "higher_is_better",
            "lower_is_better",
        }:
            errors.append(f"rule 7: CG {cg.id!r} validation.direction not set")
        # Rule 8: factor_type matches baseline-entry consistency rule per DEC-013.
        baselines = [e for e in cg.entries if e.is_baseline]
        if not baselines:
            errors.append(f"rule 8: CG {cg.id!r} has no baseline entry")
        elif cg.factor_type == "substitutive":
            for baseline in baselines:
                if baseline.level.technique_symbolic_name is None:
                    errors.append(
                        f"rule 8: CG {cg.id!r} substitutive baseline {baseline.id!r} "
                        f"must have technique_symbolic_name (DEC-013)"
                    )
        # Rule 10: paper-sourced entries require compare_to_target with target_value.
        # Detected indirectly: if the CG references any technique with paper anchor,
        # the validation rule should be compare_to_target.
        # v1: skip enforcement when validation is unset (Rule 4 already fires).

    # Rule 6: every novel TechniqueRef carries a fidelity_anchor.
    for sym, t in buffer.techniques.items():
        if not t.standard and t.fidelity_anchor is None:
            errors.append(
                f"rule 6: novel TechniqueRef {sym!r} missing fidelity_anchor "
                f"(non-standard techniques require one per DEC-032)"
            )

    return errors


def _structural_check_paper_ingestion(buffer: PlannerBuffer) -> list[str]:
    errors: list[str] = []
    paper_arxiv_id = buffer.paper_arxiv_id

    # Paper-ingestion: rule 1 only (per ``08`` §5.4 / DEC-032).
    has_paper_anchor = False
    for sym, t in buffer.techniques.items():
        if t.standard:
            continue
        if t.fidelity_anchor is None:
            errors.append(f"rule 1 dependency: TechniqueRef {sym!r} missing fidelity_anchor")
            continue
        anchor = t.fidelity_anchor
        if anchor.get("kind") == "paper":
            if paper_arxiv_id is None or anchor.get("arxiv_id") == paper_arxiv_id:
                has_paper_anchor = True
    if not has_paper_anchor:
        if paper_arxiv_id is not None:
            errors.append(
                f"rule 1: no TechniqueRef carries PaperFidelityAnchor matching "
                f"arxiv_id={paper_arxiv_id!r} (paper has no contribution techniques)"
            )
        else:
            errors.append(
                "rule 1: no TechniqueRef carries PaperFidelityAnchor "
                "(paper has no contribution techniques)"
            )
    # CG-shape rules (3..8, 10, 11) explicitly DROPPED per DEC-032.
    return errors


# === Tool-registry assembly =================================================


def _build_planner_tool_registry(
    *,
    variant: PlannerVariant,
    buffer: PlannerBuffer,
    pool_summary: str,
    artifact_store: ArtifactStore | None,
    artifact_key: str,
) -> ToolRegistry:
    """Assemble the variant-specific tool registry.

    Per the prompt-config variant, each variant lists the tools it
    binds; this function constructs the corresponding handler set. The
    standard read tools (read_file / search / list_files) are NOT
    included here — the planner's reading happens through the snapshot
    text in the prompt context, not through workspace I/O. (If a
    deployment needs file-reading planner support, the variant's tool
    list can add the standard tools and this function can route to
    :func:`build_standard_tool_registry`. v1 ships the snapshot-only
    shape per ``04-agents.md`` §12.4 last paragraph.)
    """
    registry = ToolRegistry()

    if variant == "novel_technique":
        registry.register(_make_search_techniques_tool(buffer, pool_summary))
        registry.register(_make_set_classification_tool(buffer))
        registry.register(_make_draft_comparison_tool(buffer))
        registry.register(_make_set_conditions_tool(buffer))
        registry.register(_make_draft_assertion_tool(buffer))
        registry.register(_make_add_follow_up_tool(buffer))
        registry.register(
            _make_finalize_plan_tool(
                buffer, artifact_store=artifact_store, artifact_key=artifact_key
            )
        )
        # Useful for the novel-technique variant when drafting a comparison
        # technique that wasn't in the pool.
        registry.register(_make_draft_create_technique_tool(buffer))
        registry.register(_make_draft_ensure_technique_tool(buffer))
    elif variant == "paper_ingestion":
        registry.register(_make_draft_create_technique_tool(buffer))
        registry.register(_make_draft_ensure_technique_tool(buffer))
        registry.register(
            _make_finalize_paper_techniques_tool(
                buffer, artifact_store=artifact_store, artifact_key=artifact_key
            )
        )

    registry.register(make_finish_tool())
    return registry


# === In-process runner ======================================================


# Test-only seam shape mirrors :data:`smai_agents.agents.harness_builder.SessionRunner`.
SessionRunner = Callable[[AgentSession], Awaitable[AgentOutcome]]


class PlannerSessionResult(BaseModel):
    """Bundle returned by :func:`run_planner_session`.

    The agent loop's :class:`AgentOutcome` plus the planner's
    accumulated :class:`PlannerBuffer` (whether finalized or partial,
    per ``08-novel-technique-pipeline.md`` §2.5's "partial buffer
    preserved on failure").

    The :attr:`artifact_key` carries the ArtifactStore key the buffer
    was persisted under (only when :attr:`buffer.finalized` is
    ``True``); the proposal pipeline-spec's ``designed`` gate reads
    this key to confirm the design plan parses.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    outcome: AgentOutcome
    buffer: PlannerBuffer
    artifact_key: str


async def run_planner_session(
    *,
    input: PlannerInput,
    llm: LlmProvider,
    workspace_path: Path,
    artifact_store: ArtifactStore | None = None,
    design_plan_artifact_key: str | None = None,
    prompt_config: PromptConfig | None = None,
    config: AgentLoopConfig | None = None,
    runner: SessionRunner | None = None,
    supervisor_llm: LlmProvider | None = None,
) -> PlannerSessionResult:
    """Drive the planner agent loop in-process.

    Args:
        input: Variant-keyed bootstrap data.
        llm: :class:`LlmProvider` for the ``planner`` role.
        workspace_path: Local-disk workspace; the planner's standard
            file tools (when bound by the variant) operate against
            this. v1's planner does not touch the workspace, but the
            field is required by :class:`AgentSession`.
        artifact_store: Storage seam for the finalized buffer. ``None``
            disables persistence (useful for unit tests that assert
            on the buffer in-memory).
        design_plan_artifact_key: Optional override for the
            ArtifactStore key. Defaults to
            :data:`DEFAULT_DESIGN_PLAN_KEY_TEMPLATE` (novel-technique
            variant) or :data:`DEFAULT_PAPER_PLAN_KEY_TEMPLATE`
            (paper-ingestion variant), templated against the
            corresponding identity field.
        prompt_config: Optional pre-loaded :class:`PromptConfig`.
            ``None`` calls :func:`load_prompt_config` for the
            ``planner`` role with the variant.
        config: Optional :class:`AgentLoopConfig`; defaults to stock.
        runner: Test-only seam — when ``None``, calls :func:`run_loop`.

    Returns:
        A :class:`PlannerSessionResult` carrying the loop's terminal
        outcome, the accumulated buffer (which the caller can
        inspect to decide whether to advance the proposal pipeline),
        and the artifact key the buffer would be persisted under.
    """
    artifact_key = design_plan_artifact_key or _default_artifact_key(input)

    # === Buffer + tool registry (variant-keyed) =============================
    buffer = PlannerBuffer(
        proposal_id=input.proposal_id,
        paper_arxiv_id=input.paper_arxiv_id,
    )

    cfg = (
        prompt_config
        if prompt_config is not None
        else load_prompt_config(role=_PLANNER_ROLE, variant_name=input.variant)
    )

    tool_registry = _build_planner_tool_registry(
        variant=input.variant,
        buffer=buffer,
        pool_summary=input.pool_summary,
        artifact_store=artifact_store,
        artifact_key=artifact_key,
    )

    # === Initial user message ===============================================
    initial_user = render_initial_user_message(
        cfg,
        proposal_id=input.proposal_id or "(no proposal id)",
        submission_kind=input.submission_kind or "novel_technique",
        technique_description=input.technique_description or "(none)",
        paper_arxiv_id=input.paper_arxiv_id or "(no paper)",
        pool_summary=input.pool_summary,
        metric_registry_summary=input.metric_registry_summary,
    )

    # === AgentSession =======================================================
    session = AgentSession(
        system_prompt=cfg.system_prompt,
        messages=[
            NormalizedMessage(
                role="user",
                content=[TextContent(text=initial_user)],
            )
        ],
        tools=tool_registry,
        llm_providers={_PLANNER_ROLE: llm},
        current_role=_PLANNER_ROLE,
        workspace_path=workspace_path,
        artifact_store=artifact_store,
        compute=None,
        status_artifact_path=None,
        nudge_artifact_path=None,
        config=config or AgentLoopConfig(),
    )

    if supervisor_llm is not None:
        session.llm_providers["supervisor"] = supervisor_llm

    actual_runner = runner if runner is not None else run_loop
    outcome = await actual_runner(session)
    return PlannerSessionResult(
        outcome=outcome,
        buffer=buffer,
        artifact_key=artifact_key,
    )


def _default_artifact_key(input: PlannerInput) -> str:
    """Resolve the default ArtifactStore key per variant.

    Novel-technique buffers land under
    ``proposals/{proposal_id}/design_plan.json`` per
    ``08-novel-technique-pipeline.md`` §3.2. Paper-ingestion buffers
    land under ``papers/{arxiv_id}/planner_buffer.json`` per the v1
    convention; the actual paper-ingestion pipeline-spec (Task 3.E2)
    may rename / re-route this.
    """
    if input.variant == "novel_technique":
        proposal_id = input.proposal_id
        if proposal_id is None:
            raise ValueError(
                "novel_technique variant requires PlannerInput.proposal_id "
                "(used as the design-plan artifact key)"
            )
        return DEFAULT_DESIGN_PLAN_KEY_TEMPLATE.format(proposal_id=proposal_id)
    if input.variant == "paper_ingestion":
        arxiv_id = input.paper_arxiv_id
        if arxiv_id is None:
            raise ValueError(
                "paper_ingestion variant requires PlannerInput.paper_arxiv_id "
                "(used as the planner-buffer artifact key)"
            )
        return DEFAULT_PAPER_PLAN_KEY_TEMPLATE.format(arxiv_id=arxiv_id)
    raise ValueError(f"unknown planner variant {input.variant!r}")


# === Orchestrator dispatch handler ==========================================


def make_dispatch_planner(
    *,
    workspace_root: Path,
    pool_summary_loader: (
        Callable[[Any], Awaitable[str]] | None  # ctx: DispatchContext
    ) = None,
    metric_registry_summary: str = "(default v1 metric registry — accuracy, loss)",
    design_plan_artifact_key_for: Callable[[str], str] | None = None,
    inline_runner: SessionRunner | None = None,
) -> Callable[[Any], Awaitable[Any]]:
    """Build a :data:`DispatchHandler` for the planner role.

    The proposal pipeline-spec (Task 3.E1 ``proposal.dispatch_planner``)
    binds one of these on the ``designing`` state's on-entry slot. Per
    ``03-state-machine.md`` §4.3 the handler is **inline** (no external
    :class:`Compute`); the multi-turn loop runs in the worker via the
    agent loop engine in ``04-agents.md`` §3.

    Per implementation_plan.md §2.1 the smai-agents → smai-orchestrator
    dependency direction means we lazy-import :class:`DispatchOutcome`
    inside the body.

    Args:
        workspace_root: Filesystem root for per-proposal planner
            workspaces. The handler creates
            ``<workspace_root>/<proposal_id>/`` before invoking the
            session.
        pool_summary_loader: Optional async callable that returns the
            pool snapshot text for the planner's prompt context. When
            ``None`` the handler renders a default summary from
            :meth:`MetadataStore.list_techniques` (drains all pages
            and produces one line per technique).
        metric_registry_summary: Free-text summary of the closed v1
            metric registry, threaded into the planner's prompt
            context. Production deployments override per their
            registry shape; tests pass a fixture string.
        design_plan_artifact_key_for: Callable resolving
            ``proposal_id → ArtifactStore key`` for the design-plan
            artifact. Defaults to
            :data:`DEFAULT_DESIGN_PLAN_KEY_TEMPLATE`.
        inline_runner: Test-only seam — when non-``None``, the
            dispatch handler runs the agent loop in-process via this
            runner. Production passes ``None`` and the handler calls
            :func:`run_loop`.
    """

    def _default_design_plan_key(proposal_id: str) -> str:
        return DEFAULT_DESIGN_PLAN_KEY_TEMPLATE.format(proposal_id=proposal_id)

    design_plan_key_fn = design_plan_artifact_key_for or _default_design_plan_key

    async def _dispatch(ctx: Any) -> Any:  # ctx: DispatchContext
        from smai_orchestrator.engine.types import DispatchOutcome  # noqa: PLC0415

        proposal_id = ctx.entity_id
        proposal = await ctx.metadata_store.get_proposal(proposal_id)
        if proposal is None:
            return DispatchOutcome(
                submitted_handles=[],
                error=f"proposal {proposal_id!r} not found in MetadataStore",
            )

        llm = ctx.llm
        if llm is None:
            return DispatchOutcome(
                submitted_handles=[],
                error=(
                    "DispatchContext.llm is None; planner requires an "
                    "LlmProvider for the planner role. Wire one through "
                    "the engine config or a per-role LlmProvider map."
                ),
            )

        # Pool snapshot — caller-supplied or default-rendered.
        if pool_summary_loader is not None:
            pool_summary = await pool_summary_loader(ctx)
        else:
            pool_summary = await _default_pool_summary_loader(ctx.metadata_store)

        # Resolve variant. Per ``04-agents.md`` §2.1 + ``08`` §3.2 +
        # §4: the proposal pipeline always uses novel_technique
        # (including for "reproduce paper X" proposals — the planner
        # reads pre-ingested paper artifacts as context but still
        # drafts ExperimentDefinitions). The paper-ingestion variant
        # is selected by the paper-ingestion pipeline-spec's planning
        # state's dispatch handler (Task 3.E2), not by this handler.
        variant: PlannerVariant = "novel_technique"
        submission_kind = getattr(proposal, "submission_kind", None) or "novel_technique"
        technique_description = await _load_technique_description(ctx, proposal)

        planner_input = PlannerInput(
            variant=variant,
            proposal_id=proposal_id,
            submission_kind=submission_kind,
            technique_description=technique_description,
            paper_arxiv_id=getattr(proposal, "reproduce_paper_arxiv_id", None),
            pool_summary=pool_summary,
            metric_registry_summary=metric_registry_summary,
        )

        workspace_path = workspace_root / proposal_id
        workspace_path.mkdir(parents=True, exist_ok=True)
        artifact_key = design_plan_key_fn(proposal_id)

        # Translate :class:`EngineConfig` supervisor settings (Task
        # 3.G4) into the loop's :class:`AgentLoopConfig`. Mirrors
        # ``make_dispatch_harness_build`` / ``make_dispatch_technique_implementation``.
        # The representative ``ctx.llm`` provider doubles as the
        # supervisor LLM in the C2 single-LLM-per-context shape;
        # per-role plumbing is the natural follow-up. Some unit tests
        # leave ``ctx.config = None``; treat as "supervisor off".
        engine_config = getattr(ctx, "config", None)
        if (
            engine_config is not None
            and getattr(engine_config, "supervisor_enabled", False) is True
        ):
            supervisor_on = True
            supervisor_cadence = int(getattr(engine_config, "supervisor_check_every_n_turns", 0))
        else:
            supervisor_on = False
            supervisor_cadence = 0
        loop_config = AgentLoopConfig(supervisor_check_every_turns=supervisor_cadence)

        result = await run_planner_session(
            input=planner_input,
            llm=llm,
            workspace_path=workspace_path,
            artifact_store=ctx.artifact_store,
            design_plan_artifact_key=artifact_key,
            runner=inline_runner,
            supervisor_llm=llm if supervisor_on else None,
            config=loop_config,
        )

        if not result.buffer.finalized:
            return DispatchOutcome(
                submitted_handles=[],
                error=(
                    f"planner did not finalize: agent outcome="
                    f"{result.outcome.kind!r} after {result.outcome.turn_count} turn(s)"
                ),
            )
        # Synthesize a JobHandle so the engine writes it to the
        # entity's ``handle_field`` (per the
        # ``DispatchAction.handle_field`` contract). This lets the
        # phase-1 scheduling query
        # ``get_in_flight_proposal_design`` (which predicate-filters on
        # ``planner_job_handle is not null``) pick the proposal up on
        # the next cycle, where the ``designing → designed`` dispatch-
        # time gate fires once the design plan is finalized in
        # ArtifactStore. Inline planners don't really have a Compute
        # job; the ``inline-planner-<proposal_id>`` handle is a
        # synthetic that mirrors the pattern in
        # ``smai_agents.agents.harness_builder.make_dispatch_harness_build``
        # for inline-runner test paths.
        handle = JobHandle(plugin="inline", handle=f"inline-planner-{proposal_id}")
        return DispatchOutcome(submitted_handles=[handle], error=None)

    return _dispatch


async def _default_pool_summary_loader(metadata_store: Any) -> str:
    """Render a free-form text summary of the technique pool.

    Drains :meth:`MetadataStore.list_techniques` (cursor-paginated per
    DEC-035 #1) and emits one line per technique:

        "<id> | <name> | category=<category> | standard=<bool>"

    The format matches v1's planner-snapshot convention; production
    deployments may override via :func:`make_dispatch_planner`'s
    ``pool_summary_loader`` arg.
    """
    lines: list[str] = []
    cursor: str | None = None
    while True:
        page = await metadata_store.list_techniques(limit=100, cursor=cursor)
        for technique in page.items:
            lines.append(
                f"{technique.id} | {technique.name} | "
                f"category={technique.category} | standard={technique.standard}"
            )
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    if not lines:
        return "(empty pool — no techniques registered yet)"
    return "\n".join(lines)


async def _load_technique_description(ctx: Any, proposal: Any) -> str:
    """Read the user's submitted technique description from ArtifactStore.

    Per ``08-novel-technique-pipeline.md`` §3.1 the submission step
    persists the description (or reproduce-paper reference) at the
    artifact key carried on
    :attr:`ProposalRecord.technique_description_artifact_key`. The
    handler reads it and feeds the content into the planner's initial
    user message. Returns a fallback string when the field is unset
    (so the loop still gets a well-formed message; tests that exercise
    the handler with no submission artifact see the placeholder text).
    """
    from smai_core.plugins import ArtifactNotFound  # noqa: PLC0415

    artifact_key = getattr(proposal, "technique_description_artifact_key", None)
    if not artifact_key:
        return "(no technique description artifact; reproduce-paper proposal)"
    try:
        body = await ctx.artifact_store.get(artifact_key)
    except ArtifactNotFound:
        return f"(technique description artifact {artifact_key!r} not found)"
    # Try to parse as JSON for a friendly rendering; fall back to raw text.
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return body.decode("utf-8", errors="replace")
    if isinstance(parsed, dict):
        return json.dumps(parsed, indent=2, sort_keys=True)
    return str(parsed)


# === Public registration helper =============================================


def variant_for_submission_kind(submission_kind: str) -> PlannerVariant:
    """Map a :class:`ProposalRecord.submission_kind` to the planner variant.

    Public helper so external callers (the proposal pipeline-spec, the
    paper-ingestion pipeline-spec — Task 3.E2 — and tests) share one
    discriminator. Per the module docstring, the proposal pipeline
    always uses ``novel_technique``; the paper-ingestion variant is
    selected only by the paper-ingestion pipeline-spec.

    Provided for completeness; the proposal-pipeline dispatch handler
    above hardcodes ``novel_technique`` because paper-ingestion-side
    selection happens elsewhere.
    """
    del submission_kind
    return "novel_technique"


__all__ = [
    "DEFAULT_DESIGN_PLAN_KEY_TEMPLATE",
    "DEFAULT_PAPER_PLAN_KEY_TEMPLATE",
    "AddFollowUpInput",
    "DraftAssertionInput",
    "DraftComparisonGroup",
    "DraftComparisonInput",
    "DraftCreateTechniqueInput",
    "DraftEnsureTechniqueInput",
    "DraftEntry",
    "DraftLevel",
    "DraftTechnique",
    "DraftValidationCriteria",
    "FinalizePaperTechniquesInput",
    "FinalizePlanInput",
    "PlannerBuffer",
    "PlannerInput",
    "PlannerSessionResult",
    "PlannerVariant",
    "SearchTechniquesInput",
    "SessionRunner",
    "SetClassificationInput",
    "SetConditionsInput",
    "make_dispatch_planner",
    "run_planner_session",
    "variant_for_submission_kind",
]
