"""Body-generation input + output Pydantic schemas (D7a + D7b).

Lifted from the design-note sketches at
``~/projects/Yaarp/designs/smai/agent_refactor/design_notes/harness_body_generation_schema.py``
(D7a) and ``technique_body_generation_schemas.py`` (D7b). The design-note
files are documentation; this module is the canonical Pydantic source.

Per architectural_decisions.md §12 #1: bundle-completeness belongs in
the schema, not in agent escape-hatch tools. Body-generation steps have
zero ``read_file`` access; if the agent appears to need information not
in the bundle, the fix is to extend these schemas, not to give the
agent free-form workspace access.

Schemas land here:

* :class:`HarnessBuilderBodyGenerationInput` + :class:`HarnessBuilderBodyGenerationOutput` (D7a)
* :class:`TechniqueBodyGenerationBundle` + :class:`TechniqueBodyOutput` (D7b)
* Five-variant :data:`GroundingContext` discriminated union (D7b)
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
from smai_core.artifacts.harness_contract import FixedVariable
from smai_core.entities.factor import Factor

# Closed v1 ABI of harness/__init__.py per 10-runtime-and-templates.md §8.2
# and smai_runtime.runner steps 5 / 9 / 10. A Literal locks the discriminator
# to the byte-stable surface; if the ABI ever grows to 4 functions, this
# Literal is the one place to update.
ABIFunctionName = Literal["build_harness", "run_training_loop", "evaluate"]

# Mirrors smai_runtime.manifest.IntegrationPattern (closed v1). Re-stated
# here as a Literal so the schema's JSON Schema (which PydanticAI
# auto-renders into the agent prompt) names the admissible values inline.
IntegrationPattern = Literal["append", "replace", "wrap", "override_dict"]


# === D7a: harness builder body-generation schema =============================


class FunctionSignature(BaseModel):
    """The ABI function the step is filling."""

    model_config = ConfigDict(extra="forbid")

    name: ABIFunctionName
    signature: str = Field(
        description=(
            "The full Python signature line (e.g. "
            "`def build_harness(config: dict[str, Any]) -> HarnessComponents:`). "
            "Verbatim from the skeleton; the agent does not change it."
        )
    )
    purpose: str = Field(
        description=(
            "One sentence describing what the function must do at runtime "
            "(e.g. 'Construct the experiment harness from `config`, called "
            "once per run by smai_runtime.runner step 5')."
        )
    )


class ExtensionPointSpec(BaseModel):
    """One manifest extension point the function must accommodate.

    Mini-orchestrator constructs this list from ``HarnessContract.body.factor``
    plus ``smai_runtime.COMPONENT_FIELD_FOR_KEY`` +
    ``ADMISSIBLE_PATTERNS_FOR_KEY`` (round-15 introspection pattern).
    """

    model_config = ConfigDict(extra="forbid")

    key: str = Field(
        description=(
            "Manifest extension-point key (closed v1 set: train_transforms, "
            "model_wrapper, loss_fn, training_overrides, callbacks)."
        )
    )
    component_field: str = Field(
        description="HarnessComponents field the splicer writes into (e.g. 'train_transforms')."
    )
    admissible_patterns: list[IntegrationPattern]
    factor_role: Literal["additive_baseline_default", "substitutive_slot"] = Field(
        description=(
            "'additive_baseline_default': build_harness must initialize the "
            "field with a working no-op (the additive baseline runs through "
            "the harness as-is per 10-runtime-and-templates.md §8.4). "
            "'substitutive_slot': the slot is mandatory and the splicer fills it."
        )
    )


class LintFailure(BaseModel):
    """Lint-step failure record for prior-attempt context."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["lint"] = "lint"
    linter: str = Field(description="e.g. 'ruff'.")
    output: str = Field(
        description=(
            "Captured linter stdout/stderr. Mini-orchestrator caps this at "
            "~3KB (head + tail); the cap is a workflow-level convention, not "
            "schema-enforced, so retries on different linters can adjust."
        )
    )


class ValidationFailure(BaseModel):
    """Runtime --mode validation failure (round-20 stderr-to-agent pattern)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["validation"] = "validation"
    exit_code: int
    canonical_status: str = Field(
        description=(
            "Mirrors smai_runtime.runner exit-status canonical string "
            "(e.g. 'no_go_zone_violation', 'technique_output_contract_error', "
            "'harness_build_error', 'training_error')."
        )
    )
    stderr_tail: str = Field(
        description=(
            "Last ~100 lines of container stderr per the round-20 fix and "
            "research_report.md §4.3 #1 tail convention."
        )
    )


PriorFailureDetail = LintFailure | ValidationFailure


class PriorFailedAttempt(BaseModel):
    """One prior fill attempt's full code + structured failure context.

    Full module_source, not summary: the agent learns more from diffing its
    own prior code than from a paraphrase. 3 retries x ~2KB code stays well
    under the 50K-token bundle budget.
    """

    model_config = ConfigDict(extra="forbid")

    attempt_index: int = Field(ge=0)
    prior_module_source: str
    failure: PriorFailureDetail = Field(discriminator="kind")


class HarnessBuilderBodyGenerationInput(BaseModel):
    """Input bundle to one HarnessBuilderBodyGenerationStep.

    Instantiated once per ABI function in HarnessContract (3 today; N if
    the ABI grows). Bundle-complete per architectural_decisions.md §12 #1:
    no ``read_file`` at body-generation steps. If the agent appears to need
    information not in the bundle, extend the schema to include it.
    """

    model_config = ConfigDict(extra="forbid")

    target_file_path: str = Field(
        description=(
            "Workspace-relative path the mini-orchestrator writes the "
            "structured-output module_source to (e.g. 'harness/__init__.py')."
        )
    )
    function_signature: FunctionSignature
    extension_points: list[ExtensionPointSpec] = Field(
        description=(
            "Manifest extension points the function must accommodate. "
            "Constructed by the mini-orchestrator from "
            "smai_runtime.COMPONENT_FIELD_FOR_KEY filtered by which keys "
            "the contract's factor type implicates."
        )
    )
    factor: Factor = Field(
        description=(
            "HarnessContract.body.factor verbatim. Carries the operator's "
            "prose factor description the agent grounds against."
        )
    )
    locked_config: list[FixedVariable] = Field(
        description=(
            "HarnessContract.body.fixed_variables verbatim. The "
            "ceteris-paribus values the function must honor "
            "(02-dsl-and-contracts.md §7.4)."
        )
    )
    harness_api_reference: str = Field(
        description=(
            "Generated markdown from the round-15 harness API reference "
            "generator. HarnessComponents field-set + extension-point "
            "mapping + ABI documentation."
        )
    )
    current_module_source: str = Field(
        description=(
            "Current contents of target_file_path. Skeleton on the first "
            "body-generation step; partially filled on subsequent steps."
        )
    )
    prior_failed_attempts: list[PriorFailedAttempt] = Field(
        default_factory=list[PriorFailedAttempt],
        description=(
            "Empty on attempt 0. On retry: the mini-orchestrator appends "
            "the previous (source, failure) pair. RetryPolicy on the "
            "WorkflowStep caps list length."
        ),
    )


class HarnessBuilderBodyGenerationOutput(BaseModel):
    """Structured output the agent returns. PydanticAI Agent(output_type=...) bound.

    Single field for the new module source: agent rewrites the full file
    with the target function filled and other functions preserved. AST-
    splicing one body into the file from a body-only string is brittle
    in pure Python (docstring parsing, decorator preservation, import-
    order handling); the whole-file rewrite is simpler and matches what
    the D7a sub-spike showed agents naturally produce.
    """

    model_config = ConfigDict(extra="forbid")

    function_name: ABIFunctionName = Field(
        description=(
            "Echo of the target function for downstream dispatch and "
            "telemetry. Mini-orchestrator asserts equality with the input "
            "function_signature.name."
        )
    )
    module_source: str = Field(
        description=(
            "The full new contents of target_file_path. Mini-orchestrator "
            "writes verbatim, then runs lint + (optionally) the next "
            "WorkflowStep."
        )
    )
    reasoning: str = Field(
        max_length=1000,
        description=(
            "One- or two-paragraph note on how the agent satisfied the "
            "extension-point constraints. Bounded by max_length so the "
            "reasoning surface stays compact; persists into the per-step "
            "message_history for debugging."
        ),
    )


# === D7b: technique / baseline body-generation schema ========================


class PaperExtractGrounding(BaseModel):
    """Technique grounded in a paper-extracted method body.

    Source: ``papers/{arxiv_id}/techniques/{technique_id}/method_extraction.json``
    (an :class:`EnrichmentResult` serialization). Maps to
    PaperFidelityAnchor on the upstream TechniqueContract.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["paper_extract"] = "paper_extract"
    arxiv_id: str
    technique_id: str
    method_extraction: str = Field(
        description=(
            "The free-form Markdown method body the enricher produced "
            "(verbatim EnrichmentResult.method_extraction)."
        )
    )
    implementability: Literal["high", "medium", "blocked"]
    refined_description: str | None = None
    blocked_reason: str | None = None


class ProposalGrounding(BaseModel):
    """Technique grounded in a novel-technique proposal."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["proposal"] = "proposal"
    proposal_id: str
    technique_description: str = Field(
        description="Planner-authored prose from `TechniqueRef.description`."
    )
    proposal_text: str | None = None


class ReviewerAttestedGrounding(BaseModel):
    """Technique grounded in a reviewer-vetted inline spec.

    Source: inline ``spec_text`` on :class:`ReviewerAttestedFidelityAnchor`
    (no separate artifact lookup needed).
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["reviewer_attested"] = "reviewer_attested"
    spec_text: str
    attested_by: str | None = None


class StandardLibraryGrounding(BaseModel):
    """Standard technique (DEC-015): use library APIs from model knowledge."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["standard"] = "standard"
    technique_description: str


class NoOpBaselineGrounding(BaseModel):
    """Additive-factor baseline: produce a working no-op default."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["no_op_baseline"] = "no_op_baseline"


GroundingContext = Annotated[
    PaperExtractGrounding
    | ProposalGrounding
    | ReviewerAttestedGrounding
    | StandardLibraryGrounding
    | NoOpBaselineGrounding,
    Field(discriminator="kind"),
]
"""Five-variant discriminated union (DEC-017 + DEC-013).

Replaces the current 3-way ``ContextKind`` Literal in the legacy
technique_implementer code path. The variant ladder is strictly broader:
``proposal`` and ``reviewer_attested`` were both folded into
``method_description`` in the old code; here they stay distinct so the
prompt can reason about which grounding flavor it has."""


class PriorTechniqueAttempt(BaseModel):
    """One prior attempt's structured failure record."""

    model_config = ConfigDict(extra="forbid")

    attempt_index: int = Field(ge=0)
    prior_source: str = Field(
        description=(
            "Technique-module body the prior attempt produced. Tail-capped "
            "at bundle construction so retry-3 bundles do not grow "
            "unboundedly."
        )
    )
    failure_kind: Literal[
        "validation_failed",
        "container_stderr",
        "lint",
        "import_error",
        "review_findings",
        "other",
    ]
    failure_excerpt: str = Field(
        description=("Tail of stderr / validation output / review prose (~100 lines).")
    )


class TechniqueBodyGenerationBundle(BaseModel):
    """Input bundle shared by ``TechniqueImplementerBodyGenerationStep`` and
    ``BaselineGenerationStep``.

    Shape discipline (architectural_decisions §12 #1): everything the
    agent needs to produce a conforming ``techniques/<technique_name>.py``
    body is materialized inline. No path the agent must dereference; no
    expectation that the agent will read files outside this bundle.

    The schema is shared across the two step types because they produce
    identical artifact shape (a ``techniques/<name>.py`` body satisfying
    the same manifest). The ``step_kind`` discriminator lets the per-step
    prompts branch on "you are producing the baseline arm" vs "you are
    producing a variation arm".
    """

    model_config = ConfigDict(extra="forbid")

    step_kind: Literal["technique", "baseline"]

    cg_id: str
    entry_id: str
    technique_name: str = Field(
        description=("File stem the agent must produce: writes ``techniques/<technique_name>.py``.")
    )
    is_baseline: bool

    factor_dimension: str
    factor_type: Literal["additive", "substitutive"]

    technique_id: str | None
    technique_params: dict[str, str | int | float | bool | None] | None = Field(
        default=None,
        description=(
            "Per-entry parameter configuration (mirrors TechniqueContractBody.technique_params)."
        ),
    )

    grounding: GroundingContext

    harness_api_manifest_json: str = Field(
        description=(
            "Serialized HarnessAPIManifest. Carried as a JSON string rather "
            "than embedded Pydantic so the bundle's own JSON schema stays "
            "stable as the manifest evolves."
        )
    )

    harness_source: dict[str, str] = Field(
        description=(
            "Map relative path -> file content (UTF-8 text). Every file "
            "under ``harness/``. Cap: ~50KB total at bundle construction; "
            "overrun -> warn-and-truncate-largest."
        )
    )

    baseline_source: str | None = Field(
        default=None,
        description=(
            "Content of ``techniques/baseline.py`` from the prior step. "
            "None when (a) this IS the baseline step, or (b) the harness "
            "builder did not ship a baseline."
        ),
    )

    prior_failed_attempts: list[PriorTechniqueAttempt] = Field(
        default_factory=list[PriorTechniqueAttempt],
    )


class TechniqueBodyOutput(BaseModel):
    """Structured output of a single body-generation step's agent reasoning.

    Sufficient to be PydanticAI ``output_type``-bound: the agent produces
    exactly one of these per step invocation.
    """

    model_config = ConfigDict(extra="forbid")

    technique_py_source: str = Field(
        description=(
            "Full UTF-8 source of ``techniques/<technique_name>.py``. Must "
            "be syntactically valid Python and import the symbols the "
            "manifest's extension points declare."
        )
    )
    reasoning: str | None = Field(
        default=None,
        description=(
            "Optional short reasoning trace (~200 tokens cap by prompt). "
            "Aids debugging; not load-bearing."
        ),
    )


__all__ = [
    "ABIFunctionName",
    "ExtensionPointSpec",
    "FunctionSignature",
    "GroundingContext",
    "HarnessBuilderBodyGenerationInput",
    "HarnessBuilderBodyGenerationOutput",
    "IntegrationPattern",
    "LintFailure",
    "NoOpBaselineGrounding",
    "PaperExtractGrounding",
    "PriorFailedAttempt",
    "PriorFailureDetail",
    "PriorTechniqueAttempt",
    "ProposalGrounding",
    "ReviewerAttestedGrounding",
    "StandardLibraryGrounding",
    "TechniqueBodyGenerationBundle",
    "TechniqueBodyOutput",
    "ValidationFailure",
]
