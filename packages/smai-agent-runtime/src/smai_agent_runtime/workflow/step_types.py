"""``WorkflowStep`` type hierarchy + registry.

Per ``designs/smai/agent_refactor/architectural_decisions.md`` §12 items 3
and 4 and the D9 design note. Two parallel surfaces over the same set of
step types:

* a Pydantic discriminated union (``WorkflowStep``) keyed on the
  ``step_type`` literal — used by ``generate_workflow`` to return a typed
  list and by ``WorkflowStepAdapter`` to deserialize step payloads across
  the host-worker / sandbox boundary.
* a module-level registry (``STEP_REGISTRY``) keyed by the same literal —
  used by tests that assert "is ``apply_review_feedback`` registered even
  though no generator rule emits one?" and by the future mini-orchestrator
  dispatcher's per-step lookup.

The two body-generation variants live in the union as distinct
discriminator values (``harness_body_generation`` and
``technique_body_generation``) per D9's "two schemas not one" rationale:
the harness-builder step carries function-level identification while the
technique-implementer step carries technique_id + grounding. A shared
``BodyGenerationStep`` base lets ``isinstance(s, BodyGenerationStep)``
return True for either variant, so generator outputs can be filtered in
tests without enumerating both concrete types.

``ApplyReviewFeedbackStep`` is registered with no generator rule
producing it — the resume-prep hedge per arch §12 item 4. Sub-PR A only
lands the step type; the input/output Pydantic schemas land alongside
the real PydanticAI integration in a follow-up sub-PR.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

StepKind = Literal["agent_reasoning", "scripted"]
"""Whether a step's body invokes a PydanticAI Agent (``agent_reasoning``)
or a deterministic Python helper (``scripted``). Read by the
mini-orchestrator's per-step dispatcher (sub-PR B)."""


class WorkflowStepBase(BaseModel):
    """Common Pydantic base for every WorkflowStep variant.

    Subclasses declare ``step_type`` as a ``Literal[...]`` whose value is
    the discriminator. ``kind`` is a ``ClassVar`` (not a Pydantic field,
    not in the serialized output) used by the mini-orchestrator to
    branch agent-reasoning vs scripted dispatch without an isinstance
    ladder over every concrete type.
    """

    model_config = ConfigDict(extra="forbid")
    kind: ClassVar[StepKind]


# === Agent-reasoning steps ====================================================


class BodyGenerationStep(WorkflowStepBase):
    """Shared base for the two body-generation variants.

    Not directly emitted by ``generate_workflow``; used as the
    isinstance target when a caller wants "any body-generation step"
    without enumerating both concrete subclasses.
    """

    kind: ClassVar[StepKind] = "agent_reasoning"

    write_to_path: str


class HarnessBuilderBodyGenerationStep(BodyGenerationStep):
    """One harness ABI function's body.

    ``generate_workflow`` emits N instances for an N-function contract
    (arch §12 item 3). The input bundle assembled at dispatch time
    (sub-PR C) materializes the per-function context from these
    identification fields plus the contract.
    """

    step_type: Literal["harness_body_generation"] = "harness_body_generation"

    function_index: int
    function_name: str
    function_signature: str


class TechniqueImplementerBodyGenerationStep(BodyGenerationStep):
    """The single technique-implementation step in a technique_implementer
    workflow. One instance per generated workflow; not parametric on the
    ABI."""

    step_type: Literal["technique_body_generation"] = "technique_body_generation"

    technique_id: str


class BaselineGenerationStep(WorkflowStepBase):
    """The baseline-technique-body step in harness_builder workflows.

    ``factor_type`` is read from ``contract.body.factor.type`` and
    passed through so sub-PR C's dispatcher can assemble the right
    bundle without re-reading the contract.
    """

    step_type: Literal["baseline_generation"] = "baseline_generation"
    kind: ClassVar[StepKind] = "agent_reasoning"

    factor_type: Literal["additive", "substitutive"]
    baseline_technique_id: str
    write_to_path: str


class DiagnoseOnFailureStep(WorkflowStepBase):
    """The diagnose step that runs after a scripted ValidationStep failure.

    Always emitted by the generator immediately after each
    ValidationStep. The mini-orchestrator invokes its body only when the
    anchor step's outcome was failure; on success it passes through as a
    no-op. ``max_retries`` bounds the diagnose-then-fix-then-retry loop
    that runs inside the step.
    """

    step_type: Literal["diagnose_on_failure"] = "diagnose_on_failure"
    kind: ClassVar[StepKind] = "agent_reasoning"

    anchor_step_index: int
    max_retries: int = 3


# === Scripted steps ===========================================================


class ValidationStep(WorkflowStepBase):
    """Scripted CPU validation smoke (``python experiment.py --mode validation``).

    ``dispatch_target`` carries the architectural_decisions §1 escape
    hatch as a step-level field: ``subprocess`` runs validation
    in-sandbox; ``compute_submit`` instead calls back to the host-side
    Compute Protocol. The §1 escape hatches plug in by flipping this
    field, not by changing the workflow shape.
    """

    step_type: Literal["validation"] = "validation"
    kind: ClassVar[StepKind] = "scripted"

    technique_id: str
    seed: int
    dispatch_target: Literal["subprocess", "compute_submit"] = "subprocess"


class ManifestEmitStep(WorkflowStepBase):
    """Scripted ``HarnessAPIManifest`` emit (harness_builder only).

    ``runtime_template_version`` is pinned at generator time from
    ``smai_runtime.no_go_zone.RUNTIME_TEMPLATE_VERSION`` (round-16
    drift-guard pattern: introspect the live runtime, do not hardcode
    the version string in the workflow output).
    ``parent_harness_contract_hash`` carries the contract's
    ``envelope.content_hash`` so the emitted manifest can reference its
    parent contract without re-loading it.
    """

    step_type: Literal["manifest_emit"] = "manifest_emit"
    kind: ClassVar[StepKind] = "scripted"

    runtime_template_version: str
    parent_harness_contract_hash: str


# === Resume-prep step (registered, not produced) ==============================


class ApplyReviewFeedbackStep(WorkflowStepBase):
    """Future-extension step: re-resolve a prior body / baseline step's
    output given code_review feedback.

    Registered in ``STEP_REGISTRY`` and included in the discriminated
    union for type-system completeness, but no ``generate_workflow``
    rule emits one in sub-PR A. The architecture-only hedge per
    arch §12 item 4: when the multi-cycle review-feedback loop ships,
    adding the generator rule + outer-orchestrator wiring is a small
    diff against an already-registered type.
    """

    step_type: Literal["apply_review_feedback"] = "apply_review_feedback"
    kind: ClassVar[StepKind] = "agent_reasoning"

    target_step_index: int
    max_retries: int = 1


# === Discriminated union + registry ===========================================


WorkflowStep = Annotated[
    HarnessBuilderBodyGenerationStep
    | TechniqueImplementerBodyGenerationStep
    | BaselineGenerationStep
    | DiagnoseOnFailureStep
    | ValidationStep
    | ManifestEmitStep
    | ApplyReviewFeedbackStep,
    Field(discriminator="step_type"),
]
"""Pydantic discriminated union over every registered step type.

Use ``WorkflowStepAdapter.validate_python(...)`` to parse a step dict;
the concrete subclasses' ``model_dump`` / ``model_validate_json``
round-trip through the union by construction (the ``step_type``
discriminator is preserved on the wire)."""


WorkflowStepAdapter: TypeAdapter[WorkflowStep] = TypeAdapter(WorkflowStep)


STEP_REGISTRY: dict[str, type[WorkflowStepBase]] = {}
"""Module-level dict keyed by ``step_type`` literal value.

Intentionally redundant with the discriminated union: the union gives
Pydantic parsing, the registry gives runtime introspection (tests
asserting registry presence without importing every concrete type;
mini-orchestrator dispatcher looking up handlers by step-type key).

Registration is import-time and explicit. No entry-point indirection
(DEC-026 dbt-adapter pattern) because step types are sandbox-side leaf
code, not a substrate plugin surface; future variants register in this
module."""


def register_step(cls: type[WorkflowStepBase]) -> type[WorkflowStepBase]:
    """Register ``cls`` in :data:`STEP_REGISTRY` keyed by its
    ``step_type`` literal default.

    Raises ``TypeError`` if ``cls`` does not declare a ``step_type``
    Literal field with a string default, and ``ValueError`` on
    discriminator collisions.
    """
    step_type_field = cls.model_fields.get("step_type")
    if step_type_field is None:
        raise TypeError(f"{cls.__name__} is missing a step_type Literal field")
    discriminator = step_type_field.default
    if not isinstance(discriminator, str) or not discriminator:
        raise TypeError(f"{cls.__name__}.step_type has no string default discriminator")
    existing = STEP_REGISTRY.get(discriminator)
    if existing is not None and existing is not cls:
        raise ValueError(f"step_type {discriminator!r} already registered to {existing.__name__}")
    STEP_REGISTRY[discriminator] = cls
    return cls


# Explicit registration (rather than decorator-at-definition) so source order
# and registry-population order stay independent — easier to diff and easier
# to extend without re-reading every class body for the decorator.
for _cls in (
    HarnessBuilderBodyGenerationStep,
    TechniqueImplementerBodyGenerationStep,
    BaselineGenerationStep,
    DiagnoseOnFailureStep,
    ValidationStep,
    ManifestEmitStep,
    ApplyReviewFeedbackStep,
):
    register_step(_cls)


__all__ = [
    "STEP_REGISTRY",
    "ApplyReviewFeedbackStep",
    "BaselineGenerationStep",
    "BodyGenerationStep",
    "DiagnoseOnFailureStep",
    "HarnessBuilderBodyGenerationStep",
    "ManifestEmitStep",
    "StepKind",
    "TechniqueImplementerBodyGenerationStep",
    "ValidationStep",
    "WorkflowStep",
    "WorkflowStepAdapter",
    "WorkflowStepBase",
    "register_step",
]
