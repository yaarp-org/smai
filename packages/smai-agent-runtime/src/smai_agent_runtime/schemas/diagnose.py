"""D7c diagnose step input + output Pydantic schemas.

Lifted from the design-note sketch at
``~/projects/Yaarp/designs/smai/agent_refactor/design_notes/diagnostic_bundle.md``
and ``spikes/d7c/spike.py``. The design-note artifact is documentation;
this module is the canonical Pydantic source.

Per architectural_decisions.md §12 #1 "Design discipline": the bundle is
the agent's complete informational world; ``read_file`` is the diagnose-
step-only escape hatch for genuinely unforeseen lookups. Bundle-
completeness lives in the schema, not in tool latitude. If the agent's
diagnose output suggests "I would need to see X," the fix is to extend
this schema, not to widen the tool surface.

Schemas:

* :class:`PriorDiagnosis` — structured summary of one prior diagnose
  attempt in the same retry chain (not full text replay).
* :class:`HarnessOriginalBundleRef` / :class:`TechniqueOriginalBundleRef` —
  discriminated-union reference to the D7a / D7b bundle that produced
  the failing code. Carries the load-bearing context fields rather than
  re-shipping the full bundle (which would double per-retry tokens per
  design-note §3-4).
* :class:`DiagnoseBundle` — full input bundle for one diagnose step
  attempt. Mirrors the D7c spike's shape exactly.
* :class:`Diagnosis` — structured output the mini-orchestrator applies.
  ``fix_target`` literal carries all four writable surfaces; the
  (``fix_target``, ``original_input_bundle.kind``) cross-check fires in
  the mini-orchestrator's fix-application code, not as a Pydantic
  ``model_validator`` (the schema stays loose-but-typed, the
  mini-orchestrator stays the gatekeeper per the design-note open-
  question discussion).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# Single source of truth for the four-surface fix target. The mini-
# orchestrator's per-surface writer table is keyed off this literal.
FixTarget = Literal["harness", "baseline", "config", "technique"]


class PriorDiagnosis(BaseModel):
    """Structured summary of one prior diagnose attempt.

    Structured summary, not full-text replay, so a retry-3 bundle stays
    well under the 50 KiB token budget (design-note rationale §2). The
    ``outcome`` enum lets the agent see whether the prior fix was
    applied at all (``did_not_apply`` = mini-orchestrator rejected for
    fix_target / kind mismatch) and what flavor of follow-up failure
    (lint vs validation) hit.
    """

    model_config = ConfigDict(extra="forbid")

    attempt_index: int = Field(..., ge=0)
    diagnosis_summary: str = Field(
        ...,
        description="One-line summary of what the prior diagnosis said.",
    )
    fix_target: FixTarget
    fix_payload_preview: str = Field(
        ...,
        description="First ~200 chars of the prior fix_payload; full text not retained.",
    )
    outcome: Literal[
        "did_not_apply",
        "applied_but_validation_failed_again",
        "applied_but_lint_failed_again",
    ]


class HarnessOriginalBundleRef(BaseModel):
    """Structured summary of the D7a harness body-generation bundle that
    produced the failing code.

    Reference shape (load-bearing fields) rather than the full bundle,
    because the failing ``current_code`` already shows what the agent
    produced from those fields. The ``harness_api_reference_pointer``
    is a workspace-relative path the agent can ``read_file`` from if it
    needs the verbatim reference doc.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["harness"] = "harness"
    function_signature: str = Field(
        ...,
        description=(
            "Signature of the function the body-generation step filled "
            "(e.g., 'build_harness(config: dict[str, Any]) -> dict[str, Any]')."
        ),
    )
    factor_type: Literal["additive", "substitutive"]
    extension_points_summary: str = Field(
        ...,
        description=(
            "Compact textual summary of the manifest extension_points "
            "(key + type_signature + integration_pattern), one per line. "
            "Empty for validation-mode runs against the stub manifest."
        ),
    )
    harness_api_reference_pointer: str = Field(
        default="workspace://harness_api_reference.md",
        description="Workspace-relative pointer; agent may read via the read_file escape hatch.",
    )


class TechniqueOriginalBundleRef(BaseModel):
    """Structured summary of the D7b technique body-generation bundle.

    Symmetric to :class:`HarnessOriginalBundleRef` for technique-flow
    failures. Carries the technique identity + extension-point binding
    + grounding flavor so the diagnose agent can reason about whether
    the technique misunderstood the contract.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["technique"] = "technique"
    technique_name: str
    extension_point_binding: str
    factor_type_context: Literal["additive", "substitutive"]
    grounding_kind: Literal["paper_extract", "method_description", "description_only"]
    grounding_summary: str = Field(
        ...,
        description="One-paragraph compaction of the grounding context.",
    )


OriginalInputBundleRef = Annotated[
    HarnessOriginalBundleRef | TechniqueOriginalBundleRef,
    Field(discriminator="kind"),
]
"""Discriminated union over the two original-bundle reference shapes."""


class DiagnoseBundle(BaseModel):
    """Input bundle for one :class:`DiagnoseOnFailureStep` attempt.

    Per architectural_decisions.md §12 #1: the bundle is the agent's
    complete informational world. ``read_file`` exists at this step
    only as a controlled escape hatch (5-call per-attempt cap, workspace-
    rooted paths only).

    Stderr-tail cap: 100 lines + 16 KiB byte ceiling per the design-note
    §1 reasoning (research_report §4.3 #1 line cap plus byte-defense
    against pathological single-line tracebacks).
    """

    model_config = ConfigDict(extra="forbid")

    failure_reason: str = Field(
        ...,
        description=(
            "Structured exit-status string from the experiment.py runner "
            "(EXIT_RUNTIME_FAILURE / EXIT_USAGE_ERROR codes) plus a "
            "one-line synopsis of what went wrong."
        ),
    )
    container_stderr: str = Field(
        ...,
        description=(
            "Tail of the validation container's stderr. Last 100 lines, "
            "byte-capped to 16 KiB per design-note §1."
        ),
    )
    validation_results: dict[str, object] | None = Field(
        default=None,
        description=(
            "Parsed validation_results.json contents if present. The "
            "runtime writes this only on the success path, so on a "
            "failed validation it is typically absent (None)."
        ),
    )
    current_code: dict[str, str] = Field(
        ...,
        description=(
            "Workspace files the failing run produced, keyed by "
            "workspace-relative path. Always includes the most recently-"
            "written file at minimum."
        ),
    )
    prior_diagnoses: list[PriorDiagnosis] = Field(
        default_factory=list[PriorDiagnosis],
        description=(
            "Structured summaries of prior diagnose-step attempts in "
            "this retry chain. Empty on the first attempt."
        ),
    )
    original_input_bundle: OriginalInputBundleRef = Field(
        ...,
        description=(
            "Structured summary of the body-generation bundle (D7a or "
            "D7b) that produced the failing code. Discriminated by the "
            "'kind' field. Carries enough to ground the diagnose "
            "without re-shipping the full bundle."
        ),
    )


class Diagnosis(BaseModel):
    """Output of one :class:`DiagnoseOnFailureStep` agent call.

    The mini-orchestrator dispatches ``fix_payload`` to the surface
    named by ``fix_target`` (overwrite harness/__init__.py, rewrite
    techniques/baseline.py, edit config.yaml, or — for technique flows —
    rewrite techniques/<name>.py).
    """

    model_config = ConfigDict(extra="forbid")

    diagnosis: str = Field(
        ...,
        min_length=20,
        description=(
            "One-paragraph explanation of the failure's root cause. "
            "Names the offending construct concretely (file:line if "
            "identifiable from the stderr tail)."
        ),
    )
    fix_target: FixTarget = Field(
        ...,
        description=(
            "Which writable surface the fix lands on. The harness-builder "
            "workflow accepts harness / baseline / config; the "
            "technique-implementer workflow additionally accepts technique."
        ),
    )
    fix_payload: str = Field(
        ...,
        min_length=1,
        description=(
            "Full replacement content for the target surface. For "
            "harness / baseline / technique: complete file source. For "
            "config: complete config.yaml content. The mini-orchestrator "
            "writes the payload to the target path verbatim."
        ),
    )


# Stderr-tail discipline. Mirrors the design-note §1 / research_report
# §4.3 #1 conventions; lives as a module-level constant so the diagnose
# bundle assembly call site reads "tail this many lines, but cap at this
# many bytes" against names rather than magic numbers.
STDERR_TAIL_LINES: int = 100
STDERR_TAIL_BYTES: int = 16 * 1024  # 16 KiB

# Per-attempt read_file budget. The agent that needs more than this is
# signalling a bundle-completeness gap; mini-orchestrator surfaces the
# overrun rather than silently allowing arbitrary read-file traffic.
READ_FILE_BUDGET_PER_ATTEMPT: int = 5

# Max bytes a single read_file response returns to the agent. Long
# files routinely blow context windows; truncate-with-flag at this
# boundary so the agent sees a structured "file truncated at N bytes"
# signal.
READ_FILE_MAX_BYTES: int = 50 * 1024  # 50 KiB


__all__ = [
    "READ_FILE_BUDGET_PER_ATTEMPT",
    "READ_FILE_MAX_BYTES",
    "STDERR_TAIL_BYTES",
    "STDERR_TAIL_LINES",
    "DiagnoseBundle",
    "Diagnosis",
    "FixTarget",
    "HarnessOriginalBundleRef",
    "OriginalInputBundleRef",
    "PriorDiagnosis",
    "TechniqueOriginalBundleRef",
]
