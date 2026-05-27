"""``generate_workflow(contract, role)`` — pure deterministic mapping
from ``(HarnessContract, TaskRole)`` to a workflow step list.

Per the D9 design note and arch §12 item 3: the mini-orchestrator's
workflow shape is methodology-driven Python, not an agent-decided
sequence. Same input produces a byte-identical step list; the output
is structurally hashable (every step is a Pydantic
``BaseModel(extra="forbid")``).

Two seams worth flagging for the future-extension story:

1. **ABI declarations.** The current ``HarnessContract`` does not
   carry per-function ABI declarations; the 3-function shape is
   implicit in ``RUNTIME_TEMPLATE_VERSION = "1.0.0"``. The generator
   reads the ABI from a module-level table keyed by runtime template
   version, with an ``abi_functions`` keyword-only override for
   testability. Production callers do not pass the override; the
   synthetic-N-function tests do. A future contract-shape change that
   lifts ABI declarations onto the contract collapses the table to a
   one-line accessor.

2. **Baseline-id resolution.** The baseline ``technique_id`` for a CG
   lives on the CG's baseline ``Entry``, not on ``HarnessContract``
   alone. The brief-mandated signature is ``(contract, role)``, so
   the generator emits the baseline step with a placeholder id; sub-PR
   B / C refine the resolution by passing the compiled CG (or its
   baseline-entry view) through to the generator.

Both seams are documented as open questions in the D9 design note.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

from smai_core.artifacts.harness_contract import HarnessContract
from smai_runtime.no_go_zone import RUNTIME_TEMPLATE_VERSION

from smai_agent_runtime.workflow.step_types import (
    BaselineGenerationStep,
    DiagnoseOnFailureStep,
    HarnessBuilderBodyGenerationStep,
    ManifestEmitStep,
    TechniqueImplementerBodyGenerationStep,
    ValidationStep,
    WorkflowStep,
)


class TaskRole(StrEnum):
    """The two sandboxed roles whose workflow shape this generator
    produces.

    Distinct from ``smai_agents.model_selection.TaskRole`` (which is a
    broader Literal over the eight-role fleet, host-side). The
    sandbox-side runtime only deals with the two sandboxed roles; the
    Enum form lets the generator's role-branching read as
    ``TaskRole.HARNESS_BUILDER`` rather than a stringly-typed
    comparison. Sub-PR B translates from the host-side Literal to this
    Enum at dispatch time.
    """

    HARNESS_BUILDER = "harness_builder"
    TECHNIQUE_IMPLEMENTER = "technique_implementer"


# === ABI declarations per runtime template version ============================


class _AbiFunction:
    """One ABI-function declaration: name, signature, write target.

    Plain dataclass-style holder rather than a Pydantic model — the
    table values are module-level constants, never deserialized from
    JSON, and the lighter-weight ``__slots__`` shape keeps the
    table noise-free.
    """

    __slots__ = ("name", "signature", "write_to_path")

    def __init__(self, name: str, signature: str, write_to_path: str) -> None:
        self.name = name
        self.signature = signature
        self.write_to_path = write_to_path


_ABI_BY_RUNTIME_VERSION: dict[str, tuple[_AbiFunction, ...]] = {
    "1.0.0": (
        _AbiFunction(
            name="build_harness",
            signature="def build_harness(config: dict) -> HarnessComponents:",
            write_to_path="harness/__init__.py",
        ),
        _AbiFunction(
            name="run_training_loop",
            signature=(
                "def run_training_loop(components: HarnessComponents, config: dict, seed: int):"
            ),
            write_to_path="harness/__init__.py",
        ),
        _AbiFunction(
            name="evaluate",
            signature=(
                "def evaluate(trained_model, components: HarnessComponents, config: dict) -> dict:"
            ),
            write_to_path="harness/__init__.py",
        ),
    ),
}
"""ABI shape per ``RUNTIME_TEMPLATE_VERSION``. v1 (1.0.0) is the round-15
+ round-19 3-function ABI lifted from
``smai_runtime/templates/_files/``. Future runtime versions register
additional entries here; the generator dispatches on the contract's
runtime version (currently the live ``RUNTIME_TEMPLATE_VERSION``
constant — round-16 drift-guard discipline)."""


_BASELINE_PLACEHOLDER_ID: str = "BASELINE_PLACEHOLDER"
"""Placeholder baseline technique_id emitted into
:class:`BaselineGenerationStep`. Sub-PR B/C replaces this with the real
resolution by passing the compiled CG's baseline entry through to the
generator (the contract alone does not carry baseline-entry identity).
"""


_TECHNIQUE_PLACEHOLDER_ID: str = "TECHNIQUE_PLACEHOLDER"
"""Placeholder technique_id emitted into
:class:`TechniqueImplementerBodyGenerationStep`. Sub-PR B/C replaces
this with the entry-id-driven resolution; the workflow shape is the
same either way."""


def _resolve_abi(contract: HarnessContract) -> tuple[_AbiFunction, ...]:
    """Look up the ABI for the contract's runtime template version.

    The contract does not currently carry the runtime template version
    (the version pins manifest emission, not contract compile), so the
    generator uses ``RUNTIME_TEMPLATE_VERSION`` from
    ``smai_runtime.no_go_zone`` as the live source. Round-16's
    drift-guard discipline: introspect the version from the runtime at
    generator import time rather than hardcoding a version string into
    the workflow output.
    """
    _ = contract  # placeholder; future contracts may carry a version field
    abi = _ABI_BY_RUNTIME_VERSION.get(RUNTIME_TEMPLATE_VERSION)
    if abi is None:
        raise ValueError(
            f"no ABI registered for runtime template version {RUNTIME_TEMPLATE_VERSION!r}"
        )
    return abi


def generate_workflow(
    contract: HarnessContract,
    role: TaskRole,
    *,
    abi_functions: Iterable[_AbiFunction] | None = None,
) -> list[WorkflowStep]:
    """Pure deterministic mapping from ``(contract, role)`` to a
    workflow step list.

    Same input produces byte-identical output. The mini-orchestrator
    iterates the returned list; no mid-iteration mutation happens
    upstream.

    ``abi_functions`` is a testability override: pass it to exercise a
    synthetic N-function contract without registering a new
    ``RUNTIME_TEMPLATE_VERSION``. Production callers pass ``None``.
    """
    abi = tuple(abi_functions) if abi_functions is not None else _resolve_abi(contract)

    if role is TaskRole.HARNESS_BUILDER:
        return _generate_harness_builder_workflow(contract, abi)
    if role is TaskRole.TECHNIQUE_IMPLEMENTER:
        return _generate_technique_implementer_workflow(contract)
    raise ValueError(f"role {role!r} has no sandboxed workflow shape")


def _generate_harness_builder_workflow(
    contract: HarnessContract,
    abi: tuple[_AbiFunction, ...],
) -> list[WorkflowStep]:
    """Harness_builder shape per research_report §6.1 + arch §12 item 3.

    * One :class:`HarnessBuilderBodyGenerationStep` per ABI function
      (3 instances for v1; 5 for a hypothetical 5-function future).
    * One :class:`BaselineGenerationStep` (factor-type-conditional
      emission is a sub-PR B/C refinement; sub-PR A emits one
      unconditionally with the placeholder baseline id).
    * One :class:`ValidationStep` against the baseline.
    * One :class:`DiagnoseOnFailureStep` anchored to the validation step.
    * One :class:`ManifestEmitStep`.

    Scripted between-step actions (workspace materialize, write-to-file,
    ruff lint, ArtifactStore publish) are the mini-orchestrator main's
    responsibility, not first-class step types — keeps the registry
    aligned with the D9 brief's narrow six-type list.
    """
    steps: list[WorkflowStep] = []

    for idx, fn in enumerate(abi):
        steps.append(
            HarnessBuilderBodyGenerationStep(
                function_index=idx,
                function_name=fn.name,
                function_signature=fn.signature,
                write_to_path=fn.write_to_path,
            )
        )

    factor_type = contract.body.factor.type
    steps.append(
        BaselineGenerationStep(
            factor_type=factor_type,
            baseline_technique_id=_BASELINE_PLACEHOLDER_ID,
            write_to_path="techniques/baseline.py",
        )
    )

    seed = contract.body.seeds[0] if contract.body.seeds else 0
    validation_index = len(steps)
    steps.append(
        ValidationStep(
            technique_id=_BASELINE_PLACEHOLDER_ID,
            seed=seed,
            dispatch_target="subprocess",
        )
    )
    steps.append(DiagnoseOnFailureStep(anchor_step_index=validation_index))

    steps.append(
        ManifestEmitStep(
            runtime_template_version=RUNTIME_TEMPLATE_VERSION,
            parent_harness_contract_hash=contract.envelope.content_hash,
        )
    )

    return steps


def _generate_technique_implementer_workflow(
    contract: HarnessContract,
) -> list[WorkflowStep]:
    """Technique_implementer shape per D9.

    Three entries, fixed shape:

    * Exactly one :class:`TechniqueImplementerBodyGenerationStep` (one
      technique implementation = one body-generation step; not
      parametric on the ABI).
    * One :class:`ValidationStep` against the implemented technique.
    * One :class:`DiagnoseOnFailureStep` anchored to the validation step.

    The seed comes from the contract; the technique_id is dispatch-time
    state (the entry-id the outer orchestrator assigned), placeholder
    here per the D9 sketch — sub-PR B/C threads the real id through.
    """
    seed = contract.body.seeds[0] if contract.body.seeds else 0

    steps: list[WorkflowStep] = [
        TechniqueImplementerBodyGenerationStep(
            technique_id=_TECHNIQUE_PLACEHOLDER_ID,
            write_to_path=f"techniques/{_TECHNIQUE_PLACEHOLDER_ID}.py",
        ),
    ]
    validation_index = len(steps)
    steps.append(
        ValidationStep(
            technique_id=_TECHNIQUE_PLACEHOLDER_ID,
            seed=seed,
            dispatch_target="subprocess",
        )
    )
    steps.append(DiagnoseOnFailureStep(anchor_step_index=validation_index))

    return steps


__all__ = [
    "TaskRole",
    "generate_workflow",
]
