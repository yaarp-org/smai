"""Tests for ``smai_agent_runtime.workflow``.

Covers the sub-PR A acceptance criteria from the agent-layer refactor
Step 4 brief:

* Ergonomic imports from ``smai_agent_runtime.workflow``.
* Parametric-on-ABI body-step count (3-function v1 contract → 3 body
  steps; synthetic 5-function override → 5 body steps).
* ``generate_workflow`` is deterministic (same inputs → equal outputs).
* ``ApplyReviewFeedbackStep`` is registered in :data:`STEP_REGISTRY`
  but never produced by ``generate_workflow``.
* Technique-implementer role produces the D9-mandated 3-step shape.
* Discriminated-union round-trip survives :data:`WorkflowStepAdapter`
  serialization.
"""

from __future__ import annotations

from typing import Literal

import pytest
from _workflow_fixtures import (  # type: ignore[import-not-found]
    SYNTHETIC_5_ABI,
    V1_ABI,
    make_contract,
)

# Ergonomic-import surface (acceptance criterion). Tests that fail to
# import here pre-empt the rest of the file.
from smai_agent_runtime.workflow import (
    STEP_REGISTRY,
    ApplyReviewFeedbackStep,
    BaselineGenerationStep,
    BodyGenerationStep,
    DiagnoseOnFailureStep,
    HarnessBuilderBodyGenerationStep,
    ManifestEmitStep,
    TaskRole,
    TechniqueImplementerBodyGenerationStep,
    ValidationStep,
    WorkflowStepAdapter,
    generate_workflow,
)
from smai_runtime.no_go_zone import RUNTIME_TEMPLATE_VERSION

# === Parametric-on-ABI body-step count =======================================


@pytest.mark.parametrize(
    "abi, expected_body_count",
    [
        pytest.param(V1_ABI, 3, id="v1_three_function_abi"),
        pytest.param(SYNTHETIC_5_ABI, 5, id="synthetic_five_function_abi"),
    ],
)
def test_body_generation_step_count_is_parametric_on_abi(
    abi: tuple[object, ...],
    expected_body_count: int,
) -> None:
    """Per arch §12 item 3: N-function contract → N body steps. The
    explicit ``abi_functions`` override is the testability seam that
    exercises a synthetic future contract without registering a new
    ``RUNTIME_TEMPLATE_VERSION``."""
    workflow = generate_workflow(
        contract=make_contract(),
        role=TaskRole.HARNESS_BUILDER,
        abi_functions=abi,  # type: ignore[arg-type]
    )
    body_steps = [s for s in workflow if isinstance(s, HarnessBuilderBodyGenerationStep)]
    assert len(body_steps) == expected_body_count
    assert [s.function_index for s in body_steps] == list(range(expected_body_count))


def test_v1_harness_builder_workflow_has_expected_tail_shape() -> None:
    """V1 contract (3 functions) emits: 3 body + 1 baseline + 1
    validation + 1 diagnose + 1 manifest = 7 steps. Tail shape locked
    so sub-PR B can rely on the structural positions."""
    workflow = generate_workflow(make_contract(), TaskRole.HARNESS_BUILDER)
    assert len(workflow) == 7
    assert isinstance(workflow[0], HarnessBuilderBodyGenerationStep)
    assert isinstance(workflow[1], HarnessBuilderBodyGenerationStep)
    assert isinstance(workflow[2], HarnessBuilderBodyGenerationStep)
    assert isinstance(workflow[3], BaselineGenerationStep)
    assert isinstance(workflow[4], ValidationStep)
    assert isinstance(workflow[5], DiagnoseOnFailureStep)
    assert isinstance(workflow[6], ManifestEmitStep)
    # The diagnose step anchors to the validation step's index so the
    # mini-orchestrator can look up the right prior outcome.
    assert workflow[5].anchor_step_index == 4


def test_bodygenerationstep_isinstance_matches_either_concrete_variant() -> None:
    """``BodyGenerationStep`` is the shared base; both concrete body
    steps inherit from it so generator consumers can filter on the
    category without enumerating both subclasses."""
    harness_workflow = generate_workflow(make_contract(), TaskRole.HARNESS_BUILDER)
    technique_workflow = generate_workflow(make_contract(), TaskRole.TECHNIQUE_IMPLEMENTER)

    harness_body = [s for s in harness_workflow if isinstance(s, BodyGenerationStep)]
    technique_body = [s for s in technique_workflow if isinstance(s, BodyGenerationStep)]
    assert all(isinstance(s, HarnessBuilderBodyGenerationStep) for s in harness_body)
    assert all(isinstance(s, TechniqueImplementerBodyGenerationStep) for s in technique_body)


# === Determinism =============================================================


def test_generate_workflow_is_deterministic_for_harness_builder() -> None:
    """Same contract + role pair produces equal step lists across
    calls. Pydantic ``BaseModel`` equality is by field value, so the
    asserted-equal lists round-trip through the deterministic-up-front
    framing."""
    contract = make_contract()
    a = generate_workflow(contract, TaskRole.HARNESS_BUILDER)
    b = generate_workflow(contract, TaskRole.HARNESS_BUILDER)
    assert a == b


def test_generate_workflow_is_deterministic_for_technique_implementer() -> None:
    contract = make_contract()
    a = generate_workflow(contract, TaskRole.TECHNIQUE_IMPLEMENTER)
    b = generate_workflow(contract, TaskRole.TECHNIQUE_IMPLEMENTER)
    assert a == b


# === ApplyReviewFeedbackStep registered-but-not-produced ======================


def test_apply_review_feedback_step_is_registered() -> None:
    """Per arch §12 item 4: the type must be discoverable in the
    registry so the future multi-cycle-review wiring lands as a
    generator-rule update plus outer-orchestrator wiring, not a
    type-system restructure."""
    assert "apply_review_feedback" in STEP_REGISTRY
    assert STEP_REGISTRY["apply_review_feedback"] is ApplyReviewFeedbackStep


@pytest.mark.parametrize(
    "factor_type",
    [
        pytest.param("additive", id="additive_factor"),
        pytest.param("substitutive", id="substitutive_factor"),
    ],
)
@pytest.mark.parametrize(
    "role",
    [
        pytest.param(TaskRole.HARNESS_BUILDER, id="harness_builder"),
        pytest.param(TaskRole.TECHNIQUE_IMPLEMENTER, id="technique_implementer"),
    ],
)
def test_no_generator_rule_emits_apply_review_feedback_step(
    factor_type: Literal["additive", "substitutive"],
    role: TaskRole,
) -> None:
    """Scans several plausible contract/role combinations: no rule
    in :func:`generate_workflow` produces an
    :class:`ApplyReviewFeedbackStep`. Sub-PR A's hedge is type-only;
    if a future contract shape accidentally triggers emission this
    test catches the regression."""
    workflow = generate_workflow(make_contract(factor_type=factor_type), role)
    assert not any(isinstance(s, ApplyReviewFeedbackStep) for s in workflow)
    assert not any(s.step_type == "apply_review_feedback" for s in workflow)


# === Technique-implementer role shape =========================================


def test_technique_implementer_workflow_shape() -> None:
    """Per D9: technique_implementer always emits exactly 3 steps —
    one body-generation + one validation + one diagnose-on-failure.
    Not parametric on ABI; the technique implementation is one file."""
    workflow = generate_workflow(make_contract(), TaskRole.TECHNIQUE_IMPLEMENTER)
    assert len(workflow) == 3
    assert isinstance(workflow[0], TechniqueImplementerBodyGenerationStep)
    assert isinstance(workflow[1], ValidationStep)
    assert isinstance(workflow[2], DiagnoseOnFailureStep)
    assert workflow[2].anchor_step_index == 1


# === Validation + manifest field assertions ==================================


def test_validation_step_defaults_to_subprocess_dispatch() -> None:
    """CPU subprocess is the default per arch §1; escape hatches flip
    ``dispatch_target`` rather than mutating the workflow shape."""
    workflow = generate_workflow(make_contract(), TaskRole.HARNESS_BUILDER)
    validations = [s for s in workflow if isinstance(s, ValidationStep)]
    assert len(validations) == 1
    assert validations[0].dispatch_target == "subprocess"


def test_manifest_emit_pins_live_runtime_template_version() -> None:
    """Round-16 drift-guard discipline: the version comes from
    introspecting the live runtime, not from a hardcoded string in the
    generator output."""
    workflow = generate_workflow(make_contract(), TaskRole.HARNESS_BUILDER)
    manifests = [s for s in workflow if isinstance(s, ManifestEmitStep)]
    assert len(manifests) == 1
    assert manifests[0].runtime_template_version == RUNTIME_TEMPLATE_VERSION


def test_manifest_emit_carries_contract_envelope_content_hash() -> None:
    """The emitted manifest references its parent contract by
    ``content_hash`` so sub-PR B's mini-orchestrator can write the
    parent-pointer without re-loading the contract."""
    contract = make_contract(content_hash="0123456789abcdef" * 4)
    workflow = generate_workflow(contract, TaskRole.HARNESS_BUILDER)
    manifests = [s for s in workflow if isinstance(s, ManifestEmitStep)]
    assert manifests[0].parent_harness_contract_hash == contract.envelope.content_hash


def test_baseline_generation_step_carries_contract_factor_type() -> None:
    """``BaselineGenerationStep.factor_type`` is read from
    ``contract.body.factor.type``. Sub-PR C's dispatcher reads this off
    the step rather than re-traversing the contract."""
    for factor_type in ("additive", "substitutive"):
        contract = make_contract(factor_type=factor_type)  # type: ignore[arg-type]
        workflow = generate_workflow(contract, TaskRole.HARNESS_BUILDER)
        baselines = [s for s in workflow if isinstance(s, BaselineGenerationStep)]
        assert len(baselines) == 1
        assert baselines[0].factor_type == factor_type


# === Discriminated-union round-trip ==========================================


def test_workflow_step_adapter_round_trip_preserves_concrete_types() -> None:
    """``WorkflowStepAdapter.validate_python`` deserializes step dumps
    back into the right concrete subclasses via the ``step_type``
    discriminator. Load-bearing for the cross-process boundary where
    sub-PR B serializes the workflow on the host and sub-PR B's
    mini-orchestrator deserializes it sandbox-side."""
    workflow = generate_workflow(make_contract(), TaskRole.HARNESS_BUILDER)
    for step in workflow:
        dumped = step.model_dump()
        revived = WorkflowStepAdapter.validate_python(dumped)
        assert type(revived) is type(step)
        assert revived == step


# === No-empty-seeds defensiveness ============================================


def test_validation_step_seed_falls_back_to_zero_when_contract_has_no_seeds() -> None:
    """The generator pulls ``seed`` from ``contract.body.seeds[0]``;
    an empty list falls back to ``0`` rather than raising. This keeps
    the generator usable against pathological contracts during the
    refactor's incremental landing."""
    workflow = generate_workflow(make_contract(seeds=[]), TaskRole.HARNESS_BUILDER)
    validations = [s for s in workflow if isinstance(s, ValidationStep)]
    assert validations[0].seed == 0


# === Registry / union completeness ===========================================


def test_step_registry_contains_every_concrete_step_type() -> None:
    """Every concrete step type lands in :data:`STEP_REGISTRY` at
    import time. The expected keys are the seven discriminator values
    per D9's union layout."""
    expected = {
        "harness_body_generation",
        "technique_body_generation",
        "baseline_generation",
        "diagnose_on_failure",
        "validation",
        "manifest_emit",
        "apply_review_feedback",
    }
    assert set(STEP_REGISTRY.keys()) == expected


# === ABI drift guard (round-21) ==============================================


def test_abi_function_signatures_use_real_smai_runtime_type_names() -> None:
    """Round-21 finding (2026-05-26): the generator originally hardcoded
    function signatures using ``Harness`` / ``Technique`` /
    ``TrainingResult`` / ``Metrics`` — none of which the ``smai_runtime``
    package exports. The agent dutifully wrote ``from smai_runtime
    import Harness`` and the validation subprocess crashed with
    ``ImportError``.

    Drift guard: every type name referenced in
    ``_ABI_BY_RUNTIME_VERSION[RUNTIME_TEMPLATE_VERSION]`` must be one of
    (1) a Python built-in (``dict``, ``int``, ``str``, ``None``,
    ``Any``, ...), (2) a public export of ``smai_runtime``, or (3) a
    parameter name (not a type annotation at all, e.g. the trailing
    ``trained_model`` argument).

    The test parses each signature, extracts type-annotation tokens,
    and asserts each is either a builtin or importable from
    ``smai_runtime``.
    """
    import re  # noqa: PLC0415

    import smai_runtime  # noqa: PLC0415
    from smai_agent_runtime.workflow.generator import (  # noqa: PLC0415
        _ABI_BY_RUNTIME_VERSION,
    )

    exported = set(getattr(smai_runtime, "__all__", []))
    if not exported:
        exported = {name for name in dir(smai_runtime) if not name.startswith("_")}
    builtins = {
        "dict",
        "list",
        "tuple",
        "set",
        "int",
        "float",
        "str",
        "bool",
        "bytes",
        "None",
        "Any",
        "object",
    }

    # Match ``: TypeName`` (parameter type) or ``-> TypeName`` (return type).
    # Type names start with a letter and may include underscores or further
    # qualifiers; subscripted generics (``dict[str, int]``) are stripped to
    # the base.
    type_token_re = re.compile(r"(?:->|:)\s*([A-Za-z_][A-Za-z0-9_]*)")

    abi = _ABI_BY_RUNTIME_VERSION[RUNTIME_TEMPLATE_VERSION]
    failures: list[str] = []
    for fn in abi:
        for token in type_token_re.findall(fn.signature):
            if token in builtins or token in exported:
                continue
            failures.append(
                f"function {fn.name!r} references type {token!r} which is "
                f"neither a Python builtin nor a public export of smai_runtime; "
                f"signature: {fn.signature!r}"
            )

    assert not failures, "\n".join(failures)


# === Baseline file-stem ↔ validation technique_id drift guard (round-22) =====


def test_baseline_file_stem_matches_validation_technique_id() -> None:
    """Round-22 Wall #3 (project_round22_real_llm_dogfood.md): the
    harness_builder workflow's :class:`BaselineGenerationStep`
    ``write_to_path`` and the immediately-following
    :class:`ValidationStep` ``technique_id`` must reference the SAME
    on-disk filename stem. The runner's
    :func:`smai_runtime.templates._files.techniques_init.load_technique`
    does ``importlib.import_module(f"techniques.{technique_id}")``,
    so a mismatch means the first validation ALWAYS fails with
    ``ModuleNotFoundError`` regardless of the agent's baseline quality.

    Drift guard: assert the baseline-step's file stem (extracted from
    its ``write_to_path``) equals the validation-step's ``technique_id``
    for the v1 harness_builder workflow.
    """
    from pathlib import Path  # noqa: PLC0415

    workflow = generate_workflow(make_contract(), TaskRole.HARNESS_BUILDER)
    baselines = [s for s in workflow if isinstance(s, BaselineGenerationStep)]
    validations = [s for s in workflow if isinstance(s, ValidationStep)]
    assert len(baselines) == 1
    assert len(validations) == 1
    baseline_stem = Path(baselines[0].write_to_path).stem
    assert validations[0].technique_id == baseline_stem, (
        f"ValidationStep.technique_id={validations[0].technique_id!r} does not match "
        f"BaselineGenerationStep.write_to_path stem={baseline_stem!r}; the runner's "
        f"load_technique would try to import techniques.{validations[0].technique_id!r} "
        f"but the file on disk is techniques/{baseline_stem}.py — round-22 Wall #3"
    )


def test_technique_implementer_write_path_stem_matches_validation_technique_id() -> None:
    """Same drift-guard pattern for the technique_implementer workflow.

    The technique_implementer workflow already uses the same
    :data:`_TECHNIQUE_PLACEHOLDER_ID` for both its body-generation
    step's ``write_to_path`` and the validation step's
    ``technique_id`` (consistent by construction), but pinning the
    invariant means a future refactor that diverges them gets caught
    here — the harness_builder side was broken for ~3 rounds before
    Wall #3 surfaced because nothing pinned the invariant.
    """
    from pathlib import Path  # noqa: PLC0415

    workflow = generate_workflow(make_contract(), TaskRole.TECHNIQUE_IMPLEMENTER)
    body_steps = [s for s in workflow if isinstance(s, TechniqueImplementerBodyGenerationStep)]
    validations = [s for s in workflow if isinstance(s, ValidationStep)]
    assert len(body_steps) == 1
    assert len(validations) == 1
    body_stem = Path(body_steps[0].write_to_path).stem
    assert validations[0].technique_id == body_stem, (
        f"ValidationStep.technique_id={validations[0].technique_id!r} does not match "
        f"TechniqueImplementerBodyGenerationStep.write_to_path stem={body_stem!r}"
    )
