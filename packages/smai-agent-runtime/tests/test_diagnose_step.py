"""Per-step tests for :func:`_run_diagnose_step` (sub-PR C2 D7c).

Asserts the D7c bundle is built correctly (every field name matches),
the agent is constructed with the diagnose-only ``read_file`` tool
(scoped to the workspace), the fix is applied per ``fix_target``, the
revalidation loop runs, and budget exhaustion surfaces a clear error.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _diagnose_fixtures import (  # type: ignore[import-not-found]
    DiagnoseFakeAgentRunner,
    make_diagnosis,
)
from _harness_builder_workspace_fixtures import (  # type: ignore[import-not-found]
    write_failing_experiment_shim,
    write_minimal_harness_workspace,
)
from smai_agent_runtime.harness_builder import _main as harness_builder_main
from smai_agent_runtime.harness_builder._main import (
    _build_diagnose_bundle,
    _DispatchContext,
    _load_contract,
    _run_diagnose_step,
    _StepOutcome,
)
from smai_agent_runtime.schemas import (
    DiagnoseBundle,
    HarnessOriginalBundleRef,
    PriorDiagnosis,
)
from smai_agent_runtime.workflow.step_types import DiagnoseOnFailureStep


def _make_context(workspace: Path) -> _DispatchContext:
    contract = _load_contract(workspace)
    assert contract is not None
    return _DispatchContext(
        cg_id="cg-test-diagnose",
        workspace=workspace,
        contract=contract,
        technique_contract=None,
        overrides=None,
    )


def _make_diagnose_step(*, max_retries: int = 3) -> DiagnoseOnFailureStep:
    return DiagnoseOnFailureStep(anchor_step_index=4, max_retries=max_retries)


def _failed_validation_outcome(stderr: str = "fake stderr tail") -> _StepOutcome:
    return _StepOutcome(
        step_index=4,
        step_type="validation",
        succeeded=False,
        error="validation subprocess exited with code 1",
        captured_stderr=stderr,
    )


def _passing_validation_outcome() -> _StepOutcome:
    return _StepOutcome(
        step_index=4,
        step_type="validation",
        succeeded=True,
    )


# === Pass-through path =======================================================


def test_diagnose_step_passes_through_when_anchor_succeeded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the validation step succeeded the diagnose step does nothing
    — no agent call, no fix applied, no revalidation."""
    write_minimal_harness_workspace(tmp_path)
    fake = DiagnoseFakeAgentRunner(responses=[])  # would explode if called
    monkeypatch.setattr(harness_builder_main, "_run_agent_sync", fake)

    outcome = _run_diagnose_step(
        _make_diagnose_step(),
        index=5,
        context=_make_context(tmp_path),
        prior_outcomes=[
            _passing_validation_outcome(),
            _passing_validation_outcome(),
            _passing_validation_outcome(),
            _passing_validation_outcome(),
            _passing_validation_outcome(),
        ],
    )
    assert outcome.succeeded is True
    assert len(fake.calls) == 0


# === Bundle shape ============================================================


def test_diagnose_bundle_is_built_with_correct_field_names(
    tmp_path: Path,
) -> None:
    """The dispatcher's bundle assembly populates every D7c field with
    the right shape: failure_reason from the anchor's error, stderr
    from the anchor's captured_stderr, current_code from the on-disk
    harness/ + techniques/ sources, original_input_bundle as the
    harness-discriminated reference."""
    write_minimal_harness_workspace(tmp_path)
    (tmp_path / "harness" / "__init__.py").write_text("def build_harness(c): return {}\n")
    context = _make_context(tmp_path)
    anchor = _failed_validation_outcome("stderr line\nanother stderr line")

    bundle = _build_diagnose_bundle(
        anchor=anchor,
        context=context,
        prior_diagnoses=[],
    )

    assert isinstance(bundle, DiagnoseBundle)
    assert bundle.failure_reason.startswith("validation subprocess exited with code")
    assert "stderr line" in bundle.container_stderr
    # current_code surfaces the harness/__init__.py we just wrote.
    assert "harness/__init__.py" in bundle.current_code
    assert "def build_harness" in bundle.current_code["harness/__init__.py"]
    # Discriminator routes correctly to the harness branch.
    assert isinstance(bundle.original_input_bundle, HarnessOriginalBundleRef)
    assert bundle.original_input_bundle.kind == "harness"
    # Substitutive factor (the workflow_fixtures default) flows through.
    assert bundle.original_input_bundle.factor_type == "substitutive"
    # No prior diagnoses on the first attempt.
    assert bundle.prior_diagnoses == []


def test_diagnose_bundle_carries_prior_diagnoses_across_retries(
    tmp_path: Path,
) -> None:
    """On a retry the bundle includes the structured PriorDiagnosis
    summaries the prior loop iterations recorded."""
    write_minimal_harness_workspace(tmp_path)
    context = _make_context(tmp_path)
    prior = [
        PriorDiagnosis(
            attempt_index=0,
            diagnosis_summary="prior attempt: cuda hardcoded",
            fix_target="harness",
            fix_payload_preview="def build_harness(c): return {}",
            outcome="applied_but_validation_failed_again",
        )
    ]

    bundle = _build_diagnose_bundle(
        anchor=_failed_validation_outcome(),
        context=context,
        prior_diagnoses=prior,
    )

    assert len(bundle.prior_diagnoses) == 1
    assert bundle.prior_diagnoses[0].diagnosis_summary.startswith("prior attempt")


def test_diagnose_bundle_renders_through_the_relocated_prompt(
    tmp_path: Path,
) -> None:
    """The D7c prompt template (step_7_diagnose.yaml, relocated from
    the design notes to the package) renders against the bundle with
    no UndefinedError, confirming field-name parity with the schema."""
    from smai_agent_runtime.prompts import load_step_prompt
    from smai_agent_runtime.prompts._loader import render_user_message

    write_minimal_harness_workspace(tmp_path)
    context = _make_context(tmp_path)
    bundle = _build_diagnose_bundle(
        anchor=_failed_validation_outcome(),
        context=context,
        prior_diagnoses=[],
    )
    prompt = load_step_prompt("harness_builder", "step_7_diagnose.yaml")
    rendered = render_user_message(
        prompt.initial_user_message_template,
        bundle.model_dump(mode="json"),
    )
    assert "diagnose validation failure" in rendered
    assert "kind=harness" in rendered


# === Apply-then-revalidate flow ==============================================


def test_diagnose_step_applies_fix_and_revalidates_to_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single diagnose call: the agent returns a Diagnosis with
    fix_target='harness', the dispatcher writes the payload to
    harness/__init__.py, re-runs validation (passing-shim => success),
    and the step returns succeeded=True."""
    write_minimal_harness_workspace(tmp_path)
    fake = DiagnoseFakeAgentRunner(responses=[make_diagnosis()])
    monkeypatch.setattr(harness_builder_main, "_run_agent_sync", fake)

    prior_outcomes = [
        _passing_validation_outcome(),
        _passing_validation_outcome(),
        _passing_validation_outcome(),
        _passing_validation_outcome(),
        _failed_validation_outcome("traceback here"),
    ]
    outcome = _run_diagnose_step(
        _make_diagnose_step(),
        index=5,
        context=_make_context(tmp_path),
        prior_outcomes=prior_outcomes,
    )

    assert outcome.succeeded is True
    assert len(fake.calls) == 1
    # The fix landed at the harness path.
    written = (tmp_path / "harness" / "__init__.py").read_text()
    assert "Diagnose-applied harness body" in written
    # The original anchor outcome was patched to reflect successful
    # revalidation so manifest_emit's precondition can pass.
    assert prior_outcomes[4].succeeded is True


def test_diagnose_step_baseline_fix_target_writes_to_baseline_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """fix_target='baseline' routes the payload to techniques/baseline.py."""
    write_minimal_harness_workspace(tmp_path)
    fake = DiagnoseFakeAgentRunner(
        responses=[
            make_diagnosis(
                fix_target="baseline",
                fix_payload="def baseline():\n    return None\n",
            )
        ]
    )
    monkeypatch.setattr(harness_builder_main, "_run_agent_sync", fake)

    prior_outcomes = [_passing_validation_outcome()] * 4 + [_failed_validation_outcome()]
    outcome = _run_diagnose_step(
        _make_diagnose_step(),
        index=5,
        context=_make_context(tmp_path),
        prior_outcomes=prior_outcomes,
    )

    assert outcome.succeeded is True
    assert (tmp_path / "techniques" / "baseline.py").read_text().startswith("def baseline")


def test_diagnose_step_technique_target_rejected_in_harness_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The harness_builder workflow has no per-technique writable
    surface. A diagnosis with fix_target='technique' records as
    did_not_apply and the loop continues."""
    write_minimal_harness_workspace(tmp_path)
    fake = DiagnoseFakeAgentRunner(
        responses=[
            make_diagnosis(fix_target="technique", fix_payload="..."),
            # Second attempt picks the right target.
            make_diagnosis(fix_target="harness"),
        ]
    )
    monkeypatch.setattr(harness_builder_main, "_run_agent_sync", fake)

    prior_outcomes = [_passing_validation_outcome()] * 4 + [_failed_validation_outcome()]
    outcome = _run_diagnose_step(
        _make_diagnose_step(max_retries=3),
        index=5,
        context=_make_context(tmp_path),
        prior_outcomes=prior_outcomes,
    )

    assert outcome.succeeded is True
    # Two calls: one rejected, one applied.
    assert len(fake.calls) == 2


# === Budget exhaustion =======================================================


def test_diagnose_step_budget_exhaustion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent revalidation failure across the budget exhausts the
    retry counter; the step surfaces a clear failure rather than
    looping forever."""
    write_minimal_harness_workspace(tmp_path)
    write_failing_experiment_shim(tmp_path)  # revalidation always fails
    fake = DiagnoseFakeAgentRunner(
        responses=[make_diagnosis() for _ in range(5)],
    )
    monkeypatch.setattr(harness_builder_main, "_run_agent_sync", fake)

    prior_outcomes = [_passing_validation_outcome()] * 4 + [_failed_validation_outcome()]
    outcome = _run_diagnose_step(
        _make_diagnose_step(max_retries=3),
        index=5,
        context=_make_context(tmp_path),
        prior_outcomes=prior_outcomes,
    )

    assert outcome.succeeded is False
    assert outcome.error is not None
    assert "budget exhausted" in outcome.error
    # Three diagnose attempts before giving up.
    assert len(fake.calls) == 3


# === read_file tool registration =============================================


def test_diagnose_agent_registers_read_file_tool_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The diagnose Agent gets the ``read_file`` tool and nothing else.
    Patch the build_agent factory to capture the constructed agent so
    we can introspect its tool registry."""
    captured_agents: list[object] = []
    real_register = harness_builder_main._register_read_file_tool

    def _capture_register(agent: object, workspace: Path) -> None:
        captured_agents.append(agent)
        real_register(agent, workspace)

    monkeypatch.setattr(harness_builder_main, "_register_read_file_tool", _capture_register)

    fake = DiagnoseFakeAgentRunner(responses=[make_diagnosis()])
    monkeypatch.setattr(harness_builder_main, "_run_agent_sync", fake)

    write_minimal_harness_workspace(tmp_path)
    prior_outcomes = [_passing_validation_outcome()] * 4 + [_failed_validation_outcome()]
    _run_diagnose_step(
        _make_diagnose_step(),
        index=5,
        context=_make_context(tmp_path),
        prior_outcomes=prior_outcomes,
    )

    assert len(captured_agents) == 1
    # PydanticAI's Agent exposes a ``_function_toolset`` / tools surface
    # via ``.tools`` or via the toolsets registry; we walk both shapes
    # so this test stays compatible across minor-version differences.
    agent = captured_agents[0]
    tool_names = _collect_agent_tool_names(agent)
    assert "read_file" in tool_names
    # The diagnose agent has ONE tool (read_file). The output-type
    # surface itself may add a synthetic "final_result" tool that
    # PydanticAI registers internally for structured output binding;
    # we filter that out so the assertion captures the agent-callable
    # surface.
    callable_tools = [t for t in tool_names if not t.startswith("final_result")]
    assert callable_tools == ["read_file"]


def test_body_generation_agent_does_not_register_read_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``read_file`` is the diagnose-step-only escape hatch per
    architectural_decisions §12 #1. Body-generation agents must NOT
    register it. Patches the body-generation dispatch path's agent
    factory to capture the constructed agent."""
    from _body_generation_fixtures import (  # type: ignore[import-not-found]
        FakeAgentRunner,
        make_body_generation_output,
    )
    from smai_agent_runtime.harness_builder._main import _run_body_generation_step
    from smai_agent_runtime.workflow.step_types import HarnessBuilderBodyGenerationStep

    write_minimal_harness_workspace(tmp_path)
    fake = FakeAgentRunner(responses=[make_body_generation_output()])
    monkeypatch.setattr(harness_builder_main, "_run_agent_sync", fake)

    captured_agents: list[object] = []
    real_build = harness_builder_main.build_agent

    def _capture_build(**kwargs: object) -> object:
        agent = real_build(**kwargs)  # type: ignore[arg-type]
        captured_agents.append(agent)
        return agent

    monkeypatch.setattr(harness_builder_main, "build_agent", _capture_build)

    step = HarnessBuilderBodyGenerationStep(
        function_index=0,
        function_name="build_harness",
        function_signature="def build_harness(config: dict) -> HarnessComponents:",
        write_to_path="harness/__init__.py",
    )
    _run_body_generation_step(step, index=0, context=_make_context(tmp_path))

    assert len(captured_agents) == 1
    tool_names = _collect_agent_tool_names(captured_agents[0])
    callable_tools = [t for t in tool_names if not t.startswith("final_result")]
    assert callable_tools == []


# === read_file tool behavior =================================================


def test_read_file_tool_rejects_workspace_escape(tmp_path: Path) -> None:
    """The registered ``read_file`` tool rejects ``..`` escapes and
    absolute paths outside the workspace per the design discipline."""
    from pydantic_ai import Agent
    from smai_agent_runtime.schemas import Diagnosis

    write_minimal_harness_workspace(tmp_path)
    agent: Agent[None, Diagnosis] = Agent(
        model="test",
        output_type=Diagnosis,
        system_prompt="t",
    )
    harness_builder_main._register_read_file_tool(agent, tmp_path)

    # Pull the registered read_file callable via the agent's tools surface.
    read_fn = _find_tool_callable(agent, "read_file")
    assert read_fn is not None
    # Escape via .. is rejected.
    result = read_fn("../../../etc/passwd")
    assert "outside workspace" in result
    # Absolute path outside the workspace is rejected.
    result = read_fn("/etc/passwd")
    assert "outside workspace" in result


def test_read_file_tool_caps_per_attempt_budget(tmp_path: Path) -> None:
    """5-call cap per architectural_decisions §12 #1 design discipline."""
    from pydantic_ai import Agent
    from smai_agent_runtime.schemas import (
        READ_FILE_BUDGET_PER_ATTEMPT,
        Diagnosis,
    )

    write_minimal_harness_workspace(tmp_path)
    (tmp_path / "harness" / "ok.py").write_text("content")
    agent: Agent[None, Diagnosis] = Agent(
        model="test",
        output_type=Diagnosis,
        system_prompt="t",
    )
    harness_builder_main._register_read_file_tool(agent, tmp_path)

    read_fn = _find_tool_callable(agent, "read_file")
    assert read_fn is not None
    for _ in range(READ_FILE_BUDGET_PER_ATTEMPT):
        assert read_fn("harness/ok.py") == "content"
    # Sixth call hits the budget guard.
    result = read_fn("harness/ok.py")
    assert "budget exhausted" in result


# === Helpers =================================================================


def _collect_agent_tool_names(agent: object) -> list[str]:
    """Walk a PydanticAI :class:`Agent`'s registered tool surface.

    PydanticAI ships tools through ``_function_toolset`` (an internal
    Toolset instance). The walker is forgiving of internal shape
    changes — it returns whatever ``.tools`` keys it can find.
    """
    names: list[str] = []
    toolset = getattr(agent, "_function_toolset", None)
    if toolset is not None:
        tools_attr = getattr(toolset, "tools", None)
        if isinstance(tools_attr, dict):
            for name in tools_attr.keys():
                if isinstance(name, str):
                    names.append(name)
    return names


def _find_tool_callable(agent: object, name: str):
    """Return the registered callable for the tool ``name``, or None."""
    toolset = getattr(agent, "_function_toolset", None)
    if toolset is None:
        return None
    tools_attr = getattr(toolset, "tools", None)
    if not isinstance(tools_attr, dict):
        return None
    tool = tools_attr.get(name)
    if tool is None:
        return None
    # PydanticAI's Tool object exposes the underlying callable via
    # ``.function`` (preferred) or ``.callable`` (older).
    fn = getattr(tool, "function", None) or getattr(tool, "callable", None)
    return fn
