"""Per-step fake-LLM tests for :func:`_run_body_generation_step`.

Sub-PR C1 acceptance criterion: each replaced fake handler lands with a
fake-LLM test that asserts the bundle is built correctly, the agent
call is made with the right inputs, the file gets written, and lint
runs. The lint-retry path is exercised; budget-exhaustion raises a
clear step-failure error.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _body_generation_fixtures import (  # type: ignore[import-not-found]
    FakeAgentRunner,
    make_body_generation_output,
)
from _harness_builder_workspace_fixtures import (  # type: ignore[import-not-found]
    write_minimal_harness_workspace,
)
from smai_agent_runtime.harness_builder import _main as harness_builder_main
from smai_agent_runtime.harness_builder._main import (
    _DispatchContext,
    _load_contract,
    _run_body_generation_step,
)
from smai_agent_runtime.workflow.step_types import HarnessBuilderBodyGenerationStep


def _make_context(workspace: Path) -> _DispatchContext:
    contract = _load_contract(workspace)
    assert contract is not None
    return _DispatchContext(
        cg_id="cg-test-bodygen",
        workspace=workspace,
        contract=contract,
        technique_contract=None,
        overrides=None,
    )


def test_body_generation_step_writes_module_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: agent returns conforming output, dispatcher writes
    module_source to ``write_to_path``, lint passes, step succeeds."""
    write_minimal_harness_workspace(tmp_path)
    fake = FakeAgentRunner(
        responses=[make_body_generation_output(function_name="build_harness")],
    )
    monkeypatch.setattr(harness_builder_main, "_run_agent_sync", fake)

    step = HarnessBuilderBodyGenerationStep(
        function_index=0,
        function_name="build_harness",
        function_signature="def build_harness(config: dict) -> HarnessComponents:",
        write_to_path="harness/__init__.py",
    )
    outcome = _run_body_generation_step(step, index=0, context=_make_context(tmp_path))

    assert outcome.succeeded is True
    assert outcome.error is None
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call.output_type == "HarnessBuilderBodyGenerationOutput"
    # The bundle the dispatcher constructed renders into the user
    # message via Jinja2; assert the per-step discriminator appears in
    # the rendered text (proves the bundle was built with the right
    # function-name and the prompt template wired through).
    assert "body for `build_harness`" in call.user_message
    # The factor under study comes from the staged HarnessContract.
    assert "test_factor" in call.user_message
    # The module source landed on disk at write_to_path.
    written = (tmp_path / "harness" / "__init__.py").read_text()
    assert "def build_harness" in written
    # Sub-PR D resume-prep: the dispatcher threads workspace +
    # ``trace_step_name`` so :func:`_run_agent_sync` persists the
    # message history under ``conversation_traces/`` per
    # architectural_decisions §12 #4.
    assert call.workspace == tmp_path
    assert call.trace_step_name == "00_body_build_harness_attempt_0"


def test_body_generation_step_carries_extension_points_into_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dispatcher resolves the contract's extension points (round-15
    COMPONENT_FIELD_FOR_KEY introspection) into the D7a bundle; the
    rendered user message exposes every closed-v1 key."""
    write_minimal_harness_workspace(tmp_path)
    fake = FakeAgentRunner(responses=[make_body_generation_output()])
    monkeypatch.setattr(harness_builder_main, "_run_agent_sync", fake)

    step = HarnessBuilderBodyGenerationStep(
        function_index=0,
        function_name="build_harness",
        function_signature="def build_harness(config: dict) -> HarnessComponents:",
        write_to_path="harness/__init__.py",
    )
    _run_body_generation_step(step, index=0, context=_make_context(tmp_path))

    user_message = fake.calls[0].user_message
    # Every closed-v1 extension-point key surfaces in the rendered prompt.
    for key in (
        "train_transforms",
        "model_wrapper",
        "loss_fn",
        "training_overrides",
        "callbacks",
    ):
        assert key in user_message, f"extension-point key {key!r} missing from prompt"


def test_body_generation_step_substitutive_factor_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Substitutive factor (the default fixture's setting) flows
    ``factor_role=substitutive_slot`` into every extension point spec."""
    write_minimal_harness_workspace(tmp_path)
    fake = FakeAgentRunner(responses=[make_body_generation_output()])
    monkeypatch.setattr(harness_builder_main, "_run_agent_sync", fake)

    step = HarnessBuilderBodyGenerationStep(
        function_index=0,
        function_name="build_harness",
        function_signature="def build_harness(config: dict) -> HarnessComponents:",
        write_to_path="harness/__init__.py",
    )
    _run_body_generation_step(step, index=0, context=_make_context(tmp_path))

    user_message = fake.calls[0].user_message
    assert "substitutive_slot" in user_message
    assert "additive_baseline_default" not in user_message


def test_body_generation_step_retries_on_lint_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First attempt's module_source is lint-broken; the dispatcher
    appends a PriorFailedAttempt and re-prompts. Second attempt clean,
    step succeeds. The second user_message references prior attempts."""
    write_minimal_harness_workspace(tmp_path)
    bad_source = "def build_harness(config) -> :\n  invalid syntax\n"  # syntax error
    fake = FakeAgentRunner(
        responses=[
            make_body_generation_output(module_source=bad_source),
            make_body_generation_output(),  # clean
        ],
    )
    monkeypatch.setattr(harness_builder_main, "_run_agent_sync", fake)

    step = HarnessBuilderBodyGenerationStep(
        function_index=0,
        function_name="build_harness",
        function_signature="def build_harness(config: dict) -> HarnessComponents:",
        write_to_path="harness/__init__.py",
    )
    outcome = _run_body_generation_step(step, index=0, context=_make_context(tmp_path))

    assert outcome.succeeded is True
    assert len(fake.calls) == 2, "expected one retry after the lint failure"
    # The retry's user_message renders the PriorFailedAttempt.
    second_call = fake.calls[1]
    assert "Prior failed attempts" in second_call.user_message
    assert "attempt_index" in second_call.user_message or "Attempt 0" in second_call.user_message


def test_body_generation_step_budget_exhaustion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent lint failures across the budget exhaust the retry
    counter; the step surfaces a clear step-failure error rather than
    looping forever."""
    write_minimal_harness_workspace(tmp_path)
    bad_source = "def build_harness(config) -> :\n  invalid syntax\n"
    # Provide 4 bad responses (one attempt + 3 retries = budget = 4
    # invocations); the dispatcher exhausts after the 4th.
    fake = FakeAgentRunner(
        responses=[make_body_generation_output(module_source=bad_source) for _ in range(4)],
    )
    monkeypatch.setattr(harness_builder_main, "_run_agent_sync", fake)

    step = HarnessBuilderBodyGenerationStep(
        function_index=0,
        function_name="build_harness",
        function_signature="def build_harness(config: dict) -> HarnessComponents:",
        write_to_path="harness/__init__.py",
    )
    outcome = _run_body_generation_step(step, index=0, context=_make_context(tmp_path))

    assert outcome.succeeded is False
    assert outcome.error is not None
    assert "budget exhausted" in outcome.error
    # The dispatcher tried the full 1 + 3 attempts before giving up.
    assert len(fake.calls) == 4


def test_body_generation_step_function_name_mismatch_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The agent's echoed ``function_name`` must match the step's
    target; mismatch is a hard step-failure (the bundle is the agent's
    contract, not a hint)."""
    write_minimal_harness_workspace(tmp_path)
    # Agent echoes "evaluate" but the step targets "build_harness".
    fake = FakeAgentRunner(
        responses=[make_body_generation_output(function_name="evaluate")],
    )
    monkeypatch.setattr(harness_builder_main, "_run_agent_sync", fake)

    step = HarnessBuilderBodyGenerationStep(
        function_index=0,
        function_name="build_harness",
        function_signature="def build_harness(config: dict) -> HarnessComponents:",
        write_to_path="harness/__init__.py",
    )
    outcome = _run_body_generation_step(step, index=0, context=_make_context(tmp_path))

    assert outcome.succeeded is False
    assert outcome.error is not None
    assert "function_name" in outcome.error
