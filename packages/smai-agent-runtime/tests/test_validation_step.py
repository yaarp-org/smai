"""Per-step tests for :func:`_run_validation_step` (sub-PR C2).

Real subprocess invocation against staged ``experiment.py`` shims:
the passing-shim fixture covers the happy path; the failing-shim
fixture covers the diagnose-handoff path (captured_stderr populated
for D7c bundle assembly); missing-experiment.py / missing-results /
malformed-results / soft-fail / timeout cover the boundary cases.
"""

from __future__ import annotations

from pathlib import Path

from _harness_builder_workspace_fixtures import (  # type: ignore[import-not-found]
    write_failing_experiment_shim,
    write_minimal_harness_workspace,
)
from smai_agent_runtime.harness_builder._main import (
    _DispatchContext,
    _load_contract,
    _run_validation_step,
)
from smai_agent_runtime.workflow.step_types import ValidationStep


def _make_context(workspace: Path) -> _DispatchContext:
    contract = _load_contract(workspace)
    assert contract is not None
    return _DispatchContext(
        cg_id="cg-test-validation",
        workspace=workspace,
        contract=contract,
        technique_contract=None,
        overrides=None,
    )


def _make_step(
    *,
    dispatch_target: str = "subprocess",
    technique_id: str = "baseline",
    seed: int = 42,
) -> ValidationStep:
    return ValidationStep(
        technique_id=technique_id,
        seed=seed,
        dispatch_target=dispatch_target,  # type: ignore[arg-type]
    )


def test_validation_step_succeeds_with_passing_shim(tmp_path: Path) -> None:
    """Real subprocess invocation: shim writes passing
    validation_results.json; step succeeds with no captured_stderr."""
    write_minimal_harness_workspace(tmp_path)
    outcome = _run_validation_step(_make_step(), index=4, context=_make_context(tmp_path))

    assert outcome.succeeded is True
    assert outcome.error is None
    assert outcome.captured_stderr is None
    # The shim's payload landed in validation_results.json.
    results_path = tmp_path / "validation_results.json"
    assert results_path.exists()


def test_validation_step_missing_experiment_py_fails_cleanly(tmp_path: Path) -> None:
    """No staged experiment.py => the step surfaces a clear
    "missing experiment.py" error without launching a subprocess."""
    write_minimal_harness_workspace(tmp_path, with_experiment_shim=False)
    outcome = _run_validation_step(_make_step(), index=4, context=_make_context(tmp_path))

    assert outcome.succeeded is False
    assert outcome.error is not None
    assert "experiment.py" in outcome.error
    # No subprocess, no captured_stderr.
    assert outcome.captured_stderr is None


def test_validation_step_failing_subprocess_captures_stderr(tmp_path: Path) -> None:
    """Failing shim => the step records the stderr tail on
    :attr:`_StepOutcome.captured_stderr` so the diagnose step can
    build a D7c bundle without re-running validation."""
    write_minimal_harness_workspace(tmp_path)
    write_failing_experiment_shim(tmp_path)
    outcome = _run_validation_step(_make_step(), index=4, context=_make_context(tmp_path))

    assert outcome.succeeded is False
    assert outcome.error is not None
    assert "exited with code 1" in outcome.error
    assert outcome.captured_stderr is not None
    # The shim's signature lines made it into the tail.
    assert "EXIT_RUNTIME_FAILURE" in outcome.captured_stderr
    assert ".cuda()" in outcome.captured_stderr


def test_validation_step_compute_submit_dispatch_target_not_implemented(
    tmp_path: Path,
) -> None:
    """The architectural §1 GPU-validation escape hatch is not
    implemented in sub-PR C2; the workflow generator emits
    dispatch_target='subprocess' and the alternative path surfaces a
    structured error so a future workflow flip doesn't silently skip
    validation."""
    write_minimal_harness_workspace(tmp_path)
    outcome = _run_validation_step(
        _make_step(dispatch_target="compute_submit"),
        index=4,
        context=_make_context(tmp_path),
    )

    assert outcome.succeeded is False
    assert outcome.error is not None
    assert "compute_submit" in outcome.error
    assert "not implemented" in outcome.error


def test_validation_step_missing_results_after_zero_exit_fails(tmp_path: Path) -> None:
    """Shim exits 0 but does not write validation_results.json => the
    step surfaces the structural contract gap (the runtime contract
    says exit 0 implies results.json on the success path)."""
    write_minimal_harness_workspace(tmp_path, with_experiment_shim=False)
    # Stage a shim that exits 0 with no validation_results.json output.
    (tmp_path / "experiment.py").write_text(
        '"""Test shim exit-0-no-results."""\n'
        "import sys\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    sys.exit(0)\n"
    )
    outcome = _run_validation_step(_make_step(), index=4, context=_make_context(tmp_path))

    assert outcome.succeeded is False
    assert outcome.error is not None
    assert "validation_results.json" in outcome.error


def test_validation_step_malformed_results_after_zero_exit_fails(tmp_path: Path) -> None:
    """Exit 0 but validation_results.json is not valid JSON => the step
    surfaces the parse error rather than silently advancing."""
    write_minimal_harness_workspace(tmp_path, with_experiment_shim=False)
    (tmp_path / "experiment.py").write_text(
        '"""Test shim that writes garbled results."""\n'
        "import sys\n"
        "from pathlib import Path\n"
        "import argparse\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--technique', required=True)\n"
        "    parser.add_argument('--seed', type=int, required=True)\n"
        "    parser.add_argument('--mode', required=True)\n"
        "    parser.add_argument('--workspace', required=True)\n"
        "    parser.add_argument('--epochs', type=int, default=None)\n"
        "    parser.add_argument('--subset', default=None)\n"
        "    args = parser.parse_args()\n"
        "    Path(args.workspace, 'validation_results.json').write_text('not json')\n"
        "    sys.exit(0)\n"
    )
    outcome = _run_validation_step(_make_step(), index=4, context=_make_context(tmp_path))

    assert outcome.succeeded is False
    assert outcome.error is not None
    assert "malformed" in outcome.error


def test_validation_step_soft_fail_passed_false_fails(tmp_path: Path) -> None:
    """Exit 0 and parseable validation_results.json but ``passed: false``
    => the step surfaces the soft fail (this is the runtime's future
    structured-failure mode the D7c schema's ``validation_results``
    field carries)."""
    write_minimal_harness_workspace(tmp_path, with_experiment_shim=False)
    (tmp_path / "experiment.py").write_text(
        '"""Test shim that writes a soft-fail results.json."""\n'
        "import sys\n"
        "import json\n"
        "from pathlib import Path\n"
        "import argparse\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--technique', required=True)\n"
        "    parser.add_argument('--seed', type=int, required=True)\n"
        "    parser.add_argument('--mode', required=True)\n"
        "    parser.add_argument('--workspace', required=True)\n"
        "    parser.add_argument('--epochs', type=int, default=None)\n"
        "    parser.add_argument('--subset', default=None)\n"
        "    args = parser.parse_args()\n"
        "    Path(args.workspace, 'validation_results.json').write_text(\n"
        "        json.dumps({'passed': False, 'note': 'soft fail'})\n"
        "    )\n"
        "    sys.exit(0)\n"
    )
    outcome = _run_validation_step(_make_step(), index=4, context=_make_context(tmp_path))

    assert outcome.succeeded is False
    assert outcome.error is not None
    assert "passed=True" in outcome.error
