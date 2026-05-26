"""Entry-point dispatch tests.

Sub-PR B replaced the Step 3 "not yet implemented" stub with a real
mini-orchestrator skeleton iterating the sub-PR-A workflow generator
with FAKE per-step handlers. Sub-PR C1 replaced two of those fakes
(harness body generation, baseline generation) with real PydanticAI
Agent calls. The other three step types (validation, diagnose,
manifest_emit) still run sub-PR B's fake handlers (sub-PR C2 fills them).

These tests exercise the dispatch wiring in
:mod:`smai_agent_runtime.__main__` plus the sub-PR C1 harness_builder
entry behavior end-to-end. They rely on the deterministic-fake-LLM
substitution toggle (``SMAI_AGENT_RUNTIME_FAKE_LLM=1``) so the
cross-process subprocess test can drive the workflow without real
Bedrock credentials.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from _harness_builder_workspace_fixtures import (  # type: ignore[import-not-found]
    write_minimal_harness_workspace,
)
from smai_agent_runtime.__main__ import (
    EXIT_INTERNAL_ERROR,
    EXIT_NOT_IMPLEMENTED,
    main,
)
from smai_agent_runtime.harness_builder._main import (
    EXIT_BAD_WORKSPACE,
    EXIT_OK,
    EXIT_RESUME_NOT_IMPLEMENTED,
)


@pytest.fixture(autouse=True)
def _enable_fake_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set ``SMAI_AGENT_RUNTIME_FAKE_LLM=1`` for every test in this module.

    Sub-PR C1's body-generation handlers consult this env var to decide
    whether to call the real LLM or substitute a deterministic stub.
    These end-to-end / dispatch-wiring tests do not need real LLM
    outputs; the per-step fake-LLM tests in
    ``test_body_generation_step.py`` / ``test_baseline_generation_step.py``
    exercise the agent-call surface via in-process monkeypatch.
    """
    monkeypatch.setenv("SMAI_AGENT_RUNTIME_FAKE_LLM", "1")


# === Programmatic dispatch ===================================================


def test_harness_builder_runs_workflow_with_workspace(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Sub-PR C2: with a properly-staged workspace the mini-orchestrator
    iterates the generated workflow and exits :data:`EXIT_OK`. The
    body-generation steps run via fake LLM stubs, the validation step
    runs the staged shim's ``experiment.py`` via real subprocess, the
    diagnose step passes through (validation succeeded), and the
    manifest emit step writes ``manifest.json``.
    """
    write_minimal_harness_workspace(tmp_path)
    rc = main(
        [
            "--role",
            "harness_builder",
            "--cg-id",
            "cg-test-001",
            "--workspace",
            str(tmp_path),
        ]
    )
    assert rc == EXIT_OK, capsys.readouterr().err
    captured = capsys.readouterr()
    # status.json landed in outputs/ with the per-step outcomes
    status_payload = json.loads((tmp_path / "outputs" / "status.json").read_text())
    assert status_payload["succeeded"] is True
    assert status_payload["cg_id"] == "cg-test-001"
    # 3 body + baseline + validation + diagnose + manifest = 7
    assert len(status_payload["step_outcomes"]) == 7
    # D10 session.start + session.end status lines surface on stdout.
    assert '"event_type":"session.start"' in captured.out
    assert '"event_type":"session.end"' in captured.out
    assert '"outcome":"success"' in captured.out


def test_harness_builder_workflow_writes_canned_artifacts(
    tmp_path: Path,
) -> None:
    """Sub-PR C2: the workflow materializes the per-step artifacts:
    fake-LLM body source under ``harness/__init__.py`` + baseline
    source under ``techniques/baseline.py``; the real ValidationStep
    drives the experiment-shim to produce ``validation_results.json``;
    the real ManifestEmitStep writes ``manifest.json`` against the
    recomputed harness_version_hash."""
    write_minimal_harness_workspace(tmp_path)
    rc = main(
        [
            "--role",
            "harness_builder",
            "--cg-id",
            "cg-test-002",
            "--workspace",
            str(tmp_path),
        ]
    )
    assert rc == EXIT_OK
    # body-generation steps wrote to harness/__init__.py
    harness_body = (tmp_path / "harness" / "__init__.py").read_text()
    assert "build_harness" in harness_body
    assert "run_training_loop" in harness_body
    assert "evaluate" in harness_body
    # baseline step wrote to techniques/baseline.py
    baseline_body = (tmp_path / "techniques" / "baseline.py").read_text()
    assert "baseline" in baseline_body
    # validation step's real subprocess invocation drove the shim's
    # experiment.py, which wrote a passing payload at the workspace root.
    validation_payload = json.loads((tmp_path / "validation_results.json").read_text())
    assert validation_payload["passed"] is True
    # manifest step wrote manifest.json with a recomputed
    # harness_version_hash from the on-disk harness/ contents.
    manifest_payload = json.loads((tmp_path / "manifest.json").read_text())
    assert "runtime_template_version" in manifest_payload
    assert manifest_payload["harness_version_hash"]  # non-empty
    assert manifest_payload["content_hash"]  # freeze_manifest populated


def test_harness_builder_resume_flag_is_no_op_stub(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Sub-PR B accepts ``--resume`` at argparse and routes to a no-op
    stub (architectural hedge per arch §12 item 4). Sub-PR D wires
    the resume-mode workflow logic.
    """
    rc = main(
        [
            "--role",
            "harness_builder",
            "--cg-id",
            "cg-test-003",
            "--workspace",
            str(tmp_path),
            "--resume",
            "as-prior-session-id",
        ]
    )
    assert rc == EXIT_RESUME_NOT_IMPLEMENTED
    captured = capsys.readouterr()
    assert "resume_not_implemented" in captured.out


def test_harness_builder_missing_workspace_exits_bad_workspace(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Sub-PR B: a missing or absent ``--workspace`` argv exits
    :data:`EXIT_BAD_WORKSPACE` rather than crashing on the contract
    read. Surfaces a clear diagnostic so the host worker can log
    "wiring bug" instead of "container crashed"."""
    rc = main(["--role", "harness_builder", "--cg-id", "cg-test-004"])
    assert rc == EXIT_BAD_WORKSPACE
    captured = capsys.readouterr()
    assert "--workspace" in captured.out or "workspace" in captured.out


def test_harness_builder_missing_cg_id_exits_bad_workspace(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--cg-id`` is required per-role (argparse can't reject it
    upstream); the entry surfaces the missing arg cleanly."""
    rc = main(["--role", "harness_builder", "--workspace", "/tmp"])
    assert rc == EXIT_BAD_WORKSPACE
    captured = capsys.readouterr()
    assert "--cg-id" in captured.out


def test_technique_implementer_stub_exits_not_implemented(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Step 7 fills technique_implementer; sub-PR B leaves it as the
    Step 3 stub."""
    rc = main(["--role", "technique_implementer", "--entry-id", "entry-test-001"])
    assert rc == EXIT_NOT_IMPLEMENTED
    captured = capsys.readouterr()
    assert "not yet implemented" in captured.err
    assert "technique_implementer" in captured.err


def test_planner_role_rejected_with_inline_role_pointer(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The dispatch table accepts ``planner`` so an argparse rejection
    triggers on typos, but the handler immediately raises with a
    pointer back to the host-side inline-role surface per
    ``architectural_decisions.md`` §6.
    """
    rc = main(["--role", "planner"])
    assert rc == EXIT_NOT_IMPLEMENTED
    captured = capsys.readouterr()
    assert "inline" in captured.err
    assert "planner" in captured.err


def test_unknown_role_argparse_rejected() -> None:
    """A typo on ``--role`` gets argparse's "invalid choice" rejection,
    NOT this module's stub error. Confirms the dispatch table is a
    closed set so typos can't silently fall through.
    """
    with pytest.raises(SystemExit) as exc_info:
        main(["--role", "harness_buidler"])  # deliberate typo
    # argparse exits with code 2 on usage errors.
    assert exc_info.value.code == 2


# === ``python -m smai_agent_runtime`` subprocess ==============================


def test_python_m_subprocess_runs_workflow(tmp_path: Path) -> None:
    """Invoke the module via ``python -m smai_agent_runtime`` in a
    subprocess to confirm the ``__main__`` shim + the mini-orchestrator
    wire up correctly. This mirrors the container-runtime invocation
    the dispatch round-trip will exercise (the agent-runtime image runs
    ``python -m smai_agent_runtime ...``). The autouse fixture's env-var
    setting does not survive across the subprocess boundary, so pass
    it explicitly here.
    """
    import os

    write_minimal_harness_workspace(tmp_path)
    env = {**os.environ, "SMAI_AGENT_RUNTIME_FAKE_LLM": "1"}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "smai_agent_runtime",
            "--role",
            "harness_builder",
            "--cg-id",
            "cg-subprocess-test",
            "--workspace",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert proc.returncode == EXIT_OK, (
        f"unexpected exit code {proc.returncode}; stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    # No Python traceback survives — the entry point catches and prints
    # a clean diagnostic instead.
    assert "Traceback" not in proc.stderr
    # D10 session.end line visible on stdout (sub-PR C2 replaced the
    # placeholder "session_end_success" event_type with the structured
    # event-discriminator catalog).
    assert '"event_type":"session.end"' in proc.stdout
    assert '"outcome":"success"' in proc.stdout


# === Internal-error fallthrough ==============================================


def test_unexpected_exception_routes_to_internal_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When a per-role ``main`` raises something other than
    :class:`RoleNotImplementedError`, the entry point exits with
    :data:`EXIT_INTERNAL_ERROR` (70) — distinct from the stub path so
    the host worker can tell "stub" from "crashed".
    """
    from smai_agent_runtime import harness_builder as hb_module

    def _boom(_args: object) -> int:
        raise ValueError("simulated crash")

    monkeypatch.setattr(hb_module, "main", _boom)
    # The entry point imports via ``from smai_agent_runtime.harness_builder
    # import main``; that import binding is module-local to ``_resolve_role``
    # at call time, so the monkeypatch on the package re-export wins.
    rc = main(["--role", "harness_builder", "--cg-id", "x"])
    assert rc == EXIT_INTERNAL_ERROR
    captured = capsys.readouterr()
    assert "crashed" in captured.err
    assert "ValueError" in captured.err
