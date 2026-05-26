"""Entry-point dispatch tests.

Per Step 3 of the agent-layer refactor (see
``designs/smai/agent_refactor/implementation_plan.md`` Step 3 acceptance
criteria): ``python -m smai_agent_runtime --role harness_builder --cg-id
<id>`` must exit with a clear "not yet implemented" error and NOT an
import error.

These tests exercise the dispatch wiring in
:mod:`smai_agent_runtime.__main__`. Step 4 of the refactor will replace
the harness_builder stub with the real mini-orchestrator; Step 7 will
do the same for technique_implementer. Until then the per-role
``main`` raises :class:`RoleNotImplementedError` and the entry point
exits with :data:`EXIT_NOT_IMPLEMENTED` (64).
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from smai_agent_runtime.__main__ import (
    EXIT_INTERNAL_ERROR,
    EXIT_NOT_IMPLEMENTED,
    main,
)

# === Programmatic dispatch ===================================================


def test_harness_builder_stub_exits_not_implemented(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--role harness_builder --cg-id <id>`` exits with
    :data:`EXIT_NOT_IMPLEMENTED` (64) and surfaces a recognizable
    "not yet implemented" message — NOT an import error or
    AttributeError. Step 4 of the refactor replaces this path with the
    real mini-orchestrator.
    """
    rc = main(["--role", "harness_builder", "--cg-id", "cg-test-001"])
    assert rc == EXIT_NOT_IMPLEMENTED
    captured = capsys.readouterr()
    assert "not yet implemented" in captured.err
    assert "harness_builder" in captured.err


def test_technique_implementer_stub_exits_not_implemented(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Same shape as harness_builder; replaced in Step 7."""
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


def test_harness_builder_missing_cg_id_surfaces_not_implemented(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--cg-id`` is per-role (argparse can't reject it upstream); the
    stub diagnoses the missing arg cleanly rather than dropping into
    an AttributeError downstream.
    """
    rc = main(["--role", "harness_builder"])
    assert rc == EXIT_NOT_IMPLEMENTED
    captured = capsys.readouterr()
    assert "harness_builder" in captured.err
    assert "--cg-id" in captured.err


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


def test_python_m_subprocess_exits_with_expected_code() -> None:
    """Invoke the module via ``python -m smai_agent_runtime`` in a
    subprocess to confirm the ``__main__`` shim wires up correctly.
    This is the acceptance path the Step 3 docker-run smoke test
    exercises inside the image; running it here without the image
    confirms the wiring at the Python layer.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "smai_agent_runtime",
            "--role",
            "harness_builder",
            "--cg-id",
            "cg-subprocess-test",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == EXIT_NOT_IMPLEMENTED, (
        f"unexpected exit code {proc.returncode}; stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "not yet implemented" in proc.stderr
    # No Python traceback survives — the entry point catches the raise
    # and prints a clean diagnostic instead.
    assert "Traceback" not in proc.stderr


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
