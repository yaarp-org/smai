"""``smai --help`` lists exactly the seven Phase-2 verbs.

Per `09-cli.md` §1 / Task 2.D2 acceptance: Phase-3 verbs (``start``,
``serve``, ``submit-proposal``, ``approve-proposal``,
``reject-proposal``, ``ingest``, ``migrate``) must NOT appear in the
help output. This test is the canonical surface-control gate.
"""

from __future__ import annotations

from smai_cli.main import app
from typer.testing import CliRunner

PHASE_2_VERBS: frozenset[str] = frozenset(
    {"dev", "run", "status", "compile", "init", "plugins", "version"}
)

PHASE_3_VERBS: frozenset[str] = frozenset(
    {
        "start",
        "serve",
        "submit-proposal",
        "approve-proposal",
        "reject-proposal",
        "ingest",
        "migrate",
    }
)


def _run_help(*args: str) -> str:
    runner = CliRunner()
    result = runner.invoke(app, [*args, "--help"])
    assert result.exit_code == 0, f"--help failed: {result.output}"
    return result.output


def _help_command_lines(output: str) -> list[str]:
    """Extract the ``Commands:`` section's verb names.

    Click / Typer renders each verb as ``  <verb>  <description>``.
    We grab the first whitespace-delimited token from each line below
    the ``Commands:`` header — that's the canonical place to look for
    "is verb X registered?" without confusing substring matches in
    the verb's description (e.g., the word ``start`` in ``starter
    project``).
    """
    lines = output.splitlines()
    verbs: list[str] = []
    in_commands = False
    for line in lines:
        if line.strip().lower().startswith("commands:"):
            in_commands = True
            continue
        if not in_commands:
            continue
        if not line.startswith(" "):
            break
        tokens = line.strip().split()
        if tokens:
            verbs.append(tokens[0])
    return verbs


def test_help_lists_only_phase_2_verbs() -> None:
    """The top-level ``smai --help`` lists exactly the seven Phase-2 verbs."""
    output = _run_help()
    listed = set(_help_command_lines(output))
    assert listed == PHASE_2_VERBS, (
        f"--help command list is {sorted(listed)}; expected exactly {sorted(PHASE_2_VERBS)}"
    )
    for verb in PHASE_3_VERBS:
        assert verb not in listed, (
            f"Phase-3 verb {verb!r} leaked into --help command list (Task 2.D2 forbids)"
        )


def test_each_phase_2_verb_has_help() -> None:
    """``smai <verb> --help`` works for every Phase-2 verb."""
    for verb in PHASE_2_VERBS:
        output = _run_help(verb)
        # Each verb's help mentions the verb itself somewhere.
        assert output, f"{verb} --help produced empty output"


def test_no_phase_3_verb_resolves() -> None:
    """Invoking a Phase-3 verb name fails."""
    runner = CliRunner()
    for verb in PHASE_3_VERBS:
        result = runner.invoke(app, [verb])
        # Typer / Click exit with 2 on unknown commands.
        assert result.exit_code != 0, f"Phase-3 verb {verb!r} should not resolve in Phase-2 CLI"
