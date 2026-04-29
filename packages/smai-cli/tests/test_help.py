"""``smai --help`` lists exactly the verbs landed so far.

Per `09-cli.md` §1: this is the canonical surface-control gate.
Phase-2 landed seven verbs (``dev``, ``run``, ``status``,
``compile``, ``init``, ``plugins``, ``version``); Task 3.E1
(Phase-3 W1.a) landed three proposal verbs (``submit-proposal``,
``approve-proposal``, ``reject-proposal``) per DEC-032's primary-
input-path framing. Task 3.E2 (Phase-3 W1.b) landed ``ingest``
per DEC-032 OQ1's "explicit `smai ingest` only" rule. Remaining
Phase-3 verbs (``start``, ``serve``, ``migrate``) must NOT appear
yet.
"""

from __future__ import annotations

from smai_cli.main import app
from typer.testing import CliRunner

EXPECTED_VERBS: frozenset[str] = frozenset(
    {
        # Phase-2
        "dev",
        "run",
        "status",
        "compile",
        "init",
        "plugins",
        "version",
        # Phase-3 W1.a / Task 3.E1 (DEC-032)
        "submit-proposal",
        "approve-proposal",
        "reject-proposal",
        # Phase-3 W1.b / Task 3.E2 (DEC-032 OQ1)
        "ingest",
    }
)

FORBIDDEN_PHASE_3_VERBS: frozenset[str] = frozenset(
    {
        "start",
        "serve",
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


def test_help_lists_only_expected_verbs() -> None:
    """The top-level ``smai --help`` lists exactly the verbs landed so far."""
    output = _run_help()
    listed = set(_help_command_lines(output))
    assert listed == EXPECTED_VERBS, (
        f"--help command list is {sorted(listed)}; expected exactly {sorted(EXPECTED_VERBS)}"
    )
    for verb in FORBIDDEN_PHASE_3_VERBS:
        assert verb not in listed, (
            f"Forbidden Phase-3 verb {verb!r} leaked into --help command list"
        )


def test_each_expected_verb_has_help() -> None:
    """``smai <verb> --help`` works for every landed verb."""
    for verb in EXPECTED_VERBS:
        output = _run_help(verb)
        # Each verb's help mentions the verb itself somewhere.
        assert output, f"{verb} --help produced empty output"


def test_no_forbidden_verb_resolves() -> None:
    """Invoking a not-yet-landed Phase-3 verb name fails."""
    runner = CliRunner()
    for verb in FORBIDDEN_PHASE_3_VERBS:
        result = runner.invoke(app, [verb])
        # Typer / Click exit with 2 on unknown commands.
        assert result.exit_code != 0, (
            f"Forbidden Phase-3 verb {verb!r} should not resolve in current CLI"
        )
