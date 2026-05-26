"""Sandbox-side agent-runtime entry point.

Dispatches on ``--role`` to a per-role ``main(args)`` callable. Invoked
by the host worker's unified ``make_compute_dispatcher`` (Step 4 of the
agent-layer refactor) as::

    python -m smai_agent_runtime --role harness_builder --cg-id <id>
    python -m smai_agent_runtime --role technique_implementer --entry-id <id>

At Step 3 of the refactor the role handlers are unimplemented stubs;
both per-role ``main`` functions raise :class:`RoleNotImplementedError`
with a clear message so the substrate gate can confirm dispatch wiring
without depending on agent-reasoning code (Step 4 fills
``harness_builder``, Step 7 fills ``technique_implementer``).

Exit codes:

* ``0`` — the role completed cleanly (no role currently does so;
  reserved for Steps 4 / 7).
* ``2`` — argparse rejected the invocation (standard argparse exit code
  for usage errors).
* ``64`` — a known role is not yet implemented (Step 3 stub path).
* ``70`` — an unexpected internal error escaped the role's ``main``
  (mirrors :data:`os.EX_SOFTWARE`).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from smai_agent_runtime.errors import RoleNotImplementedError

# Per-role error/exit constants. Exit codes follow BSD-style sysexits.h
# semantics so the host worker can distinguish "role-not-implemented"
# (a stub path during the agent-layer refactor) from "role crashed"
# (a real bug) without parsing stderr.
EXIT_NOT_IMPLEMENTED = 64
EXIT_INTERNAL_ERROR = 70

# Recognized roles. ``planner`` is listed because the entry point's
# dispatch table is intentionally a closed set (so a typo gets argparse-
# rejected rather than silently exiting), but it never lands in the
# sandbox image — the planner is an inline role per
# ``agent_refactor/architectural_decisions.md`` §6 and runs in the host
# worker process. Including it here documents the closed set; the
# handler raises :class:`RoleNotImplementedError` with a pointer back to
# the inline-role surface.
_ROLES: tuple[str, ...] = ("planner", "harness_builder", "technique_implementer")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m smai_agent_runtime",
        description=(
            "Sandbox-side agent runtime entry point. Dispatches on --role "
            "to a per-role mini-orchestrator. Invoked by the host worker's "
            "unified compute-dispatcher inside the smai-agent-runtime "
            "container image."
        ),
    )
    parser.add_argument(
        "--role",
        required=True,
        choices=_ROLES,
        help=(
            "Which sandboxed agent role to run. ``harness_builder`` and "
            "``technique_implementer`` are the two sandboxed roles per "
            "agent_refactor/architectural_decisions.md §6; ``planner`` is "
            "an inline role and is rejected here with a pointer to the "
            "host-side surface."
        ),
    )
    parser.add_argument(
        "--cg-id",
        default=None,
        help=("Comparison-group id (required for ``--role harness_builder``)."),
    )
    parser.add_argument(
        "--entry-id",
        default=None,
        help=("Entry id (required for ``--role technique_implementer``)."),
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help=(
            "In-container bind-mount path of the staged workspace "
            "(e.g. ``/workspace``). Required for sandboxed roles "
            "(harness_builder, technique_implementer) at the sub-PR B "
            "boundary of the agent-layer refactor."
        ),
    )
    parser.add_argument(
        "--resume",
        default=None,
        help=(
            "Prior agent-session id to resume from. Sub-PR D wires the "
            "resume-mode workflow logic; future PRs add the multi-cycle "
            "review-feedback loop (architectural hedge per "
            "architectural_decisions §12 item 4)."
        ),
    )
    parser.add_argument(
        "--fake-llm",
        action="store_true",
        default=False,
        help=(
            "Test-only seam: substitute a deterministic fake AgentRunner "
            "for the PydanticAI Agent calls so cross-process subprocess "
            "tests can drive the mini-orchestrator without real LLM "
            "credentials. Sub-PR D thread 5 replaced the previous "
            "SMAI_AGENT_RUNTIME_FAKE_LLM env-var read with this "
            "dependency-injected flag (env-var-as-mode-switch is the "
            "kind of smell that hides in production code paths)."
        ),
    )
    return parser


def _resolve_role(role: str) -> Callable[[argparse.Namespace], int]:
    """Map a ``--role`` value to its per-role ``main`` callable.

    Imported lazily so the entry point does not pull every role's
    module on dispatch — keeps the cold-start surface small and isolates
    import failures to the role that owns them.
    """
    if role == "harness_builder":
        from smai_agent_runtime.harness_builder import main  # noqa: PLC0415

        return main
    if role == "technique_implementer":
        from smai_agent_runtime.technique_implementer import main  # noqa: PLC0415

        return main
    if role == "planner":
        # The planner is an inline role per architectural_decisions §6.
        # The dispatch table accepts it so a typo on ``--role`` gets an
        # argparse "invalid choice" rejection rather than this branch's
        # softer "wrong place" diagnostic.
        def _planner_main(_args: argparse.Namespace) -> int:
            raise RoleNotImplementedError(
                "planner is an inline role (see "
                "agent_refactor/architectural_decisions.md §6); it runs "
                "in the host worker process, not inside the agent sandbox "
                "image. Submitting --role planner here is a wiring bug."
            )

        return _planner_main
    raise AssertionError(f"unreachable: argparse rejected role {role!r}")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse argv, dispatch on ``--role``, return the role's exit code.

    Wrapped by ``if __name__ == "__main__"`` below for ``python -m
    smai_agent_runtime`` invocation; importable for tests.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        handler = _resolve_role(args.role)
        return handler(args)
    except RoleNotImplementedError as exc:
        # Clear, recognizable message that lets the Step 3 acceptance
        # test assert "not yet implemented" without depending on stderr
        # tracebacks. The message includes the role name so the host-
        # side log harvester can attribute the stub path to the right
        # role.
        print(
            f"smai-agent-runtime: role {args.role!r} not yet implemented — {exc}",
            file=sys.stderr,
        )
        return EXIT_NOT_IMPLEMENTED
    except Exception as exc:  # noqa: BLE001 — internal-error escape hatch
        print(
            f"smai-agent-runtime: role {args.role!r} crashed with {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return EXIT_INTERNAL_ERROR


if __name__ == "__main__":  # pragma: no cover - module-entry shim
    raise SystemExit(main())
