"""Sandbox-side error types.

Kept in a leaf module so :mod:`smai_agent_runtime.__main__` and the
per-role modules can share types without a circular import.
"""

from __future__ import annotations


class RoleNotImplementedError(RuntimeError):
    """Raised by a per-role ``main`` when the role's body has not yet
    landed in the refactor.

    Step 4 of the agent-layer refactor replaces the raise in
    :func:`smai_agent_runtime.harness_builder.main` with the real
    mini-orchestrator; Step 7 replaces
    :func:`smai_agent_runtime.technique_implementer.main`. The
    ``__main__`` entry point catches this and exits with
    :data:`smai_agent_runtime.__main__.EXIT_NOT_IMPLEMENTED` so the
    Step 3 substrate gate can distinguish "stub path" from "crash".
    """


__all__ = ["RoleNotImplementedError"]
