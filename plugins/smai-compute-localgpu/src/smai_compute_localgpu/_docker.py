"""Thin async wrapper over the ``docker`` CLI.

Per Task 2.A4: the plugin shells out to the ``docker`` binary rather
than depending on the Python Docker SDK. Rationale (echoed in the task
report): keeping the plugin dep-free preserves the
``tools/check_deps.py`` Rule 3 invariant trivially and removes a
maintenance vector on top of an OCI substrate that already exposes a
stable CLI surface. Compatible OCI runtimes (Podman, Apple
``container``, containerd via ``nerdctl``) typically alias ``docker``
on ``$PATH`` so the same shell-out works without code changes.

All helpers are ``async`` (use :func:`asyncio.create_subprocess_exec`)
to avoid blocking the worker loop. ``shell=True`` is never used; every
argv list is fully escaped by the kernel.
"""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Result of a single ``docker`` invocation."""

    returncode: int
    stdout: str
    stderr: str


def docker_binary() -> str | None:
    """Return the absolute path to ``docker`` on ``$PATH`` or ``None``."""
    return shutil.which("docker")


async def run_docker(*args: str, timeout: float | None = 30.0) -> CommandResult:
    """Run ``docker <args...>`` and return stdout / stderr / exit code.

    A missing binary surfaces as ``returncode=127`` with a stderr hint;
    the caller decides whether to translate that into
    :class:`ComputeUnavailable` (preflight) or a generic substrate error
    (mid-run).
    """
    binary = docker_binary()
    if binary is None:
        return CommandResult(
            returncode=127,
            stdout="",
            stderr="docker binary not found on PATH",
        )
    proc = await asyncio.create_subprocess_exec(
        binary,
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return CommandResult(
            returncode=124,
            stdout="",
            stderr=f"docker {args[0] if args else ''} timed out after {timeout}s",
        )
    return CommandResult(
        returncode=proc.returncode if proc.returncode is not None else -1,
        stdout=stdout_bytes.decode("utf-8", errors="replace"),
        stderr=stderr_bytes.decode("utf-8", errors="replace"),
    )
