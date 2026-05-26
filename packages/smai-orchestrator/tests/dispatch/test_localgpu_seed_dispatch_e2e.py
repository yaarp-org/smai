"""End-to-end LocalGpu seed-dispatch through ``make_compute_dispatcher``.

Real-substrate gate for agent-refactor Step 2
(``implementation_plan.md`` Step 2 acceptance criteria; compute_dispatch
§3 "Implementation discipline call"): the unified factory must drive a
real container submit → status-poll → terminal-failure-logs cycle
against :class:`LocalGpuCompute`, NOT just fake-tested. Round-14's
lesson is the load-bearing one: fake-only coverage misses substrate-side
bugs that only surface against real Docker.

Skip-if-no-Docker discipline mirrors
``plugins/smai-compute-localgpu/tests/test_conformance.py``. Docker
must be present + reachable; the runtime CPU image
(``smai-runtime-cpu:dev``) must be loaded locally (round-19 build
runbook in ``packages/smai-cli/OPERATIONS.md``). The test pulls a tiny
Python image instead of ``smai-runtime-cpu:dev`` for the substrate
mechanics — the factory's behavior is image-agnostic, and pulling
``python:3.12-slim`` from Docker Hub is more portable across dev
machines than asserting the SMAI image is staged.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess

import pytest
from _compute_dispatcher_fakes import (  # type: ignore[import-not-found]
    make_dispatch_context,
    static_command_builder,
    static_image_resolver,
)
from smai_compute_localgpu import LocalGpuCompute
from smai_core.plugins import ComputeUnavailable, JobStatus
from smai_core.plugins.conformance import assert_logs_on_failure
from smai_orchestrator.dispatch import (
    CommandSpec,
    WorkspaceInputs,
    WorkspaceOutputs,
    format_stderr_tail,
    make_compute_dispatcher,
)


def _docker_daemon_reachable() -> bool:
    """``docker info`` succeeds → True. Cheap pre-flight."""
    if shutil.which("docker") is None:
        return False
    try:
        completed = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


_DOCKER_AVAILABLE = _docker_daemon_reachable()

pytestmark = pytest.mark.skipif(
    not _DOCKER_AVAILABLE,
    reason=(
        "Docker daemon not reachable; skipping real-substrate "
        "make_compute_dispatcher gate. Install Docker Desktop / OrbStack "
        "(or a compatible OCI runtime) and start the daemon to run."
    ),
)


# ``python:3.12-slim`` is on Docker Hub; ~50MB; pulls on first use.
# OrbStack on macOS makes container startup ~0.5-2s
# (compute_dispatch_decisions.md §7), so a poll loop in seconds is
# plenty.
_FIXTURE_IMAGE = "python:3.12-slim"

# Poll budget: real container starts in 1-3s; the workload itself runs
# in <1s; +safety margin for Docker daemon latency variability.
_POLL_BUDGET_SECONDS = 30.0
_POLL_INTERVAL = 0.5


async def _poll_until_terminal(compute, handle):  # type: ignore[no-untyped-def]
    """Spin on :meth:`Compute.status` until the job leaves the
    ``submitted``/``running`` set."""
    deadline = asyncio.get_event_loop().time() + _POLL_BUDGET_SECONDS
    while True:
        status = await compute.status(handle)
        if status.state not in ("submitted", "running"):
            return status
        if asyncio.get_event_loop().time() > deadline:
            raise TimeoutError(f"job did not reach terminal in {_POLL_BUDGET_SECONDS}s")
        await asyncio.sleep(_POLL_INTERVAL)


async def test_make_compute_dispatcher_real_localgpu_success_roundtrip() -> None:
    """Real-substrate gate: factory submits a successful job, returns a
    :class:`JobHandle`, status converges to ``succeeded``, logs are
    fetchable.

    Exercises the full call chain: ``make_compute_dispatcher`` returns
    a :data:`DispatchHandler`; the handler invokes ``stage_workspace``
    skip path (empty inputs) and ``submit``; the handle is then driven
    through the existing phase-1-equivalent poll loop and ``logs`` is
    fetched on success.
    """
    try:
        compute = LocalGpuCompute()
    except ComputeUnavailable as exc:  # pragma: no cover — guarded by pytestmark
        pytest.skip(f"Docker preflight failed: {exc}")

    dispatcher = make_compute_dispatcher(
        role="seed_run",
        image_resolver=static_image_resolver(_FIXTURE_IMAGE),
        command_builder=static_command_builder(
            CommandSpec(
                command=["python", "-c", "print('factory-dispatch-success')"],
                env={"SMAI_TEST_LANE": "factory-success"},
                gpu=False,
                timeout_seconds=60,
            )
        ),
        inputs=WorkspaceInputs.empty(),
        outputs=WorkspaceOutputs.empty(),
    )

    ctx = await make_dispatch_context(compute=compute)
    outcome = await dispatcher.handler(ctx)

    assert outcome.error is None
    assert len(outcome.submitted_handles) == 1
    handle = outcome.submitted_handles[0]
    assert handle.plugin == compute.name

    status: JobStatus = await _poll_until_terminal(compute, handle)
    assert status.state == "succeeded", (
        f"expected success, got {status!r}; logs={await compute.logs(handle)!r}"
    )

    logs = await compute.logs(handle)
    assert "factory-dispatch-success" in logs

    await compute.cancel(handle)  # idempotent on terminal jobs


async def test_make_compute_dispatcher_real_localgpu_failure_surfaces_stderr() -> None:
    """Real-substrate failure path: a non-zero exit produces logs the
    round-20 generalization can surface into ``last_error``.

    The unit-level ``last_error`` write lives in :func:`phase1_step`;
    here we assert the LocalGpu-level surface (the substrate side of
    the contract): on a failed exit, ``Compute.logs(handle)`` returns
    non-empty content AND :func:`format_stderr_tail` extracts a tail
    that contains the program's stderr — pinning the round-20 HIGH-#1
    generalization at the substrate boundary.

    Uses :func:`assert_logs_on_failure` from
    :mod:`smai_core.plugins.conformance` per
    ``implementation_plan.md`` Step 2's "Step 1's
    assert_logs_on_failure fixture pattern" reference, making the
    consistency-of-use contract testable.
    """
    try:
        compute = LocalGpuCompute()
    except ComputeUnavailable as exc:  # pragma: no cover
        pytest.skip(f"Docker preflight failed: {exc}")

    # Program writes a recognizable string to stderr and exits 2.
    forced_failure_program = (
        "import sys; sys.stderr.write('SMAI_TEST_STDERR_TAIL: forced-failure path\\n'); sys.exit(2)"
    )

    dispatcher = make_compute_dispatcher(
        role="seed_run",
        image_resolver=static_image_resolver(_FIXTURE_IMAGE),
        command_builder=static_command_builder(
            CommandSpec(
                command=["python", "-c", forced_failure_program],
                env={"SMAI_TEST_LANE": "factory-failure"},
                gpu=False,
                timeout_seconds=60,
            )
        ),
        inputs=WorkspaceInputs.empty(),
        outputs=WorkspaceOutputs.empty(),
    )

    ctx = await make_dispatch_context(compute=compute)
    outcome = await dispatcher.handler(ctx)
    assert outcome.error is None  # submit itself succeeded
    handle = outcome.submitted_handles[0]

    status = await _poll_until_terminal(compute, handle)
    assert status.state == "failed", (
        f"expected failure, got {status!r}; logs={await compute.logs(handle)!r}"
    )
    assert status.exit_code == 2

    # Conformance helper pins the round-20 contract — non-empty logs
    # on failure.
    logs = await assert_logs_on_failure(compute, handle)
    assert "SMAI_TEST_STDERR_TAIL" in logs

    # Tail formatter is the round-20 surface phase-1 uses to compose
    # ``last_error``. Short logs round-trip unchanged through the
    # formatter.
    tail = format_stderr_tail(logs)
    assert "SMAI_TEST_STDERR_TAIL" in tail

    await compute.cancel(handle)
