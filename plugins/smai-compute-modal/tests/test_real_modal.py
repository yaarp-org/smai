"""Opt-in production-readiness check against real Modal Sandboxes.

Skips by default. Runs only when ``MODAL_TOKEN_ID`` and
``MODAL_TOKEN_SECRET`` are set in the environment (Modal's SDK reads
them automatically via its default credential chain).

Marked ``@pytest.mark.credentialed`` per Task 3.G3's no-credentials-in-CI
convention — the marker is registered in the root ``pyproject.toml``
and CI never runs ``pytest -m credentialed``. The test is
local-manual-only: a credential-holder runs it pre-merge to verify
:class:`ModalCompute` works end-to-end against the real substrate
before reporting Task 3.F3 complete.

Round-trips :meth:`submit` → :meth:`status` → :meth:`logs` →
:meth:`cancel` against a real Modal Sandbox running a tiny
``python:3.12-slim`` job. No GPU is requested — the credentialed
test is meant to validate the SDK plumbing, not Modal's GPU
inventory; GPU lanes are inherently flakier (capacity-bound) and
spending GPU-seconds on a smoke test isn't justified.

To run locally::

    export MODAL_TOKEN_ID=...
    export MODAL_TOKEN_SECRET=...
    uv run pytest plugins/smai-compute-modal/tests/test_real_modal.py -v -m credentialed

The test does NOT take a ``MODAL_TEST_*``-prefixed env var because
Modal's SDK already gates everything on ``MODAL_TOKEN_ID`` /
``MODAL_TOKEN_SECRET`` — there is no "wrong account" failure mode the
way ``AWS_TEST_BUCKET`` protects S3.
"""

from __future__ import annotations

import asyncio
import os

import pytest
from smai_compute_modal import ModalCompute
from smai_core.plugins import JobStatus

# The marker registration lives in root pyproject.toml per Task 3.G3.
# Each credentialed test ALSO carries skipif-on-env so it skips
# cleanly without creds — `pytest -m credentialed` against a clean
# environment yields a noop, NOT a failure.
_REQUIRED_CREDS = ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")
_missing_creds = [name for name in _REQUIRED_CREDS if name not in os.environ]

pytestmark = [
    pytest.mark.credentialed,
    pytest.mark.skipif(
        bool(_missing_creds),
        reason=(
            "credentialed-only; missing env: " + ", ".join(_missing_creds)
            if _missing_creds
            else "skipped"
        ),
    ),
]


# Modal Sandboxes running a tiny ``python:3.12-slim`` typically
# pull-and-launch in 10-30s. We wait up to 3 minutes — enough for
# image-cache cold starts and Modal-side scheduling jitter without
# hanging the test indefinitely.
_TERMINAL_TIMEOUT_SECONDS = 180.0
_POLL_INTERVAL_SECONDS = 3.0
_TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled", "timeout"})

_FIXTURE_IMAGE = "python:3.12-slim"
_FIXTURE_MARKER = "smai-real-modal-marker-79b3"


async def _poll_until_terminal(
    compute: ModalCompute,
    handle: object,
) -> JobStatus:
    """Poll ``status`` until terminal or the per-test budget elapses."""
    deadline = asyncio.get_event_loop().time() + _TERMINAL_TIMEOUT_SECONDS
    while True:
        status = await compute.status(handle)  # type: ignore[arg-type]
        if status.state in _TERMINAL_STATES:
            return status
        if asyncio.get_event_loop().time() > deadline:
            raise TimeoutError(
                f"sandbox did not reach terminal within {_TERMINAL_TIMEOUT_SECONDS}s; "
                f"last state={status.state!r}"
            )
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)


async def test_real_modal_submit_status_logs_cycle() -> None:
    """End-to-end submit → status → logs against a real Modal Sandbox.

    Exercises the production code path:

    * :meth:`ModalCompute.submit` creates a real Sandbox.
    * :meth:`ModalCompute.status` polls Modal until terminal.
    * :meth:`ModalCompute.logs` reads the Sandbox's stdout — the
      fixture marker token must appear.
    * :meth:`ModalCompute.cancel` is called as cleanup (idempotent on
      a terminated sandbox).
    """
    compute = ModalCompute(app_name="smai-real-modal-test")
    handle = await compute.submit(
        image=_FIXTURE_IMAGE,
        command=["python", "-c", f"print({_FIXTURE_MARKER!r})"],
        env={},
        timeout_seconds=120,
    )
    try:
        status = await _poll_until_terminal(compute, handle)
        assert status.state == "succeeded", (
            f"expected succeeded; got state={status.state!r} "
            f"exit_code={status.exit_code} reason={status.failure_reason}"
        )
        assert status.exit_code == 0
        logs = await compute.logs(handle)
        assert _FIXTURE_MARKER in logs, (
            f"expected marker {_FIXTURE_MARKER!r} in logs; got: {logs[:500]!r}"
        )
    finally:
        # Idempotent on a terminated sandbox — Modal's terminate is a
        # no-op when the sandbox has already exited.
        await compute.cancel(handle)


async def test_real_modal_cancel_terminates_running_job() -> None:
    """Submit a long-sleeping job, cancel mid-run, expect ``cancelled``.

    The §7.5 cancel contract: eventually-consistent — the next
    ``status`` call returns ``cancelled`` once Modal confirms.
    """
    compute = ModalCompute(app_name="smai-real-modal-test")
    handle = await compute.submit(
        image=_FIXTURE_IMAGE,
        command=["python", "-c", "import time; time.sleep(120)"],
        env={},
        timeout_seconds=180,
    )
    # Give Modal a beat to actually start the sandbox before cancelling
    # — cancelling a not-yet-running sandbox is a valid path but harder
    # to verify-end-to-end (the sandbox may exit before Modal records
    # state).
    await asyncio.sleep(8)
    await compute.cancel(handle)
    status = await _poll_until_terminal(compute, handle)
    assert status.state == "cancelled", (
        f"expected cancelled; got state={status.state!r} "
        f"exit_code={status.exit_code} reason={status.failure_reason}"
    )
    # Idempotent — second cancel on a terminal sandbox is a no-op.
    await compute.cancel(handle)
