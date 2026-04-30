"""Opt-in production-readiness check against real RunPod.

Skips by default. Runs only when ``RUNPOD_API_KEY`` is set in the
environment. Marked ``@pytest.mark.credentialed`` so CI lanes that
filter on the marker (``-m "not credentialed"``) skip cleanly per the
no-credentials-in-CI convention from Task 3.G3.

Round-trips submit / status / logs / cancel against a real RunPod pod
so a developer with credentials can verify end-to-end behavior before
reporting Task 3.F4 complete. Uses a tiny ``python:3.12-slim`` image
and the cheapest configured GPU tier; per-run cost should be measured
in fractions of a cent (the pod runs for a few seconds at most).
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest
from smai_compute_runpod import RunPodCompute

_API_KEY_ENV = "RUNPOD_API_KEY"

pytestmark = [
    pytest.mark.credentialed,
    pytest.mark.skipif(
        _API_KEY_ENV not in os.environ,
        reason=(
            f"Set ${_API_KEY_ENV} to run real-RunPod validation. See "
            "plugins/smai-compute-runpod/README.md for setup."
        ),
    ),
]


# Cheap GPU tier for the validation lane. Override via env var if the
# operator's account has different availability.
_TEST_GPU_TYPE = os.environ.get("RUNPOD_TEST_GPU_TYPE", "NVIDIA RTX A4000")
_TEST_IMAGE = os.environ.get("RUNPOD_TEST_IMAGE", "python:3.12-slim")
_REAL_TIMEOUT_SECONDS = 300  # 5 minutes — large enough for cold pulls


@pytest.fixture
async def real_compute() -> RunPodCompute:
    """Construct a real-credentialed :class:`RunPodCompute`."""
    return RunPodCompute(default_gpu_type=_TEST_GPU_TYPE)


async def test_real_runpod_round_trip(real_compute: RunPodCompute) -> None:
    """Exercise submit / status / logs / cancel against the live substrate."""
    token = "smai-real-runpod-test-token"
    handle = await real_compute.submit(
        image=_TEST_IMAGE,
        command=["python", "-c", f"print({token!r})"],
        env={},
        timeout_seconds=_REAL_TIMEOUT_SECONDS,
    )
    try:
        # Poll until terminal — RunPod cold-starts can take a minute.
        deadline = time.monotonic() + _REAL_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            status = await real_compute.status(handle)
            if status.state in ("succeeded", "failed", "cancelled", "timeout"):
                break
            await asyncio.sleep(5)
        else:
            pytest.fail(f"job {handle.handle} did not reach terminal state in budget")
        assert status.state == "succeeded", (
            f"expected succeeded, got {status.state} (reason={status.failure_reason})"
        )
        # Log retrieval may take a beat after the pod exits.
        for _ in range(6):
            logs = await real_compute.logs(handle)
            if token in logs:
                return
            await asyncio.sleep(5)
        pytest.fail(f"token {token!r} did not appear in logs after pod completion")
    finally:
        # Idempotent cleanup — cancel a finished pod is a no-op.
        await real_compute.cancel(handle)
        await real_compute.aclose()


async def test_real_runpod_cancel_long_running(real_compute: RunPodCompute) -> None:
    """Cancel a long-running pod mid-flight."""
    handle = await real_compute.submit(
        image=_TEST_IMAGE,
        command=["python", "-c", "import time; time.sleep(600)"],
        env={},
        timeout_seconds=_REAL_TIMEOUT_SECONDS,
    )
    try:
        await asyncio.sleep(10)  # let the pod boot
        await real_compute.cancel(handle)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            status = await real_compute.status(handle)
            if status.state in ("succeeded", "failed", "cancelled", "timeout"):
                break
            await asyncio.sleep(5)
        else:
            pytest.fail("job did not terminate after cancel within 2-minute budget")
        assert status.state == "cancelled", (
            f"expected cancelled, got {status.state} (reason={status.failure_reason})"
        )
    finally:
        await real_compute.aclose()
