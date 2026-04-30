"""Plugin-internal unit tests for :class:`RunPodCompute`.

Covers translation tables and quirks the conformance suite doesn't
exercise — RunPod-specific status mappings, GPU-type dispatch, the
shell-args quoter, the ``RUNPOD_API_KEY`` env-var fallback, and
pod-not-found / image-validation error paths.
"""

from __future__ import annotations

import os
from typing import cast

import httpx
import pytest
from _runpod_fakes import FakeRunPodBackend
from smai_compute_runpod import (
    DEFAULT_GPU_TYPE_ID,
    GPU_DISPATCH,
    RunPodCompute,
)
from smai_compute_runpod._compute import _shell_join
from smai_core.plugins import (
    ComputeUnavailable,
    JobHandle,
    JobNotFound,
)


def _build_compute(backend: FakeRunPodBackend, **kwargs: object) -> RunPodCompute:
    transport = backend.transport()
    client = httpx.AsyncClient(transport=transport, base_url="https://rest.runpod.io")
    init_kwargs: dict[str, object] = {
        "api_key": "rpa_fake_test_key",
        "api_base": "https://rest.runpod.io/v1",
        "client": client,
    }
    init_kwargs.update(kwargs)
    return RunPodCompute(**cast("dict[str, object]", init_kwargs))  # type: ignore[arg-type]


def test_constructor_reads_runpod_api_key_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``RunPodCompute()`` reads ``RUNPOD_API_KEY`` when ``api_key`` not given."""
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_env_test")
    compute = RunPodCompute()
    # The plugin doesn't expose the key — confirm construction succeeded
    # and the capability surface is correct.
    assert compute.name == "runpod"
    assert compute.capabilities.supports_gpu is True


def test_constructor_raises_when_no_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing ``RUNPOD_API_KEY`` and no ``api_key=`` raises clean error."""
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(ComputeUnavailable, match="RUNPOD_API_KEY"):
        RunPodCompute()


def test_capabilities_reflect_runpod_substrate() -> None:
    """RunPod always advertises GPU support; max_timeout matches Modal cap."""
    monkeypatch_key = os.environ.get("RUNPOD_API_KEY")
    try:
        os.environ["RUNPOD_API_KEY"] = "rpa_test"
        compute = RunPodCompute()
        assert compute.capabilities.supports_gpu is True
        assert compute.capabilities.max_timeout_seconds == 24 * 60 * 60
        assert compute.capabilities.supports_log_streaming is False
    finally:
        if monkeypatch_key is None:
            os.environ.pop("RUNPOD_API_KEY", None)
        else:
            os.environ["RUNPOD_API_KEY"] = monkeypatch_key


def test_gpu_dispatch_table_covers_v1_tiers() -> None:
    """GPU dispatch table has the documented tiers + maps to non-empty ids."""
    expected_tiers = {"default", "small", "medium", "large", "a100", "h100"}
    assert set(GPU_DISPATCH) == expected_tiers
    for tier, gpu_id in GPU_DISPATCH.items():
        assert gpu_id, f"GPU dispatch tier {tier!r} maps to empty id"
    assert GPU_DISPATCH["default"] == DEFAULT_GPU_TYPE_ID


def test_shell_join_passes_safe_args_unquoted() -> None:
    """Safe shell args (alnum, dots, slashes) survive without quoting."""
    assert _shell_join(["python", "script.py", "--flag=1"]) == "python script.py --flag=1"


def test_shell_join_quotes_unsafe_args() -> None:
    """Shell-special characters force single-quote escaping."""
    rendered = _shell_join(["python", "-c", "import sys; sys.exit(0)"])
    assert "python" in rendered
    assert "-c" in rendered
    assert "'import sys; sys.exit(0)'" in rendered


def test_shell_join_escapes_embedded_single_quote() -> None:
    """Single-quote inside a quoted arg uses the POSIX ``'\\''`` idiom."""
    assert _shell_join(["echo", "it's"]) == r"echo 'it'\''s'"


async def test_status_raises_job_not_found_on_unknown_handle(
    fake_runpod_backend: FakeRunPodBackend,
) -> None:
    """An unknown pod id surfaces as :class:`JobNotFound`."""
    compute = _build_compute(fake_runpod_backend)
    handle = JobHandle(plugin="runpod", handle="pod-does-not-exist", metadata={})
    with pytest.raises(JobNotFound):
        await compute.status(handle)


async def test_logs_raises_job_not_found_on_unknown_handle(
    fake_runpod_backend: FakeRunPodBackend,
) -> None:
    """``logs`` on an unknown pod raises :class:`JobNotFound`."""
    compute = _build_compute(fake_runpod_backend)
    handle = JobHandle(plugin="runpod", handle="pod-does-not-exist", metadata={})
    with pytest.raises(JobNotFound):
        await compute.logs(handle)


async def test_cancel_is_idempotent_against_already_terminated(
    fake_runpod_backend: FakeRunPodBackend,
) -> None:
    """Cancelling a non-existent pod is a no-op (per §7.2 idempotence)."""
    compute = _build_compute(fake_runpod_backend)
    handle = JobHandle(
        plugin="runpod",
        handle="pod-does-not-exist",
        metadata={"pod_id": "pod-does-not-exist"},
    )
    # 404 from terminate path — must not raise.
    await compute.cancel(handle)


async def test_submit_records_metadata(
    fake_runpod_backend: FakeRunPodBackend,
) -> None:
    """Submit-time metadata round-trips on the returned :class:`JobHandle`."""
    compute = _build_compute(fake_runpod_backend)
    handle = await compute.submit(
        image="python:3.12-slim",
        command=["python", "-c", "import sys; sys.exit(0)"],
        env={"FOO": "bar"},
        timeout_seconds=120,
    )
    assert handle.plugin == "runpod"
    assert handle.handle.startswith("pod-")
    assert handle.metadata["image"] == "python:3.12-slim"
    assert handle.metadata["timeout_seconds"] == 120
    assert handle.metadata["gpu"] is False
    assert handle.metadata["gpu_type"] == DEFAULT_GPU_TYPE_ID


async def test_submit_honors_explicit_gpu_type_option(
    fake_runpod_backend: FakeRunPodBackend,
) -> None:
    """``plugin_options['gpu_type']`` overrides the constructor default."""
    compute = _build_compute(fake_runpod_backend)
    handle = await compute.submit(
        image="python:3.12-slim",
        command=["python", "-c", "import sys; sys.exit(0)"],
        env={},
        timeout_seconds=60,
        gpu_type="NVIDIA H100 80GB HBM3",
    )
    assert handle.metadata["gpu_type"] == "NVIDIA H100 80GB HBM3"


async def test_submit_rejects_invalid_gpu_type_kwarg(
    fake_runpod_backend: FakeRunPodBackend,
) -> None:
    """Non-string ``gpu_type`` raises :class:`ValueError` before the API call."""
    compute = _build_compute(fake_runpod_backend)
    with pytest.raises(ValueError, match="gpu_type"):
        await compute.submit(
            image="python:3.12-slim",
            command=["python", "-c", "import sys; sys.exit(0)"],
            env={},
            timeout_seconds=60,
            gpu_type=123,  # type: ignore[arg-type]
        )


async def test_status_translation_terminated_without_cancel_is_failed(
    fake_runpod_backend: FakeRunPodBackend,
) -> None:
    """Pod terminated by substrate (no ``cancel_requested``) → ``failed``.

    Distinguishes "operator killed pod externally" from "user cancelled
    via plugin." The metadata flag is the only authoritative signal —
    the substrate itself doesn't carry user intent.
    """
    compute = _build_compute(fake_runpod_backend)
    handle = await compute.submit(
        image="python:3.12-slim",
        command=["python", "-c", "import time; time.sleep(60)"],
        env={},
        timeout_seconds=600,
    )
    # Reach into the fake to mark TERMINATED without going through
    # plugin.cancel() (which would set ``cancel_requested``).
    pod = fake_runpod_backend.pods[handle.handle]
    pod.desired_status = "TERMINATED"
    status = await compute.status(handle)
    assert status.state == "failed"
    assert status.failure_reason is not None
    assert "TERMINATED" in status.failure_reason
