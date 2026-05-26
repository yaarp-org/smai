"""Unit tests for :class:`LocalGpuCompute` that don't need Docker.

These cover the plugin's pure logic:

* Preflight error shape when ``docker`` is missing on PATH.
* Image-hint message format for the SMAI default tags.
* Mac GPU caveat — ``submit(gpu=True)`` on Darwin raises before the
  daemon is consulted.
* :class:`JobHandle` round-trip preserves timeout / submitted-at /
  container metadata so cross-process reconnection works.

The full lifecycle contract (submit → status → logs → cancel) lives
in :mod:`tests.test_conformance` against a real Docker daemon.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest
from smai_compute_localgpu import (
    DEFAULT_RUNTIME_CPU_IMAGE,
    DEFAULT_RUNTIME_IMAGE,
    LocalGpuCompute,
)
from smai_core.plugins import (
    ComputeUnavailable,
    JobHandle,
    JobImageInvalid,
)


def test_default_image_tags_match_documented_defaults() -> None:
    """The build-hint format depends on these tag names — pin them."""
    assert DEFAULT_RUNTIME_IMAGE == "smai-runtime:dev"
    assert DEFAULT_RUNTIME_CPU_IMAGE == "smai-runtime-cpu:dev"


def test_preflight_raises_compute_unavailable_when_docker_missing() -> None:
    """``LocalGpuCompute()`` fails closed with :class:`ComputeUnavailable`
    when the docker binary isn't on PATH (per Task 2.A4 acceptance:
    "docker info daemon check fails closed when Docker is unavailable")."""
    with patch("smai_compute_localgpu._compute.docker_binary", return_value=None):
        with pytest.raises(ComputeUnavailable) as exc_info:
            LocalGpuCompute()
        assert "docker" in str(exc_info.value).lower()


def test_skip_preflight_constructor_flag_does_not_call_docker() -> None:
    """``skip_preflight=True`` is a test-only escape hatch used by
    :mod:`tests.test_import` to construct a Protocol-conforming object
    without touching Docker."""
    with patch("smai_compute_localgpu._compute.docker_binary") as mock_which:
        instance = LocalGpuCompute(skip_preflight=True)
        mock_which.assert_not_called()
    assert instance.name == "localgpu"


def test_capabilities_advertise_gpu_off_on_mac() -> None:
    """Per the Mac caveat: Docker Desktop on macOS / Apple Silicon
    cannot pass GPU through, so capabilities reflect that on Darwin."""
    instance = LocalGpuCompute(skip_preflight=True)
    assert instance.capabilities.supports_gpu == (sys.platform != "darwin")
    # 24h cap matches Modal Sandbox per ``07`` §7.2.
    assert instance.capabilities.max_timeout_seconds == 24 * 60 * 60
    assert instance.capabilities.supports_log_streaming is False


@pytest.mark.skipif(sys.platform != "darwin", reason="Mac-only caveat")
async def test_submit_gpu_true_on_mac_raises_with_modal_runpod_hint() -> None:
    """``submit(gpu=True)`` on Darwin raises before reaching docker
    and points the user at Modal/RunPod."""
    instance = LocalGpuCompute(skip_preflight=True)
    with pytest.raises(ComputeUnavailable) as exc_info:
        await instance.submit(
            image="smai-runtime:dev",
            command=["true"],
            env={},
            gpu=True,
            timeout_seconds=60,
        )
    msg = str(exc_info.value)
    assert "macOS" in msg or "Mac" in msg
    assert "modal" in msg.lower() or "runpod" in msg.lower()


def test_build_hint_for_default_runtime_image() -> None:
    """The SMAI-default tag check produces the documented build command
    in the missing-image error (Task 2.A4 acceptance: "missing-image
    error contains the exact build command")."""
    instance = LocalGpuCompute(skip_preflight=True)
    err = instance._make_image_invalid(  # pyright: ignore[reportPrivateUsage]
        DEFAULT_RUNTIME_IMAGE, "image not found"
    )
    assert isinstance(err, JobImageInvalid)
    assert err.image == DEFAULT_RUNTIME_IMAGE
    assert "docker build" in err.reason
    assert "runtime.Dockerfile" in err.reason
    assert DEFAULT_RUNTIME_IMAGE in err.reason


def test_build_hint_for_default_runtime_cpu_image() -> None:
    """``compute.gpu=false`` experiments get the CPU image; a missing one
    should point at ``runtime-cpu.Dockerfile`` (round-4 friction (A))."""
    instance = LocalGpuCompute(skip_preflight=True)
    err = instance._make_image_invalid(  # pyright: ignore[reportPrivateUsage]
        DEFAULT_RUNTIME_CPU_IMAGE, "image not found"
    )
    assert isinstance(err, JobImageInvalid)
    assert "runtime-cpu.Dockerfile" in err.reason
    assert DEFAULT_RUNTIME_CPU_IMAGE in err.reason


def test_unknown_image_omits_build_hint() -> None:
    """Third-party images get a generic 'not reachable' message, not a
    SMAI-specific build command."""
    instance = LocalGpuCompute(skip_preflight=True)
    err = instance._make_image_invalid(  # pyright: ignore[reportPrivateUsage]
        "totally-not-real.invalid/whatever:foo", "manifest unknown"
    )
    assert "docker build" not in err.reason
    assert "not reachable" in err.reason


def test_job_handle_round_trips_through_pydantic_json() -> None:
    """Cross-process reconnection (§7.5 / conformance ``test_job_handle_
    reconnection``): a ``JobHandle`` serialized to JSON and re-parsed
    must preserve container_id, submitted_at, timeout_seconds."""
    original = JobHandle(
        plugin="localgpu",
        handle="abcdef0123456789",
        metadata={
            "container_id": "abcdef0123456789",
            "container_name": "smai-localgpu-abc",
            "submitted_at": "2026-04-28T12:00:00Z",
            "timeout_seconds": 300,
            "image": "python:3.12-slim",
            "gpu": False,
        },
    )
    round_tripped = JobHandle.model_validate_json(original.model_dump_json())
    assert round_tripped.handle == original.handle
    assert round_tripped.metadata["container_id"] == "abcdef0123456789"
    assert round_tripped.metadata["timeout_seconds"] == 300
    assert round_tripped.metadata["submitted_at"] == "2026-04-28T12:00:00Z"


def test_container_id_resolution_prefers_metadata() -> None:
    """When ``metadata['container_id']`` is set, it wins over the
    public ``handle.handle`` field."""
    instance = LocalGpuCompute(skip_preflight=True)
    handle = JobHandle(
        plugin="localgpu",
        handle="public-id",
        metadata={"container_id": "metadata-id"},
    )
    assert instance._container_id(handle) == "metadata-id"  # pyright: ignore[reportPrivateUsage]


def test_container_id_falls_back_to_handle_field() -> None:
    """When ``metadata`` lacks ``container_id`` (e.g., a hand-crafted
    handle), the public ``handle.handle`` field is used."""
    instance = LocalGpuCompute(skip_preflight=True)
    handle = JobHandle(plugin="localgpu", handle="just-the-id", metadata={})
    assert instance._container_id(handle) == "just-the-id"  # pyright: ignore[reportPrivateUsage]
