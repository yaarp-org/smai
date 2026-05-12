""":class:`LocalGpuCompute` — :class:`Compute` adapter for local Docker.

Per ``07-plugin-interfaces.md`` §7 (the full Protocol surface), the
``smai-compute-localgpu`` reference per ``07`` §7.4 / §7.5 and
``10-runtime-and-templates.md`` §8.5 (substrate contract). The Phase 2
reference plugin per the implementation_plan §3.3 Task 2.A4 brief.

Substrate is **Docker** (commit a9e57bd). Compatible OCI runtimes —
Podman, Apple ``container``, containerd via ``nerdctl`` — typically
alias the ``docker`` binary on ``$PATH`` so the same shell-out works
without code changes.

The plugin shells out to ``docker`` via :mod:`asyncio.create_subprocess_exec`
rather than depending on the Python Docker SDK. Rationale:

* Keeps the plugin's runtime dep set empty (only ``smai-core``), trivially
  satisfying ``tools/check_deps.py`` Rule 3.
* Removes a maintenance vector on top of an OCI substrate that already
  exposes a stable CLI surface.
* Lets compatible OCI runtimes drop in by aliasing ``docker`` on
  ``$PATH``.

**Image distribution:** SMAI does NOT publish prebuilt images for the
OSS plugin in v1 per ``07`` §7.4 / commit a9e57bd. Users build the two
reference Dockerfiles (``agent.Dockerfile``, ``runtime.Dockerfile``)
themselves before first use; the plugin's missing-image error embeds
the exact ``docker build`` command.

**Mac caveat:** Docker Desktop on macOS / Apple Silicon does NOT pass
through GPU hardware. ``submit(..., gpu=True)`` on Darwin raises a
:class:`ComputeUnavailable` pointing the user at Modal or RunPod for
the experiment side. The agent side (``gpu=False``) still works.

**Not in scope** for this plugin:
* Multi-host GPU pools (single-host only).
* Cost accounting (per ``07`` §7.4).
* Spot vs on-demand selection (per ``07`` §7.4).
* Trust boundaries / IAM (per ``07`` §7.4 — local plugin runs as the
  invoking user; no credential surface).
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import uuid
from datetime import UTC, datetime
from typing import Any, cast

from smai_core.plugins import (
    ComputeCapabilities,
    ComputeUnavailable,
    JobHandle,
    JobImageInvalid,
    JobNotFound,
    JobState,
    JobStatus,
)

from smai_compute_localgpu._docker import docker_binary, run_docker

# === Constants ==============================================================

# Default image tags expected by the plugin. Users build these locally
# from the three reference Dockerfiles shipped under ``dockerfiles/``;
# SMAI does not publish prebuilt images for the OSS plugin in v1
# (commit a9e57bd / ``07`` §7.4 / Task 2.A4).
#
# * ``smai-agent:dev`` (``agent.Dockerfile``) — agent-side code exec, no
#   ML stack, lean.
# * ``smai-runtime:dev`` (``runtime.Dockerfile``) — GPU experiment runs,
#   ``nvidia/cuda`` base + ML stack, ~5-8 GB, amd64-only.
# * ``smai-runtime-cpu:dev`` (``runtime-cpu.Dockerfile``) — CPU-only
#   experiment runs, lean multi-arch base + ML stack. Used when a CG's
#   ``controlled_conditions.compute.gpu`` is ``false`` (the orchestrator's
#   run-record dispatcher passes this image; round-4 friction A).
DEFAULT_AGENT_IMAGE = "smai-agent:dev"
DEFAULT_RUNTIME_IMAGE = "smai-runtime:dev"
DEFAULT_RUNTIME_CPU_IMAGE = "smai-runtime-cpu:dev"

# Container-name prefix used when invoking ``docker run --name``.
# ``--label smai-localgpu=1`` is also applied so users can prune via::
#
#     docker container prune --filter label=smai-localgpu=1
_CONTAINER_NAME_PREFIX = "smai-localgpu"
_CONTAINER_LABEL = "smai-localgpu=1"

# ``docker stop`` grace period before SIGKILL. Decision: graceful, then
# kill — the substrate sends SIGTERM first, waits N seconds, then SIGKILL.
# Matches the Compute Protocol's ``cancel`` semantics ("eventually
# consistent" — caller polls ``status`` until ``cancelled``).
_CANCEL_GRACE_SECONDS = 10

# Per-call timeouts for the various ``docker`` commands. ``run -d`` can
# take a while when pulling; everything else is quick. These guard
# against a wedged daemon, not the user's job timeout.
_DOCKER_INFO_TIMEOUT = 10.0
_DOCKER_INSPECT_TIMEOUT = 10.0
_DOCKER_LOGS_TIMEOUT = 30.0
_DOCKER_STOP_TIMEOUT = float(_CANCEL_GRACE_SECONDS + 5)
_DOCKER_RUN_TIMEOUT = 600.0  # accommodates first-time image pulls
_DOCKER_PULL_TIMEOUT = 600.0

# Substrate-imposed maximum timeout for a job. v1 plugins all return
# False for ``supports_log_streaming`` per ``07`` §7.2.
_MAX_TIMEOUT_SECONDS = 24 * 60 * 60  # 24h, matches Modal Sandbox cap

# Map ``docker inspect`` ``State.Status`` strings → :class:`JobState`.
# ``"created"`` / ``"running"`` are non-terminal; all others are terminal.
# ``"dead"`` and ``"removing"`` are docker-internal transient terminal
# shapes — treated as ``failed``.
_DOCKER_STATE_TO_JOB_STATE: dict[str, JobState] = {
    "created": "submitted",
    "running": "running",
    "paused": "running",
    "restarting": "running",
    "exited": "succeeded",  # exit code refines this; see translate_status
    "dead": "failed",
    "removing": "failed",
}

# Regex that matches docker's "no such image" / "manifest unknown"
# error messages. Used to translate ``docker pull`` failures into
# :class:`JobImageInvalid`.
_IMAGE_NOT_FOUND_RE = re.compile(
    r"(?:not found|manifest unknown|repository does not exist|"
    r"pull access denied|no such image|unable to find image)",
    re.IGNORECASE,
)


# === Implementation =========================================================


class LocalGpuCompute:
    """Local-Docker reference implementation of :class:`Compute`.

    Constructor::

        LocalGpuCompute()  # uses smai-agent:dev / smai-runtime:dev / smai-runtime-cpu:dev defaults
        LocalGpuCompute(
            agent_image="smai-agent:dev",
            runtime_image="smai-runtime:dev",
            runtime_cpu_image="smai-runtime-cpu:dev",
            workspace="/path/to/workspace",  # bind-mounted at /workspace
        )

    The image kwargs are used only to make ``submit``'s "image not
    found" error actionable (it appends a ``docker build -t <name> ...``
    hint when the failing image matches one of these). The actual image
    for any given job is whatever the caller passes to ``submit`` — the
    orchestrator's run-record dispatcher hands ``runtime_image`` for
    GPU jobs and ``runtime_cpu_image`` for ``compute.gpu=false`` ones.

    The constructor performs the ``docker info`` preflight (per ``07``
    §7 / Task 2.A4); a missing daemon raises :class:`ComputeUnavailable`
    so callers can fall back or skip cleanly.
    """

    name: str = "localgpu"

    capabilities: ComputeCapabilities

    def __init__(
        self,
        *,
        agent_image: str = DEFAULT_AGENT_IMAGE,
        runtime_image: str = DEFAULT_RUNTIME_IMAGE,
        runtime_cpu_image: str = DEFAULT_RUNTIME_CPU_IMAGE,
        workspace: str | None = None,
        skip_preflight: bool = False,
    ) -> None:
        self._agent_image = agent_image
        self._runtime_image = runtime_image
        self._runtime_cpu_image = runtime_cpu_image
        self._workspace = workspace
        # Capabilities reflect host shape: GPU is reported as "supported"
        # whenever the host isn't macOS — Docker Desktop on Mac cannot
        # pass through GPU hardware, period. Non-Mac hosts may still
        # fail at submit() with a clearer ComputeUnavailable if NVIDIA
        # Container Toolkit isn't installed; capabilities is a static
        # advertisement, not a hard guarantee.
        self.capabilities = ComputeCapabilities(
            supports_gpu=sys.platform != "darwin",
            max_timeout_seconds=_MAX_TIMEOUT_SECONDS,
            supports_log_streaming=False,
        )
        if not skip_preflight:
            self._preflight_docker_info_sync()

    # --- public surface (Compute Protocol) ---------------------------------

    async def submit(
        self,
        image: str,
        command: list[str],
        env: dict[str, str],
        gpu: bool = False,
        timeout_seconds: int = 3600,
        **plugin_options: object,
    ) -> JobHandle:
        """Submit a containerized job.

        Implements ``07`` §7.2's ``submit``: returns once the substrate
        accepts the job (NOT when it begins executing). Eager image
        validation per ``07`` §7.4 ("eager fail on bad image references,
        not lazy fail mid-run"). Timeout is enforced by ``status`` —
        ``submitted_at`` is encoded in :attr:`JobHandle.metadata` so the
        elapsed-time check survives cross-process reconnection.

        ``plugin_options`` accepts:

        * ``workspace: str`` — overrides the constructor's ``workspace``
          for this submission. Bind-mounted at ``/workspace`` per
          ``10-runtime-and-templates.md`` §8.5.
        """
        # Mac GPU caveat: hard-fail before invoking docker.
        if gpu and sys.platform == "darwin":
            raise ComputeUnavailable(
                "GPU passthrough is not available on Docker Desktop for macOS / "
                "Apple Silicon. Use a Linux host with NVIDIA Container Toolkit, "
                "or point Compute at smai-compute-modal / smai-compute-runpod for "
                "the experiment side. The agent side (gpu=False) still works on Mac."
            )

        await self._validate_image_eager(image)

        if gpu:
            await self._preflight_nvidia_smi()

        workspace_arg = plugin_options.get("workspace", self._workspace)
        if workspace_arg is not None and not isinstance(workspace_arg, str):
            raise ValueError(
                f"plugin_options['workspace'] must be a str path, got "
                f"{type(workspace_arg).__name__}"
            )

        container_name = f"{_CONTAINER_NAME_PREFIX}-{uuid.uuid4().hex[:12]}"
        argv: list[str] = ["run", "--detach", "--name", container_name]
        argv.extend(["--label", _CONTAINER_LABEL])
        if gpu:
            argv.extend(["--gpus", "all"])
        if workspace_arg is not None:
            argv.extend(["-v", f"{workspace_arg}:/workspace"])
        for key, value in env.items():
            # ``-e KEY=VALUE`` — list args, no shell expansion.
            argv.extend(["-e", f"{key}={value}"])
        argv.append(image)
        argv.extend(command)

        result = await run_docker(*argv, timeout=_DOCKER_RUN_TIMEOUT)
        if result.returncode != 0:
            # Fallback image-invalid translation: even after eager check,
            # ``docker run`` may still surface a pull-time failure on a
            # racy daemon or partial-pull. Map known image-failure
            # patterns to JobImageInvalid; everything else flows through
            # as ComputeUnavailable.
            if _IMAGE_NOT_FOUND_RE.search(result.stderr):
                raise self._make_image_invalid(image, result.stderr)
            raise ComputeUnavailable(
                f"docker run failed (exit {result.returncode}): "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )

        container_id = result.stdout.strip()
        if not container_id:
            raise ComputeUnavailable(
                "docker run returned no container id; daemon state is unexpected"
            )

        submitted_at = _utc_now_iso()
        metadata: dict[str, Any] = {
            "container_id": container_id,
            "container_name": container_name,
            "submitted_at": submitted_at,
            "timeout_seconds": int(timeout_seconds),
            "image": image,
            "gpu": bool(gpu),
        }
        # ``handle`` (the public id) is the container-id prefix — short
        # enough to log readably; ``metadata`` carries the rest. Both are
        # round-tripped via Pydantic JSON in cross-process reconnection.
        return JobHandle(
            plugin=self.name,
            handle=container_id,
            metadata=metadata,
        )

    async def status(self, handle: JobHandle) -> JobStatus:
        """Poll the container's state.

        Per ``07`` §7.2: MUST raise :class:`JobNotFound` when the
        substrate has no record of the handle; MUST NOT block waiting
        for state changes. Timeout enforcement happens here — when the
        elapsed wall-clock exceeds ``timeout_seconds`` and the container
        is still running, the plugin sends ``docker kill`` and reports
        ``state='timeout'`` so the conformance contract holds.
        """
        container_id = self._container_id(handle)
        inspect = await self._inspect(container_id)
        if inspect is None:
            raise JobNotFound(handle)

        state = inspect.get("State", {})
        docker_status = state.get("Status", "")
        started_at = state.get("StartedAt") or None
        finished_at = state.get("FinishedAt") or None
        # Docker emits "0001-01-01T00:00:00Z" for unset timestamps.
        started_at = _normalize_timestamp(started_at)
        finished_at = _normalize_timestamp(finished_at)
        exit_code_raw = state.get("ExitCode")
        exit_code: int | None = int(exit_code_raw) if isinstance(exit_code_raw, int) else None
        oom_killed = bool(state.get("OOMKilled"))
        was_killed_by_us = bool(state.get("smai_killed_by_timeout"))  # never set

        # Timeout enforcement: if elapsed > timeout_seconds and the
        # container is still running, kill it and report timeout.
        timeout_seconds = int(handle.metadata.get("timeout_seconds", 3600))
        submitted_at_raw = handle.metadata.get("submitted_at")
        if (
            docker_status == "running"
            and isinstance(submitted_at_raw, str)
            and _elapsed_seconds(submitted_at_raw) > timeout_seconds
        ):
            await self._kill(container_id)
            return JobStatus(
                state="timeout",
                exit_code=None,
                started_at=started_at,
                finished_at=_utc_now_iso(),
                failure_reason=(
                    f"job exceeded timeout_seconds={timeout_seconds}; container killed by plugin"
                ),
            )

        # Cancellation: ``docker inspect`` may return ``exited`` with
        # exit code 137 (SIGKILL) or 143 (SIGTERM) when stopped. The
        # ``handle.metadata`` flag (``cancel_requested``) lets us
        # distinguish "user-cancelled" from "naturally failed with
        # 137". v1 stores it on the handle metadata via in-memory
        # mutation — see ``cancel`` below.
        cancel_requested = bool(handle.metadata.get("cancel_requested"))

        # Translate docker State.Status → JobState.
        job_state: JobState
        failure_reason: str | None = None
        if docker_status == "exited":
            if cancel_requested:
                job_state = "cancelled"
            elif exit_code == 0:
                job_state = "succeeded"
            else:
                job_state = "failed"
                if oom_killed:
                    failure_reason = "container was OOM-killed by the host"
                elif was_killed_by_us:
                    failure_reason = "container was killed by the plugin"
                else:
                    failure_reason = f"container exited with non-zero status {exit_code}"
        elif docker_status in _DOCKER_STATE_TO_JOB_STATE:
            job_state = _DOCKER_STATE_TO_JOB_STATE[docker_status]
            if job_state == "failed":
                failure_reason = f"docker reports container in state {docker_status!r}"
        else:
            # Unknown status — surface as failed rather than silently
            # accepting it.
            job_state = "failed"
            failure_reason = f"unknown docker State.Status: {docker_status!r}"

        return JobStatus(
            state=job_state,
            exit_code=exit_code,
            started_at=started_at,
            finished_at=finished_at,
            failure_reason=failure_reason,
        )

    async def logs(self, handle: JobHandle) -> str:
        """Return ``stdout`` + ``stderr`` concatenated.

        Per ``07`` §7.2: long logs may be returned tail-only. v1 returns
        the full content — local Docker's ``docker logs`` is bounded by
        the container's per-stream JSON file (default ~10 MB).
        """
        container_id = self._container_id(handle)
        # ``docker logs`` writes container stderr to the host's stderr
        # stream by default (per the Docker engine: the daemon
        # demultiplexes the two streams). Combine via ``2>&1``-equivalent
        # (the asyncio CLI wrapper captures both already).
        result = await run_docker("logs", container_id, timeout=_DOCKER_LOGS_TIMEOUT)
        if result.returncode != 0:
            if _is_no_such_container(result.stderr):
                raise JobNotFound(handle)
            raise ComputeUnavailable(
                f"docker logs failed (exit {result.returncode}): {result.stderr.strip()}"
            )
        return result.stdout + result.stderr

    async def cancel(self, handle: JobHandle) -> None:
        """Send ``SIGTERM`` (with grace period) to the container.

        Per ``07`` §7.2: eventually-consistent — the next ``status``
        call will return ``state='cancelled'`` once docker confirms the
        terminal state. Idempotent: cancelling an already-terminal
        container is a no-op (``docker stop`` succeeds with no-op).
        """
        container_id = self._container_id(handle)
        # Mark the handle so ``status`` translates the eventual
        # ``exited`` Docker state into ``cancelled`` rather than
        # ``failed``. Mutating the metadata in-place is fine — the
        # caller persists the handle once after ``submit`` and
        # subsequent calls flow through the same in-memory object.
        # Cross-process reconnection still gets the right answer
        # because ``docker stop`` exit codes (137/143) tell us nothing
        # about user intent — the only authoritative signal is "the
        # caller asked for cancel."
        try:
            handle.metadata["cancel_requested"] = True
        except (TypeError, AttributeError):  # pragma: no cover — defensive
            pass

        result = await run_docker(
            "stop",
            "--time",
            str(_CANCEL_GRACE_SECONDS),
            container_id,
            timeout=_DOCKER_STOP_TIMEOUT,
        )
        if result.returncode != 0:
            if _is_no_such_container(result.stderr):
                # Idempotent — the container is already gone.
                return
            # ``docker stop`` of an already-stopped container exits 0,
            # so a non-zero here is a real substrate problem.
            raise ComputeUnavailable(
                f"docker stop failed (exit {result.returncode}): {result.stderr.strip()}"
            )

    # --- preflight ---------------------------------------------------------

    def _preflight_docker_info_sync(self) -> None:
        """Synchronous ``docker info`` preflight (constructor path).

        Constructor is sync, so this uses :func:`subprocess.run`. The
        ``docker info`` daemon check fails closed: a missing binary or
        a stopped daemon raises :class:`ComputeUnavailable` with an
        actionable message.
        """
        import subprocess  # noqa: PLC0415 — sync path only

        binary = docker_binary()
        if binary is None:
            raise ComputeUnavailable(
                "docker binary not found on PATH; install Docker (or a "
                "compatible OCI runtime — Podman, Apple `container`, "
                "containerd) before using LocalGpuCompute."
            )
        try:
            completed = subprocess.run(
                [binary, "info"],
                capture_output=True,
                text=True,
                timeout=_DOCKER_INFO_TIMEOUT,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ComputeUnavailable(f"docker info preflight failed: {exc!r}") from exc
        if completed.returncode != 0:
            raise ComputeUnavailable(
                "docker daemon is not reachable. Start Docker Desktop or the "
                "docker daemon and retry. "
                f"`docker info` exited {completed.returncode}: "
                f"{completed.stderr.strip()[:300]}"
            )

    async def _preflight_nvidia_smi(self) -> None:
        """``nvidia-smi`` preflight on ``gpu=True`` dispatch.

        Per Task 2.A4: verifies the NVIDIA Container Toolkit is
        installed by running ``nvidia-smi`` on the host. Mac is rejected
        upstream by :meth:`submit`.
        """
        import shutil  # noqa: PLC0415

        if shutil.which("nvidia-smi") is None:
            raise ComputeUnavailable(
                "nvidia-smi not found on PATH. Install NVIDIA drivers + the "
                "NVIDIA Container Toolkit (https://docs.nvidia.com/datacenter/"
                "cloud-native/container-toolkit/) to dispatch GPU jobs locally, "
                "or point Compute at smai-compute-modal / smai-compute-runpod."
            )
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        except TimeoutError as exc:
            proc.kill()
            await proc.wait()
            raise ComputeUnavailable(
                "nvidia-smi timed out; the NVIDIA driver may be wedged"
            ) from exc
        if proc.returncode != 0:
            raise ComputeUnavailable(
                f"nvidia-smi exited {proc.returncode}: "
                f"{stderr.decode('utf-8', errors='replace').strip()[:300]}"
            )

    # --- image validation --------------------------------------------------

    async def _validate_image_eager(self, image: str) -> None:
        """Eager image-existence check per ``07`` §7.4.

        Strategy:

        * ``docker image inspect <image>`` — returns 0 if the image is
          already cached locally; we're done.
        * Otherwise ``docker pull <image>`` — handles
          registry-hosted images (Docker Hub, etc.) including the
          conformance suite's ``python:3.12-slim``. Failure here means
          the image isn't reachable; raise :class:`JobImageInvalid`
          with the SMAI-specific build hint when the missing tag
          matches one of the configured defaults.
        """
        inspect_result = await run_docker(
            "image", "inspect", image, timeout=_DOCKER_INSPECT_TIMEOUT
        )
        if inspect_result.returncode == 0:
            return
        # Cached miss; try pulling. The ``smai-*:dev`` defaults are
        # local-only — a pull will fail; let it fall through so the
        # build-hint surfaces.
        pull_result = await run_docker("pull", image, timeout=_DOCKER_PULL_TIMEOUT)
        if pull_result.returncode == 0:
            return
        raise self._make_image_invalid(image, pull_result.stderr or pull_result.stdout)

    def _make_image_invalid(self, image: str, reason: str) -> JobImageInvalid:
        """Build a :class:`JobImageInvalid` with the SMAI-specific build
        hint when the missing tag is one of the configured defaults."""
        hint = self._build_hint_for(image)
        if hint:
            full_reason = (
                f"{reason.strip()[:300] or 'image not found'}. Build the image with:\n  {hint}"
            )
        else:
            full_reason = (
                f"image not reachable from substrate: {reason.strip()[:300] or 'unknown failure'}"
            )
        return JobImageInvalid(image, full_reason)

    def _build_hint_for(self, image: str) -> str | None:
        """Return the ``docker build`` command for one of the configured
        SMAI default tags, or ``None`` if the tag isn't recognized."""
        if image == self._agent_image:
            return (
                f"docker build -t {self._agent_image} "
                "-f plugins/smai-compute-localgpu/dockerfiles/agent.Dockerfile ."
            )
        if image == self._runtime_image:
            return (
                f"docker build -t {self._runtime_image} "
                "-f plugins/smai-compute-localgpu/dockerfiles/runtime.Dockerfile ."
            )
        if image == self._runtime_cpu_image:
            return (
                f"docker build -t {self._runtime_cpu_image} "
                "-f plugins/smai-compute-localgpu/dockerfiles/runtime-cpu.Dockerfile ."
            )
        return None

    # --- internal helpers --------------------------------------------------

    def _container_id(self, handle: JobHandle) -> str:
        """Read the container id off a :class:`JobHandle`.

        Falls back to ``handle.handle`` when ``metadata['container_id']``
        is absent — round-tripping the handle through Pydantic JSON
        preserves the metadata, but the ``handle`` field is the
        canonical reference.
        """
        meta_id = handle.metadata.get("container_id")
        if isinstance(meta_id, str) and meta_id:
            return meta_id
        return handle.handle

    async def _inspect(self, container_id: str) -> dict[str, Any] | None:
        """Run ``docker inspect`` and return the parsed JSON dict, or
        ``None`` when the container is not found."""
        result = await run_docker("inspect", container_id, timeout=_DOCKER_INSPECT_TIMEOUT)
        if result.returncode != 0:
            if _is_no_such_container(result.stderr) or _is_no_such_object(result.stderr):
                return None
            raise ComputeUnavailable(
                f"docker inspect failed (exit {result.returncode}): {result.stderr.strip()}"
            )
        try:
            payload: object = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ComputeUnavailable(f"docker inspect returned malformed JSON: {exc}") from exc
        if not isinstance(payload, list) or not payload:
            return None
        first = cast(object, payload[0])
        if not isinstance(first, dict):
            return None
        return cast(dict[str, Any], first)

    async def _kill(self, container_id: str) -> None:
        """SIGKILL the container, ignoring "no such container" errors."""
        result = await run_docker("kill", container_id, timeout=_DOCKER_STOP_TIMEOUT)
        # ``docker kill`` of a stopped container exits non-zero; we
        # don't care since the goal is "container is not running" and
        # that's already true.
        if result.returncode != 0 and not (
            _is_no_such_container(result.stderr) or _is_not_running(result.stderr)
        ):
            # Soft-fail: log via stderr but don't raise. The next
            # ``status`` call will surface whatever the substrate's
            # state is.
            pass


# === Module-level helpers ===================================================


def _utc_now_iso() -> str:
    """ISO 8601 timestamp in UTC, always with explicit ``Z`` suffix."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_timestamp(raw: str | None) -> str | None:
    """Convert Docker's "0001-01-01T..." sentinel to ``None``."""
    if not raw or raw.startswith("0001-01-01"):
        return None
    return raw


def _elapsed_seconds(submitted_at_iso: str) -> float:
    """Return wall-clock seconds since ``submitted_at_iso``.

    Tolerant of trailing ``Z`` and microsecond components — uses
    :func:`datetime.fromisoformat` after stripping the ``Z`` (Python
    3.11's ``fromisoformat`` accepts ``Z`` in 3.12+ but not 3.11).
    """
    iso = submitted_at_iso.replace("Z", "+00:00")
    submitted = datetime.fromisoformat(iso)
    now = datetime.now(UTC)
    if submitted.tzinfo is None:
        submitted = submitted.replace(tzinfo=UTC)
    return (now - submitted).total_seconds()


def _is_no_such_container(stderr: str) -> bool:
    return "No such container" in stderr or "no such container" in stderr.lower()


def _is_no_such_object(stderr: str) -> bool:
    return "No such object" in stderr or "no such object" in stderr.lower()


def _is_not_running(stderr: str) -> bool:
    return "is not running" in stderr.lower()


__all__ = [
    "DEFAULT_AGENT_IMAGE",
    "DEFAULT_RUNTIME_CPU_IMAGE",
    "DEFAULT_RUNTIME_IMAGE",
    "LocalGpuCompute",
]
