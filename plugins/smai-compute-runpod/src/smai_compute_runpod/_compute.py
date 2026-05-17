""":class:`RunPodCompute` — :class:`Compute` adapter for RunPod's REST API.

Per ``07-plugin-interfaces.md`` §7 (the full Protocol surface) and the
Task 3.F4 brief in ``designs/smai/implementation_plan.md`` §3.4.

**Substrate API.** RunPod exposes two distinct execution surfaces:

* **Pods** — long-lived containerized GPU instances that accept arbitrary
  images + commands at create time. Lifecycle: create → run → terminate.
* **Serverless endpoints** — pre-deployed worker pools that accept job
  payloads via ``/run`` and return results from ``/status``.

This plugin targets the **Pods API**. Reasoning:

* The Compute Protocol's contract is "submit an image + command, get a
  handle back" — the Pods API maps onto that natively (no
  pre-deployment of a serverless worker is needed).
* Serverless requires the operator to build + deploy a worker per
  workload shape, which moves substantial work from the Compute layer
  back to the operator. SMAI's "operator chooses an image and a
  command per dispatch" is materially closer to Pods semantics.
* See the README's "Choosing Pods vs Serverless" section for
  operators who want the serverless path.

**HTTP layer choice.** Raw ``httpx.AsyncClient`` rather than the
``runpod`` Python SDK (same shape as ``smai-compute-localgpu`` shelling
out to ``docker`` rather than depending on the Python Docker SDK):

* Smaller install footprint (only ``httpx`` on top of ``smai-core``).
* No SDK version-pin matrix; ``httpx.MockTransport`` is a first-class
  shape for the conformance suite.
* The plugin's HTTP surface is small (4 endpoints) and decoupling from
  the SDK lets us absorb RunPod API changes without an SDK upgrade
  cycle.

**Authentication.** Reads ``RUNPOD_API_KEY`` from the environment by
default (same hygiene as ``smai-llm-bedrock`` / ``smai-artifacts-s3`` —
no credentials in shell history). The constructor's ``api_key=`` kwarg
is a test seam.

**State mapping** (RunPod → SMAI :class:`JobState`)::

    IN_QUEUE                 → submitted
    INITIALIZING / STARTING  → submitted
    IN_PROGRESS / RUNNING    → running
    COMPLETED / EXITED:
        exitCode == 0        → succeeded
        exitCode != 0        → failed
        not user-cancelled   → succeeded / failed by exitCode
        user-cancelled       → cancelled
    TERMINATED / STOPPED:
        user-cancelled       → cancelled
        otherwise            → failed
    TIMED_OUT                → timeout
    FAILED                   → failed

Mapping ambiguity: RunPod's ``EXITED`` status alone doesn't distinguish
"job ran to completion (exit 0/N)" from "operator stopped the pod
mid-run." The plugin marks the JobHandle metadata when ``cancel()`` is
called and uses that flag to translate ``EXITED`` correctly — same
shape as :class:`LocalGpuCompute`'s ``cancel_requested`` metadata.

**Image validation.** RunPod has no pre-pull validation surface
analogous to ``docker pull`` — pod creation accepts any image string
and surfaces image failures only after the pod tries to pull. The
conformance test ``test_invalid_image_raises`` therefore takes the
"plugin defers image validation to substrate" branch and skips. A
future plugin task could add an opt-in eager check by calling Docker
Registry's ``HEAD /v2/<name>/manifests/<tag>``; per Task 3.F4's "out
of scope" list this is deferred.

**Not in scope** for this plugin:

* Multi-region routing — RunPod auto-selects from the configured GPU
  type's available data centers.
* Spot vs on-demand pricing logic — operators pass ``cloud_type=`` via
  ``plugin_options`` if needed.
* Serverless endpoints (see "Substrate API" above).
* Network volumes / persistent storage — per-job stateless.
* Cost accounting (per ``07`` §7.4).

The conformance suite runs against an ``httpx.MockTransport`` that
simulates a small RunPod backend; the credentialed real-RunPod test
(``tests/test_real_runpod.py``) round-trips against the live substrate
when ``RUNPOD_API_KEY`` is set.
"""

from __future__ import annotations

import os
import re
import uuid
from datetime import UTC, datetime
from typing import Any, cast

import httpx
from smai_core.plugins import (
    ComputeCapabilities,
    ComputeUnavailable,
    JobHandle,
    JobNotFound,
    JobState,
    JobStatus,
)

# === Constants ==============================================================

# RunPod's REST API base. The ``rest.runpod.io/v1`` host is the modern
# REST surface; the legacy GraphQL surface is at ``api.runpod.io/graphql``
# and is not used here.
DEFAULT_API_BASE = "https://rest.runpod.io/v1"

# Environment variable carrying the RunPod API key. Same hygiene as
# bedrock / s3 — no constructor-arg surface for the secret.
_API_KEY_ENV = "RUNPOD_API_KEY"

# Default GPU type id used when neither the constructor nor the per-call
# plugin_options pin one. The conformance suite passes through this
# default; operators override per-deployment.
DEFAULT_GPU_TYPE_ID = "NVIDIA RTX A4000"

# Substrate-imposed maximum timeout for a job. RunPod doesn't publish a
# hard cap, but 24h matches Modal Sandbox + LocalGpu so the conformance
# capability surface is consistent across plugins.
_MAX_TIMEOUT_SECONDS = 24 * 60 * 60  # 24h

# Default per-call HTTP timeout. RunPod's API is generally fast (sub-
# second for status / cancel), but pod creation includes the API's own
# scheduling decision and can take a few seconds when GPU types are
# tight on capacity.
_DEFAULT_HTTP_TIMEOUT_SECONDS = 30.0

# Default container disk and pod naming prefix. Container disk default
# is 10 GiB — small enough to be cheap, big enough for a tiny Python
# image plus typical caches.
_DEFAULT_CONTAINER_DISK_GB = 10
_POD_NAME_PREFIX = "smai-runpod"


# === GPU dispatch table =====================================================

# Maps SMAI's generic GPU specs to RunPod's GPU type ids. See README.md
# for the full catalog and the rationale for each tier.
#
# The keys are SMAI-side spec strings (operator-facing); the values are
# RunPod-side GPU type ids (substrate-facing). Operators who need a
# RunPod GPU id not on this table pass it directly via
# ``plugin_options["gpu_type"]``.
GPU_DISPATCH: dict[str, str] = {
    "default": "NVIDIA RTX A4000",
    "small": "NVIDIA RTX A4000",
    "medium": "NVIDIA RTX A5000",
    "large": "NVIDIA RTX A6000",
    "a100": "NVIDIA A100 80GB PCIe",
    "h100": "NVIDIA H100 80GB HBM3",
}


# === RunPod state mapping ===================================================

# RunPod pod statuses that mean "still in the pre-running queue / boot
# phase." Mapped to SMAI ``submitted``.
_RUNPOD_STATES_SUBMITTED: frozenset[str] = frozenset(
    {"IN_QUEUE", "INITIALIZING", "STARTING", "PENDING"}
)

# RunPod pod statuses that mean "actively executing." Mapped to SMAI
# ``running``.
_RUNPOD_STATES_RUNNING: frozenset[str] = frozenset({"RUNNING", "IN_PROGRESS"})

# RunPod pod statuses that mean "the substrate killed the pod from the
# outside" (terminal). The ``cancel_requested`` flag on the handle
# metadata distinguishes user-cancelled from substrate-killed.
_RUNPOD_STATES_TERMINATED: frozenset[str] = frozenset({"TERMINATED", "STOPPED", "CANCELLED"})

# RunPod pod status that means "the substrate enforced its own timeout."
# Mapped to SMAI ``timeout``. (We also enforce caller-side timeout via
# elapsed-wall-clock checks; see :meth:`status`.)
_RUNPOD_STATE_TIMED_OUT = "TIMED_OUT"

# RunPod pod statuses that mean "the pod ran and finished." The exit
# code on the runtime-info subdocument refines this into succeeded /
# failed; user cancellation can promote it to ``cancelled``.
_RUNPOD_STATES_COMPLETED: frozenset[str] = frozenset({"COMPLETED", "EXITED"})

# RunPod pod status that's a hard failure outside the
# completed-with-exit-code path.
_RUNPOD_STATE_FAILED = "FAILED"


# === Implementation =========================================================


class RunPodCompute:
    """RunPod REST-API implementation of :class:`Compute`.

    Constructor::

        RunPodCompute()                                # uses RUNPOD_API_KEY
        RunPodCompute(api_key="rpa_...")               # explicit key
        RunPodCompute(default_gpu_type="NVIDIA H100 80GB HBM3")
        RunPodCompute(client=mock_client)              # test seam

    The constructor is sync; it does NOT make a network round-trip on
    construction (RunPod has no equivalent of ``docker info`` to run as
    a preflight). Auth failures surface on the first ``submit`` /
    ``status`` / ``cancel`` / ``logs`` call.
    """

    name: str = "runpod"

    capabilities: ComputeCapabilities

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str = DEFAULT_API_BASE,
        default_gpu_type: str = DEFAULT_GPU_TYPE_ID,
        default_timeout_seconds: int = 3600,
        max_timeout_seconds: int = _MAX_TIMEOUT_SECONDS,
        default_container_disk_gb: int = _DEFAULT_CONTAINER_DISK_GB,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved_key = api_key if api_key is not None else os.environ.get(_API_KEY_ENV)
        if not resolved_key:
            raise ComputeUnavailable(
                f"RunPod API key not provided; set ${_API_KEY_ENV} in the environment "
                "or pass api_key=... to the constructor."
            )
        self._api_key = resolved_key
        self._api_base = api_base.rstrip("/")
        self._default_gpu_type = default_gpu_type
        self._default_timeout_seconds = int(default_timeout_seconds)
        self._default_container_disk_gb = int(default_container_disk_gb)
        self._client: httpx.AsyncClient
        self._owns_client: bool
        if client is None:
            self._client = httpx.AsyncClient(timeout=_DEFAULT_HTTP_TIMEOUT_SECONDS)
            self._owns_client = True
        else:
            self._client = client
            self._owns_client = False
        # RunPod's published GPU catalog is very wide (many tiers); v1
        # advertises ``supports_gpu=True`` always — operators who only
        # use ``gpu=False`` for the agent side pay nothing extra.
        self.capabilities = ComputeCapabilities(
            supports_gpu=True,
            max_timeout_seconds=int(max_timeout_seconds),
            supports_log_streaming=False,
            # RunPod pulls the job image from a registry — the operator
            # must publish it where RunPod can reach it.
            requires_published_image=True,
        )

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
        """Create a RunPod pod with ``image`` + ``command`` + ``env``.

        ``plugin_options`` accepts:

        * ``gpu_type: str`` — RunPod GPU type id (e.g.
          ``"NVIDIA H100 80GB HBM3"``). When unspecified, looks up
          ``default_gpu_type`` (constructor) or
          :data:`GPU_DISPATCH`[``"default"``].
        * ``gpu_count: int`` — number of GPUs in the pod (default 1).
        * ``container_disk_gb: int`` — pod root-disk size (default 10).
        * ``cloud_type: str`` — ``"SECURE"`` (default) or ``"COMMUNITY"``.

        Image validation is deferred to the substrate per §7.4's
        explicit allowance for plugins whose substrate has no eager
        image-existence surface; the conformance suite's
        ``test_invalid_image_raises`` falls through to its skip path.
        """
        gpu_type = self._resolve_gpu_type(plugin_options)
        gpu_count = self._int_option(plugin_options, "gpu_count", default=1)
        container_disk_gb = self._int_option(
            plugin_options,
            "container_disk_gb",
            default=self._default_container_disk_gb,
        )
        cloud_type = self._str_option(plugin_options, "cloud_type", default="SECURE")

        # ``dockerArgs`` is RunPod's name for the container-entrypoint
        # arg vector; it accepts a shell string. The Compute Protocol
        # gives us a ``list[str]`` per shell-list semantics, so we
        # render it back into a single shell-escaped string.
        docker_args = _shell_join(command)

        body: dict[str, Any] = {
            "name": f"{_POD_NAME_PREFIX}-{uuid.uuid4().hex[:12]}",
            "imageName": image,
            "containerDiskInGb": container_disk_gb,
            "dockerArgs": docker_args,
            "env": dict(env),
            "cloudType": cloud_type,
        }
        # The pod-creation API's GPU surface differs by request shape:
        # pods always run on GPU hardware on RunPod (there's no
        # CPU-only pod tier in v1's catalog), so ``gpu=False`` from the
        # caller maps onto the smallest configured GPU type rather
        # than no-GPU. Mac local agent dispatches that don't need GPU
        # generally route to ``smai-compute-localgpu`` instead.
        body["gpuTypeIds"] = [gpu_type]
        body["gpuCount"] = gpu_count if gpu else 1

        url = f"{self._api_base}/pods"
        response = await self._post(url, body)
        if response.status_code >= 400:
            raise self._classify_http_error(response, action="create pod")
        payload = self._parse_json(response, action="create pod")
        pod_id = payload.get("id")
        if not isinstance(pod_id, str) or not pod_id:
            raise ComputeUnavailable(f"RunPod create-pod response missing 'id' field: {payload!r}")

        submitted_at = _utc_now_iso()
        metadata: dict[str, Any] = {
            "pod_id": pod_id,
            "submitted_at": submitted_at,
            "timeout_seconds": int(timeout_seconds),
            "image": image,
            "gpu": bool(gpu),
            "gpu_type": gpu_type,
            "gpu_count": int(gpu_count),
        }
        return JobHandle(plugin=self.name, handle=pod_id, metadata=metadata)

    async def status(self, handle: JobHandle) -> JobStatus:
        """Poll the RunPod pod's status.

        Per ``07`` §7.2: MUST raise :class:`JobNotFound` on unknown
        handles; MUST NOT block. Caller-side timeout enforcement: when
        the wall-clock elapsed since ``submitted_at`` exceeds
        ``timeout_seconds``, the pod is terminated and the status
        returns ``state='timeout'``.
        """
        pod_id = self._pod_id(handle)
        url = f"{self._api_base}/pods/{pod_id}"
        response = await self._get(url)
        if response.status_code == 404:
            raise JobNotFound(handle)
        if response.status_code >= 400:
            raise self._classify_http_error(response, action="get pod status")
        payload = self._parse_json(response, action="get pod status")

        runpod_status_raw = payload.get("desiredStatus") or payload.get("status") or ""
        runpod_status = str(runpod_status_raw).upper()
        runtime_info = payload.get("runtime") or {}
        if not isinstance(runtime_info, dict):
            runtime_info = {}
        exit_code_raw = runtime_info.get("exitCode") if runtime_info else payload.get("exitCode")
        exit_code: int | None = int(exit_code_raw) if isinstance(exit_code_raw, int) else None

        started_at = _coerce_iso(payload.get("startedAt"))
        finished_at = _coerce_iso(payload.get("finishedAt") or payload.get("stoppedAt"))

        # Caller-side timeout enforcement: once elapsed > timeout_seconds
        # and the pod is still in a non-terminal state, terminate it
        # and report ``state='timeout'``.
        timeout_seconds = int(handle.metadata.get("timeout_seconds", 3600))
        submitted_at_raw = handle.metadata.get("submitted_at")
        if (
            runpod_status in _RUNPOD_STATES_RUNNING or runpod_status in _RUNPOD_STATES_SUBMITTED
        ) and isinstance(submitted_at_raw, str):
            if _elapsed_seconds(submitted_at_raw) > timeout_seconds:
                await self._terminate_pod(pod_id)
                return JobStatus(
                    state="timeout",
                    exit_code=None,
                    started_at=started_at,
                    finished_at=_utc_now_iso(),
                    failure_reason=(
                        f"job exceeded timeout_seconds={timeout_seconds}; pod terminated by plugin"
                    ),
                )

        cancel_requested = bool(handle.metadata.get("cancel_requested"))

        job_state, failure_reason = self._translate_runpod_status(
            runpod_status=runpod_status,
            exit_code=exit_code,
            cancel_requested=cancel_requested,
            raw_payload=payload,
        )

        return JobStatus(
            state=job_state,
            exit_code=exit_code,
            started_at=started_at,
            finished_at=finished_at,
            failure_reason=failure_reason,
        )

    async def logs(self, handle: JobHandle) -> str:
        """Return the pod's stdout+stderr concatenated.

        RunPod exposes pod logs via ``GET /pods/{id}/logs``; the
        response body is the raw log text.
        """
        pod_id = self._pod_id(handle)
        url = f"{self._api_base}/pods/{pod_id}/logs"
        response = await self._get(url)
        if response.status_code == 404:
            raise JobNotFound(handle)
        if response.status_code >= 400:
            raise self._classify_http_error(response, action="get pod logs")
        # Logs endpoint returns either a raw text body or a JSON wrapper
        # depending on the API generation; accept both shapes.
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            payload = self._parse_json(response, action="get pod logs")
            logs_field = payload.get("logs")
            if isinstance(logs_field, str):
                return logs_field
            if isinstance(logs_field, list):
                return "".join(str(line) for line in logs_field)  # pyright: ignore[reportUnknownVariableType]
            return ""
        return response.text

    async def cancel(self, handle: JobHandle) -> None:
        """Terminate the RunPod pod.

        Eventually-consistent per ``07`` §7.2: the next ``status`` call
        translates the pod's eventual ``TERMINATED`` / ``STOPPED``
        state into ``state='cancelled'`` (rather than ``failed``)
        because we mark the handle metadata before issuing the DELETE.
        Idempotent — a 404 from RunPod's terminate endpoint means the
        pod is already gone.
        """
        pod_id = self._pod_id(handle)
        # Mark intent BEFORE the DELETE round-trip — same shape as
        # LocalGpu's ``cancel_requested`` metadata. The flag survives
        # cross-process reconnection because it's persisted on the
        # JobHandle the caller already round-trips.
        try:
            handle.metadata["cancel_requested"] = True
        except (TypeError, AttributeError):  # pragma: no cover — defensive
            pass

        await self._terminate_pod(pod_id)

    # --- internal helpers --------------------------------------------------

    async def _terminate_pod(self, pod_id: str) -> None:
        """DELETE the pod, treating 404 as already-gone (idempotent)."""
        url = f"{self._api_base}/pods/{pod_id}"
        response = await self._delete(url)
        if response.status_code == 404:
            return
        if response.status_code >= 400:
            raise self._classify_http_error(response, action="terminate pod")

    def _pod_id(self, handle: JobHandle) -> str:
        """Read the RunPod pod id off a :class:`JobHandle`.

        Falls back to ``handle.handle`` when ``metadata['pod_id']`` is
        absent — round-tripping the handle through Pydantic JSON
        preserves both, but ``handle`` is the canonical reference.
        """
        meta_id = handle.metadata.get("pod_id")
        if isinstance(meta_id, str) and meta_id:
            return meta_id
        return handle.handle

    def _resolve_gpu_type(self, plugin_options: dict[str, object]) -> str:
        """Pick the RunPod GPU type id for a submission.

        Resolution order:

        1. ``plugin_options['gpu_type']`` — explicit per-call override.
        2. The constructor's ``default_gpu_type``.
        """
        explicit = plugin_options.get("gpu_type")
        if isinstance(explicit, str) and explicit:
            return explicit
        if explicit is not None:
            raise ValueError(
                f"plugin_options['gpu_type'] must be a non-empty str, got {type(explicit).__name__}"
            )
        return self._default_gpu_type

    @staticmethod
    def _int_option(plugin_options: dict[str, object], key: str, *, default: int) -> int:
        value = plugin_options.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"plugin_options[{key!r}] must be an int, got {type(value).__name__}")
        return value

    @staticmethod
    def _str_option(plugin_options: dict[str, object], key: str, *, default: str) -> str:
        value = plugin_options.get(key, default)
        if not isinstance(value, str):
            raise ValueError(f"plugin_options[{key!r}] must be a str, got {type(value).__name__}")
        return value

    def _translate_runpod_status(
        self,
        *,
        runpod_status: str,
        exit_code: int | None,
        cancel_requested: bool,
        raw_payload: dict[str, Any],
    ) -> tuple[JobState, str | None]:
        """Map a RunPod pod-status string + exit_code → SMAI :class:`JobState`."""
        if runpod_status in _RUNPOD_STATES_SUBMITTED:
            return "submitted", None
        if runpod_status in _RUNPOD_STATES_RUNNING:
            return "running", None
        if runpod_status == _RUNPOD_STATE_TIMED_OUT:
            return "timeout", "RunPod substrate reported TIMED_OUT"
        if runpod_status in _RUNPOD_STATES_COMPLETED:
            if cancel_requested:
                return "cancelled", None
            if exit_code is None:
                # No exit code reported — accept the substrate's framing
                # ("the pod completed") as success.
                return "succeeded", None
            if exit_code == 0:
                return "succeeded", None
            return "failed", f"pod exited with non-zero status {exit_code}"
        if runpod_status in _RUNPOD_STATES_TERMINATED:
            if cancel_requested:
                return "cancelled", None
            return (
                "failed",
                f"pod terminated by substrate (status={runpod_status!r})",
            )
        if runpod_status == _RUNPOD_STATE_FAILED:
            failure_message = raw_payload.get("lastStatusChange") or "RunPod reported FAILED"
            return "failed", str(failure_message)
        # Unknown status — fail closed rather than silently accepting.
        return "failed", f"unknown RunPod status: {runpod_status!r}"

    # --- HTTP shims --------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def _post(self, url: str, body: dict[str, Any]) -> httpx.Response:
        try:
            return await self._client.post(url, json=body, headers=self._auth_headers())
        except httpx.HTTPError as exc:
            raise ComputeUnavailable(f"RunPod API POST {url} failed: {exc!r}") from exc

    async def _get(self, url: str) -> httpx.Response:
        try:
            return await self._client.get(url, headers=self._auth_headers())
        except httpx.HTTPError as exc:
            raise ComputeUnavailable(f"RunPod API GET {url} failed: {exc!r}") from exc

    async def _delete(self, url: str) -> httpx.Response:
        try:
            return await self._client.delete(url, headers=self._auth_headers())
        except httpx.HTTPError as exc:
            raise ComputeUnavailable(f"RunPod API DELETE {url} failed: {exc!r}") from exc

    def _classify_http_error(self, response: httpx.Response, *, action: str) -> ComputeUnavailable:
        """Build a :class:`ComputeUnavailable` from a non-2xx RunPod response.

        Auth failures (401/403) surface with the same exception class
        because the contract per §7.3 only ships a single
        substrate-outage error type — operators read the message to
        diagnose. The conformance suite never hits this path.
        """
        status_code = response.status_code
        try:
            body = response.text[:500]
        except Exception:  # pragma: no cover — defensive
            body = "<unavailable>"
        return ComputeUnavailable(f"RunPod API {action} returned {status_code}: {body}")

    def _parse_json(self, response: httpx.Response, *, action: str) -> dict[str, Any]:
        try:
            payload = response.json()
        except Exception as exc:
            raise ComputeUnavailable(
                f"RunPod API {action} returned non-JSON body: {exc!r}"
            ) from exc
        if not isinstance(payload, dict):
            raise ComputeUnavailable(
                f"RunPod API {action} returned non-object JSON: {type(payload).__name__}"
            )
        return cast("dict[str, Any]", payload)

    # --- async-context lifecycle ------------------------------------------

    async def aclose(self) -> None:
        """Close the underlying ``httpx.AsyncClient`` if we own it.

        Idempotent — calling twice is safe.
        """
        if self._owns_client:
            await self._client.aclose()


# === Module-level helpers ===================================================


def _utc_now_iso() -> str:
    """ISO 8601 timestamp in UTC, always with explicit ``Z`` suffix."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _coerce_iso(raw: object) -> str | None:
    """Return a string ISO timestamp or ``None`` for missing / sentinel values."""
    if not isinstance(raw, str) or not raw:
        return None
    if raw.startswith("0001-01-01") or raw == "null":
        return None
    return raw


def _elapsed_seconds(submitted_at_iso: str) -> float:
    """Return wall-clock seconds since ``submitted_at_iso``.

    Tolerant of trailing ``Z`` and the ``+00:00`` offset spelling —
    Python 3.11's :func:`datetime.fromisoformat` rejects ``Z`` directly.
    """
    iso = submitted_at_iso.replace("Z", "+00:00")
    try:
        submitted = datetime.fromisoformat(iso)
    except ValueError:
        return 0.0
    if submitted.tzinfo is None:
        submitted = submitted.replace(tzinfo=UTC)
    return (datetime.now(UTC) - submitted).total_seconds()


_SHELL_SAFE_CHARS = re.compile(r"[\w@%+=:,./-]+\Z")


def _shell_join(args: list[str]) -> str:
    """Render a ``list[str]`` command into a single shell-safe string.

    RunPod's ``dockerArgs`` field accepts a shell string rather than an
    argv array. We single-quote each argument when it contains
    characters that aren't shell-safe; safe args go through unquoted.
    """
    parts: list[str] = []
    for arg in args:
        if _SHELL_SAFE_CHARS.match(arg):
            parts.append(arg)
        else:
            # Single-quote and escape embedded single-quotes per POSIX
            # ``'\''`` idiom.
            escaped = arg.replace("'", "'\\''")
            parts.append(f"'{escaped}'")
    return " ".join(parts)


__all__ = [
    "DEFAULT_API_BASE",
    "DEFAULT_GPU_TYPE_ID",
    "GPU_DISPATCH",
    "RunPodCompute",
]
