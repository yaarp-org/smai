"""In-process RunPod-API fake backing the plugin's mocked-HTTP test lane.

Wired into :class:`smai_compute_runpod.RunPodCompute` via
:class:`httpx.MockTransport` so the conformance suite and the unit
tests run offline without ever leaving the process. Mirrors the shape
of :mod:`smai_llm_bedrock.tests._fakes` (the per-plugin "VCR-or-
equivalent" pattern from the Phase-2 plugin tasks).

The fake understands the small set of shell commands the
:class:`smai_core.plugins.conformance.ComputeConformance` suite passes
in (``python -c "import sys; sys.exit(N)"`` /
``python -c "import time; time.sleep(N)"`` /
``python -c "print('token')"``) and produces deterministic pod
lifecycle traces so the conformance assertions hold without spinning
up real GPU pods. Real-substrate validation lives in
:mod:`tests.test_real_runpod`.
"""

from __future__ import annotations

import json
import re
import shlex
import time
import uuid
from typing import Any

import httpx

# === Pattern recognition ====================================================


# Patterns the fake recognizes in the submitted command's ``-c`` body
# (after :func:`shlex.split` undoes the plugin's shell-quoting). We
# pull these out at module level so the conformance suite's invariants
# are visible in one place.
_RE_SYS_EXIT = re.compile(r"sys\.exit\((-?\d+)\)")
_RE_TIME_SLEEP = re.compile(r"time\.sleep\((\d+)\)")
_RE_PRINT = re.compile(r"print\(\s*[\"']([^\"']*)[\"']")


def _extract_python_c_body(docker_args: str) -> str:
    """Return the body of ``python -c "<...>"`` from a shell-joined command.

    The plugin renders ``submit``'s ``list[str] command`` to a single
    shell string via :func:`smai_compute_runpod._compute._shell_join`;
    the fake undoes that here so its regex inspectors can see the
    pre-shell-escape Python source the conformance suite passed.
    Falls back to the original string if shell-parsing fails (e.g.,
    the test harness sends a non-``python -c`` shape).
    """
    try:
        tokens = shlex.split(docker_args)
    except ValueError:
        return docker_args
    if "-c" in tokens:
        idx = tokens.index("-c")
        if idx + 1 < len(tokens):
            return tokens[idx + 1]
    return docker_args


# === Fake pod ===============================================================


class _FakePod:
    """In-process simulation of one RunPod pod.

    Each pod is "started" at construction; its terminal state is
    determined by parsing the submitted command:

    * ``sys.exit(N)`` → on the first ``status`` read, jump to
      ``EXITED`` with ``exitCode=N``.
    * ``time.sleep(N)`` → stay ``RUNNING`` until ``DELETE`` arrives.
    * ``print('token')`` (with no other lifecycle pattern) → behaves
      like ``sys.exit(0)`` and the token surfaces in ``logs``.
    """

    pod_id: str
    image: str
    command: str
    desired_status: str
    exit_code: int | None
    logs: str
    cancelled: bool
    started_at: float

    def __init__(self, pod_id: str, image: str, command: str) -> None:
        self.pod_id = pod_id
        self.image = image
        self.command = command
        # Extract the ``-c`` body once at construction; the inspectors
        # below run against the unquoted Python source the conformance
        # suite originally passed (rather than the shell-escaped form
        # the plugin sends to RunPod's ``dockerArgs``).
        self._command_body = _extract_python_c_body(command)
        # The fake skips the IN_QUEUE / INITIALIZING phase — every pod
        # boots straight to RUNNING. The plugin's status() handles both
        # forms; the conformance suite never inspects the boot phase.
        self.desired_status = "RUNNING"
        self.exit_code = None
        self.logs = ""
        self.cancelled = False
        self.started_at = time.monotonic()
        # Capture print() output eagerly; it's available even before
        # the pod transitions to EXITED. The tests' commands are simple
        # (single ``print(...)``) so a raw regex extract is enough.
        match = _RE_PRINT.search(self._command_body)
        if match:
            self.logs = match.group(1) + "\n"

    def step(self) -> None:
        """Advance the pod's lifecycle on each ``status`` poll.

        ``sys.exit(N)`` jobs jump to EXITED on the first poll;
        ``time.sleep(...)`` jobs stay RUNNING until DELETE.
        """
        if self.desired_status not in ("RUNNING",):
            return
        # Long-running detection: any ``time.sleep`` keeps the pod alive.
        if _RE_TIME_SLEEP.search(self._command_body):
            return
        # Otherwise the pod has "finished" by the next poll.
        match = _RE_SYS_EXIT.search(self._command_body)
        if match is not None:
            self.exit_code = int(match.group(1))
        else:
            self.exit_code = 0
        self.desired_status = "EXITED"

    def terminate(self) -> None:
        """Apply the substrate's response to a DELETE request."""
        # ``cancelled`` is what the plugin maps to ``state='cancelled'``
        # via the ``cancel_requested`` metadata; the substrate itself
        # has no notion of "user-cancelled vs operator-killed."
        self.cancelled = True
        self.desired_status = "TERMINATED"


# === Fake backend ===========================================================


class FakeRunPodBackend:
    """In-memory simulator of the RunPod REST API.

    Wired into the plugin via :class:`httpx.MockTransport`. Tests share
    one backend across ``make_compute()`` and ``make_fresh_compute()``
    so the cross-instance reconnection contract holds (per
    ``07-plugin-interfaces.md`` §7.5: a fresh :class:`Compute` instance
    polling a serialized handle observes the same substrate state).
    """

    pods: dict[str, _FakePod]

    def __init__(self) -> None:
        self.pods = {}

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        method = request.method
        path = request.url.path
        # POST /pods — create
        if method == "POST" and path == "/v1/pods":
            return self._create_pod(request)
        # GET / DELETE /pods/{id}
        match = re.fullmatch(r"/v1/pods/([^/]+)", path)
        if match and method == "GET":
            return self._get_pod(match.group(1))
        if match and method == "DELETE":
            return self._delete_pod(match.group(1))
        # GET /pods/{id}/logs
        match = re.fullmatch(r"/v1/pods/([^/]+)/logs", path)
        if match and method == "GET":
            return self._get_logs(match.group(1))
        return httpx.Response(404, json={"error": f"unhandled {method} {path}"})

    def _create_pod(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        image = body.get("imageName", "")
        command = body.get("dockerArgs", "")
        pod_id = f"pod-{uuid.uuid4().hex[:12]}"
        pod = _FakePod(pod_id=pod_id, image=image, command=command)
        self.pods[pod_id] = pod
        return httpx.Response(
            200,
            json={
                "id": pod_id,
                "name": body.get("name", ""),
                "imageName": image,
                "desiredStatus": pod.desired_status,
            },
        )

    def _get_pod(self, pod_id: str) -> httpx.Response:
        pod = self.pods.get(pod_id)
        if pod is None:
            return httpx.Response(404, json={"error": "pod not found"})
        # Each ``status`` poll advances the pod by one step.
        pod.step()
        runtime: dict[str, Any] = {}
        if pod.desired_status == "EXITED":
            runtime["exitCode"] = pod.exit_code
        return httpx.Response(
            200,
            json={
                "id": pod.pod_id,
                "desiredStatus": pod.desired_status,
                "runtime": runtime,
                "startedAt": "2026-04-29T00:00:00Z",
                "finishedAt": (
                    "2026-04-29T00:00:01Z"
                    if pod.desired_status in ("EXITED", "TERMINATED")
                    else None
                ),
            },
        )

    def _delete_pod(self, pod_id: str) -> httpx.Response:
        pod = self.pods.get(pod_id)
        if pod is None:
            return httpx.Response(404, json={"error": "pod not found"})
        pod.terminate()
        return httpx.Response(200, json={"id": pod_id, "status": "TERMINATED"})

    def _get_logs(self, pod_id: str) -> httpx.Response:
        pod = self.pods.get(pod_id)
        if pod is None:
            return httpx.Response(404, json={"error": "pod not found"})
        return httpx.Response(
            200,
            text=pod.logs,
            headers={"content-type": "text/plain"},
        )


__all__ = ["FakeRunPodBackend"]
