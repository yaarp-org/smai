""":class:`Compute` Protocol — submit a containerized job, poll status,
retrieve logs, cancel.

Per ``designs/smai/07-plugin-interfaces.md`` §7 and DEC-021 / DEC-028.

The single seam through which the orchestrator dispatches agent jobs
(LLM-driven, CPU work) and training/evaluation runs (GPU-driven). v1's
two concrete implementations under this conceptual interface — AWS
Batch (per ``compute_monitoring.md``) and Modal Sandboxes (per DEC-021
/ ``modal_migration.md``) — proved the abstraction is sound; v2 turns
the conceptual interface into a real Protocol.

Per §7.4, the Protocol does NOT abstract container build, trust
boundaries / credentials surface, spot vs on-demand or GPU-type
selection, or cost accounting. Plugins surface only job lifecycle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

# === Job types (§7.2) ========================================================


class JobHandle(BaseModel):
    """Opaque identifier for a submitted job (§7.2).

    Plugins encode whatever they need to reconnect to the job (Batch
    ARN, Modal sandbox ID, RunPod pod ID) in this object's fields.
    Consumers persist it via the :class:`MetadataStore` and pass it
    back to ``.status`` / ``.logs`` / ``.cancel`` later.
    """

    model_config = ConfigDict(extra="forbid")

    plugin: str
    handle: str
    metadata: dict[str, Any] = Field(default_factory=dict)


JobState = Literal[
    "submitted",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "timeout",
]


class JobStatus(BaseModel):
    """Current status of a submitted job (§7.2).

    ``failure_reason`` is a plugin-supplied diagnostic string; only
    populated when ``state`` is ``failed`` / ``cancelled`` / ``timeout``
    (plugin discretion).
    """

    model_config = ConfigDict(extra="forbid")

    state: JobState
    exit_code: int | None
    started_at: str | None
    finished_at: str | None
    failure_reason: str | None


# === Capability flags (§7.2) =================================================


class ComputeCapabilities(BaseModel):
    """Static per-plugin capability flags (§7.2).

    ``supports_gpu``:
        ``True`` for Modal/RunPod/Batch GPU; ``False`` for ``LocalGpu``
        when no GPU detected.

    ``max_timeout_seconds``:
        Substrate-imposed max (e.g., Modal Sandbox 24h cap).

    ``supports_log_streaming``:
        Optional: future addition for ``tail -f`` style. v1 plugins
        all return ``False``.

    ``requires_published_image``:
        ``True`` for registry-pull substrates (Modal / RunPod) — the
        ``image`` argument to :meth:`Compute.submit` MUST be a tag the
        substrate can pull from a registry it can reach, so the
        operator owns publishing it. ``False`` for local-build
        substrates (``LocalGpu``) that build the image on the host.
        Pre-flight checks (``smai verify``, the worker boot path) read
        this flag to refuse a registry-pull substrate paired with the
        local-only built-in default runtime image, rather than letting
        the failure surface as an opaque mid-run image-build error.

    ``workspace_distribution``:
        How the plugin moves files between host and sandbox (§2
        ``agent_refactor/compute_dispatch_decisions.md``).

        ``"bind_mount"``: :meth:`Compute.stage_workspace` is a
        no-op-passthrough returning a :class:`WorkspaceHandle` wrapping
        the host path; the container sees the same files via the
        substrate's mount semantics. :meth:`Compute.harvest_workspace`
        is also a no-op (the host already reads what the container
        wrote). LocalGpu.

        ``"upload_download"``: :meth:`Compute.stage_workspace` uploads
        the local directory to a substrate-managed volume / store;
        :meth:`Compute.harvest_workspace` downloads back. Modal.

        ``"none"``: Plugin does not support workspace distribution.
        Both methods raise :class:`NotImplementedError`. The conformance
        suite's round-trip test skips cleanly. The unified dispatch
        factory (Step 2 of the agent-layer refactor) refuses to
        dispatch a role that needs a workspace through this plugin.
    """

    model_config = ConfigDict(extra="forbid")

    supports_gpu: bool
    max_timeout_seconds: int
    supports_log_streaming: bool = False
    requires_published_image: bool = False
    workspace_distribution: Literal["bind_mount", "upload_download", "none"] = "bind_mount"


# === Error contract (§7.3) ===================================================


class ComputeError(Exception):
    """Base class for all :class:`Compute` errors (§7.3)."""


class JobNotFound(ComputeError):
    """Raised by :meth:`Compute.status` when the substrate has no record
    of the handle (§7.3).

    E.g., expired or garbage-collected.
    """

    def __init__(self, handle: JobHandle) -> None:
        self.handle = handle
        super().__init__(f"job not found: {handle.plugin}/{handle.handle}")


class JobImageInvalid(ComputeError):
    """Raised by :meth:`Compute.submit` when ``image`` is not reachable
    from the substrate (§7.3).

    Eager fail on bad image references, not lazy fail mid-run.
    """

    def __init__(self, image: str, reason: str) -> None:
        self.image = image
        self.reason = reason
        super().__init__(f"invalid image {image!r}: {reason}")


class ComputeUnavailable(ComputeError):
    """Substrate-level outage (§7.3). Caller may retry."""


# === Workspace types (agent-refactor §2) ====================================


class WorkspaceHandle(BaseModel):
    """Opaque identifier for a staged workspace.

    Mirrors :class:`JobHandle`'s shape so substrates can reuse their
    handle-encoding conventions. The ``plugin`` field is the same
    plugin-identifier string the Compute plugin sets as its
    :attr:`Compute.name` ("localgpu", "modal", "runpod"); the
    ``handle`` field encodes whatever the substrate needs to reattach
    to the staged data (a host path for bind-mount substrates, a Modal
    volume name + version, a RunPod volume id, etc.); ``metadata`` is
    for non-load-bearing annotations (``staged_at``, ``byte_count``)
    that the substrate or callers find useful but the Protocol does
    not interpret.

    Consumers persist this object via the :class:`MetadataStore` and
    pass it back to :meth:`Compute.harvest_workspace` later (possibly
    from a fresh worker process, per the same cross-process
    reconnection contract as :class:`JobHandle`).
    """

    model_config = ConfigDict(extra="forbid")

    plugin: str
    handle: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkspaceNotFound(ComputeError):
    """Raised by :meth:`Compute.harvest_workspace` when the substrate
    has no record of the workspace handle.

    Mirrors :class:`JobNotFound`: substrate-side garbage collection,
    expired volumes, or a handle that was never staged on this
    substrate produce this error.
    """

    def __init__(self, handle: WorkspaceHandle) -> None:
        self.handle = handle
        super().__init__(f"workspace not found: {handle.plugin}/{handle.handle}")


# === Protocol shape (§7.2) ===================================================


@runtime_checkable
class Compute(Protocol):
    """Submit a containerized job, poll status, retrieve logs, cancel.

    Plugins register via::

        [project.entry-points."smai.computes"]
        <name> = "<module>:<class>"
    """

    name: str
    capabilities: ComputeCapabilities

    async def submit(
        self,
        image: str,
        command: list[str],
        env: dict[str, str],
        gpu: bool = False,
        timeout_seconds: int = 3600,
        **plugin_options: object,
    ) -> JobHandle:
        """Submit a job (§7.2).

        Returns once the job is accepted by the substrate (NOT when
        it begins executing). The returned :class:`JobHandle` is
        opaque; consumers persist it via the
        :class:`MetadataStore`.

        Implementations MUST:

        * Validate ``image`` is reachable from the substrate (eager
          fail on bad image references, not lazy fail mid-run).
        * Apply the timeout — jobs exceeding it transition to state
          ``'timeout'`` on the next ``.status()`` call.
        * Encode all reconnection state in ``JobHandle.handle`` /
          ``.metadata`` so a fresh process can resume polling.

        Storage / mount surface deliberately left open via
        ``**plugin_options`` — different substrates have different
        shapes (S3-via-IAM, Modal volumes, local bind mounts). The
        Protocol commits only to ``image``, ``command``, ``env``,
        ``gpu``, ``timeout``.
        """
        ...

    async def status(self, handle: JobHandle) -> JobStatus:
        """Return the current status of the job (§7.2).

        MUST raise :class:`JobNotFound` if the substrate has no
        record of the handle (e.g., expired or garbage-collected).
        MUST NOT block waiting for state changes — this is a poll,
        not a wait.
        """
        ...

    async def logs(self, handle: JobHandle) -> str:
        """Return the job's stdout+stderr concatenated (§7.2).

        For long logs, plugins MAY return only the tail (a
        plugin-specific cap). Long-form structured log streaming is
        plugin-internal.
        """
        ...

    async def cancel(self, handle: JobHandle) -> None:
        """Request cancellation (§7.2).

        Eventually-consistent: the next ``.status()`` call will
        return ``state='cancelled'`` once the substrate confirms.
        Idempotent — calling cancel on an already-terminal job is a
        no-op.
        """
        ...

    async def stage_workspace(self, local_path: Path) -> WorkspaceHandle:
        """Make ``local_path`` (a directory tree on the host) available
        inside subsequent :meth:`submit` calls under ``/workspace/``
        (or the substrate's equivalent root). Returns an opaque handle
        the caller persists and passes back to :meth:`harvest_workspace`
        later.

        Implementations MUST:

        * Treat ``local_path`` as the canonical source: every regular
          file under it is staged; symlinks are followed (since the
          container's view should be self-contained).
        * Raise :class:`FileNotFoundError` if ``local_path`` does not
          exist and :class:`NotADirectoryError` if it exists but is
          not a directory. These are caller-side bugs and use stdlib
          exceptions, not the Compute error taxonomy (which is
          reserved for substrate-side conditions).
        * Raise :class:`ComputeUnavailable` if the substrate is
          unreachable.
        * Encode reattachment state in the returned
          :class:`WorkspaceHandle` so a fresh process can later harvest
          (the same cross-process contract as :class:`JobHandle`).

        Not idempotent in the strict sense: repeat calls on the same
        ``local_path`` may produce distinct handles, and substrates
        with versioned volumes may layer the second stage on top of
        the first. Callers that need a single staged copy must persist
        the first handle.

        Plugins with ``workspace_distribution="none"`` raise
        :class:`NotImplementedError`.
        """
        ...

    async def harvest_workspace(self, handle: WorkspaceHandle, local_path: Path) -> None:
        """Copy the workspace's current contents (as the container
        last wrote them) back to the host at ``local_path``. Returns
        once the copy is complete.

        Implementations MUST:

        * Create ``local_path`` (including missing parent directories)
          when absent. Overwrite same-named files when present
          (``mkdir(parents=True, exist_ok=True)`` semantics).
        * Raise :class:`WorkspaceNotFound` when the substrate has no
          record of the handle (expired volume, garbage-collected,
          mismatched plugin identifier).
        * Raise :class:`ComputeUnavailable` if the substrate is
          unreachable.
        * Tolerate being called multiple times on the same
          ``(handle, local_path)`` pair: the second call writes the
          same files. No partial state on failure (best-effort
          all-or-nothing; substrate-dependent).

        Plugins with ``workspace_distribution="bind_mount"`` MAY
        implement this as a no-op (the host already sees what the
        container wrote via the mount); the method exists for Protocol
        uniformity at callsites.

        Plugins with ``workspace_distribution="none"`` raise
        :class:`NotImplementedError`.
        """
        ...
