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
    """

    model_config = ConfigDict(extra="forbid")

    supports_gpu: bool
    max_timeout_seconds: int
    supports_log_streaming: bool = False
    requires_published_image: bool = False


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
