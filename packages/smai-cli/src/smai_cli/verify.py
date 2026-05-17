"""Plugin-ping pre-flight helpers — backs ``smai verify``.

Per Task 3.G3: ``smai verify`` is a sibling of ``smai start`` — it
instantiates each configured plugin and calls a minimal read-only
"ping" so misconfigured ``smai.yaml`` (bad creds, unreachable bucket,
network issue, wrong region) fails fast with a clear per-plugin
diagnostic, before a worker boots and starts dispatching jobs.

Probe semantics (per the Task 3.G3 brief carry-forward #4):

* :class:`LlmProvider` — single 1-token completion. Cheap (single
  digit input + output tokens) but **does cost real tokens**; the
  ``smai verify`` verb's ``--help`` text documents this so operators
  aren't surprised. Surfaces auth + region + network in one round-
  trip.
* :class:`MetadataStore` — :meth:`MetadataStore.count_in_state` for
  the ``cg`` entity kind with a single-state list. Read-only by
  protocol contract (a ``SELECT COUNT(*) FROM comparison_groups
  WHERE state IN (...)``); requires the schema to be at head, which
  surfaces the same migration-stale failure mode as
  ``smai migrate --check``.
* :class:`ArtifactStore` — :meth:`ArtifactStore.exists` against a
  deliberately-non-existent key. Read-only (it's a HEAD-equivalent
  on S3-shaped backends). Returning ``False`` is the success case;
  any exception (auth failure, bucket missing, network unreachable)
  surfaces as a verify failure.
* :class:`Compute` — :meth:`Compute.status` against a deliberately-
  non-existent :class:`JobHandle`. Returning a clean
  :class:`JobNotFound` is the success case; any other exception
  (auth failure, substrate unreachable) surfaces as a verify
  failure.

* **Runtime-image config** (round 11) — :func:`verify_runtime_image_config`
  is a free, deterministic, always-on check: when the configured
  :class:`Compute` substrate is registry-pull
  (:attr:`ComputeCapabilities.requires_published_image`), it fails if
  ``engine.runtime_image`` / ``runtime_cpu_image`` are still the
  local-only built-in defaults the substrate cannot pull.
* **Runtime-image real probe** (round 11, opt-in) —
  :func:`verify_runtime_image_probe` is gated behind
  ``smai verify --probe-image``: it submits a trivial no-op job per
  runtime image to confirm pullability. It costs real money / time, so
  it is opt-in only.

No new Protocol methods are introduced — every probe uses an existing
method (the runtime-image probe uses ``submit`` / ``cancel``). Where
the brief named a method that doesn't exist on the Protocol (e.g.,
``ArtifactStore.head``), this module uses the nearest equivalent that
satisfies the same probe-semantics contract.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass

from smai_core.plugins import (
    ArtifactStore,
    Compute,
    JobHandle,
    LlmProvider,
    MetadataStore,
    NormalizedMessage,
    TextContent,
)
from smai_core.plugins.compute import ComputeError, JobImageInvalid, JobNotFound
from smai_orchestrator.engine import DEFAULT_RUNTIME_CPU_IMAGE, DEFAULT_RUNTIME_IMAGE

_VERIFY_PROBE_KEY = "smai-verify-probe-key-that-does-not-exist"
_VERIFY_PROBE_HANDLE = "smai-verify-probe-handle-that-does-not-exist"

# Doc pointer surfaced in the runtime-image config-check failure message
# (and the build-and-push runbook it references).
_RUNTIME_IMAGE_RUNBOOK = (
    "packages/smai-cli/OPERATIONS.md, section 'Building and publishing the runtime images'"
)


@dataclass(frozen=True)
class VerifyResult:
    """Per-plugin ping outcome.

    ``ok`` — ``True`` when the probe completed cleanly.
    ``reason`` — human-readable summary; success message on pass,
    error class + message on fail. Operators read this directly from
    the ``smai verify`` output.
    ``latency_ms`` — wall-clock duration of the probe call (None if
    the probe didn't actually issue an I/O round-trip — currently
    unused, reserved for future degraded-perf surfaces).
    """

    ok: bool
    reason: str
    latency_ms: float | None


def _format_exception(exc: BaseException) -> str:
    """Render an exception for ``VerifyResult.reason``."""
    return f"{type(exc).__name__}: {exc}"


def _llm_failure_hint(exc: BaseException) -> str:
    """Actionable hint appended to an LLM-probe failure reason.

    Bedrock's two most common ``smai verify`` failures are an invalid
    inference-profile ID (``ValidationException: The provided model
    identifier is invalid``) and an ungranted model
    (``AccessDeniedException: <model> is not available for this
    account``). Both look the same from a config-typo standpoint — the
    region is fine, the creds are fine, the ``model_id`` is wrong or not
    enabled — so a one-line pointer shortens the debug loop. Returns
    ``""`` when the error doesn't match (so non-Bedrock providers and
    real network/auth failures are unaffected).
    """
    msg = str(exc).lower()
    if "model identifier is invalid" in msg or "provided model identifier" in msg:
        return (
            " (hint: that doesn't look like a valid model ID for this provider/region; "
            "for Bedrock, model_id must be an inference-profile ID, listed by "
            "`aws bedrock list-inference-profiles --region <region>`)"
        )
    if "not available for this account" in msg or ("accessdenied" in msg and "model" in msg):
        return (
            " (hint: the model ID is recognized but not enabled for this account/region; "
            "grant model access in the Bedrock console under 'Model access', then retry)"
        )
    return ""


async def verify_llm_provider(provider: LlmProvider) -> VerifyResult:
    """Issue a single 1-token completion to surface auth + network.

    The call costs real tokens (typically <10 in + 1 out). The
    ``smai verify`` verb's ``--help`` documents this. A fail here
    surfaces :class:`LlmProviderAuthError`, network errors, or a
    region/model-id mismatch — every failure mode that would otherwise
    only manifest mid-dispatch when an agent loop fired. A model-id /
    model-access failure additionally gets an :func:`_llm_failure_hint`
    pointer appended.
    """
    started = time.monotonic()
    try:
        await provider.call(
            system="ping",
            messages=[
                NormalizedMessage(role="user", content=[TextContent(text="ping")]),
            ],
            max_tokens=1,
        )
    except Exception as exc:  # noqa: BLE001 — every plugin error type funnels here
        return VerifyResult(
            ok=False,
            reason=_format_exception(exc) + _llm_failure_hint(exc),
            latency_ms=(time.monotonic() - started) * 1000,
        )
    return VerifyResult(
        ok=True,
        reason=f"llm provider {provider.name!r} responded to 1-token ping",
        latency_ms=(time.monotonic() - started) * 1000,
    )


async def verify_metadata_store(store: MetadataStore) -> VerifyResult:
    """Surface DB connectivity + schema-at-head via a read-only count.

    :meth:`MetadataStore.count_in_state` translates to
    ``SELECT COUNT(*) FROM comparison_groups WHERE state IN (:s)`` —
    requires the schema to be at head, surfaces auth + connectivity,
    no mutations. Equivalent in failure-mode coverage to the
    ``smai migrate --check`` shape the Task 3.G3 brief named.
    """
    started = time.monotonic()
    try:
        await store.count_in_state("cg", ["draft"])
    except Exception as exc:  # noqa: BLE001
        return VerifyResult(
            ok=False,
            reason=_format_exception(exc),
            latency_ms=(time.monotonic() - started) * 1000,
        )
    return VerifyResult(
        ok=True,
        reason=f"metadata store {store.name!r} responded to count_in_state probe",
        latency_ms=(time.monotonic() - started) * 1000,
    )


async def verify_artifact_store(store: ArtifactStore) -> VerifyResult:
    """Surface bucket reachability + creds via a read-only ``exists``.

    Probes :meth:`ArtifactStore.exists` against a deliberately-non-
    existent key — returning ``False`` (the expected outcome) means
    the bucket is reachable and the credential chain is wired. Any
    exception is a fail.

    The Task 3.G3 brief named ``head(...)`` returning ``None``; the
    Protocol exposes ``exists(key) -> bool`` (S3 HEAD with a coerced
    boolean) which has the same read-only failure-mode coverage.
    """
    started = time.monotonic()
    try:
        await store.exists(_VERIFY_PROBE_KEY)
    except Exception as exc:  # noqa: BLE001
        return VerifyResult(
            ok=False,
            reason=_format_exception(exc),
            latency_ms=(time.monotonic() - started) * 1000,
        )
    return VerifyResult(
        ok=True,
        reason=f"artifact store {store.name!r} responded to exists probe",
        latency_ms=(time.monotonic() - started) * 1000,
    )


async def verify_compute(compute: Compute) -> VerifyResult:
    """Surface substrate reachability + creds via a read-only status.

    :meth:`Compute.status` against a deliberately-non-existent
    :class:`JobHandle` is expected to raise :class:`JobNotFound` —
    that's the success case (substrate reached, auth worked, just no
    such handle). Any other exception (auth error, substrate
    unreachable) surfaces as a verify failure.

    Some plugins return a synthetic :class:`JobStatus` with
    ``state="failed"`` rather than raising — :class:`LocalGpuCompute`
    in particular treats a missing PID as a no-op success per the
    Protocol's "must not block waiting for state changes" wording. We
    accept both as success: the probe established that the substrate
    accepted the call without raising an unexpected error. Plugins that
    validate the handle id client-side (e.g. ``ModalCompute``, whose
    ``Sandbox.from_id`` rejects the bogus probe handle before any
    round-trip) translate that to :class:`JobNotFound` per `07` §7.2,
    so the probe still establishes "creds + substrate reachable".
    """
    started = time.monotonic()
    handle = JobHandle(plugin=compute.name, handle=_VERIFY_PROBE_HANDLE)
    try:
        await compute.status(handle)
    except JobNotFound:
        return VerifyResult(
            ok=True,
            reason=f"compute {compute.name!r} responded to status probe (JobNotFound, expected)",
            latency_ms=(time.monotonic() - started) * 1000,
        )
    except ComputeError as exc:
        return VerifyResult(
            ok=False,
            reason=_format_exception(exc),
            latency_ms=(time.monotonic() - started) * 1000,
        )
    except Exception as exc:  # noqa: BLE001
        return VerifyResult(
            ok=False,
            reason=_format_exception(exc),
            latency_ms=(time.monotonic() - started) * 1000,
        )
    return VerifyResult(
        ok=True,
        reason=f"compute {compute.name!r} responded to status probe",
        latency_ms=(time.monotonic() - started) * 1000,
    )


def verify_runtime_image_config(
    compute: Compute,
    runtime_image: str,
    runtime_cpu_image: str,
) -> VerifyResult:
    """Hard-fail config check: a registry-pull Compute substrate paired
    with the local-only built-in default runtime image.

    Always-on, free, deterministic — no I/O, so ``latency_ms`` is
    ``None``. When the configured :class:`Compute` plugin reports
    :attr:`ComputeCapabilities.requires_published_image` (Modal /
    RunPod), the ``image`` argument to :meth:`Compute.submit` MUST be a
    registry-pullable tag. The built-in :class:`EngineConfig` defaults
    (``smai-runtime:dev`` / ``smai-runtime-cpu:dev``) are local-only
    Docker tags ``LocalGpuCompute`` builds on the host; a registry-pull
    substrate cannot pull them. Left unchecked the failure surfaces as
    an opaque ``RemoteError: Image build ... failed`` mid-CG-execution
    (round 11) — this catches it at pre-flight with a concrete pointer.

    Local-build substrates (``requires_published_image`` ``False``,
    e.g. ``LocalGpu``) always pass: building the default tag locally is
    exactly the intended flow.
    """
    capabilities = compute.capabilities
    if not getattr(capabilities, "requires_published_image", False):
        return VerifyResult(
            ok=True,
            reason=(
                f"compute {compute.name!r} builds images locally; the default "
                "runtime image tags are fine"
            ),
            latency_ms=None,
        )
    offenders: list[tuple[str, str]] = []
    if runtime_image == DEFAULT_RUNTIME_IMAGE:
        offenders.append(("engine.runtime_image", runtime_image))
    if runtime_cpu_image == DEFAULT_RUNTIME_CPU_IMAGE:
        offenders.append(("engine.runtime_cpu_image", runtime_cpu_image))
    if not offenders:
        return VerifyResult(
            ok=True,
            reason=(
                f"runtime image(s) overridden from the local-only defaults for "
                f"the {compute.name!r} registry-pull substrate"
            ),
            latency_ms=None,
        )
    parts = "; ".join(
        f"{field} is {value!r}, a local-only tag the {compute.name!r} compute substrate cannot pull"
        for field, value in offenders
    )
    return VerifyResult(
        ok=False,
        reason=(
            f"{parts}. Build the runtime image and push it to a registry the "
            f"{compute.name!r} substrate can reach, then set the field(s) in "
            f"smai.yaml. See {_RUNTIME_IMAGE_RUNBOOK}."
        ),
        latency_ms=None,
    )


async def _probe_one_runtime_image(compute: Compute, image: str, label: str) -> tuple[bool, str]:
    """Submit one trivial no-op job to confirm ``image`` is pullable.

    ``gpu=False`` — an image-build / registry pull is image-level, so
    skipping the GPU keeps the probe cheap. A small ``timeout_seconds``
    bounds the worst case; the job is cancelled right after submission
    (a successful :meth:`Compute.submit` already establishes that the
    substrate accepted and could resolve the image).
    """
    started = time.monotonic()
    try:
        handle = await compute.submit(
            image=image,
            command=["sh", "-c", "exit 0"],
            env={},
            gpu=False,
            timeout_seconds=120,
        )
    except JobImageInvalid as exc:
        return False, (
            f"{label} {image!r}: image not reachable from the substrate — {exc}. "
            f"See {_RUNTIME_IMAGE_RUNBOOK}."
        )
    except Exception as exc:  # noqa: BLE001 — every plugin error funnels here
        return False, f"{label} {image!r}: probe submit failed — {_format_exception(exc)}"
    # Best-effort cleanup: cancel the no-op job we just submitted so the
    # probe leaves nothing running. ``cancel`` is idempotent per `07` §7.2.
    with contextlib.suppress(Exception):
        await compute.cancel(handle)
    elapsed_ms = (time.monotonic() - started) * 1000
    return True, f"{label} {image!r}: reachable (no-op probe job submitted, {elapsed_ms:.0f}ms)"


async def verify_runtime_image_probe(
    compute: Compute,
    runtime_image: str,
    runtime_cpu_image: str,
) -> VerifyResult:
    """Opt-in real probe (``smai verify --probe-image``): submit a
    trivial no-op job per runtime image to confirm it is pullable.

    This is **not** a cheap registry-only check. The generic
    :class:`Compute` Protocol exposes only ``submit`` / ``status`` /
    ``cancel`` — there is no published-image-pullability call on the
    Protocol surface — so the smallest honest validation is a real
    (minimal) :meth:`Compute.submit` of a no-op command, which is where
    a registry-pull substrate actually resolves the image. The probe
    therefore **costs real money / time** and is opt-in; ``smai verify
    --help`` documents that.

    Probes :attr:`runtime_image` and :attr:`runtime_cpu_image` (deduped
    when they are equal). Fails if either image is unreachable.
    """
    started = time.monotonic()
    images: list[tuple[str, str]] = [("runtime_image", runtime_image)]
    if runtime_cpu_image != runtime_image:
        images.append(("runtime_cpu_image", runtime_cpu_image))
    messages: list[str] = []
    all_ok = True
    for label, image in images:
        ok, message = await _probe_one_runtime_image(compute, image, label)
        all_ok = all_ok and ok
        messages.append(message)
    return VerifyResult(
        ok=all_ok,
        reason="; ".join(messages),
        latency_ms=(time.monotonic() - started) * 1000,
    )


__all__ = [
    "VerifyResult",
    "verify_artifact_store",
    "verify_compute",
    "verify_llm_provider",
    "verify_metadata_store",
    "verify_runtime_image_config",
    "verify_runtime_image_probe",
]
