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

No new Protocol methods are introduced — every probe uses an existing
read-only method. Where the brief named a method that doesn't exist
on the Protocol (e.g., ``ArtifactStore.head``), this module uses the
nearest equivalent that satisfies the same probe-semantics contract.
"""

from __future__ import annotations

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
from smai_core.plugins.compute import ComputeError, JobNotFound

_VERIFY_PROBE_KEY = "smai-verify-probe-key-that-does-not-exist"
_VERIFY_PROBE_HANDLE = "smai-verify-probe-handle-that-does-not-exist"


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


__all__ = [
    "VerifyResult",
    "verify_artifact_store",
    "verify_compute",
    "verify_llm_provider",
    "verify_metadata_store",
]
