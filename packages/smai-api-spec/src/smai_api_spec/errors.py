"""Error envelope + error-code catalog per ``designs/smai/11-api.md`` §6.

Every non-2xx response from a SMAI / Yaarp v2 API carries the
:class:`ErrorEnvelope` shape — ``{"error": {"code", "message",
"issues"?, "retryable"?}}``. The wrapping ``error:`` namespace
disambiguates from happy-path responses that may contain an ``error``
field of their own.

Error ``code`` values are drawn from a fixed catalog (see :data:`ErrorCode`
below) so client code can branch on them safely. The HTTP status code
mapping is documented in ``11`` §6.2; this module is just the wire
shape.
"""

from __future__ import annotations

from typing import Literal

from smai_api_spec._common import APIBaseModel

# Catalog per ``11`` §6.2.
#
# The set is closed: client code may switch over these values.
# Adding a new code is a minor-version bump (additive); removing or
# renaming one is a major-version bump.
ErrorCode = Literal[
    "CG_NOT_FOUND",
    "PROPOSAL_NOT_FOUND",
    "PAPER_NOT_FOUND",
    "RUN_NOT_FOUND",
    "ENTRY_NOT_FOUND",
    "INVALID_STATE",
    "CAS_CONFLICT",
    "VALIDATION_ERROR",
    "LLM_UPSTREAM",
    "RUNTIME_NOT_READY",
    "TIMEOUT",
    "ARTIFACT_NOT_FOUND",
    "JOB_NOT_FOUND",
    "FORBIDDEN",
    "HOST_REJECTED",
    "INTERNAL_ERROR",
    "PAPER_NOT_READY",
]


class ValidationIssue(APIBaseModel):
    """One problem inside a ``VALIDATION_ERROR`` response.

    Mirrors :class:`pydantic.ValidationError`'s per-error shape so the
    central FastAPI exception handler can translate Pydantic errors with
    minimal massaging. ``loc`` is a JSON-pointer-shaped list (string
    field names + integer list indices); ``type`` is the Pydantic
    validator identifier (e.g. ``"value_error"`` / ``"missing"``).
    """

    loc: list[str | int]
    msg: str
    type: str


class APIError(APIBaseModel):
    """Body of an :class:`ErrorEnvelope`.

    ``code`` — drawn from :data:`ErrorCode`.
    ``message`` — human-readable; safe to render in a UI.
    ``issues`` — populated only for ``VALIDATION_ERROR`` (one entry per
        Pydantic per-field failure). ``None`` for every other code.
    ``retryable`` — populated only for plugin failures (5xx). ``True``
        when a retry-with-backoff has a chance of succeeding (transient
        network blip, throttled upstream); ``False`` when the failure
        will recur (auth misconfig, missing resource).
    """

    code: ErrorCode
    message: str
    issues: list[ValidationIssue] | None = None
    retryable: bool | None = None


class ErrorEnvelope(APIBaseModel):
    """Top-level body for every non-2xx response."""

    error: APIError


__all__ = [
    "APIError",
    "ErrorCode",
    "ErrorEnvelope",
    "ValidationIssue",
]
