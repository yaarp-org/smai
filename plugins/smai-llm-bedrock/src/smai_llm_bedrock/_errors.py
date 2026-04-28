"""Bedrock-side exception → :class:`LlmProviderError` translation.

Per ``07-plugin-interfaces.md`` §4.5: every Bedrock-side exception MUST
land in one of four canonical buckets — ``LlmProviderRateLimited`` /
``LlmProviderUnavailable`` / ``LlmProviderInvalidRequest`` /
``LlmProviderAuthError`` — or the base :class:`LlmProviderError` for
unknown shapes. The retry-once-with-30s-backoff pattern is owned by the
caller (``_provider.call``), not by this module — this module just
classifies.

The mapping uses ``botocore`` error codes (read out of
``ClientError.response['Error']['Code']``) rather than exception class
names. Bedrock's SDK emits everything as :class:`botocore.exceptions.ClientError`
with a varying ``Error.Code`` field; class-based dispatch would force a
list of ``except`` blocks against errors not exposed as named classes by
boto3. Code-based dispatch is the v1 pattern (``model-client.ts``
``isTransientError``) and the AWS docs' canonical form.
"""

from __future__ import annotations

from typing import Any, cast

from smai_core.plugins import (
    LlmProviderAuthError,
    LlmProviderError,
    LlmProviderInvalidRequest,
    LlmProviderRateLimited,
    LlmProviderUnavailable,
)

# botocore is a runtime-only dep — accept the duck-type for the
# ClientError class via :class:`Any` to avoid pulling botocore stubs
# (none ship with boto3) into the strict type-check.
ClientError = Any  # noqa: N816 — alias mirrors the upstream class name

# Per AWS Bedrock Runtime ``Converse`` API docs:
# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
#
# Each code is documented as raised by Converse; we drop them into the
# four canonical buckets per §4.5.
_RATE_LIMIT_CODES: frozenset[str] = frozenset(
    {
        "ThrottlingException",
        "TooManyRequestsException",
    }
)

# 503-shaped errors: service-side transient failures the caller should
# retry. Per §4.5, "service-unavailable / 503" maps to
# :class:`LlmProviderUnavailable`. ``ModelNotReadyException`` is the
# Bedrock-specific shape for "the inference profile is still warming
# up"; ``ModelTimeoutException`` is "the model took too long" (also
# transient — backend congestion).
_UNAVAILABLE_CODES: frozenset[str] = frozenset(
    {
        "ServiceUnavailableException",
        "ModelNotReadyException",
        "ModelTimeoutException",
        "ModelStreamErrorException",
        "InternalServerException",
    }
)

# Auth failures: not retried per §4.5.
_AUTH_CODES: frozenset[str] = frozenset(
    {
        "AccessDeniedException",
        "ExpiredTokenException",
        "UnrecognizedClientException",
        "InvalidSignatureException",
        "UnauthorizedOperation",
    }
)

# Validation / 4xx: not retried per §4.5.
_INVALID_REQUEST_CODES: frozenset[str] = frozenset(
    {
        "ValidationException",
        "ResourceNotFoundException",
        "ModelErrorException",
    }
)


def translate_client_error(exc: ClientError) -> LlmProviderError:
    """Map a botocore :class:`ClientError` to its canonical
    :class:`LlmProviderError` subclass.

    Returns the freshly constructed :class:`LlmProviderError` (with
    ``__cause__`` set to ``exc``) — the caller decides whether to ``raise
    ... from`` or store-and-forward.
    """
    code = _extract_error_code(exc)
    message = _extract_error_message(exc)
    detail = f"{code}: {message}" if code else message
    cls: type[LlmProviderError]
    if code in _RATE_LIMIT_CODES:
        cls = LlmProviderRateLimited
    elif code in _UNAVAILABLE_CODES:
        cls = LlmProviderUnavailable
    elif code in _AUTH_CODES:
        cls = LlmProviderAuthError
    elif code in _INVALID_REQUEST_CODES:
        cls = LlmProviderInvalidRequest
    else:
        cls = LlmProviderError
    out = cls(detail)
    out.__cause__ = exc
    return out


def is_transient(exc: LlmProviderError) -> bool:
    """Return ``True`` iff ``exc`` is the retry-once kind per §4.5."""
    return isinstance(exc, (LlmProviderRateLimited, LlmProviderUnavailable))


def _extract_error_code(exc: ClientError) -> str:
    response: Any = getattr(exc, "response", None)
    if isinstance(response, dict):
        error_dict: Any = cast("dict[str, Any]", response).get("Error")
        if isinstance(error_dict, dict):
            code: Any = cast("dict[str, Any]", error_dict).get("Code")
            if isinstance(code, str):
                return code
    return ""


def _extract_error_message(exc: ClientError) -> str:
    response: Any = getattr(exc, "response", None)
    if isinstance(response, dict):
        error_dict: Any = cast("dict[str, Any]", response).get("Error")
        if isinstance(error_dict, dict):
            message: Any = cast("dict[str, Any]", error_dict).get("Message")
            if isinstance(message, str):
                return message
    return str(exc)


__all__ = ["is_transient", "translate_client_error"]
