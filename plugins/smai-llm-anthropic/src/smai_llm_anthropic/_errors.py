"""Anthropic SDK exception → :class:`LlmProviderError` translation.

Per ``07-plugin-interfaces.md`` §4.5: every Anthropic-side exception
MUST land in one of four canonical buckets — ``LlmProviderRateLimited``
/ ``LlmProviderUnavailable`` / ``LlmProviderInvalidRequest`` /
``LlmProviderAuthError`` — or the base :class:`LlmProviderError` for
unknown shapes. The retry-once-with-30s-backoff pattern is owned by the
caller (``_provider.call``), not by this module — this module just
classifies.

The ``anthropic`` SDK raises typed subclasses of
:class:`anthropic.APIStatusError` keyed by HTTP status:

* 401 → :class:`AuthenticationError`
* 403 → :class:`PermissionDeniedError`
* 404 → :class:`NotFoundError`
* 422 → :class:`UnprocessableEntityError`
* 429 → :class:`RateLimitError`
* 5xx → :class:`InternalServerError` / :class:`APIStatusError`

Plus connection / timeout errors as subclasses of
:class:`anthropic.APIConnectionError`.

We classify primarily by HTTP status (``exc.status_code``) since that
is the documented contract; class-based dispatch is the fallback when
status is missing (e.g., a connection error pre-flight).
"""

from __future__ import annotations

from typing import Any

from smai_core.plugins import (
    LlmProviderAuthError,
    LlmProviderError,
    LlmProviderInvalidRequest,
    LlmProviderRateLimited,
    LlmProviderUnavailable,
)

_RATE_LIMIT_STATUSES: frozenset[int] = frozenset({429})
_UNAVAILABLE_STATUSES: frozenset[int] = frozenset({500, 502, 503, 504, 529})
_AUTH_STATUSES: frozenset[int] = frozenset({401, 403})
_INVALID_REQUEST_STATUSES: frozenset[int] = frozenset({400, 404, 405, 413, 422})


def translate_sdk_error(exc: BaseException) -> LlmProviderError:
    """Map an :class:`anthropic.APIError` (or sub-class) to a canonical
    :class:`LlmProviderError`.

    Connection / timeout errors land in :class:`LlmProviderUnavailable`
    (transient, retried once per §4.5).
    """
    status = _extract_status(exc)
    message = _extract_message(exc)

    cls: type[LlmProviderError]
    if status is not None:
        if status in _RATE_LIMIT_STATUSES:
            cls = LlmProviderRateLimited
        elif status in _UNAVAILABLE_STATUSES:
            cls = LlmProviderUnavailable
        elif status in _AUTH_STATUSES:
            cls = LlmProviderAuthError
        elif status in _INVALID_REQUEST_STATUSES:
            cls = LlmProviderInvalidRequest
        else:
            cls = LlmProviderError
    elif _looks_like_connection_error(exc):
        cls = LlmProviderUnavailable
    else:
        cls = LlmProviderError

    detail = f"{status}: {message}" if status is not None else message
    out = cls(detail)
    out.__cause__ = exc
    return out


def is_transient(exc: LlmProviderError) -> bool:
    """Return ``True`` iff ``exc`` is the retry-once kind per §4.5."""
    return isinstance(exc, (LlmProviderRateLimited, LlmProviderUnavailable))


def _extract_status(exc: BaseException) -> int | None:
    status: Any = getattr(exc, "status_code", None)
    if isinstance(status, bool):
        return None
    if isinstance(status, int):
        return status
    return None


def _extract_message(exc: BaseException) -> str:
    msg: Any = getattr(exc, "message", None)
    if isinstance(msg, str) and msg:
        return msg
    return str(exc)


def _looks_like_connection_error(exc: BaseException) -> bool:
    cls_name = type(exc).__name__
    return cls_name in {
        "APIConnectionError",
        "APITimeoutError",
        "APIConnectionTimeoutError",
    }


__all__ = ["is_transient", "translate_sdk_error"]
