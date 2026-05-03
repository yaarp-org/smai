"""Central FastAPI exception handlers — Runtime / plugin exception → ``ErrorEnvelope``.

Per ``designs/smai/11-api.md`` §6.2: route handlers raise typed
exceptions; this module's handlers translate them to ``(status_code,
ErrorEnvelope)`` responses. Adding a new exception type is a two-step
change: define it in the Runtime / plugin layer, register a handler
here.

The translation centralizes one piece of awkwardness from FastAPI's
``HTTPException``: that class only carries an integer status + a
``detail`` body. We want a structured envelope with a code + message +
optional issues list, so we install per-type handlers that return
:class:`fastapi.responses.JSONResponse` directly.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from smai_api_spec import APIError, ErrorCode, ErrorEnvelope, ValidationIssue
from smai_cli.runtime import (
    CGNotFoundError,
    PaperNotFoundError,
    PaperStateError,
    ProposalNotFoundError,
    ProposalStateError,
    RunNotFoundError,
    RuntimeNotStartedError,
    WaitTimeoutError,
)
from smai_core.plugins.artifact_store import ArtifactNotFound
from smai_core.plugins.compute import JobNotFound
from smai_core.plugins.metadata_store import ConflictError

_log = logging.getLogger(__name__)


# === In-API exception types (raised by routers) ============================


class EntryNotFoundError(LookupError):
    """Raised by the comparison-groups router when an entry id is unknown.

    The Runtime layer doesn't ship an ``EntryNotFoundError`` (entries
    don't have a top-level lookup verb in v1); the API surface needs
    one to map cleanly onto the ``ENTRY_NOT_FOUND`` code per
    ``11-api.md`` §6.2.
    """

    def __init__(self, entry_id: str) -> None:
        self.entry_id = entry_id
        super().__init__(f"no entry with id {entry_id!r} found in MetadataStore")


class PaperNotReadyError(RuntimeError):
    """Raised by the proposals router when a ``reproduce_paper`` submission
    references a missing or non-terminal paper.

    Per ``11-api.md`` §13 OQ11 RESOLVED 2026-05-03: distinct from the
    generic ``INVALID_STATE`` so clients can render a paper-specific
    error message.
    """

    def __init__(self, arxiv_id: str, *, current_state: str | None = None) -> None:
        self.arxiv_id = arxiv_id
        self.current_state = current_state
        if current_state is None:
            super().__init__(
                f"reproduce_paper proposal references arxiv_id {arxiv_id!r} which "
                "has not been ingested yet; submit it via POST /api/v1/papers first"
            )
        else:
            super().__init__(
                f"reproduce_paper proposal references arxiv_id {arxiv_id!r} which "
                f"is in state {current_state!r}; reproduce-paper requires the paper "
                "to be in terminal 'registered' state"
            )


class HostRejectedError(RuntimeError):
    """Raised by the Host-validation middleware on allowlist mismatch.

    Translated to ``421 HOST_REJECTED`` per ``11-api.md`` §7.1.
    """

    def __init__(self, host: str) -> None:
        self.host = host
        super().__init__(f"Host header {host!r} not in allowlist")


class AuthError(RuntimeError):
    """Raised by the bearer-token middleware on missing / invalid token.

    Translated to ``403 FORBIDDEN`` per ``11-api.md`` §7.3.
    """


# === Helpers ================================================================


def _envelope(
    code: ErrorCode,
    message: str,
    *,
    issues: list[ValidationIssue] | None = None,
    retryable: bool | None = None,
) -> dict[str, Any]:
    """Construct an :class:`ErrorEnvelope` and serialize for ``JSONResponse``.

    ``model_dump(mode="json")`` produces a JSON-safe dict (no datetimes,
    etc.) suitable for direct return as the response body.
    """
    envelope = ErrorEnvelope(
        error=APIError(
            code=code,
            message=message,
            issues=issues,
            retryable=retryable,
        )
    )
    return envelope.model_dump(mode="json")


def _validation_issues_from_pydantic(exc: ValidationError) -> list[ValidationIssue]:
    """Project a :class:`pydantic.ValidationError` onto the API's
    :class:`ValidationIssue` shape per ``11-api.md`` §6.1."""
    issues: list[ValidationIssue] = []
    for err in exc.errors():
        loc_raw = err.get("loc", ())
        loc: list[str | int] = []
        for part in loc_raw:
            if isinstance(part, int):
                loc.append(part)
            else:
                loc.append(str(part))
        issues.append(
            ValidationIssue(
                loc=loc,
                msg=str(err.get("msg", "")),
                type=str(err.get("type", "")),
            )
        )
    return issues


# === Handlers ===============================================================


# === Per-exception handler factories =========================================
#
# We use module-level handler factories rather than nested functions inside
# :func:`register_exception_handlers` so pyright can see they're used. Each
# factory binds a (status, code) pair to a request-and-exception → JSONResponse
# function suitable for ``app.add_exception_handler``.


def _simple_handler(status: int, code: ErrorCode) -> Any:
    """Return a handler that maps any exception to ``(status, code)``."""

    async def _handler(request: Request, exc: Exception) -> JSONResponse:
        del request
        return JSONResponse(status_code=status, content=_envelope(code, str(exc)))

    return _handler


def _retryable_handler(status: int, code: ErrorCode) -> Any:
    """Like :func:`_simple_handler` but sets ``retryable=True``."""

    async def _handler(request: Request, exc: Exception) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=status,
            content=_envelope(code, str(exc), retryable=True),
        )

    return _handler


async def _request_validation_handler(request: Request, exc: Exception) -> JSONResponse:
    """Map FastAPI's ``RequestValidationError`` to 400 + envelope.

    ``RequestValidationError`` is structurally a ``ValidationError`` (it
    wraps the same per-error list), but is a separate type in FastAPI's
    public API. ``_validation_issues_from_pydantic`` only takes a
    ``ValidationError``; we narrow at the call site.
    """
    del request
    if isinstance(exc, ValidationError):
        issues = _validation_issues_from_pydantic(exc)
    elif isinstance(exc, RequestValidationError):
        # ``RequestValidationError.errors()`` returns the same per-error
        # dict list as ``ValidationError.errors()``; project directly.
        issues: list[ValidationIssue] = []
        for err in exc.errors():
            loc_raw = err.get("loc", ())
            loc: list[str | int] = []
            for part in loc_raw:
                if isinstance(part, int):
                    loc.append(part)
                else:
                    loc.append(str(part))
            issues.append(
                ValidationIssue(
                    loc=loc,
                    msg=str(err.get("msg", "")),
                    type=str(err.get("type", "")),
                )
            )
    else:
        issues = []
    return JSONResponse(
        status_code=400,
        content=_envelope(
            "VALIDATION_ERROR",
            "request body did not match the contract schema",
            issues=issues,
        ),
    )


async def _internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unexpected exceptions per ``11`` §6.2 (bug surface)."""
    _log.exception(
        "smai-api unhandled exception serving %s %s: %s",
        request.method,
        request.url.path,
        exc,
    )
    return JSONResponse(
        status_code=500,
        content=_envelope(
            "INTERNAL_ERROR",
            "internal server error; see server logs for details",
        ),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register every (exception type → status / code) handler on ``app``.

    The set of handlers must cover every typed exception the Runtime
    service surface (and the plugin Protocols it dispatches to) can
    raise — plus a generic catch-all that maps the unexpected to
    ``500 INTERNAL_ERROR``.
    """
    # 404 NOT_FOUND family.
    app.add_exception_handler(CGNotFoundError, _simple_handler(404, "CG_NOT_FOUND"))
    app.add_exception_handler(ProposalNotFoundError, _simple_handler(404, "PROPOSAL_NOT_FOUND"))
    app.add_exception_handler(PaperNotFoundError, _simple_handler(404, "PAPER_NOT_FOUND"))
    app.add_exception_handler(RunNotFoundError, _simple_handler(404, "RUN_NOT_FOUND"))
    app.add_exception_handler(EntryNotFoundError, _simple_handler(404, "ENTRY_NOT_FOUND"))
    app.add_exception_handler(ArtifactNotFound, _simple_handler(404, "ARTIFACT_NOT_FOUND"))
    app.add_exception_handler(JobNotFound, _simple_handler(404, "JOB_NOT_FOUND"))

    # 409 INVALID_STATE / CAS_CONFLICT / PAPER_NOT_READY.
    app.add_exception_handler(ProposalStateError, _simple_handler(409, "INVALID_STATE"))
    app.add_exception_handler(PaperStateError, _simple_handler(409, "INVALID_STATE"))
    app.add_exception_handler(PaperNotReadyError, _simple_handler(409, "PAPER_NOT_READY"))
    app.add_exception_handler(ConflictError, _simple_handler(409, "CAS_CONFLICT"))

    # 400 VALIDATION_ERROR — both FastAPI's RequestValidationError (route
    # body parse failure) and a manually-raised pydantic.ValidationError
    # (e.g. from a service-layer model_validate) map to the same envelope.
    app.add_exception_handler(RequestValidationError, _request_validation_handler)
    app.add_exception_handler(ValidationError, _request_validation_handler)

    # 503 RUNTIME_NOT_READY (retryable).
    app.add_exception_handler(RuntimeNotStartedError, _retryable_handler(503, "RUNTIME_NOT_READY"))
    # 504 TIMEOUT (retryable).
    app.add_exception_handler(WaitTimeoutError, _retryable_handler(504, "TIMEOUT"))

    # 421 HOST_REJECTED / 403 FORBIDDEN. Middleware short-circuits these
    # before route dispatch; the handlers below are a safety net for any
    # raise from a route handler.
    app.add_exception_handler(HostRejectedError, _simple_handler(421, "HOST_REJECTED"))
    app.add_exception_handler(AuthError, _simple_handler(403, "FORBIDDEN"))

    # 500 INTERNAL_ERROR catch-all.
    app.add_exception_handler(Exception, _internal_error_handler)


__all__ = [
    "AuthError",
    "EntryNotFoundError",
    "HostRejectedError",
    "PaperNotReadyError",
    "register_exception_handlers",
]
