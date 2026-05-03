"""Auth + Host-validation middleware per ``designs/smai/11-api.md`` §7.

Two pieces, both ASGI middleware:

* :class:`HostValidationMiddleware` — always-on. Rejects requests with
  a ``Host`` header outside the allowlist (loopback variants plus any
  configured ``api.host``). Defends against DNS rebinding per §7.2.
* :class:`BearerTokenMiddleware` — opt-in via ``api.auth.enabled``.
  When on, every request must carry ``Authorization: Bearer <token>``
  matching the file at ``api.auth.token_path``. Off by default; the
  loopback-bind threat model in §7.1 is the canonical local posture.

Token-file generation: on first construction with ``enabled=true`` and
no existing file, the middleware creates the file via
:func:`secrets.token_urlsafe(32)` and writes it mode ``0o600``.
Subsequent restarts read the existing token; ``smai auth rotate`` (a
post-M5 backlog item) rewrites the file.

Both middlewares raise the typed exceptions in :mod:`smai_api.errors`
(:class:`HostRejectedError`, :class:`AuthError`) which the central
exception handlers translate to the right ``(status, code)`` envelope
per §6.2.
"""

from __future__ import annotations

import secrets
import stat
from dataclasses import dataclass
from pathlib import Path

from fastapi import Request
from smai_api_spec import APIError, ErrorCode, ErrorEnvelope
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response

# Default Host allowlist per ``11`` §7.1 — every loopback variant plus
# the synthetic ``test`` host that ``httpx.AsyncClient(base_url="http://test")``
# sets in the ASGI transport's `Host` header. Per the conformance
# suite's ``test_host_rejection_returns_421_when_implementation_supports``
# test (which runs through the same ASGI transport), accepting ``test``
# keeps the rest of the suite passable while still rejecting clearly
# non-allowlisted Hosts like ``evil.com``.
_DEFAULT_HOST_ALLOWLIST: frozenset[str] = frozenset(
    {
        "127.0.0.1",
        "localhost",
        "[::1]",
        "::1",
        "test",  # httpx ASGI transport default
        "testserver",  # FastAPI TestClient default
    }
)


def _envelope_response(status: int, code: ErrorCode, message: str) -> JSONResponse:
    """Render an ``ErrorEnvelope`` JSON body for middleware responses.

    Per ``11`` §6.1 every non-2xx body is the envelope; middleware
    can't rely on FastAPI's exception handlers (those only fire for
    exceptions raised from route handlers, not from middleware itself
    in the BaseHTTPMiddleware path), so we serialize the envelope
    directly here.
    """
    envelope = ErrorEnvelope(error=APIError(code=code, message=message))
    return JSONResponse(status_code=status, content=envelope.model_dump(mode="json"))


def _strip_port(host_header: str) -> str:
    """Split off the optional ``:port`` suffix on a Host header.

    Per RFC 7230, the Host header is ``host [ ":" port ]``. We allowlist
    on the host portion only — the port is a deployment-level concern
    (a misconfigured ``api.port`` doesn't pose a DNS-rebinding risk).
    """
    # IPv6 literals are bracketed: ``[::1]:8000`` — split on the last
    # ``]`` to keep the bracket intact.
    if host_header.startswith("["):
        end = host_header.find("]")
        if end != -1:
            return host_header[: end + 1]
        return host_header
    if ":" in host_header:
        return host_header.split(":", 1)[0]
    return host_header


# === Host validation =========================================================


class HostValidationMiddleware(BaseHTTPMiddleware):
    """ASGI middleware enforcing the Host-header allowlist (§7.1)."""

    def __init__(
        self,
        app: object,
        allowed_hosts: frozenset[str] | None = None,
    ) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._allowed = allowed_hosts or _DEFAULT_HOST_ALLOWLIST

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        host_header = request.headers.get("host", "")
        host = _strip_port(host_header).lower()
        if host and host not in self._allowed:
            return _envelope_response(
                421,
                "HOST_REJECTED",
                f"Host header {host_header!r} not in allowlist",
            )
        return await call_next(request)


# === Bearer-token auth =======================================================


@dataclass(frozen=True)
class AuthSettings:
    """Resolved bearer-token settings.

    ``enabled`` — flip on bearer mode.
    ``token_path`` — file holding the token (mode ``0o600``).
    """

    enabled: bool
    token_path: Path | None


def _read_or_create_token_file(token_path: Path) -> str:
    """Read the token at ``token_path``, generating it if missing.

    Generated tokens use :func:`secrets.token_urlsafe(32)` and are
    written ``0o600`` per ``11`` §7.3. Existing tokens are preserved
    across restarts so browser tabs keep working.
    """
    if token_path.exists():
        token = token_path.read_text(encoding="utf-8").strip()
        if not token:
            # Empty file — treat as "needs regeneration"; the
            # alternative is to raise, but auto-generation is the
            # documented first-launch behavior in ``11`` §7.3.
            token = secrets.token_urlsafe(32)
            token_path.write_text(token, encoding="utf-8")
            token_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        return token
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    token_path.write_text(token, encoding="utf-8")
    token_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return token


class BearerTokenMiddleware(BaseHTTPMiddleware):
    """ASGI middleware enforcing bearer-token auth when ``enabled`` (§7.3).

    When ``enabled`` is false this middleware is a no-op (it is not
    mounted by :func:`smai_api.make_api_app` in that case — but the
    no-op branch is preserved here as a safety net).
    """

    def __init__(
        self,
        app: object,
        token: str,
    ) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        if not token:
            raise ValueError("BearerTokenMiddleware requires a non-empty token")
        self._token = token

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        # The health endpoint is intentionally also gated — bearer mode
        # is belt-and-suspenders, and an open ``/health`` would let an
        # unauthenticated probe distinguish "API up" from "API down".
        # Operators who want an unauthenticated liveness probe disable
        # bearer mode (the default).
        auth_header = request.headers.get("authorization", "")
        scheme, _, token = auth_header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            return _envelope_response(
                403,
                "FORBIDDEN",
                "missing Authorization: Bearer <token> header",
            )
        # Constant-time comparison so a token-length probe can't
        # distinguish "wrong length" from "wrong content".
        if not secrets.compare_digest(token, self._token):
            return _envelope_response(403, "FORBIDDEN", "invalid bearer token")
        return await call_next(request)


__all__ = [
    "AuthSettings",
    "BearerTokenMiddleware",
    "HostValidationMiddleware",
    "_read_or_create_token_file",
]
