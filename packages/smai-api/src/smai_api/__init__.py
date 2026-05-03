"""FastAPI implementation of the SMAI v2 HTTP API contract.

Per ``designs/smai/11-api.md`` and DEC-037: this package is one of two
independent implementations of the shared contract published in
``smai-api-spec``. The sibling implementation lives in the Yaarp v2
hosted-backend codebase; both pass the parameterizable conformance
suite in ``smai-api-conformance``.

Public entry point: :func:`make_api_app` — takes a constructed
:class:`smai_cli.runtime.Runtime`, returns a ready-to-serve
:class:`fastapi.FastAPI` instance with middleware mounted, exception
handlers registered, and per-resource routers included.
"""

from __future__ import annotations

from smai_api.app import AuthConfig, make_api_app

__all__ = ["AuthConfig", "make_api_app"]
