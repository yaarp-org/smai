"""Dump ``smai-api``'s OpenAPI 3.1 spec to stdout as JSON.

Used by the SPA's ``pnpm codegen`` workflow per ``designs/smai/13-frontend.md``
§5: the typed API client under ``apps/ui/src/lib/api/generated/api-types.ts``
is regenerated from this spec via ``openapi-typescript``.

Why in-process (not uvicorn + curl): deterministic, no port held, no async
cleanup, and runs in well under a second. The chosen pattern lets a future
CI gate run the script and ``git diff --exit-code`` the result against the
committed types (post-M5 backlog item per ``13`` §5.4).

Why a placeholder runtime: ``app.openapi()`` only reflects on registered
routes and Pydantic ``response_model`` schemas. ``make_api_app`` stashes
the supplied runtime on ``app.state.runtime`` for request-time use by
``Depends(get_runtime)``, but no spec-emission code path reads it. So a
minimal sentinel is sufficient and lets the script run offline with no
plugin / database / credential setup.

Usage::

    uv run python tools/dump_openapi.py > openapi.json
    uv run python tools/dump_openapi.py | pnpm exec openapi-typescript - --output ...
"""

from __future__ import annotations

import json
import sys

from smai_api import make_api_app


class _PlaceholderRuntime:
    """Sentinel stashed on ``app.state.runtime`` for spec emission only.

    OpenAPI generation never reads ``app.state.runtime`` — only request
    handlers do, via ``Depends(get_runtime)``. So an empty class with
    no surface keeps the script free of plugin / store / compute setup.
    """


def main() -> int:
    app = make_api_app(_PlaceholderRuntime())  # type: ignore[arg-type]
    spec = app.openapi()
    json.dump(spec, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
