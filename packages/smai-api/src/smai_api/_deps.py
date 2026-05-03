"""FastAPI dependency-injection helpers.

The :class:`Runtime` is bound to ``app.state.runtime`` by
:func:`smai_api.make_api_app`; route handlers read it via the
:func:`get_runtime` dependency.
"""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Request
from smai_cli.runtime import Runtime


def get_runtime(request: Request) -> Runtime:
    """Return the :class:`Runtime` bound to the FastAPI app.

    :func:`make_api_app` writes the runtime onto ``app.state``; every
    route handler depends on this resolver to read it. Surfaces a clear
    error when the runtime isn't bound (mis-configured caller).
    """
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise RuntimeError(
            "smai-api app has no runtime bound on app.state — was it "
            "constructed via smai_api.make_api_app(runtime)?"
        )
    return cast("Runtime", runtime)


# The :class:`Annotated` form is the Ruff-friendly FastAPI dependency
# spelling — ``runtime: RuntimeDep`` instead of
# ``runtime: Runtime = Depends(get_runtime)``. Same runtime behavior;
# avoids the ``B008 Depends in default-value`` lint that fires on the
# legacy form.
RuntimeDep = Annotated[Runtime, Depends(get_runtime)]


__all__ = ["RuntimeDep", "get_runtime"]
