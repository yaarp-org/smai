"""SPA bundle wrapper for SMAI v2.

Per ``designs/smai/12-ui-process.md`` §8.2 + ``13-frontend.md`` §12.3:
this package ships the React SPA's Vite build output as Python package
data so a single ``pip install smai-ui`` lands the SPA on disk where
:func:`smai_api.app.make_api_app` can mount it at ``/``. No separate
Node toolchain is needed on production hosts.

The bundle is staged at wheel-build time by ``build_hook.py`` (Hatch
custom build hook); at runtime, callers resolve the on-disk path via
:func:`get_static_bundle_path`.

Public surface: :func:`get_static_bundle_path`. Everything else is an
implementation detail of the build hook.
"""

from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path

__all__ = ["get_static_bundle_path"]


def get_static_bundle_path() -> Path:
    """Return the on-disk path to the staged SPA static bundle.

    Resolution: ``importlib.resources.files("smai_ui") / "static_spa"``.
    For both wheel and editable installs the directory exists on disk
    (the bundle is package data, not zipped resources), so we materialize
    it via :func:`importlib.resources.as_file` and return a
    :class:`pathlib.Path` that callers can hand to
    :class:`fastapi.staticfiles.StaticFiles`.

    Raises
    ------
    FileNotFoundError
        When ``static_spa/`` is missing or empty (typically: developer
        installed from source without pnpm on PATH and without a
        pre-built ``apps/ui/dist/``). The error message points at the
        recovery path (run ``pnpm build`` from ``apps/ui/`` then
        re-install).

    Notes
    -----
    The function does not cache. ``importlib.resources.files`` is cheap
    enough that repeated calls during process startup are fine, and not
    caching keeps the surface trivial for tests that swap the bundle
    out via monkeypatch.
    """
    bundle = files("smai_ui") / "static_spa"
    # `as_file()` is the future-proof way to get a Path even if the
    # package is ever loaded from a zip; for normal on-disk installs it
    # returns a context manager that yields the same path the resource
    # already lives at, so we do not need to actually keep the context
    # open.
    with as_file(bundle) as path:
        resolved = Path(path)

    if not resolved.is_dir():
        raise FileNotFoundError(
            f"smai-ui static bundle not found at {resolved}. If installing from "
            "source, run `pnpm build` from apps/ui/ first, then re-install "
            "smai-ui (`pip install -e packages/smai-ui`)."
        )
    if not (resolved / "index.html").is_file():
        raise FileNotFoundError(
            f"smai-ui static bundle at {resolved} is missing index.html. The "
            "bundle staging step (build_hook.py) likely failed; check the "
            "install log for warnings about pnpm or apps/ui/dist/."
        )
    return resolved
