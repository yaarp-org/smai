"""Hatchling custom build hook that stages the SPA bundle as package data.

Per ``designs/smai/13-frontend.md`` §12.3 / §14 OQ3 RESOLVED 2026-05-03
(Hatch build hook on the ``smai-ui`` ``pyproject.toml``): on every
wheel / editable build, this hook copies the SPA's Vite build output
(``apps/ui/dist/*``) into ``src/smai_ui/static_spa/`` so the final
wheel ships with the SPA inline. ``smai-api`` then resolves the bundle
via :func:`smai_ui.get_static_bundle_path` at startup and mounts it at
``/`` (per ``12-ui-process.md`` §8.2).

Behavior summary (the "rough edges" warning from `13` §14 OQ3 informs
the tolerance posture below):

1. The hook tries to run ``pnpm build`` from ``apps/ui/`` if pnpm is on
   ``PATH`` AND ``apps/ui/`` exists. Build failures (non-zero exit) are
   propagated — a broken SPA build should fail the package install
   loudly, not silently ship a stale bundle.
2. If pnpm is NOT on ``PATH`` and ``apps/ui/dist/`` already exists with
   an ``index.html`` (a developer pre-built it), the hook copies the
   existing dist. This is the CI-on-pre-built-bundle path and the
   "developer pre-built once, now installs without rebuilding" path.
3. If neither (no pnpm AND no pre-built dist), the hook logs a warning
   and skips. The package still installs cleanly. Downstream
   :func:`smai_ui.get_static_bundle_path` raises ``FileNotFoundError``
   on access and ``smai-api``'s SPA mount detects this and degrades
   gracefully to API-only mode. This is the CI path on a fresh
   workspace clone (``uv sync`` on a runner without Node).
4. ``apps/ui/`` not found at all (e.g., installing from an sdist that
   does not bundle the SPA source) is the same skip path as case 3.

The hook is idempotent: re-running it overwrites ``static_spa/``
contents wholesale via :func:`shutil.copytree` with
``dirs_exist_ok=True``.

Approach B fallback (per the brief): if Approach A surfaces real
problems on a particular developer's machine, run
``cd apps/ui && pnpm install && pnpm build`` manually before
``pip install -e packages/smai-ui``; case 2 above (pre-built dist)
short-circuits the hook's pnpm dependency.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

_log = logging.getLogger(__name__)


class SpaBundleBuildHook(BuildHookInterface):
    """Stage the SPA bundle into ``src/smai_ui/static_spa/``."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:  # noqa: ARG002 - hatchling-API signature
        package_root = Path(self.root).resolve()
        # The `smai-ui` package lives at packages/smai-ui/ in the
        # workspace; apps/ui/ sits at the workspace root next to
        # packages/. Two `.parent` hops up from packages/smai-ui/
        # land at the workspace root.
        workspace_root = package_root.parent.parent
        apps_ui = workspace_root / "apps" / "ui"
        dist = apps_ui / "dist"
        target = package_root / "src" / "smai_ui" / "static_spa"

        if not apps_ui.is_dir():
            _log.warning(
                "smai-ui build_hook: apps/ui/ not found at %s; skipping SPA bundle "
                "staging. The wheel will install but smai_ui.get_static_bundle_path() "
                "will raise FileNotFoundError on access.",
                apps_ui,
            )
            return

        pnpm_on_path = shutil.which("pnpm") is not None

        if pnpm_on_path:
            # Run `pnpm install` only when node_modules is missing — saves
            # a few seconds on warm dev installs while keeping the cold
            # path correct.
            node_modules = apps_ui / "node_modules"
            if not node_modules.is_dir():
                _log.info("smai-ui build_hook: running `pnpm install` in %s", apps_ui)
                subprocess.run(
                    ["pnpm", "install", "--frozen-lockfile"],
                    cwd=apps_ui,
                    check=True,
                )
            _log.info("smai-ui build_hook: running `pnpm build` in %s", apps_ui)
            subprocess.run(
                ["pnpm", "build"],
                cwd=apps_ui,
                check=True,
            )
        else:
            _log.warning(
                "smai-ui build_hook: pnpm not on PATH; cannot run `pnpm build`. "
                "Falling back to existing apps/ui/dist/ if present."
            )

        if not (dist / "index.html").is_file():
            _log.warning(
                "smai-ui build_hook: %s missing or empty; skipping SPA bundle "
                "staging. Install pnpm + run `pnpm build` from apps/ui/ before "
                "re-installing if the SPA mount is required.",
                dist,
            )
            return

        # Wipe stale build outputs so Vite's content-hashed filenames
        # do not accumulate across rebuilds — but preserve the
        # `.gitkeep` marker so editable installs that re-run this hook
        # do not show a tracked-file deletion in `git status`.
        if target.exists():
            for child in target.iterdir():
                if child.name == ".gitkeep":
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        shutil.copytree(dist, target, dirs_exist_ok=True)
        _log.info(
            "smai-ui build_hook: staged %d files from %s into %s",
            sum(1 for _ in target.rglob("*") if _.is_file()),
            dist,
            target,
        )
