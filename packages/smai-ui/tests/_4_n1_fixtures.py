"""Test helpers for Task 4.N1's smai-ui package.

Per the workspace convention (Task 3.F4 / 3.F5 lessons): per-task fixture
filenames use the ``_<task>_<purpose>.py`` shape so sibling-plugin /
sibling-package fakes do not collide under
``--import-mode=importlib``.
"""

from __future__ import annotations

from pathlib import Path


def write_stub_bundle(target: Path) -> Path:
    """Populate ``target`` with a minimally valid SPA bundle layout.

    Creates ``index.html`` (matching the shape Vite emits — head + body
    + a script tag we can inject the bearer-token bootstrap into per
    ``13-frontend.md`` §12.4) and an empty ``assets/`` directory.
    Returns ``target`` for chaining.
    """
    target.mkdir(parents=True, exist_ok=True)
    (target / "index.html").write_text(
        "<!doctype html>\n"
        "<html>\n"
        "  <head>\n"
        "    <title>smai</title>\n"
        "  </head>\n"
        "  <body>\n"
        '    <div id="root"></div>\n'
        '    <script type="module" src="/assets/index.js"></script>\n'
        "  </body>\n"
        "</html>\n"
    )
    (target / "assets").mkdir(exist_ok=True)
    (target / "assets" / "index.js").write_text("// stub asset\n")
    return target
