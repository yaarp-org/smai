"""Drift guards for the two runtime Dockerfiles.

The locked fixed-template ``experiment.py`` (the agent's
``run_experiment`` tool entrypoint) and the seed-run dispatcher's
``python -m smai_runtime.runner`` both import :mod:`smai_runtime`.
Both consumers run *inside* one of the two runtime images
(``smai-runtime:dev`` / ``smai-runtime-cpu:dev``), so each image must
carry a ``smai_runtime`` install (which transitively requires
``smai_core``).

Round 19 added the COPY + pip install layer to both Dockerfiles. These
tests pin the install in place so a future Dockerfile edit can't
silently drop it. Same drift-guard discipline as rounds 15/16/17 —
asserts the Dockerfile content directly rather than the built image
(the pytest suite does not build Docker images; the real test is the
operator rebuild plus a test-env re-run).
"""

from __future__ import annotations

from pathlib import Path

import pytest

_DOCKERFILES_DIR = Path(__file__).resolve().parents[1] / "dockerfiles"
_RUNTIME_DOCKERFILES = ("runtime-cpu.Dockerfile", "runtime.Dockerfile")


@pytest.mark.parametrize("name", _RUNTIME_DOCKERFILES)
def test_runtime_image_installs_smai_packages(name: str) -> None:
    """Each runtime Dockerfile must COPY + pip install smai-core then
    smai-runtime. Order is load-bearing: pip resolves smai-runtime's
    ``smai-core`` dep locally only if smai-core is already installed
    (the workspace packages are not published to PyPI)."""
    content = (_DOCKERFILES_DIR / name).read_text()
    assert "COPY packages/smai-core/ /tmp/smai-core/" in content, (
        f"{name}: missing smai-core COPY layer (Round 19)"
    )
    assert "COPY packages/smai-runtime/ /tmp/smai-runtime/" in content, (
        f"{name}: missing smai-runtime COPY layer (Round 19)"
    )
    core_install_pos = content.find("pip install --no-cache-dir /tmp/smai-core")
    runtime_install_pos = content.find("pip install --no-cache-dir /tmp/smai-runtime")
    assert core_install_pos != -1, f"{name}: missing smai-core pip install (Round 19)"
    assert runtime_install_pos != -1, f"{name}: missing smai-runtime pip install (Round 19)"
    assert core_install_pos < runtime_install_pos, (
        f"{name}: smai-core must be installed before smai-runtime "
        "(pip resolves the workspace-only dep locally only when the "
        "earlier package is already installed)"
    )


@pytest.mark.parametrize("name", _RUNTIME_DOCKERFILES)
def test_runtime_image_does_not_leave_source_behind(name: str) -> None:
    """The COPY-into-/tmp pattern is only acceptable if the same RUN
    layer also removes the source — otherwise the final image carries
    the packages twice (once in ``/tmp``, once installed). Pin the
    ``rm -rf`` cleanup so a future edit that splits the RUN into two
    can't silently inflate the image."""
    content = (_DOCKERFILES_DIR / name).read_text()
    assert "rm -rf /tmp/smai-core /tmp/smai-runtime" in content, (
        f"{name}: smai sources must be removed in the same RUN as the "
        "pip install (Round 19, image-size hygiene)"
    )


def test_dockerignore_excludes_known_bloat() -> None:
    """The repo-root .dockerignore must keep the build context lean —
    the runtime Dockerfiles now COPY package sources every rebuild, so
    sending the whole repo (including ``.venv/``, ``.git/``,
    ``node_modules/``) wastes daemon transfer on every build."""
    dockerignore = _DOCKERFILES_DIR.parents[2] / ".dockerignore"
    assert dockerignore.exists(), "repo-root .dockerignore missing (Round 19)"
    content = dockerignore.read_text()
    for entry in (
        ".venv/",
        ".git/",
        "node_modules/",
        "__pycache__/",
        "tests/",
    ):
        assert entry in content, f".dockerignore missing required exclude {entry!r} (Round 19)"
