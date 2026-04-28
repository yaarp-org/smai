"""No-go-zone hash check.

Per ``10-runtime-and-templates.md`` §7. The runtime asserts at startup that
the fixed-template files have not been modified. Path-not-found is a
failure (the agent can't be allowed to delete the template); content-mismatch
is a failure (the agent can't be allowed to edit the template).

Two sources contribute to the no-go-zone file list (§7.1):
1. ``HarnessContract.body.no_go_zones`` — the agent-visible names.
2. The runtime template package's internal manifest — the source of truth
   for expected content for the runtime's own template-package version.
"""

from __future__ import annotations

import hashlib
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

from smai_runtime.errors import NoGoZoneHashError

RUNTIME_TEMPLATE_VERSION: str = "1.0.0"
"""Bumped on any byte-level change to the shipped templates (§3.3)."""

# Workspace-relative paths the runtime hashes — these are also the canonical
# ``HarnessContract.body.no_go_zones`` agent-visible names (§2.2).
TEMPLATE_WORKSPACE_PATHS: tuple[str, ...] = (
    "experiment.py",
    "techniques/__init__.py",
)

# Mapping from the workspace-relative path to the resource filename inside
# ``smai_runtime.templates._files`` (the package-resource location). Resource
# filenames carry a ``.template`` suffix so pyright does not type-check them
# as part of smai-runtime's source tree.
_RESOURCE_FOR_PATH: dict[str, str] = {
    "experiment.py": "experiment.py.template",
    "techniques/__init__.py": "techniques_init.py.template",
}


def _resource_root() -> Traversable:
    return files("smai_runtime.templates._files")


def read_template_bytes(workspace_path: str) -> bytes:
    """Return the template's expected on-disk bytes for ``workspace_path``.

    Used by :func:`workspace.materialize_workspace` to drop the template
    file at run-start and by :func:`compute_expected_hashes` to compute
    expected hashes (both refer to the same source of truth — the template
    file shipped in the package resource directory).
    """
    resource = _RESOURCE_FOR_PATH[workspace_path]
    return _resource_root().joinpath(resource).read_bytes()


def compute_expected_hashes() -> dict[str, str]:
    """Map workspace-relative path → expected SHA-256 hex digest (§7.1)."""
    return {
        path: hashlib.sha256(read_template_bytes(path)).hexdigest()
        for path in TEMPLATE_WORKSPACE_PATHS
    }


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_no_go_zones(
    workspace: Path,
    *,
    no_go_zones: list[str] | None = None,
    runtime_template_version: str = RUNTIME_TEMPLATE_VERSION,
) -> None:
    """Verify every fixed-template file in ``workspace`` matches its expected
    hash (§7).

    Path-not-found → :class:`NoGoZoneHashError` with code ``file_missing``.
    Content-mismatch → :class:`NoGoZoneHashError` with code ``hash_mismatch``.

    The two sources contributing to the checked list (§7.1) are merged: the
    methodology contract's ``no_go_zones`` (passed in) and the runtime's
    own template paths. The runtime is the source of truth for expected
    content; the contract just contributes the file-name surface.
    """
    expected = compute_expected_hashes()

    # Union of methodology-declared names + runtime's own template paths
    # (deduplicated, sorted for deterministic error ordering).
    contract_paths: list[str] = list(no_go_zones) if no_go_zones is not None else []
    candidates: list[str] = sorted(set(contract_paths) | set(TEMPLATE_WORKSPACE_PATHS))

    for rel in candidates:
        target = workspace / rel
        if not target.is_file():
            raise NoGoZoneHashError(
                code="file_missing",
                path=rel,
                expected_hash=expected.get(rel),
                actual_hash=None,
                runtime_template_version=runtime_template_version,
            )

        # Files outside the runtime-known set are listed by the methodology
        # contract but not hashed by the runtime — methodology declares the
        # *names*; runtime knows the *contents* of files it ships (§7.2).
        # Such paths are sanity-checked for existence only.
        if rel not in expected:
            continue

        actual = _hash_file(target)
        if actual != expected[rel]:
            raise NoGoZoneHashError(
                code="hash_mismatch",
                path=rel,
                expected_hash=expected[rel],
                actual_hash=actual,
                runtime_template_version=runtime_template_version,
            )
