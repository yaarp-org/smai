"""Entry-point discovery + import smoke tests."""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points

from smai_compute_localgpu import LocalGpuCompute
from smai_core.plugins import Compute


def test_module_imports() -> None:
    module = importlib.import_module("smai_compute_localgpu")
    assert module.LocalGpuCompute is LocalGpuCompute


def test_localgpu_compute_runtime_checkable_against_protocol() -> None:
    """``runtime_checkable`` requires attribute presence; the class
    itself must satisfy the :class:`Compute` Protocol shape (without
    needing a real Docker daemon, which would fail in CI).

    The ``skip_preflight=True`` constructor flag bypasses the
    ``docker info`` daemon check — used here and by the conformance
    subclass when checking the Protocol structurally.
    """
    instance = LocalGpuCompute(skip_preflight=True)
    assert isinstance(instance, Compute)


def test_entry_point_advertises_localgpu_compute() -> None:
    eps = entry_points(group="smai.computes")
    matching = [ep for ep in eps if ep.name == "localgpu"]
    assert matching, "localgpu entry point not registered under smai.computes"
    loaded = matching[0].load()
    assert loaded is LocalGpuCompute
