"""Smoke tests for :mod:`smai_compute_modal` import + entry-point shape.

Mirrors the import-test pattern of the LocalGpu reference plugin
(:mod:`plugins.smai-compute-localgpu.tests.test_import`):

* The package imports cleanly.
* ``ModalCompute`` is exposed at the top level (matches the
  ``smai.computes`` entry-point declaration in ``pyproject.toml``).
* The class structurally satisfies the
  :class:`smai_core.plugins.Compute` Protocol (``runtime_checkable``).

Construction in this test does NOT touch real Modal — we pass a
duck-typed fake module via ``modal_module=...`` so the import test can
run without Modal credentials.
"""

from __future__ import annotations

import importlib

from _modal_fakes import FakeModal  # type: ignore[import-not-found]


def test_module_imports() -> None:
    module = importlib.import_module("smai_compute_modal")
    assert module is not None
    assert hasattr(module, "ModalCompute")


def test_modal_compute_is_compute_protocol() -> None:
    """Structural conformance to :class:`smai_core.plugins.Compute`.

    The Protocol is ``runtime_checkable`` per ``07-plugin-interfaces.md``
    §3 — ``isinstance(plugin, Compute)`` checks attribute presence
    only, not method signatures or types. The conformance suite is the
    actual contract enforcement; this is the fail-fast smoke check
    Tier A integrators rely on at startup.
    """
    from smai_compute_modal import ModalCompute  # noqa: PLC0415
    from smai_core.plugins import Compute  # noqa: PLC0415

    instance = ModalCompute(modal_module=FakeModal())
    assert isinstance(instance, Compute)
    assert instance.name == "modal"
