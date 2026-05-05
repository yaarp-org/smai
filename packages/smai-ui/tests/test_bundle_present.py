"""Tests for :mod:`smai_ui` (Task 4.N1).

Two acceptance probes:

* :func:`get_static_bundle_path` raises ``FileNotFoundError`` with a
  helpful message when the bundle is missing.
* :func:`get_static_bundle_path` returns the staged bundle directory
  when one is present (this leg is gated on a real bundle existing —
  skip cleanly when the developer has not run ``pnpm build`` yet, so
  CI on a fresh checkout passes without flake).
"""

from __future__ import annotations

from importlib.resources import files

import pytest
from _4_n1_fixtures import write_stub_bundle  # type: ignore[import-not-found]
from smai_ui import get_static_bundle_path


def _patch_resource_anchor(monkeypatch, target_dir):
    """Patch ``smai_ui.files(...)`` so the ``/`` traversal lands at ``target_dir``.

    The accessor calls ``files("smai_ui") / "static_spa"`` — so we
    return an anchor whose ``__truediv__("static_spa")`` resolves to
    ``target_dir`` itself.
    """

    class _Anchor:
        def __truediv__(self, _other: str):
            return target_dir

    import smai_ui as smai_ui_mod

    monkeypatch.setattr(smai_ui_mod, "files", lambda _pkg: _Anchor())


def test_returns_path_when_bundle_present(tmp_path, monkeypatch) -> None:
    """The accessor returns the staged directory when index.html exists."""
    bundle = write_stub_bundle(tmp_path / "static_spa")
    _patch_resource_anchor(monkeypatch, bundle)
    resolved = get_static_bundle_path()
    assert resolved == bundle
    assert (resolved / "index.html").is_file()


def test_raises_when_bundle_missing(tmp_path, monkeypatch) -> None:
    """The accessor raises ``FileNotFoundError`` when the dir is missing."""
    _patch_resource_anchor(monkeypatch, tmp_path / "does_not_exist")
    with pytest.raises(FileNotFoundError, match="not found"):
        get_static_bundle_path()


def test_raises_when_index_html_missing(tmp_path, monkeypatch) -> None:
    """An empty static_spa/ dir (e.g. .gitkeep only) is treated as missing."""
    empty_dir = tmp_path / "static_spa"
    empty_dir.mkdir()
    (empty_dir / ".gitkeep").write_text("")
    _patch_resource_anchor(monkeypatch, empty_dir)
    with pytest.raises(FileNotFoundError, match="missing index.html"):
        get_static_bundle_path()


def test_real_bundle_if_staged() -> None:
    """When the real bundle has been staged (developer ran the hook
    or pre-built ``apps/ui/dist/``), the accessor must succeed.

    Skips when the bundle hasn't been staged — that's a valid state on
    a fresh CI checkout that has no Node toolchain.
    """
    real = files("smai_ui") / "static_spa"
    has_bundle = (real / "index.html").is_file()  # type: ignore[attr-defined]
    if not has_bundle:
        pytest.skip(
            "smai-ui static_spa bundle not staged; run `pnpm build` from apps/ui/ "
            "and re-install smai-ui to exercise this leg."
        )
    resolved = get_static_bundle_path()
    assert resolved.is_dir()
    assert (resolved / "index.html").is_file()
