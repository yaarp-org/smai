"""No-go-zone hash check tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from smai_runtime import (
    NoGoZoneHashError,
    check_no_go_zones,
    compute_expected_hashes,
    create_workspace_skeleton,
    write_template_files,
)


def _populate(workspace: Path) -> None:
    create_workspace_skeleton(workspace)
    write_template_files(workspace)


def test_template_hashes_match_after_write(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    _populate(workspace)
    check_no_go_zones(workspace, no_go_zones=["experiment.py", "techniques/__init__.py"])


def test_check_passes_when_methodology_lists_extra_unknown_files(tmp_path: Path) -> None:
    """Methodology declares names; runtime hashes only the ones it knows.

    Files outside the runtime-known set are sanity-checked for existence
    only (§7.2). We add an extra file via the contract no_go_zones list
    and confirm presence-only enforcement.
    """
    workspace = tmp_path / "ws"
    _populate(workspace)
    extra = workspace / "harness" / "config.yaml"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text("seed: 1\n")
    check_no_go_zones(
        workspace,
        no_go_zones=["experiment.py", "techniques/__init__.py", "harness/config.yaml"],
    )


def test_missing_template_raises_file_missing(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    create_workspace_skeleton(workspace)
    # Intentionally do not call write_template_files.
    with pytest.raises(NoGoZoneHashError) as exc:
        check_no_go_zones(workspace, no_go_zones=["experiment.py"])
    assert exc.value.code == "file_missing"
    assert exc.value.actual_hash is None


def test_modified_template_raises_hash_mismatch(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    _populate(workspace)
    target = workspace / "experiment.py"
    target.write_text(target.read_text() + "\n# tampered\n")
    with pytest.raises(NoGoZoneHashError) as exc:
        check_no_go_zones(workspace, no_go_zones=["experiment.py"])
    assert exc.value.code == "hash_mismatch"
    assert exc.value.actual_hash is not None
    assert exc.value.expected_hash is not None
    assert exc.value.actual_hash != exc.value.expected_hash


def test_techniques_init_tampering_detected(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    _populate(workspace)
    target = workspace / "techniques" / "__init__.py"
    target.write_bytes(target.read_bytes().replace(b"importlib", b"# tampered"))
    with pytest.raises(NoGoZoneHashError) as exc:
        check_no_go_zones(workspace, no_go_zones=["techniques/__init__.py"])
    assert exc.value.code == "hash_mismatch"
    assert exc.value.path == "techniques/__init__.py"


def test_compute_expected_hashes_covers_all_template_paths() -> None:
    expected = compute_expected_hashes()
    assert set(expected.keys()) == {"experiment.py", "techniques/__init__.py"}
    for digest in expected.values():
        assert len(digest) == 64
