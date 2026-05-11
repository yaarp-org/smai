"""Unit tests for ``_reset_dev_state`` — the helper behind ``smai dev --reset``.

End-to-end CLI invocation of ``smai dev`` boots a long-running worker
(blocks on the OS signal handler), which is awkward to drive from a
``CliRunner``; the load-bearing piece is the deletion itself. We
exercise the helper directly with a tmp ``$SMAI_HOME``.
"""

from __future__ import annotations

from pathlib import Path

from smai_cli.main import _reset_dev_state


def test_removes_state_db_and_managed_dirs(tmp_path: Path) -> None:
    home = tmp_path / "smai_home"
    home.mkdir()
    state_db = home / "state.db"
    state_db.write_bytes(b"sqlite-stub")
    artifacts = home / "artifacts"
    (artifacts / "cgs" / "cg_x").mkdir(parents=True)
    (artifacts / "cgs" / "cg_x" / "f.json").write_text("{}")
    workspaces = home / "workspaces"
    (workspaces / "ws_x").mkdir(parents=True)
    (workspaces / "ws_x" / "scratch.py").write_text("print(1)")

    removed = _reset_dev_state(home)

    assert sorted(removed) == sorted([state_db, artifacts, workspaces])
    assert not state_db.exists()
    assert not artifacts.exists()
    assert not workspaces.exists()
    # The home directory itself is preserved.
    assert home.is_dir()


def test_is_a_no_op_when_paths_are_missing(tmp_path: Path) -> None:
    """Calling --reset on a fresh / never-booted $SMAI_HOME doesn't error."""
    home = tmp_path / "smai_home"
    home.mkdir()
    # No state.db / artifacts / workspaces have been created.
    removed = _reset_dev_state(home)
    assert removed == []


def test_leaves_unrelated_files_alone(tmp_path: Path) -> None:
    """Stray files in $SMAI_HOME (notes, manual backups) are preserved.

    ``_reset_dev_state`` only knows the three canonical paths; anything
    the user has stashed alongside them is theirs.
    """
    home = tmp_path / "smai_home"
    home.mkdir()
    (home / "state.db").write_bytes(b"x")
    (home / "artifacts").mkdir()
    (home / "workspaces").mkdir()
    keep_me = home / "notes.md"
    keep_me.write_text("manual notes")
    keep_backup_dir = home / "backups"
    keep_backup_dir.mkdir()
    (keep_backup_dir / "state.db.bak").write_bytes(b"backup")

    _reset_dev_state(home)

    assert keep_me.exists()
    assert keep_backup_dir.is_dir()
    assert (keep_backup_dir / "state.db.bak").exists()


def test_handles_partial_state(tmp_path: Path) -> None:
    """When only some of the three paths exist, the helper removes those
    and skips the missing ones cleanly."""
    home = tmp_path / "smai_home"
    home.mkdir()
    state_db = home / "state.db"
    state_db.write_bytes(b"x")
    # No artifacts/ or workspaces/ this time.

    removed = _reset_dev_state(home)
    assert removed == [state_db]
    assert not state_db.exists()
