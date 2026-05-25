"""Tests for the file-operation standard tools (Task 2.B2 / §12.1)."""

from __future__ import annotations

from pathlib import Path

import pytest
from _b2_helpers import make_test_context, make_test_session  # type: ignore[import-not-found]
from smai_agents.std_tools.files import (
    MAX_LIST_ENTRIES,
    MAX_SEARCH_MATCHES,
    EditFileInput,
    ListFilesInput,
    ReadFileInput,
    SearchInput,
    WriteFileInput,
    make_edit_file_tool,
    make_list_files_tool,
    make_read_file_tool,
    make_search_tool,
    make_write_file_tool,
)

# ---------- read_file --------------------------------------------------------


@pytest.mark.asyncio
async def test_read_file_returns_numbered_content(tmp_path: Path) -> None:
    target = tmp_path / "hello.txt"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = make_test_session(tmp_path)
    ctx = make_test_context(session)
    tool = make_read_file_tool()

    result = await tool.handler(ReadFileInput(path="hello.txt"), ctx)

    assert result.is_error is False
    assert "1\talpha" in result.content
    assert "2\tbeta" in result.content
    assert "3\tgamma" in result.content


@pytest.mark.asyncio
async def test_read_file_path_escape_rejected(tmp_path: Path) -> None:
    session = make_test_session(tmp_path)
    ctx = make_test_context(session)
    tool = make_read_file_tool()

    result = await tool.handler(ReadFileInput(path="../escape.txt"), ctx)

    assert result.is_error is True
    assert "outside workspace" in result.content


@pytest.mark.asyncio
async def test_read_file_directory_rejected(tmp_path: Path) -> None:
    (tmp_path / "subdir").mkdir()
    session = make_test_session(tmp_path)
    ctx = make_test_context(session)
    tool = make_read_file_tool()

    result = await tool.handler(ReadFileInput(path="subdir"), ctx)

    assert result.is_error is True
    assert "directory" in result.content
    assert "list_files" in result.content


@pytest.mark.asyncio
async def test_read_file_missing_file(tmp_path: Path) -> None:
    session = make_test_session(tmp_path)
    ctx = make_test_context(session)
    tool = make_read_file_tool()

    result = await tool.handler(ReadFileInput(path="nope.txt"), ctx)

    assert result.is_error is True
    assert "file not found" in result.content


# ---------- write_file -------------------------------------------------------


@pytest.mark.asyncio
async def test_write_file_creates_parent_dirs(tmp_path: Path) -> None:
    session = make_test_session(tmp_path)
    ctx = make_test_context(session)
    tool = make_write_file_tool()

    result = await tool.handler(
        WriteFileInput(path="nested/dir/file.txt", content="hello"),
        ctx,
    )

    assert result.is_error is False
    assert (tmp_path / "nested" / "dir" / "file.txt").read_text() == "hello"
    assert "wrote 5 bytes" in result.content


@pytest.mark.asyncio
async def test_write_file_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    target.write_text("old", encoding="utf-8")
    session = make_test_session(tmp_path)
    ctx = make_test_context(session)
    tool = make_write_file_tool()

    result = await tool.handler(
        WriteFileInput(path="f.txt", content="new content"),
        ctx,
    )

    assert result.is_error is False
    assert target.read_text() == "new content"


@pytest.mark.asyncio
async def test_write_file_path_escape_rejected(tmp_path: Path) -> None:
    session = make_test_session(tmp_path)
    ctx = make_test_context(session)
    tool = make_write_file_tool()

    result = await tool.handler(
        WriteFileInput(path="/etc/passwd", content="x"),
        ctx,
    )

    assert result.is_error is True
    assert "outside workspace" in result.content


@pytest.mark.asyncio
async def test_write_file_lint_hook_registered() -> None:
    """The write_file tool factory binds lint_after_python_write."""
    tool = make_write_file_tool()
    from smai_agents.between_turn import lint_after_python_write

    assert lint_after_python_write in tool.post_result_hooks


@pytest.mark.asyncio
async def test_write_file_lint_hook_fires_on_python(tmp_path: Path) -> None:
    """Lint hook surfaces ruff output in the tool result for .py files.

    Skips when ruff is not on PATH (the lint hook is best-effort per
    its own contract — no-op when ruff is absent).
    """
    import shutil

    if shutil.which("ruff") is None:
        pytest.skip("ruff not on PATH; lint-on-write hook is best-effort")

    session = make_test_session(tmp_path)
    ctx = make_test_context(session)
    tool = make_write_file_tool()

    bad_python = "import os\n\nundefined_name\n"
    parsed = WriteFileInput(path="bad.py", content=bad_python)

    result = await tool.handler(parsed, ctx)
    # Mimic what the loop does — run post_result_hooks sequentially.
    for hook in tool.post_result_hooks:
        result = await hook(tool, parsed, result, ctx)

    assert "[ruff check bad.py]" in result.content


# ---------- edit_file --------------------------------------------------------


@pytest.mark.asyncio
async def test_edit_file_replaces_unique_match(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = make_test_session(tmp_path)
    ctx = make_test_context(session)
    tool = make_edit_file_tool()

    result = await tool.handler(
        EditFileInput(path="f.txt", old_string="beta", new_string="DELTA"),
        ctx,
    )

    assert result.is_error is False
    assert target.read_text() == "alpha\nDELTA\ngamma\n"


@pytest.mark.asyncio
async def test_edit_file_multiple_matches_rejected(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    target.write_text("foo\nfoo\nfoo\n", encoding="utf-8")
    session = make_test_session(tmp_path)
    ctx = make_test_context(session)
    tool = make_edit_file_tool()

    result = await tool.handler(
        EditFileInput(path="f.txt", old_string="foo", new_string="bar"),
        ctx,
    )

    assert result.is_error is True
    assert "matches 3 times" in result.content


@pytest.mark.asyncio
async def test_edit_file_no_matches_rejected(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    target.write_text("alpha\n", encoding="utf-8")
    session = make_test_session(tmp_path)
    ctx = make_test_context(session)
    tool = make_edit_file_tool()

    result = await tool.handler(
        EditFileInput(path="f.txt", old_string="nope", new_string="bar"),
        ctx,
    )

    assert result.is_error is True
    assert "not found" in result.content


@pytest.mark.asyncio
async def test_edit_file_lint_hook_registered() -> None:
    tool = make_edit_file_tool()
    from smai_agents.between_turn import lint_after_python_write

    assert lint_after_python_write in tool.post_result_hooks


# ---------- search ----------------------------------------------------------


@pytest.mark.asyncio
async def test_search_returns_matches(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello world\nfoo bar\nhello again\n")
    (tmp_path / "b.txt").write_text("nothing to see\nhello once\n")
    session = make_test_session(tmp_path)
    ctx = make_test_context(session)
    tool = make_search_tool()

    result = await tool.handler(SearchInput(pattern=r"hello"), ctx)

    assert result.is_error is False
    assert "a.txt:1:hello world" in result.content
    assert "a.txt:3:hello again" in result.content
    assert "b.txt:2:hello once" in result.content


@pytest.mark.asyncio
async def test_search_invalid_regex_returns_error(tmp_path: Path) -> None:
    session = make_test_session(tmp_path)
    ctx = make_test_context(session)
    tool = make_search_tool()

    result = await tool.handler(SearchInput(pattern="[invalid"), ctx)

    assert result.is_error is True
    assert "invalid regex" in result.content


@pytest.mark.asyncio
async def test_search_no_matches(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("nothing\n")
    session = make_test_session(tmp_path)
    ctx = make_test_context(session)
    tool = make_search_tool()

    result = await tool.handler(SearchInput(pattern=r"absent"), ctx)

    assert result.is_error is False
    assert result.content == "no matches found"


@pytest.mark.asyncio
async def test_search_caps_at_max_matches(tmp_path: Path) -> None:
    # Write more matches than MAX_SEARCH_MATCHES — verifies the cap.
    body = "\n".join("MATCH" for _ in range(MAX_SEARCH_MATCHES + 50))
    (tmp_path / "many.txt").write_text(body)
    session = make_test_session(tmp_path)
    ctx = make_test_context(session)
    tool = make_search_tool()

    result = await tool.handler(SearchInput(pattern=r"MATCH"), ctx)

    assert result.is_error is False
    assert "[truncated:" in result.content


@pytest.mark.asyncio
async def test_search_path_escape_rejected(tmp_path: Path) -> None:
    session = make_test_session(tmp_path)
    ctx = make_test_context(session)
    tool = make_search_tool()

    result = await tool.handler(
        SearchInput(pattern="x", path="../somewhere"),
        ctx,
    )

    assert result.is_error is True
    assert "outside workspace" in result.content


# ---------- list_files ------------------------------------------------------


@pytest.mark.asyncio
async def test_list_files_returns_relative_paths(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("x")
    session = make_test_session(tmp_path)
    ctx = make_test_context(session)
    tool = make_list_files_tool()

    result = await tool.handler(ListFilesInput(), ctx)

    assert result.is_error is False
    assert "a.txt" in result.content
    assert "sub/b.py" in result.content


@pytest.mark.asyncio
async def test_list_files_glob_filter(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.txt").write_text("x")
    (tmp_path / "c.py").write_text("x")
    session = make_test_session(tmp_path)
    ctx = make_test_context(session)
    tool = make_list_files_tool()

    result = await tool.handler(ListFilesInput(glob="*.py"), ctx)

    assert result.is_error is False
    assert "a.py" in result.content
    assert "c.py" in result.content
    assert "b.txt" not in result.content


@pytest.mark.asyncio
async def test_list_files_skips_pycache(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "noise.pyc").write_text("x")
    session = make_test_session(tmp_path)
    ctx = make_test_context(session)
    tool = make_list_files_tool()

    result = await tool.handler(ListFilesInput(), ctx)

    assert result.is_error is False
    assert "noise.pyc" not in result.content


@pytest.mark.asyncio
async def test_list_files_caps_at_max_entries(tmp_path: Path) -> None:
    for i in range(MAX_LIST_ENTRIES + 50):
        (tmp_path / f"f_{i:04d}.txt").write_text("x")
    session = make_test_session(tmp_path)
    ctx = make_test_context(session)
    tool = make_list_files_tool()

    result = await tool.handler(ListFilesInput(), ctx)

    assert result.is_error is False
    assert "[truncated:" in result.content


@pytest.mark.asyncio
async def test_list_files_non_recursive(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("x")
    session = make_test_session(tmp_path)
    ctx = make_test_context(session)
    tool = make_list_files_tool()

    result = await tool.handler(ListFilesInput(recursive=False), ctx)

    assert result.is_error is False
    assert "a.txt" in result.content
    assert "sub/b.txt" not in result.content


# ---------- Round 17: tool-description required-field drift guards ----------
#
# The round-16 re-test agent burned 4 supervisor nudges (turns 8-11) calling
# ``edit_file`` without the required ``path`` field. The error message alone
# was not enough to recover; the descriptions now preemptively flag the
# required fields. These drift-guard tests pin the call-out so a future
# terseness regression fails loudly here rather than burning agent turns.


def test_edit_file_description_flags_required_path() -> None:
    tool = make_edit_file_tool()
    desc = tool.description
    assert "REQUIRED" in desc
    assert "`path`" in desc
    assert "Pydantic validation error" in desc


def test_read_file_description_flags_required_path() -> None:
    tool = make_read_file_tool()
    desc = tool.description
    assert "REQUIRED" in desc
    assert "`path`" in desc
    assert "Pydantic validation error" in desc


def test_write_file_description_flags_required_fields() -> None:
    tool = make_write_file_tool()
    desc = tool.description
    assert "REQUIRED" in desc
    assert "`path`" in desc
    assert "`content`" in desc
    assert "Pydantic validation error" in desc


def test_search_description_flags_required_pattern() -> None:
    tool = make_search_tool()
    desc = tool.description
    assert "REQUIRED" in desc
    assert "`pattern`" in desc
    assert "Pydantic validation error" in desc
