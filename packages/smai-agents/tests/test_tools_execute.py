"""Tests for the ``execute`` standard tool (Task 2.B2 / §12.2)."""

from __future__ import annotations

from pathlib import Path

import pytest
from _b2_helpers import make_test_context, make_test_session  # type: ignore[import-not-found]
from smai_agents.std_tools.execute import (
    EXECUTE_DEFAULT_TIMEOUT_SECONDS,
    EXECUTE_TOOL_NAME,
    ExecuteInput,
    make_execute_tool,
)


@pytest.mark.asyncio
async def test_execute_runs_command(tmp_path: Path) -> None:
    session = make_test_session(tmp_path)
    ctx = make_test_context(session)
    tool = make_execute_tool()

    result = await tool.handler(ExecuteInput(command="echo hello"), ctx)

    assert result.is_error is False
    assert "exit code: 0" in result.content
    assert "hello" in result.content


@pytest.mark.asyncio
async def test_execute_captures_nonzero_exit(tmp_path: Path) -> None:
    """Non-zero exit is NOT a tool error — agents read exit_code in body."""
    session = make_test_session(tmp_path)
    ctx = make_test_context(session)
    tool = make_execute_tool()

    result = await tool.handler(ExecuteInput(command="false"), ctx)

    assert result.is_error is False
    assert "exit code: 1" in result.content


@pytest.mark.asyncio
async def test_execute_runs_in_workspace_cwd(tmp_path: Path) -> None:
    (tmp_path / "marker").write_text("here")
    session = make_test_session(tmp_path)
    ctx = make_test_context(session)
    tool = make_execute_tool()

    result = await tool.handler(ExecuteInput(command="ls"), ctx)

    assert result.is_error is False
    assert "marker" in result.content


@pytest.mark.asyncio
async def test_execute_blocks_experiment_py_direct(tmp_path: Path) -> None:
    """DEC-021 / §12.2 guard — python experiment.py rejected."""
    session = make_test_session(tmp_path)
    ctx = make_test_context(session)
    tool = make_execute_tool()

    result = await tool.handler(
        ExecuteInput(command="python experiment.py --seed 0"),
        ctx,
    )

    assert result.is_error is True
    assert "run_experiment" in result.content


@pytest.mark.asyncio
async def test_execute_blocks_python3_dash_m_experiment(tmp_path: Path) -> None:
    session = make_test_session(tmp_path)
    ctx = make_test_context(session)
    tool = make_execute_tool()

    result = await tool.handler(
        ExecuteInput(command="python3 -m experiment"),
        ctx,
    )

    assert result.is_error is True


@pytest.mark.asyncio
async def test_execute_blocks_dot_slash_experiment(tmp_path: Path) -> None:
    session = make_test_session(tmp_path)
    ctx = make_test_context(session)
    tool = make_execute_tool()

    result = await tool.handler(
        ExecuteInput(command="./experiment.py --seed 1"),
        ctx,
    )

    assert result.is_error is True


@pytest.mark.asyncio
async def test_execute_allows_other_python_invocations(tmp_path: Path) -> None:
    """The guard is narrow — ``python -c …`` and other patterns pass."""
    session = make_test_session(tmp_path)
    ctx = make_test_context(session)
    tool = make_execute_tool()

    result = await tool.handler(
        ExecuteInput(command="python3 -c \"print('ok')\""),
        ctx,
    )

    assert result.is_error is False
    assert "ok" in result.content


@pytest.mark.asyncio
async def test_execute_truncates_long_output(tmp_path: Path) -> None:
    """Output longer than 200 lines / 5000 tokens is truncated.

    Builds a command that emits many lines and verifies the
    ``[output truncated …]`` marker appears.
    """
    session = make_test_session(tmp_path)
    ctx = make_test_context(session)
    tool = make_execute_tool()

    # 300 echoed lines triggers the 200-line cap.
    result = await tool.handler(
        ExecuteInput(command="for i in $(seq 1 300); do echo line$i; done"),
        ctx,
    )

    assert result.is_error is False
    assert "[output truncated" in result.content


@pytest.mark.asyncio
async def test_execute_timeout_kills_command(tmp_path: Path) -> None:
    session = make_test_session(tmp_path)
    ctx = make_test_context(session)
    tool = make_execute_tool()

    result = await tool.handler(
        ExecuteInput(command="sleep 30", timeout_seconds=1),
        ctx,
    )

    assert result.is_error is True
    assert "timeout" in result.content


@pytest.mark.asyncio
async def test_execute_default_timeout_is_120s() -> None:
    """The default timeout matches v1's settled value."""
    assert EXECUTE_DEFAULT_TIMEOUT_SECONDS == 120


def test_execute_tool_name_constant_matches() -> None:
    tool = make_execute_tool()
    assert tool.name == EXECUTE_TOOL_NAME
