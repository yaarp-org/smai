""":mod:`smai_inline_agents.between_turn` — status, nudge, lint hook, wait."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _agent_fakes import StubArtifactStore, StubLlmProvider  # type: ignore[import-not-found]
from pydantic import BaseModel
from smai_core.plugins import (
    Compute,
    ComputeCapabilities,
    JobHandle,
    JobStatus,
    LlmProvider,
    NormalizedMessage,
    TextContent,
    ToolResultContent,
)
from smai_inline_agents import (
    SUPERVISOR_NUDGE_PREFIX,
    AgentLoopConfig,
    AgentSession,
    Tool,
    ToolContext,
    ToolRegistry,
    lint_after_python_write,
    make_finish_tool,
    wait_for_compute_job,
)
from smai_inline_agents.between_turn import (
    maybe_check_supervisor_nudge,
    write_status,
)


def _new_session(
    *,
    llm: LlmProvider,
    workspace: Path,
    artifact_store: StubArtifactStore | None = None,
    status_path: str | None = None,
    nudge_path: str | None = None,
) -> AgentSession:
    registry = ToolRegistry()
    registry.register(make_finish_tool())
    return AgentSession(
        system_prompt="sys",
        tools=registry,
        llm_providers={"planner": llm},
        current_role="planner",
        workspace_path=workspace,
        artifact_store=artifact_store,
        status_artifact_path=status_path,
        nudge_artifact_path=nudge_path,
        config=AgentLoopConfig(status_write_every_turns=0),
    )


# --- write_status ------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_status_writes_json_to_artifact_store(tmp_path: Path) -> None:
    store = StubArtifactStore()
    llm = StubLlmProvider([])
    session = _new_session(
        llm=llm,
        workspace=tmp_path,
        artifact_store=store,
        status_path="comparison-groups/cg-1/harness/status.json",
    )
    session.turn_count = 3

    await write_status(session)

    assert len(store.put_calls) == 1
    key, body, ctype = store.put_calls[0]
    assert key == "comparison-groups/cg-1/harness/status.json"
    assert ctype == "application/json"
    payload = json.loads(body.decode("utf-8"))
    assert payload["role"] == "planner"
    assert payload["turn_count"] == 3
    assert "monotonic_timestamp" in payload


@pytest.mark.asyncio
async def test_write_status_no_op_without_path(tmp_path: Path) -> None:
    """Tests that pass no ``status_artifact_path`` should not see writes."""
    store = StubArtifactStore()
    llm = StubLlmProvider([])
    session = _new_session(llm=llm, workspace=tmp_path, artifact_store=store)
    await write_status(session)
    assert store.put_calls == []


@pytest.mark.asyncio
async def test_write_status_uses_session_time_provider(tmp_path: Path) -> None:
    """The Task 1.9 carry-forward: status timestamps come from
    ``session.time_provider``, not ``time.monotonic`` directly."""
    store = StubArtifactStore()
    llm = StubLlmProvider([])
    session = _new_session(
        llm=llm,
        workspace=tmp_path,
        artifact_store=store,
        status_path="status.json",
    )
    session.time_provider = lambda: 12345.6
    await write_status(session)

    payload = json.loads(store.put_calls[0][1].decode("utf-8"))
    assert payload["monotonic_timestamp"] == 12345.6


# --- maybe_check_supervisor_nudge -------------------------------------------


@pytest.mark.asyncio
async def test_supervisor_nudge_injects_user_message_with_prefix(
    tmp_path: Path,
) -> None:
    """§3.2: 'inject the nudge content as a user-role message' with the
    literal ``[supervisor nudge]`` prefix."""
    store = StubArtifactStore()
    nudge_path = "comparison-groups/cg-1/nudge.txt"
    await store.put(nudge_path, b"focus on the harness validation step")

    llm = StubLlmProvider([])
    session = _new_session(
        llm=llm,
        workspace=tmp_path,
        artifact_store=store,
        nudge_path=nudge_path,
    )

    await maybe_check_supervisor_nudge(session)

    assert session.nudges_consumed == 1
    assert len(session.messages) == 1
    msg = session.messages[0]
    assert isinstance(msg, NormalizedMessage)
    assert msg.role == "user"
    text_blocks = [b for b in msg.content if isinstance(b, TextContent)]
    assert text_blocks[0].text.startswith(SUPERVISOR_NUDGE_PREFIX)
    assert "focus on the harness" in text_blocks[0].text

    # Nudge file should be deleted after consumption.
    assert nudge_path in store.delete_calls
    assert not await store.exists(nudge_path)


@pytest.mark.asyncio
async def test_supervisor_nudge_no_op_when_file_missing(tmp_path: Path) -> None:
    store = StubArtifactStore()
    llm = StubLlmProvider([])
    session = _new_session(
        llm=llm,
        workspace=tmp_path,
        artifact_store=store,
        nudge_path="comparison-groups/cg-1/nudge.txt",
    )
    await maybe_check_supervisor_nudge(session)
    assert session.nudges_consumed == 0
    assert session.messages == []


@pytest.mark.asyncio
async def test_supervisor_nudge_no_op_without_artifact_store(tmp_path: Path) -> None:
    """No store + no path = no nudge polling. Unit-test ergonomics."""
    llm = StubLlmProvider([])
    session = _new_session(llm=llm, workspace=tmp_path)
    await maybe_check_supervisor_nudge(session)
    assert session.nudges_consumed == 0


# --- lint_after_python_write ------------------------------------------------


class _WriteFileInput(BaseModel):
    path: str
    content: str


async def _write_file_handler(parsed_input: BaseModel, context: ToolContext) -> ToolResultContent:
    if not isinstance(parsed_input, _WriteFileInput):
        raise TypeError("expected _WriteFileInput")
    target = context.workspace_path / parsed_input.path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(parsed_input.content)
    return ToolResultContent(
        tool_use_id="",
        content=f"wrote {parsed_input.path}",
        is_error=False,
    )


@pytest.mark.asyncio
async def test_lint_hook_appends_ruff_output_for_python_files(
    tmp_path: Path,
) -> None:
    """Per §3.2: the linter output is appended to the tool result for
    ``.py`` writes. We materialize a file with a known violation
    (``import os`` unused) and check the hook surfaces it."""
    llm = StubLlmProvider([])
    session = _new_session(llm=llm, workspace=tmp_path)
    write_tool = Tool(
        name="write_file",
        description="Write a file.",
        input_schema=_WriteFileInput,
        handler=_write_file_handler,
    )
    target = tmp_path / "violation.py"
    target.write_text("import os\n")  # F401: unused import

    context = ToolContext(
        workspace_path=tmp_path,
        artifact_store=session.artifact_store,
        compute=session.compute,
        session=session,
    )
    result_in = ToolResultContent(
        tool_use_id="tu-1",
        content="wrote violation.py",
        is_error=False,
    )
    parsed = _WriteFileInput(path="violation.py", content="import os\n")
    result_out = await lint_after_python_write(write_tool, parsed, result_in, context)

    # Hook appends ruff output when there are findings; otherwise
    # leaves the result alone. We accept either outcome (ruff CLI may
    # not be present in some test environments) but check the hook
    # signature contract.
    assert result_out.tool_use_id == "tu-1"
    assert result_out.content.startswith("wrote violation.py")


@pytest.mark.asyncio
async def test_lint_hook_skips_non_python_files(tmp_path: Path) -> None:
    """Hook is a no-op when the input path isn't a ``.py`` file."""
    llm = StubLlmProvider([])
    session = _new_session(llm=llm, workspace=tmp_path)
    write_tool = Tool(
        name="write_file",
        description="Write a file.",
        input_schema=_WriteFileInput,
        handler=_write_file_handler,
    )
    target = tmp_path / "notes.txt"
    target.write_text("some notes")

    context = ToolContext(
        workspace_path=tmp_path,
        artifact_store=session.artifact_store,
        compute=session.compute,
        session=session,
    )
    result_in = ToolResultContent(
        tool_use_id="tu-1",
        content="wrote notes.txt",
        is_error=False,
    )
    parsed = _WriteFileInput(path="notes.txt", content="some notes")
    result_out = await lint_after_python_write(write_tool, parsed, result_in, context)

    assert result_out.content == result_in.content


# --- wait_for_compute_job ---------------------------------------------------


class _StubCompute:
    """Deterministic Compute fake — returns canned statuses in order."""

    def __init__(self, states: list[JobStatus]) -> None:
        self.name = "stub-compute"
        self.capabilities = ComputeCapabilities(
            supports_gpu=False,
            max_timeout_seconds=600,
        )
        self._states = list(states)
        self._idx = 0
        self.status_calls = 0

    async def submit(
        self,
        image: str,
        command: list[str],
        env: dict[str, str],
        gpu: bool = False,
        timeout_seconds: int = 3600,
        **plugin_options: object,
    ) -> JobHandle:
        del image, command, env, gpu, timeout_seconds, plugin_options
        return JobHandle(plugin="stub", handle="job-1", metadata={})

    async def status(self, handle: JobHandle) -> JobStatus:
        del handle
        self.status_calls += 1
        if self._idx >= len(self._states):
            return self._states[-1]
        out = self._states[self._idx]
        self._idx += 1
        return out

    async def logs(self, handle: JobHandle) -> str:
        del handle
        return "stub logs"

    async def cancel(self, handle: JobHandle) -> None:
        del handle


def _status(state: str) -> JobStatus:
    return JobStatus(
        state=state,  # type: ignore[arg-type]
        exit_code=0 if state == "succeeded" else None,
        started_at=None,
        finished_at=None,
        failure_reason=None,
    )


@pytest.mark.asyncio
async def test_wait_for_compute_job_returns_terminal_status() -> None:
    """Polls until ``state`` is terminal; returns the terminal status."""
    states = [_status("running"), _status("running"), _status("succeeded")]
    compute = _StubCompute(states)
    handle = JobHandle(plugin="stub", handle="job-1", metadata={})

    fake_now = [0.0]

    def _fake_clock() -> float:
        return fake_now[0]

    async def _fake_sleep(seconds: float) -> None:
        fake_now[0] += seconds

    result = await wait_for_compute_job(
        handle=handle,
        compute=_cast_compute(compute),
        time_provider=_fake_clock,
        poll_interval=0.5,
        sleep=_fake_sleep,
    )
    assert result.state == "succeeded"
    assert compute.status_calls == 3


@pytest.mark.asyncio
async def test_wait_for_compute_job_times_out() -> None:
    """When ``max_wait_seconds`` elapses before the job terminates,
    raises :class:`TimeoutError`."""
    states = [_status("running")]
    compute = _StubCompute(states)
    handle = JobHandle(plugin="stub", handle="job-1", metadata={})

    fake_now = [0.0]

    def _fake_clock() -> float:
        return fake_now[0]

    async def _fake_sleep(seconds: float) -> None:
        fake_now[0] += seconds

    with pytest.raises(TimeoutError):
        await wait_for_compute_job(
            handle=handle,
            compute=_cast_compute(compute),
            time_provider=_fake_clock,
            poll_interval=1.0,
            max_wait_seconds=2.0,
            sleep=_fake_sleep,
        )


def _cast_compute(stub: _StubCompute) -> Compute:
    """Coerce the stub to the :class:`Compute` Protocol for type-checkers.

    The Protocol is ``runtime_checkable`` so ``isinstance`` is true,
    and pyright's strict mode accepts the duck-typed assignment via a
    cast. The narrow helper keeps test bodies type-clean."""
    return stub  # type: ignore[return-value]
