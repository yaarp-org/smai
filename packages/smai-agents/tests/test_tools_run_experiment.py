"""Tests for the ``run_experiment`` structured tool (Task 2.B2 / §12.3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _b2_fakes import (  # type: ignore[import-not-found]
    FakeClock,
    FakeCompute,
    make_clock_advancing_sleep,
)
from _b2_helpers import make_test_context, make_test_session  # type: ignore[import-not-found]
from smai_agents.std_tools.run_experiment import (
    MAX_VALIDATION_EPOCHS,
    MAX_VALIDATION_SUBSET,
    RUN_EXPERIMENT_TOOL_NAME,
    RunExperimentInput,
    RunExperimentResult,
    make_run_experiment_tool,
)
from smai_core.plugins import (
    ComputeUnavailable,
    JobImageInvalid,
    JobStatus,
)


def _terminal_succeeded() -> JobStatus:
    return JobStatus(
        state="succeeded",
        exit_code=0,
        started_at="2026-04-28T10:00:00Z",
        finished_at="2026-04-28T10:00:30Z",
        failure_reason=None,
    )


def _terminal_failed(reason: str = "OOM") -> JobStatus:
    return JobStatus(
        state="failed",
        exit_code=1,
        started_at="2026-04-28T10:00:00Z",
        finished_at="2026-04-28T10:00:05Z",
        failure_reason=reason,
    )


# ---------- happy path ------------------------------------------------------


@pytest.mark.asyncio
async def test_run_experiment_succeeds_and_harvests_metrics(tmp_path: Path) -> None:
    """End-to-end: submit → wait → harvest metrics.json → return result."""
    metrics_payload = {"accuracy": 0.85, "loss": 0.42}

    async def drop_metrics(workspace: Path) -> None:
        (workspace / "metrics.json").write_text(json.dumps(metrics_payload))

    fake_compute = FakeCompute(
        status_queue=[_terminal_succeeded()],
        on_submit=drop_metrics,
    )
    fake_compute.set_workspace(tmp_path)
    clock = FakeClock()
    session = make_test_session(
        tmp_path,
        compute=fake_compute,
        time_provider=clock.now,
    )
    ctx = make_test_context(session)
    tool = make_run_experiment_tool(
        image="smai-runtime:test",
        gpu=True,
        wait_helper=_make_immediate_wait_helper(clock),
    )

    result = await tool.handler(
        RunExperimentInput(technique="cutout", seed=42, epochs=2, subset_fraction=0.1),
        ctx,
    )

    assert result.is_error is False
    parsed = RunExperimentResult.model_validate_json(result.content)
    assert parsed.completed is True
    assert parsed.metrics == metrics_payload
    assert parsed.run_status["state"] == "succeeded"
    assert parsed.failure_reason is None

    # Submit was called with the correct image / gpu / command.
    assert len(fake_compute.submit_calls) == 1
    call = fake_compute.submit_calls[0]
    assert call["image"] == "smai-runtime:test"
    assert call["gpu"] is True
    cmd = call["command"]
    assert "experiment.py" in cmd
    assert "--technique" in cmd and "cutout" in cmd
    assert "--seed" in cmd and "42" in cmd
    assert "--epochs" in cmd and "2" in cmd


@pytest.mark.asyncio
async def test_run_experiment_baseline_omits_technique(tmp_path: Path) -> None:
    """``technique=None`` (additive baseline) skips the --technique flag."""

    async def drop_metrics(workspace: Path) -> None:
        (workspace / "metrics.json").write_text(json.dumps({"accuracy": 0.5}))

    fake_compute = FakeCompute(
        status_queue=[_terminal_succeeded()],
        on_submit=drop_metrics,
    )
    fake_compute.set_workspace(tmp_path)
    clock = FakeClock()
    session = make_test_session(
        tmp_path,
        compute=fake_compute,
        time_provider=clock.now,
    )
    ctx = make_test_context(session)
    tool = make_run_experiment_tool(
        image="smai-runtime:test",
        gpu=True,
        wait_helper=_make_immediate_wait_helper(clock),
    )

    result = await tool.handler(
        RunExperimentInput(technique=None, seed=1, epochs=1, subset_fraction=0.05),
        ctx,
    )

    assert result.is_error is False
    cmd = fake_compute.submit_calls[0]["command"]
    assert "--technique" not in cmd


# ---------- hard caps -------------------------------------------------------


@pytest.mark.asyncio
async def test_run_experiment_caps_epochs_above_max(tmp_path: Path) -> None:
    """Per DEC-021 / §12.3: epochs > MAX_VALIDATION_EPOCHS is silently
    clamped at the tool layer."""

    async def drop_metrics(workspace: Path) -> None:
        (workspace / "metrics.json").write_text(json.dumps({"accuracy": 1.0}))

    fake_compute = FakeCompute(
        status_queue=[_terminal_succeeded()],
        on_submit=drop_metrics,
    )
    fake_compute.set_workspace(tmp_path)
    clock = FakeClock()
    session = make_test_session(
        tmp_path,
        compute=fake_compute,
        time_provider=clock.now,
    )
    ctx = make_test_context(session)
    tool = make_run_experiment_tool(
        image="smai-runtime:test",
        gpu=True,
        wait_helper=_make_immediate_wait_helper(clock),
    )

    result = await tool.handler(
        RunExperimentInput(
            technique="t",
            seed=1,
            epochs=100,
            subset_fraction=0.05,
        ),
        ctx,
    )

    assert result.is_error is False
    # Cap notice prepended.
    assert "[capped] epochs capped from 100 to" in result.content
    # The actual command sent to compute uses the capped value.
    cmd = fake_compute.submit_calls[0]["command"]
    epochs_idx = cmd.index("--epochs")
    assert cmd[epochs_idx + 1] == str(MAX_VALIDATION_EPOCHS)


@pytest.mark.asyncio
async def test_run_experiment_caps_subset_above_max(tmp_path: Path) -> None:
    async def drop_metrics(workspace: Path) -> None:
        (workspace / "metrics.json").write_text(json.dumps({"accuracy": 1.0}))

    fake_compute = FakeCompute(
        status_queue=[_terminal_succeeded()],
        on_submit=drop_metrics,
    )
    fake_compute.set_workspace(tmp_path)
    clock = FakeClock()
    session = make_test_session(
        tmp_path,
        compute=fake_compute,
        time_provider=clock.now,
    )
    ctx = make_test_context(session)
    tool = make_run_experiment_tool(
        image="smai-runtime:test",
        gpu=True,
        wait_helper=_make_immediate_wait_helper(clock),
    )

    result = await tool.handler(
        RunExperimentInput(
            technique="t",
            seed=1,
            epochs=1,
            subset_fraction=0.9,
        ),
        ctx,
    )

    assert result.is_error is False
    assert "[capped] subset_fraction capped from 0.9 to" in result.content
    cmd = fake_compute.submit_calls[0]["command"]
    subset_idx = cmd.index("--subset")
    assert cmd[subset_idx + 1] == str(MAX_VALIDATION_SUBSET)


# ---------- failure paths ---------------------------------------------------


@pytest.mark.asyncio
async def test_run_experiment_handles_submit_failure(tmp_path: Path) -> None:
    """JobImageInvalid → completed=False with failure_reason populated."""
    fake_compute = FakeCompute(
        submit_should_raise=JobImageInvalid("bad", "no such image"),
    )
    clock = FakeClock()
    session = make_test_session(
        tmp_path,
        compute=fake_compute,
        time_provider=clock.now,
    )
    ctx = make_test_context(session)
    tool = make_run_experiment_tool(
        image="bad",
        gpu=True,
        wait_helper=_make_immediate_wait_helper(clock),
    )

    result = await tool.handler(
        RunExperimentInput(technique="t", seed=1),
        ctx,
    )

    assert result.is_error is True
    parsed = RunExperimentResult.model_validate_json(result.content)
    assert parsed.completed is False
    assert parsed.metrics is None
    assert parsed.failure_reason is not None
    assert "JobImageInvalid" in parsed.failure_reason


@pytest.mark.asyncio
async def test_run_experiment_handles_compute_unavailable(tmp_path: Path) -> None:
    fake_compute = FakeCompute(submit_should_raise=ComputeUnavailable("outage"))
    clock = FakeClock()
    session = make_test_session(
        tmp_path,
        compute=fake_compute,
        time_provider=clock.now,
    )
    ctx = make_test_context(session)
    tool = make_run_experiment_tool(
        image="x",
        gpu=True,
        wait_helper=_make_immediate_wait_helper(clock),
    )

    result = await tool.handler(RunExperimentInput(technique="t", seed=1), ctx)

    assert result.is_error is True
    parsed = RunExperimentResult.model_validate_json(result.content)
    assert parsed.completed is False
    assert "ComputeUnavailable" in (parsed.failure_reason or "")


@pytest.mark.asyncio
async def test_run_experiment_failed_state_returns_failure(tmp_path: Path) -> None:
    """Substrate state=failed → completed=False with substrate's reason."""
    fake_compute = FakeCompute(status_queue=[_terminal_failed("CUDA OOM")])
    fake_compute.set_workspace(tmp_path)
    clock = FakeClock()
    session = make_test_session(
        tmp_path,
        compute=fake_compute,
        time_provider=clock.now,
    )
    ctx = make_test_context(session)
    tool = make_run_experiment_tool(
        image="x",
        gpu=True,
        wait_helper=_make_immediate_wait_helper(clock),
    )

    result = await tool.handler(RunExperimentInput(technique="t", seed=1), ctx)

    assert result.is_error is True
    parsed = RunExperimentResult.model_validate_json(result.content)
    assert parsed.completed is False
    assert parsed.metrics is None
    assert "CUDA OOM" in (parsed.failure_reason or "")


@pytest.mark.asyncio
async def test_run_experiment_no_compute_in_session(tmp_path: Path) -> None:
    """Tool requires Compute on session — explicit error if absent."""
    session = make_test_session(tmp_path)  # no compute
    ctx = make_test_context(session)
    tool = make_run_experiment_tool(image="x", gpu=True)

    result = await tool.handler(RunExperimentInput(technique="t", seed=1), ctx)

    assert result.is_error is True
    assert "Compute plugin" in result.content


# ---------- harvest edge cases ----------------------------------------------


@pytest.mark.asyncio
async def test_run_experiment_succeeds_but_no_metrics(tmp_path: Path) -> None:
    """State=succeeded but metrics.json missing → completed=False."""
    fake_compute = FakeCompute(status_queue=[_terminal_succeeded()])
    fake_compute.set_workspace(tmp_path)
    clock = FakeClock()
    session = make_test_session(
        tmp_path,
        compute=fake_compute,
        time_provider=clock.now,
    )
    ctx = make_test_context(session)
    tool = make_run_experiment_tool(
        image="x",
        gpu=True,
        wait_helper=_make_immediate_wait_helper(clock),
    )

    result = await tool.handler(RunExperimentInput(technique="t", seed=1), ctx)

    assert result.is_error is True
    parsed = RunExperimentResult.model_validate_json(result.content)
    assert parsed.completed is False
    assert parsed.metrics is None
    assert "metrics.json" in (parsed.failure_reason or "")


@pytest.mark.asyncio
async def test_run_experiment_harvests_validation_results_on_failure(
    tmp_path: Path,
) -> None:
    """Failed run with validation_results.json → snippet in failure_reason."""

    async def drop_validation(workspace: Path) -> None:
        (workspace / "validation_results.json").write_text(
            json.dumps({"loss_decreased": False, "error": "diverged"})
        )

    fake_compute = FakeCompute(
        status_queue=[_terminal_failed("loss diverged")],
        on_submit=drop_validation,
    )
    fake_compute.set_workspace(tmp_path)
    clock = FakeClock()
    session = make_test_session(
        tmp_path,
        compute=fake_compute,
        time_provider=clock.now,
    )
    ctx = make_test_context(session)
    tool = make_run_experiment_tool(
        image="x",
        gpu=True,
        wait_helper=_make_immediate_wait_helper(clock),
    )

    result = await tool.handler(RunExperimentInput(technique="t", seed=1), ctx)

    parsed = RunExperimentResult.model_validate_json(result.content)
    assert parsed.completed is False
    assert "diverged" in (parsed.failure_reason or "")


@pytest.mark.asyncio
async def test_run_experiment_metrics_with_invalid_value_treated_as_missing(
    tmp_path: Path,
) -> None:
    """Non-numeric metrics.json values → metrics=None (failure)."""

    async def drop_bad_metrics(workspace: Path) -> None:
        (workspace / "metrics.json").write_text(json.dumps({"accuracy": "not_a_number"}))

    fake_compute = FakeCompute(
        status_queue=[_terminal_succeeded()],
        on_submit=drop_bad_metrics,
    )
    fake_compute.set_workspace(tmp_path)
    clock = FakeClock()
    session = make_test_session(
        tmp_path,
        compute=fake_compute,
        time_provider=clock.now,
    )
    ctx = make_test_context(session)
    tool = make_run_experiment_tool(
        image="x",
        gpu=True,
        wait_helper=_make_immediate_wait_helper(clock),
    )

    result = await tool.handler(RunExperimentInput(technique="t", seed=1), ctx)

    parsed = RunExperimentResult.model_validate_json(result.content)
    assert parsed.completed is False
    assert parsed.metrics is None


def test_run_experiment_tool_name_matches() -> None:
    tool = make_run_experiment_tool(image="x", gpu=True)
    assert tool.name == RUN_EXPERIMENT_TOOL_NAME


def _make_immediate_wait_helper(clock: FakeClock):
    """Build a wait helper that uses the fake clock + advance-on-sleep
    semantics so the wait loop terminates deterministically.

    Uses the real :func:`smai_agents.between_turn.wait_for_compute_job`
    so the test exercises the real wait-loop logic, just with a fake
    clock + non-blocking sleep.
    """
    from smai_agents.between_turn import wait_for_compute_job

    advancing_sleep = make_clock_advancing_sleep(clock)

    async def _wait(
        *, handle, compute, time_provider, poll_interval=0.0, max_wait_seconds=None, sleep=None
    ):
        del sleep, poll_interval  # we control these
        return await wait_for_compute_job(
            handle=handle,
            compute=compute,
            time_provider=time_provider,
            poll_interval=0.001,
            max_wait_seconds=max_wait_seconds,
            sleep=advancing_sleep,
        )

    return _wait


# === Round 17: tool-description required-field drift guard ===================
#
# See ``test_tools_files.py`` for the cluster note. ``run_experiment`` has
# one required field, ``seed``; the rest of the input shape (technique /
# epochs / subset_fraction) has defaults.


def test_run_experiment_description_flags_required_seed() -> None:
    tool = make_run_experiment_tool(image="smai-runtime:test", gpu=True)
    desc = tool.description
    assert "REQUIRED" in desc
    assert "`seed`" in desc
    assert "Pydantic validation error" in desc


# === Round 18: workspace mount always flows to compute.submit ================
#
# The agent writes experiment.py / harness/ / techniques/ to the host
# workspace; the container that runs experiment.py must see the same
# path under /workspace via a bind-mount. The tool always forwards
# ``workspace=str(context.workspace_path)`` to ``Compute.submit`` via
# plugin_options, and that value cannot be overridden by anything the
# factory-time caller (or, by extension, the agent) supplies in
# ``plugin_options``.


@pytest.mark.asyncio
async def test_run_experiment_forwards_workspace_to_compute_submit(
    tmp_path: Path,
) -> None:
    async def drop_metrics(workspace: Path) -> None:
        (workspace / "metrics.json").write_text(json.dumps({"accuracy": 0.5}))

    fake_compute = FakeCompute(
        status_queue=[_terminal_succeeded()],
        on_submit=drop_metrics,
    )
    fake_compute.set_workspace(tmp_path)
    clock = FakeClock()
    session = make_test_session(
        tmp_path,
        compute=fake_compute,
        time_provider=clock.now,
    )
    ctx = make_test_context(session)
    tool = make_run_experiment_tool(
        image="smai-runtime:test",
        gpu=True,
        wait_helper=_make_immediate_wait_helper(clock),
    )

    await tool.handler(RunExperimentInput(technique=None, seed=1), ctx)

    assert len(fake_compute.submit_calls) == 1
    submitted = fake_compute.submit_calls[0]["plugin_options"]
    assert submitted["workspace"] == str(tmp_path)


@pytest.mark.asyncio
async def test_run_experiment_workspace_not_overridable_by_plugin_options(
    tmp_path: Path,
) -> None:
    """A caller-supplied ``workspace`` in plugin_options cannot redirect
    the container's bind-mount away from the agent's workspace. The
    tool's ``context.workspace_path`` always wins."""

    async def drop_metrics(workspace: Path) -> None:
        (workspace / "metrics.json").write_text(json.dumps({"accuracy": 0.5}))

    fake_compute = FakeCompute(
        status_queue=[_terminal_succeeded()],
        on_submit=drop_metrics,
    )
    fake_compute.set_workspace(tmp_path)
    clock = FakeClock()
    session = make_test_session(
        tmp_path,
        compute=fake_compute,
        time_provider=clock.now,
    )
    ctx = make_test_context(session)
    tool = make_run_experiment_tool(
        image="smai-runtime:test",
        gpu=True,
        plugin_options={"workspace": "/some/wrong/path"},
        wait_helper=_make_immediate_wait_helper(clock),
    )

    await tool.handler(RunExperimentInput(technique=None, seed=1), ctx)

    submitted = fake_compute.submit_calls[0]["plugin_options"]
    assert submitted["workspace"] == str(tmp_path)
    assert submitted["workspace"] != "/some/wrong/path"


# === Round 20: --mode validation always present ==============================
#
# The runtime defaults --mode to "full"; full mode requires the
# HarnessAPIManifest at contracts/harness_api_manifest.json. But the
# harness builder runs validation BEFORE emitting the manifest (manifest
# is the OUTCOME of a passing validation per 04-agents.md §9 step 5), so
# the only way the agent's run_experiment call can succeed against the
# runtime's contract loader is with --mode validation, which lets the
# runtime synthesize a stub manifest. The tool always passes the flag
# (the agent has no business overriding it — this tool's whole job is
# the in-loop validation check, never the orchestrator's seed-run path).


@pytest.mark.asyncio
async def test_run_experiment_command_includes_mode_validation(
    tmp_path: Path,
) -> None:
    async def drop_metrics(workspace: Path) -> None:
        (workspace / "metrics.json").write_text(json.dumps({"accuracy": 0.5}))

    fake_compute = FakeCompute(
        status_queue=[_terminal_succeeded()],
        on_submit=drop_metrics,
    )
    fake_compute.set_workspace(tmp_path)
    clock = FakeClock()
    session = make_test_session(
        tmp_path,
        compute=fake_compute,
        time_provider=clock.now,
    )
    ctx = make_test_context(session)
    tool = make_run_experiment_tool(
        image="smai-runtime:test",
        gpu=False,
        wait_helper=_make_immediate_wait_helper(clock),
    )

    await tool.handler(RunExperimentInput(technique="cutout", seed=7), ctx)

    cmd = fake_compute.submit_calls[0]["command"]
    assert "--mode" in cmd
    mode_idx = cmd.index("--mode")
    assert cmd[mode_idx + 1] == "validation"


@pytest.mark.asyncio
async def test_run_experiment_command_includes_mode_validation_for_baseline(
    tmp_path: Path,
) -> None:
    """Baseline (``technique=None``) still carries --mode validation —
    the flag is unconditional, NOT tied to the --technique branch.
    """

    async def drop_metrics(workspace: Path) -> None:
        (workspace / "metrics.json").write_text(json.dumps({"accuracy": 0.5}))

    fake_compute = FakeCompute(
        status_queue=[_terminal_succeeded()],
        on_submit=drop_metrics,
    )
    fake_compute.set_workspace(tmp_path)
    clock = FakeClock()
    session = make_test_session(
        tmp_path,
        compute=fake_compute,
        time_provider=clock.now,
    )
    ctx = make_test_context(session)
    tool = make_run_experiment_tool(
        image="smai-runtime:test",
        gpu=False,
        wait_helper=_make_immediate_wait_helper(clock),
    )

    await tool.handler(RunExperimentInput(technique=None, seed=1), ctx)

    cmd = fake_compute.submit_calls[0]["command"]
    assert "--mode" in cmd
    assert cmd[cmd.index("--mode") + 1] == "validation"
    assert "--technique" not in cmd
