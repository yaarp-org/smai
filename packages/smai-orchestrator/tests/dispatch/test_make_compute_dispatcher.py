"""Unit tests for ``make_compute_dispatcher`` (agent-refactor Step 2).

Covers:

* Call-order pinning: ``stage_workspace`` fires before ``submit`` and
  the staged :class:`WorkspaceHandle` is threaded through ``submit``'s
  ``workspace=`` plugin_option.
* Empty inputs: ``stage_workspace`` is skipped; ``submit`` runs without
  ``workspace=`` in plugin_options.
* The ``image_resolver`` / ``command_builder`` outputs (image, command,
  env, gpu, timeout) reach ``submit`` unchanged.
* Submit-time exceptions propagate to the engine's dispatch wrapper.
* The factory's value wins over caller-supplied ``workspace`` in
  ``plugin_options`` (round-18 / round-20 same-PR discipline).

The round-20 stderr-tail generalization in phase-1 is exercised by
:mod:`test_phase1_last_error_round20` (sibling file).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _compute_dispatcher_fakes import (  # type: ignore[import-not-found]
    RecordingCompute,
    make_dispatch_context,
    static_command_builder,
    static_image_resolver,
    static_workspace_inputs,
)
from smai_core.plugins import (
    ComputeUnavailable,
    JobHandle,
    WorkspaceHandle,
)
from smai_orchestrator.dispatch import (
    CommandSpec,
    WorkspaceInputs,
    WorkspaceOutputs,
    make_compute_dispatcher,
)


async def test_dispatcher_calls_stage_then_submit_when_inputs_present(
    tmp_path: Path,
) -> None:
    """With non-empty inputs, ``stage_workspace`` fires before ``submit``;
    the staged :class:`WorkspaceHandle` is threaded into ``submit``'s
    ``plugin_options['workspace']``.
    """
    workspace_dir = tmp_path / "ws"
    workspace_dir.mkdir()
    handle = JobHandle(plugin="recording-compute", handle="job-abc")
    compute = RecordingCompute(submit_handle=handle)

    dispatcher = make_compute_dispatcher(
        role="harness_builder",
        image_resolver=static_image_resolver("smai-agent-runtime:dev"),
        command_builder=static_command_builder(
            CommandSpec(
                command=["python", "-m", "smai_agent_runtime"],
                env={"SMAI_CG_ID": "cg-1"},
            )
        ),
        inputs=static_workspace_inputs(workspace_dir),
        outputs=WorkspaceOutputs.empty(),
    )
    outcome = await dispatcher.handler(await make_dispatch_context(compute=compute))

    assert outcome.error is None
    assert outcome.submitted_handles == [handle]

    kinds = [c.kind for c in compute.calls]
    assert kinds == ["stage_workspace", "submit"], kinds

    stage_call = compute.calls[0]
    assert stage_call.payload["local_path"] == workspace_dir

    submit_call = compute.calls[1]
    assert submit_call.payload["image"] == "smai-agent-runtime:dev"
    assert submit_call.payload["command"] == ["python", "-m", "smai_agent_runtime"]
    assert submit_call.payload["env"] == {"SMAI_CG_ID": "cg-1"}
    workspace_arg = submit_call.payload["plugin_options"]["workspace"]
    assert isinstance(workspace_arg, WorkspaceHandle)
    assert workspace_arg.handle == str(workspace_dir)


async def test_dispatcher_skips_stage_when_inputs_empty() -> None:
    """Empty inputs: ``stage_workspace`` not called; ``submit`` receives
    no ``workspace=`` in plugin_options. Seed-run dispatcher migration
    relies on this — the runtime image is self-contained and no
    workspace is pushed at submit time.
    """
    handle = JobHandle(plugin="recording-compute", handle="job-seed")
    compute = RecordingCompute(submit_handle=handle)

    dispatcher = make_compute_dispatcher(
        role="seed_run",
        image_resolver=static_image_resolver("smai-runtime-cpu:dev"),
        command_builder=static_command_builder(
            CommandSpec(
                command=["python", "-m", "smai_runtime.runner"],
                env={"SMAI_SEED": "0"},
                gpu=False,
                timeout_seconds=60,
            )
        ),
        inputs=WorkspaceInputs.empty(),
        outputs=WorkspaceOutputs.empty(),
    )
    outcome = await dispatcher.handler(await make_dispatch_context(compute=compute))

    assert outcome.error is None
    assert outcome.submitted_handles == [handle]

    kinds = [c.kind for c in compute.calls]
    assert kinds == ["submit"], kinds

    submit_call = compute.calls[0]
    assert "workspace" not in submit_call.payload["plugin_options"]
    assert submit_call.payload["gpu"] is False
    assert submit_call.payload["timeout_seconds"] == 60


async def test_dispatcher_skips_stage_when_resolver_returns_none() -> None:
    """Inputs has a resolver but it returns ``None``: same as empty
    (per-dispatch opt-out — useful when the workspace doesn't exist yet).
    """
    compute = RecordingCompute()

    async def _none_resolver(ctx):  # type: ignore[no-untyped-def]
        del ctx
        return None

    dispatcher = make_compute_dispatcher(
        role="harness_builder",
        image_resolver=static_image_resolver("img"),
        command_builder=static_command_builder(CommandSpec(command=["true"], env={})),
        inputs=WorkspaceInputs(resolver=_none_resolver),
        outputs=WorkspaceOutputs.empty(),
    )
    await dispatcher.handler(await make_dispatch_context(compute=compute))

    kinds = [c.kind for c in compute.calls]
    assert kinds == ["submit"], kinds


async def test_dispatcher_factory_workspace_overrides_caller_plugin_options(
    tmp_path: Path,
) -> None:
    """If the command_builder's :class:`CommandSpec` includes a
    ``workspace`` plugin_option, the factory's staged handle wins.

    Round-18 / round-20 same-PR discipline (compute_dispatch_decisions.md
    §3): the dispatcher owns workspace threading; agent-supplied (or
    spec-supplied) options must not override session-internal
    correctness.
    """
    workspace_dir = tmp_path / "ws"
    workspace_dir.mkdir()
    compute = RecordingCompute()

    dispatcher = make_compute_dispatcher(
        role="harness_builder",
        image_resolver=static_image_resolver("img"),
        command_builder=static_command_builder(
            CommandSpec(
                command=["true"],
                env={},
                plugin_options={"workspace": "caller-supplied-junk"},
            )
        ),
        inputs=static_workspace_inputs(workspace_dir),
        outputs=WorkspaceOutputs.empty(),
    )
    await dispatcher.handler(await make_dispatch_context(compute=compute))

    submit_call = next(c for c in compute.calls if c.kind == "submit")
    workspace_arg = submit_call.payload["plugin_options"]["workspace"]
    assert isinstance(workspace_arg, WorkspaceHandle), workspace_arg
    assert workspace_arg.handle == str(workspace_dir)


async def test_dispatcher_propagates_submit_failure() -> None:
    """``Compute.submit`` raising propagates the exception. The engine's
    ``_handle_dispatch_failure`` catches it and forward-rolls-back; the
    factory itself does not swallow.
    """
    compute = RecordingCompute(submit_raises=ComputeUnavailable("substrate offline"))

    dispatcher = make_compute_dispatcher(
        role="seed_run",
        image_resolver=static_image_resolver("img"),
        command_builder=static_command_builder(CommandSpec(command=["true"], env={})),
        inputs=WorkspaceInputs.empty(),
        outputs=WorkspaceOutputs.empty(),
    )
    with pytest.raises(ComputeUnavailable, match="substrate offline"):
        await dispatcher.handler(await make_dispatch_context(compute=compute))


async def test_format_stderr_tail_short_input_returns_unchanged() -> None:
    """``format_stderr_tail`` on short input is the identity."""
    from smai_orchestrator.dispatch import format_stderr_tail

    short = "line 1\nline 2\nline 3"
    assert format_stderr_tail(short) == short


async def test_format_stderr_tail_truncates_long_input() -> None:
    """Long inputs are truncated to the last ``max_lines`` lines with a
    ``(truncated, N lines hidden)`` prefix.
    """
    from smai_orchestrator.dispatch import format_stderr_tail

    lines = [f"line {i}" for i in range(200)]
    tail = format_stderr_tail("\n".join(lines), max_lines=50)
    tail_lines = tail.splitlines()
    # 1 truncation-marker line + 50 content lines.
    assert len(tail_lines) == 51
    assert "truncated" in tail_lines[0]
    assert "150 lines hidden" in tail_lines[0]
    assert tail_lines[1] == "line 150"
    assert tail_lines[-1] == "line 199"


async def test_format_stderr_tail_empty_input_returns_empty() -> None:
    from smai_orchestrator.dispatch import format_stderr_tail

    assert format_stderr_tail("") == ""
