"""Per-step fake-LLM tests for :func:`_run_technique_body_generation_step`.

Step 7 acceptance criterion: assert the bundle is built correctly when
the technique-implementer body-generation step fires, the file gets
written to ``techniques/<technique_name>.py``, lint runs, and the
lint-retry budget exhaustion path returns a clean error.

The handler under test is a near-mirror of
:func:`smai_agent_runtime.harness_builder._main._run_baseline_step`
(both share the :class:`TechniqueBodyGenerationBundle` /
:class:`TechniqueBodyOutput` schemas — D7b). The differences are
``step_kind="technique"`` instead of ``"baseline"``, ``is_baseline=False``,
and the write target is the per-entry technique file instead of
``techniques/baseline.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _baseline_generation_fixtures import (  # type: ignore[import-not-found]
    stage_harness_api_manifest,
    stage_harness_source_module,
    stage_technique_contract,
)
from _body_generation_fixtures import (  # type: ignore[import-not-found]
    FakeAgentRunner,
    make_technique_body_output,
)
from _harness_builder_workspace_fixtures import (  # type: ignore[import-not-found]
    write_minimal_harness_workspace,
)
from smai_agent_runtime.harness_builder import _main as harness_builder_main
from smai_agent_runtime.technique_implementer import _main as technique_implementer_main
from smai_agent_runtime.technique_implementer._main import (
    _DispatchContext,
    _run_technique_body_generation_step,
)
from smai_agent_runtime.workflow.step_types import TechniqueImplementerBodyGenerationStep


def _make_context(workspace: Path) -> _DispatchContext:
    """Build the technique-implementer dispatch context the handler reads.

    Mirrors the harness-builder test pattern; both contracts come from
    the staged workspace (the baseline fixture stages both).
    """
    from smai_agent_runtime.harness_builder._main import (
        _load_contract,
        _load_technique_contract,
    )

    contract = _load_contract(workspace)
    assert contract is not None
    technique_contract = _load_technique_contract(workspace)
    assert technique_contract is not None
    return _DispatchContext(
        cg_id=technique_contract.body.parent_experiment_id,
        entry_id=technique_contract.body.entry_id,
        workspace=workspace,
        contract=contract,
        technique_contract=technique_contract,
        overrides=None,
    )


def _make_technique_step(
    *,
    technique_id: str = "tech-cutout",
    write_to_path: str = "techniques/tech-cutout.py",
) -> TechniqueImplementerBodyGenerationStep:
    return TechniqueImplementerBodyGenerationStep(
        technique_id=technique_id,
        write_to_path=write_to_path,
    )


def test_technique_body_step_writes_technique_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: fake agent returns a valid body, mini-orchestrator
    writes the file, lint passes, step.succeeded == True."""
    write_minimal_harness_workspace(tmp_path)
    stage_technique_contract(
        tmp_path,
        is_baseline=False,
        technique_id="tech-cutout",
        standard=True,
    )
    stage_harness_api_manifest(tmp_path, {"extension_points": []})
    fake = FakeAgentRunner(responses=[make_technique_body_output()])
    # Patch BOTH binding sites: the original (harness_builder._main) and
    # the technique_implementer's local re-import. The from-import in
    # technique_implementer/_main.py creates a separate module-namespace
    # binding the test's monkeypatch needs to update too.
    monkeypatch.setattr(harness_builder_main, "_run_agent_sync", fake)
    monkeypatch.setattr(technique_implementer_main, "_run_agent_sync", fake)

    outcome = _run_technique_body_generation_step(
        _make_technique_step(),
        index=0,
        context=_make_context(tmp_path),
    )

    assert outcome.succeeded is True
    written = tmp_path / "techniques" / "tech-cutout.py"
    assert written.exists()
    assert "def baseline" in written.read_text()  # FakeAgentRunner output stub
    assert len(fake.calls) == 1


def test_technique_body_step_bundle_carries_technique_step_kind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bundle's ``step_kind`` discriminator is ``"technique"`` (not
    ``"baseline"``). This is the only difference between the harness
    builder's baseline step and the technique implementer's body step
    at the prompt-rendering layer, so the per-step prompt's branching
    on ``step_kind`` depends on this value flowing through correctly.
    """
    write_minimal_harness_workspace(tmp_path)
    stage_technique_contract(
        tmp_path,
        is_baseline=False,
        technique_id="tech-cutout",
        standard=True,
    )
    stage_harness_api_manifest(tmp_path, {"extension_points": []})
    fake = FakeAgentRunner(responses=[make_technique_body_output()])
    # Patch BOTH binding sites: the original (harness_builder._main) and
    # the technique_implementer's local re-import. The from-import in
    # technique_implementer/_main.py creates a separate module-namespace
    # binding the test's monkeypatch needs to update too.
    monkeypatch.setattr(harness_builder_main, "_run_agent_sync", fake)
    monkeypatch.setattr(technique_implementer_main, "_run_agent_sync", fake)

    _run_technique_body_generation_step(
        _make_technique_step(),
        index=0,
        context=_make_context(tmp_path),
    )

    user_message = fake.calls[0].user_message
    # The bundle carries step_kind="technique" and is_baseline=False;
    # both surface in the rendered prompt's frontmatter.
    assert "step_kind: `technique`" in user_message
    assert "is_baseline: `False`" in user_message


def test_technique_body_step_reads_baseline_source_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``techniques/baseline.py`` exists in the workspace (the
    host-side dispatcher stages it for non-baseline entries), the
    bundle's ``baseline_source`` field carries the content and the
    rendered prompt surfaces it under the ``Baseline source`` heading.
    """
    write_minimal_harness_workspace(tmp_path)
    stage_technique_contract(
        tmp_path,
        is_baseline=False,
        technique_id="tech-cutout",
        standard=True,
    )
    stage_harness_api_manifest(tmp_path, {"extension_points": []})
    # Stage a baseline source file the dispatcher should read.
    techniques_dir = tmp_path / "techniques"
    techniques_dir.mkdir(parents=True, exist_ok=True)
    baseline_marker = '"""baseline module from harness builder"""\n'
    (techniques_dir / "baseline.py").write_text(baseline_marker)
    fake = FakeAgentRunner(responses=[make_technique_body_output()])
    # Patch BOTH binding sites: the original (harness_builder._main) and
    # the technique_implementer's local re-import. The from-import in
    # technique_implementer/_main.py creates a separate module-namespace
    # binding the test's monkeypatch needs to update too.
    monkeypatch.setattr(harness_builder_main, "_run_agent_sync", fake)
    monkeypatch.setattr(technique_implementer_main, "_run_agent_sync", fake)

    _run_technique_body_generation_step(
        _make_technique_step(),
        index=0,
        context=_make_context(tmp_path),
    )

    user_message = fake.calls[0].user_message
    assert "Baseline source" in user_message
    assert baseline_marker.strip() in user_message


def test_technique_body_step_lint_retry_budget_exhausted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ruff lint fails on every retry attempt (the agent keeps
    returning syntactically invalid output), the step returns a
    clean failure with ``error="lint-retry budget exhausted ..."`` so
    the outer workflow loop's fail-fast logic can surface the
    failure as a session.end ``failure`` event.
    """
    write_minimal_harness_workspace(tmp_path)
    stage_technique_contract(
        tmp_path,
        is_baseline=False,
        technique_id="tech-cutout",
        standard=True,
    )
    stage_harness_api_manifest(tmp_path, {"extension_points": []})
    # Stage harness source so the bundle assembles correctly.
    stage_harness_source_module(
        tmp_path,
        "harness/__init__.py",
        "from smai_runtime import HarnessComponents\n",
    )
    # Return a body that ruff will always reject (syntax error).
    bad_output = make_technique_body_output(
        technique_py_source="def baseline(:\n    return invalid syntax\n"
    )
    fake = FakeAgentRunner(responses=[bad_output, bad_output, bad_output, bad_output])
    # Patch BOTH binding sites: the original (harness_builder._main) and
    # the technique_implementer's local re-import. The from-import in
    # technique_implementer/_main.py creates a separate module-namespace
    # binding the test's monkeypatch needs to update too.
    monkeypatch.setattr(harness_builder_main, "_run_agent_sync", fake)
    monkeypatch.setattr(technique_implementer_main, "_run_agent_sync", fake)

    outcome = _run_technique_body_generation_step(
        _make_technique_step(),
        index=0,
        context=_make_context(tmp_path),
    )

    assert outcome.succeeded is False
    assert outcome.error is not None
    assert "lint-retry budget exhausted" in outcome.error
    # The retry budget is 3 (per _MAX_LINT_RETRIES), so 4 total calls
    # (attempt 0 + 3 retries) were made.
    assert len(fake.calls) == 4
