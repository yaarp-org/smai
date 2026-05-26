"""Per-step fake-LLM tests for :func:`_run_baseline_step`.

Sub-PR C1 acceptance criterion: assert the bundle is built correctly
across at least two :data:`GroundingContext` variants, the file gets
written, lint runs, retry-path works, budget exhaustion raises a clean
error.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _baseline_generation_fixtures import (  # type: ignore[import-not-found]
    PaperFidelityAnchor,
    ProposalFidelityAnchor,
    ReviewerAttestedFidelityAnchor,
    stage_grounding_file,
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
from smai_agent_runtime.harness_builder._main import (
    _DispatchContext,
    _load_contract,
    _load_technique_contract,
    _run_baseline_step,
)
from smai_agent_runtime.workflow.step_types import BaselineGenerationStep


def _make_context(workspace: Path) -> _DispatchContext:
    contract = _load_contract(workspace)
    assert contract is not None
    return _DispatchContext(
        cg_id="cg-test-baseline",
        workspace=workspace,
        contract=contract,
        technique_contract=_load_technique_contract(workspace),
        overrides=None,
    )


def _make_baseline_step(*, factor_type: str = "substitutive") -> BaselineGenerationStep:
    return BaselineGenerationStep(
        factor_type=factor_type,  # type: ignore[arg-type]
        baseline_technique_id="tech-baseline",
        write_to_path="techniques/baseline.py",
    )


def test_baseline_step_standard_grounding_writes_baseline_py(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No staged grounding artifact, no fidelity anchor on the technique
    contract -> the dispatcher falls back to
    :class:`StandardLibraryGrounding` and the agent receives a
    ``standard`` discriminator in the rendered prompt."""
    write_minimal_harness_workspace(tmp_path)
    stage_technique_contract(tmp_path)
    fake = FakeAgentRunner(responses=[make_technique_body_output()])
    monkeypatch.setattr(harness_builder_main, "_run_agent_sync", fake)

    outcome = _run_baseline_step(
        _make_baseline_step(),
        index=0,
        context=_make_context(tmp_path),
    )

    assert outcome.succeeded is True
    assert (tmp_path / "techniques" / "baseline.py").exists()
    assert "def baseline" in (tmp_path / "techniques" / "baseline.py").read_text()
    assert len(fake.calls) == 1
    user_message = fake.calls[0].user_message
    # The bundle carries step_kind="baseline" and the standard grounding
    # variant; both surface in the rendered prompt.
    assert "step_kind: `baseline`" in user_message
    assert "kind=standard" in user_message


def test_baseline_step_reviewer_attested_grounding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``ReviewerAttestedFidelityAnchor`` on the technique contract
    flows ``ReviewerAttestedGrounding`` into the bundle inline."""
    write_minimal_harness_workspace(tmp_path)
    anchor = ReviewerAttestedFidelityAnchor(
        spec_text="The baseline is a single-layer perceptron with no activation.",
        attested_by="senior-reviewer@example.com",
    )
    stage_technique_contract(tmp_path, fidelity_anchor=anchor)
    fake = FakeAgentRunner(responses=[make_technique_body_output()])
    monkeypatch.setattr(harness_builder_main, "_run_agent_sync", fake)

    outcome = _run_baseline_step(
        _make_baseline_step(),
        index=0,
        context=_make_context(tmp_path),
    )

    assert outcome.succeeded is True
    user_message = fake.calls[0].user_message
    assert "kind=reviewer_attested" in user_message
    assert "single-layer perceptron" in user_message
    assert "senior-reviewer@example.com" in user_message


def test_baseline_step_staged_grounding_file_wins_over_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The host-staged ``grounding/baseline_grounding.json`` overrides
    whatever the technique contract carries (sub-PR D's host-side
    dispatcher writes the staged file from upstream ArtifactStore
    lookups; the staged variant is canonical)."""
    write_minimal_harness_workspace(tmp_path)
    # Contract says proposal anchor; staged file says paper extract.
    stage_technique_contract(
        tmp_path,
        fidelity_anchor=ProposalFidelityAnchor(proposal_id="prop-123"),
    )
    stage_grounding_file(
        tmp_path,
        {
            "kind": "paper_extract",
            "arxiv_id": "2401.12345",
            "technique_id": "tech-baseline",
            "method_extraction": "Algorithm 1: do this and that.",
            "implementability": "high",
        },
    )
    fake = FakeAgentRunner(responses=[make_technique_body_output()])
    monkeypatch.setattr(harness_builder_main, "_run_agent_sync", fake)

    outcome = _run_baseline_step(
        _make_baseline_step(),
        index=0,
        context=_make_context(tmp_path),
    )

    assert outcome.succeeded is True
    user_message = fake.calls[0].user_message
    assert "kind=paper_extract" in user_message
    assert "2401.12345" in user_message
    assert "Algorithm 1" in user_message
    # The contract's ProposalGrounding shape did not win.
    assert "kind=proposal" not in user_message


def test_baseline_step_paper_anchor_falls_back_when_no_extract_staged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the contract carries a paper anchor but no extract is
    staged, the bundle gets a paper_extract variant with a
    "extract not staged" placeholder. Sub-PR D's host dispatcher fills
    the gap in production; for now this branch is observable."""
    write_minimal_harness_workspace(tmp_path)
    stage_technique_contract(
        tmp_path,
        fidelity_anchor=PaperFidelityAnchor(doi="10.1234/abcd", arxiv_id="2401.99999"),
    )
    fake = FakeAgentRunner(responses=[make_technique_body_output()])
    monkeypatch.setattr(harness_builder_main, "_run_agent_sync", fake)

    outcome = _run_baseline_step(
        _make_baseline_step(),
        index=0,
        context=_make_context(tmp_path),
    )

    assert outcome.succeeded is True
    user_message = fake.calls[0].user_message
    assert "kind=paper_extract" in user_message
    assert "2401.99999" in user_message


def test_baseline_step_bundle_carries_harness_source_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Staged harness modules + manifest flow into the bundle so the
    agent sees the integration shape it must plug into."""
    write_minimal_harness_workspace(tmp_path)
    stage_technique_contract(tmp_path)
    stage_harness_source_module(
        tmp_path,
        "harness/__init__.py",
        '"""Test harness."""\n\ndef build_harness(config):\n    return {"model": None}\n',
    )
    stage_harness_api_manifest(
        tmp_path,
        {
            "manifest_schema_version": 1,
            "harness_version_hash": "abc",
            "extension_points": [],
        },
    )
    fake = FakeAgentRunner(responses=[make_technique_body_output()])
    monkeypatch.setattr(harness_builder_main, "_run_agent_sync", fake)

    outcome = _run_baseline_step(
        _make_baseline_step(),
        index=0,
        context=_make_context(tmp_path),
    )

    assert outcome.succeeded is True
    user_message = fake.calls[0].user_message
    # Manifest JSON content present in the rendered prompt.
    assert "manifest_schema_version" in user_message
    # Staged harness module content present.
    assert "build_harness" in user_message


def test_baseline_step_retries_on_lint_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A first lint-broken response triggers a retry; the retry's
    bundle carries the ``PriorTechniqueAttempt``."""
    write_minimal_harness_workspace(tmp_path)
    stage_technique_contract(tmp_path)
    bad_source = "def baseline(:\n  not valid python\n"
    fake = FakeAgentRunner(
        responses=[
            make_technique_body_output(technique_py_source=bad_source),
            make_technique_body_output(),  # clean
        ],
    )
    monkeypatch.setattr(harness_builder_main, "_run_agent_sync", fake)

    outcome = _run_baseline_step(
        _make_baseline_step(),
        index=0,
        context=_make_context(tmp_path),
    )

    assert outcome.succeeded is True
    assert len(fake.calls) == 2
    second_call = fake.calls[1]
    assert "Prior failed attempts" in second_call.user_message


def test_baseline_step_budget_exhaustion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent lint failures exhaust the retry budget; the step
    surfaces a clear step-failure error."""
    write_minimal_harness_workspace(tmp_path)
    stage_technique_contract(tmp_path)
    bad_source = "def baseline(:\n  not valid python\n"
    fake = FakeAgentRunner(
        responses=[make_technique_body_output(technique_py_source=bad_source) for _ in range(4)],
    )
    monkeypatch.setattr(harness_builder_main, "_run_agent_sync", fake)

    outcome = _run_baseline_step(
        _make_baseline_step(),
        index=0,
        context=_make_context(tmp_path),
    )

    assert outcome.succeeded is False
    assert outcome.error is not None
    assert "budget exhausted" in outcome.error
    assert len(fake.calls) == 4


def test_baseline_step_additive_factor_no_contract_uses_no_op_grounding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Additive factor with no staged technique contract -> the
    grounding resolver falls back to :class:`NoOpBaselineGrounding`
    (DEC-013); the bundle still renders cleanly."""
    write_minimal_harness_workspace(tmp_path)
    fake = FakeAgentRunner(responses=[make_technique_body_output()])
    monkeypatch.setattr(harness_builder_main, "_run_agent_sync", fake)

    outcome = _run_baseline_step(
        _make_baseline_step(factor_type="additive"),
        index=0,
        context=_make_context(tmp_path),
    )

    assert outcome.succeeded is True
    user_message = fake.calls[0].user_message
    assert "kind=no_op_baseline" in user_message
