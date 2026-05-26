"""Per-step tests for :func:`_run_manifest_emit_step` (sub-PR C2).

Asserts the round-15 hash-recompute discipline (the manifest's
``harness_version_hash`` reflects the on-disk ``harness/`` contents at
emit time, not any value the step carries from a prior turn), the
validation-precondition guard (no successful ValidationStep in the
prior outcomes => emit refuses), and the freeze-and-write path.
"""

from __future__ import annotations

import json
from pathlib import Path

from _harness_builder_workspace_fixtures import (  # type: ignore[import-not-found]
    write_minimal_harness_workspace,
)
from smai_agent_runtime.harness_builder._main import (
    _DispatchContext,
    _load_contract,
    _run_manifest_emit_step,
    _StepOutcome,
)
from smai_agent_runtime.workflow.step_types import ManifestEmitStep
from smai_runtime.manifest import compute_harness_version_hash
from smai_runtime.no_go_zone import RUNTIME_TEMPLATE_VERSION


def _make_context(workspace: Path) -> _DispatchContext:
    contract = _load_contract(workspace)
    assert contract is not None
    return _DispatchContext(
        cg_id="cg-test-manifest",
        workspace=workspace,
        contract=contract,
        technique_contract=None,
        overrides=None,
    )


def _make_step(
    *,
    runtime_template_version: str | None = None,
    parent_harness_contract_hash: str = "cafebabe" * 8,
) -> ManifestEmitStep:
    return ManifestEmitStep(
        runtime_template_version=runtime_template_version or RUNTIME_TEMPLATE_VERSION,
        parent_harness_contract_hash=parent_harness_contract_hash,
    )


def _passing_validation_outcome() -> _StepOutcome:
    return _StepOutcome(
        step_index=4,
        step_type="validation",
        succeeded=True,
    )


def _failed_validation_outcome() -> _StepOutcome:
    return _StepOutcome(
        step_index=4,
        step_type="validation",
        succeeded=False,
        error="validation subprocess exited with code 1",
    )


# === Happy path ==============================================================


def test_manifest_emit_writes_manifest_with_recomputed_hash(tmp_path: Path) -> None:
    """Successful prior ValidationStep + harness/ contents on disk =>
    the emit recomputes ``harness_version_hash`` from those contents
    and writes ``manifest.json`` to the workspace root with the freeze
    content_hash populated."""
    write_minimal_harness_workspace(tmp_path)
    (tmp_path / "harness" / "__init__.py").write_text("def build_harness(config):\n    return {}\n")
    prior_outcomes = [_passing_validation_outcome()]

    outcome = _run_manifest_emit_step(
        _make_step(),
        index=6,
        context=_make_context(tmp_path),
        prior_outcomes=prior_outcomes,
    )

    assert outcome.succeeded is True
    assert outcome.error is None
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text())
    # Recomputed hash matches what the runtime's compute_harness_version_hash
    # produces over the on-disk harness/ files.
    expected = compute_harness_version_hash(
        {"__init__.py": (tmp_path / "harness" / "__init__.py").read_bytes()}
    )
    assert payload["harness_version_hash"] == expected
    # Freeze populated content_hash.
    assert payload["content_hash"]
    # Runtime template version matches the live constant (round-16
    # drift-guard discipline).
    assert payload["runtime_template_version"] == RUNTIME_TEMPLATE_VERSION


def test_manifest_emit_hash_reflects_current_disk_contents_not_stale(
    tmp_path: Path,
) -> None:
    """Round-15 discipline: the emit MUST recompute the hash from the
    current on-disk harness/ contents, not reuse a stale value. We
    write the harness, snapshot the hash, then write a different
    harness body and re-emit; the manifest's hash must differ."""
    write_minimal_harness_workspace(tmp_path)
    (tmp_path / "harness" / "__init__.py").write_text("v1 = 1\n")
    prior_outcomes = [_passing_validation_outcome()]
    outcome = _run_manifest_emit_step(
        _make_step(),
        index=6,
        context=_make_context(tmp_path),
        prior_outcomes=prior_outcomes,
    )
    assert outcome.succeeded is True
    hash_v1 = json.loads((tmp_path / "manifest.json").read_text())["harness_version_hash"]

    # Mutate the harness; re-emit.
    (tmp_path / "harness" / "__init__.py").write_text("v2 = 2\n")
    outcome2 = _run_manifest_emit_step(
        _make_step(),
        index=6,
        context=_make_context(tmp_path),
        prior_outcomes=prior_outcomes,
    )
    assert outcome2.succeeded is True
    hash_v2 = json.loads((tmp_path / "manifest.json").read_text())["harness_version_hash"]
    assert hash_v1 != hash_v2


# === Boundary: validation precondition =======================================


def test_manifest_emit_refuses_without_passing_validation(tmp_path: Path) -> None:
    """No successful ValidationStep in prior_outcomes => emit refuses
    with a structured error."""
    write_minimal_harness_workspace(tmp_path)
    (tmp_path / "harness" / "__init__.py").write_text("def build_harness(c): return {}\n")

    outcome = _run_manifest_emit_step(
        _make_step(),
        index=6,
        context=_make_context(tmp_path),
        prior_outcomes=[_failed_validation_outcome()],
    )
    assert outcome.succeeded is False
    assert outcome.error is not None
    assert "no prior successful ValidationStep" in outcome.error
    assert not (tmp_path / "manifest.json").exists()


def test_manifest_emit_refuses_when_validation_is_a_skip(tmp_path: Path) -> None:
    """The precondition specifically requires a SUCCESSFUL prior
    ValidationStep — outcomes with the validation step_type but
    succeeded=False (the diagnose-handoff case) do not satisfy."""
    write_minimal_harness_workspace(tmp_path)
    (tmp_path / "harness" / "__init__.py").write_text("def build_harness(c): return {}\n")

    outcome = _run_manifest_emit_step(
        _make_step(),
        index=6,
        context=_make_context(tmp_path),
        prior_outcomes=[
            _StepOutcome(
                step_index=4,
                step_type="validation",
                succeeded=False,
                error="ran but didn't pass",
            )
        ],
    )
    assert outcome.succeeded is False
    assert outcome.error is not None
    assert "no prior successful ValidationStep" in outcome.error


# === Boundary: harness/ empty ================================================


def test_manifest_emit_refuses_with_no_harness_files(tmp_path: Path) -> None:
    """Empty harness/ directory => emit refuses (the manifest documents
    the harness; emitting one without any harness source files would
    silently produce a zero-content-hash artifact)."""
    write_minimal_harness_workspace(tmp_path)
    # Don't write anything to harness/__init__.py — the dir is empty.
    prior_outcomes = [_passing_validation_outcome()]

    outcome = _run_manifest_emit_step(
        _make_step(),
        index=6,
        context=_make_context(tmp_path),
        prior_outcomes=prior_outcomes,
    )
    assert outcome.succeeded is False
    assert outcome.error is not None
    assert "harness/ directory contains no files" in outcome.error


# === Boundary: runtime template version drift ================================


def test_manifest_emit_refuses_on_runtime_template_version_drift(
    tmp_path: Path,
) -> None:
    """The step's ``runtime_template_version`` must equal the runtime's
    live ``RUNTIME_TEMPLATE_VERSION``; mismatch surfaces upstream so the
    runtime doesn't reject the manifest at training-run start."""
    write_minimal_harness_workspace(tmp_path)
    (tmp_path / "harness" / "__init__.py").write_text("def build_harness(c): return {}\n")

    outcome = _run_manifest_emit_step(
        _make_step(runtime_template_version="999.999.999"),
        index=6,
        context=_make_context(tmp_path),
        prior_outcomes=[_passing_validation_outcome()],
    )
    assert outcome.succeeded is False
    assert outcome.error is not None
    assert "runtime_template_version" in outcome.error
