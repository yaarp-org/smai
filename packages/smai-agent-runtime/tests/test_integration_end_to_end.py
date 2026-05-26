"""End-to-end integration test for sub-PR C2's mini-orchestrator.

Drives a complete harness_builder workflow against:

* a synthetic 3-function :class:`HarnessContract`,
* fake-LLM body-generation outputs,
* a real ``subprocess`` validation invocation against the
  ``experiment.py`` shim,
* the real :class:`ManifestEmitStep` hash-recompute + freeze.

Asserts the workspace ends with the expected file layout:

* ``harness/__init__.py`` (body steps)
* ``techniques/baseline.py`` (baseline step)
* ``validation_results.json`` (validation step)
* ``manifest.json`` (manifest emit step)
* ``outputs/status.json`` (summary)
* ``status/events.jsonl`` (D10 mirror)

And that the workflow completed cleanly through all 7 step types
(3 body + baseline + validation + diagnose + manifest).
"""

from __future__ import annotations

import json
from pathlib import Path

from _harness_builder_workspace_fixtures import (  # type: ignore[import-not-found]
    write_minimal_harness_workspace,
)
from pydantic import TypeAdapter
from smai_agent_runtime.__main__ import main as entry_main
from smai_agent_runtime.status import EVENTS_MIRROR_RELPATH, StatusEvent

# Sub-PR D thread 5: the previous ``SMAI_AGENT_RUNTIME_FAKE_LLM`` env-var
# seam was replaced with a dependency-injected :class:`AgentRunner`
# selected at main()-entry via ``--fake-llm``. Integration tests append
# it to their argv.
_FAKE_LLM_ARGV = ["--fake-llm"]


def test_end_to_end_workflow_produces_complete_artifact_set(tmp_path: Path) -> None:
    """A successful run through all 7 step types materializes the
    full artifact set on disk."""
    write_minimal_harness_workspace(tmp_path)
    rc = entry_main(
        [
            "--role",
            "harness_builder",
            "--cg-id",
            "cg-integration-test",
            "--workspace",
            str(tmp_path),
            *_FAKE_LLM_ARGV,
        ]
    )
    assert rc == 0

    # Body-generation steps wrote harness/__init__.py.
    harness_path = tmp_path / "harness" / "__init__.py"
    assert harness_path.exists()
    harness_body = harness_path.read_text()
    assert "build_harness" in harness_body
    assert "run_training_loop" in harness_body
    assert "evaluate" in harness_body

    # Baseline step wrote techniques/baseline.py.
    baseline_path = tmp_path / "techniques" / "baseline.py"
    assert baseline_path.exists()
    assert "baseline" in baseline_path.read_text()

    # Validation step's subprocess wrote validation_results.json.
    validation_path = tmp_path / "validation_results.json"
    assert validation_path.exists()
    validation_payload = json.loads(validation_path.read_text())
    assert validation_payload["passed"] is True

    # Manifest step wrote manifest.json with a frozen content_hash.
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    manifest_payload = json.loads(manifest_path.read_text())
    assert manifest_payload["content_hash"]
    assert manifest_payload["harness_version_hash"]

    # Workflow summary at outputs/status.json.
    status_path = tmp_path / "outputs" / "status.json"
    assert status_path.exists()
    status_payload = json.loads(status_path.read_text())
    assert status_payload["succeeded"] is True
    assert len(status_payload["step_outcomes"]) == 7

    # D10 mirror at status/events.jsonl.
    mirror_path = tmp_path / EVENTS_MIRROR_RELPATH
    assert mirror_path.exists()
    mirror_lines = mirror_path.read_text().strip().split("\n")
    # Every line parses cleanly as a StatusEvent.
    adapter: TypeAdapter[StatusEvent] = TypeAdapter(StatusEvent)
    for line in mirror_lines:
        adapter.validate_json(line)


def test_end_to_end_event_stream_step_kinds_match_workflow(tmp_path: Path) -> None:
    """The status event stream reflects every step kind in the
    workflow exactly once at step_end. Confirms the workflow's seven
    step types (3 body + baseline + validation + diagnose + manifest)
    each pushed through the per-step iteration."""
    write_minimal_harness_workspace(tmp_path)
    entry_main(
        [
            "--role",
            "harness_builder",
            "--cg-id",
            "cg-step-kinds",
            "--workspace",
            str(tmp_path),
            *_FAKE_LLM_ARGV,
        ]
    )
    mirror_path = tmp_path / EVENTS_MIRROR_RELPATH
    events = [json.loads(line) for line in mirror_path.read_text().strip().split("\n")]
    step_kinds_at_end = [e["step_kind"] for e in events if e["event_type"] == "step.end"]
    assert step_kinds_at_end == [
        "harness_body_generation",
        "harness_body_generation",
        "harness_body_generation",
        "baseline_generation",
        "validation",
        "diagnose_on_failure",
        "manifest_emit",
    ]
