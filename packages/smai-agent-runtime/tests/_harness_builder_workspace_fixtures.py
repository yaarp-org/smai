"""Per-task fixtures for the sub-PR B harness_builder mini-orchestrator tests.

Per the SMAI test convention (``tests/_<task>_<purpose>.py``); the
per-task prefix keeps the module isolated from sibling-package fakes
under ``--import-mode=importlib``.
"""

from __future__ import annotations

from pathlib import Path

from _workflow_fixtures import make_contract  # type: ignore[import-not-found]


def write_minimal_harness_workspace(workspace_path: Path) -> None:
    """Materialize the minimum on-disk shape the sub-PR B mini-orchestrator
    consumes: ``contracts/harness_contract.json`` + the directory
    skeleton so per-step writes (e.g., ``harness/__init__.py``) have a
    parent.

    Sub-PR B's mini-orchestrator does NOT consume the technique contract
    or the harness API reference (those are sub-PR C territory once the
    real agent reasoning lands); only the harness contract is required
    to drive ``generate_workflow``.
    """
    contract = make_contract()
    contracts_dir = workspace_path / "contracts"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    (contracts_dir / "harness_contract.json").write_text(contract.model_dump_json(indent=2))

    # Pre-create harness/ and techniques/ so the body-generation /
    # baseline fake handlers don't have to mkdir on every test
    # invocation (they would anyway via the helpers, but pre-creating
    # mirrors the host-side ``create_workspace_skeleton`` shape).
    (workspace_path / "harness").mkdir(parents=True, exist_ok=True)
    (workspace_path / "techniques").mkdir(parents=True, exist_ok=True)


__all__ = ["write_minimal_harness_workspace"]
