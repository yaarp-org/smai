"""Per-task fixtures for the sub-PR B/C harness_builder mini-orchestrator tests.

Per the SMAI test convention (``tests/_<task>_<purpose>.py``); the
per-task prefix keeps the module isolated from sibling-package fakes
under ``--import-mode=importlib``.
"""

from __future__ import annotations

from pathlib import Path

from _workflow_fixtures import make_contract  # type: ignore[import-not-found]

# Shim ``experiment.py`` that always exits 0 and writes a passing
# ``validation_results.json`` to the workspace. Lets the entry-point /
# dispatch tests exercise the real ValidationStep subprocess flow
# without staging a full runnable harness (the in-process per-step
# tests assert the ValidationStep handler's behavior directly).
#
# The integration test in ``test_integration_end_to_end.py`` uses this
# same shim because the integration brief targets the workflow's
# step-iteration shape, not the runtime semantics (the
# smai-runtime test suite covers the runner's real behavior).
_EXPERIMENT_PY_SHIM: str = '''"""Test shim for experiment.py.

Always exits 0 and writes a passing validation_results.json to
the --workspace path. Used by the smai-agent-runtime harness-builder
entry-point tests so the real ValidationStep subprocess flow can run
without staging a full runtime-runnable harness.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--technique", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--subset", default=None)
    args = parser.parse_args()
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    payload = {
        "passed": True,
        "mode": args.mode,
        "technique": args.technique,
        "seed": args.seed,
        "epochs": args.epochs,
        "subset": args.subset,
        "metrics": {},
    }
    (workspace / "validation_results.json").write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def write_minimal_harness_workspace(
    workspace_path: Path,
    *,
    with_experiment_shim: bool = True,
) -> None:
    """Materialize the minimum on-disk shape the mini-orchestrator consumes.

    Drops:

    * ``contracts/harness_contract.json`` from :func:`make_contract`.
    * Empty ``harness/`` and ``techniques/`` directories so the body-
      generation handlers have parent directories to write into.
    * ``experiment.py`` (test shim that always passes validation) when
      ``with_experiment_shim`` is True. The shim lets the real
      :class:`ValidationStep` subprocess flow run end-to-end against
      this workspace without staging a runnable harness.

    Pass ``with_experiment_shim=False`` to exercise the ValidationStep's
    "no experiment.py" failure path.
    """
    contract = make_contract()
    contracts_dir = workspace_path / "contracts"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    (contracts_dir / "harness_contract.json").write_text(contract.model_dump_json(indent=2))

    (workspace_path / "harness").mkdir(parents=True, exist_ok=True)
    (workspace_path / "techniques").mkdir(parents=True, exist_ok=True)

    if with_experiment_shim:
        (workspace_path / "experiment.py").write_text(_EXPERIMENT_PY_SHIM)


def write_failing_experiment_shim(workspace_path: Path) -> None:
    """Drop an ``experiment.py`` that always exits non-zero with a
    structured stderr tail.

    Used by ValidationStep failure-path tests so they exercise the real
    subprocess invocation instead of the "no experiment.py" branch.
    """
    failing = '''"""Test shim that always exits 1 with structured stderr."""
import sys

def main():
    print("EXIT_RUNTIME_FAILURE", file=sys.stderr)
    print("Traceback (most recent call last):", file=sys.stderr)
    print('  File "/workspace/harness/__init__.py", line 42, in build_harness', file=sys.stderr)
    print("    model = build_model().cuda()", file=sys.stderr)
    print("RuntimeError: Found no NVIDIA driver on your system.", file=sys.stderr)
    return 1

if __name__ == "__main__":
    sys.exit(main())
'''
    (workspace_path / "experiment.py").write_text(failing)


__all__ = [
    "_EXPERIMENT_PY_SHIM",
    "write_failing_experiment_shim",
    "write_minimal_harness_workspace",
]
