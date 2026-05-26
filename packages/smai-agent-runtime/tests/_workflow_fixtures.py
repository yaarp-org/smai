"""Per-task fixtures for ``test_workflow_generator``.

Per the SMAI test convention (``tests/_<task>_<purpose>.py``) — sibling
packages use generic ``_fakes.py`` filenames and collide under
``--import-mode=importlib``; the per-task prefix keeps this module
isolated from other packages' fixtures.
"""

from __future__ import annotations

from typing import Literal

from smai_agent_runtime.workflow.generator import _AbiFunction
from smai_core.artifacts._envelope import ArtifactEnvelope
from smai_core.artifacts.harness_contract import HarnessContract, HarnessContractBody
from smai_core.entities.factor import Factor


def make_contract(
    *,
    factor_type: Literal["additive", "substitutive"] = "substitutive",
    seeds: list[int] | None = None,
    content_hash: str = "deadbeef" * 8,
    parent_experiment_hash: str = "cafebabe" * 8,
) -> HarnessContract:
    """Build a minimal valid :class:`HarnessContract` for generator tests.

    The generator reads ``body.factor.type``, ``body.seeds[0]`` (or 0 if
    empty), and ``envelope.content_hash`` — every other field is set to
    its smallest valid value so the fixture stays focused on the
    surface the generator actually touches.
    """
    envelope = ArtifactEnvelope(
        artifact_kind="harness_contract",
        artifact_id="hc-fixture",
        schema_version=1,
        compiler_version="0.0.0-test",
        content_hash=content_hash,
        parent_experiment_id="exp-fixture",
    )
    body = HarnessContractBody(
        parent_experiment_hash=parent_experiment_hash,
        factor=Factor(
            name="test_factor",
            type=factor_type,
            description="fixture factor",
        ),
        seeds=list(seeds) if seeds is not None else [42],
        fixed_variables=[],
        required_metrics=[],
        optional_telemetry=[],
        no_go_zones=[],
    )
    return HarnessContract(envelope=envelope, body=body)


V1_ABI: tuple[_AbiFunction, ...] = (
    _AbiFunction(
        name="build_harness",
        signature="def build_harness(config: dict) -> Harness:",
        write_to_path="harness/__init__.py",
    ),
    _AbiFunction(
        name="run_training_loop",
        signature=(
            "def run_training_loop(harness: Harness, technique: Technique) -> TrainingResult:"
        ),
        write_to_path="harness/__init__.py",
    ),
    _AbiFunction(
        name="evaluate",
        signature="def evaluate(harness: Harness, result: TrainingResult) -> Metrics:",
        write_to_path="harness/__init__.py",
    ),
)
"""Mirror of the generator's v1 ABI — used by 3-function-shape tests
that want to assert against the same names without coupling to the
generator's internal table."""


SYNTHETIC_5_ABI: tuple[_AbiFunction, ...] = V1_ABI + (
    _AbiFunction(
        name="preflight_check",
        signature="def preflight_check(config: dict) -> None:",
        write_to_path="harness/__init__.py",
    ),
    _AbiFunction(
        name="postrun_telemetry",
        signature="def postrun_telemetry(result: TrainingResult) -> dict[str, float]:",
        write_to_path="harness/__init__.py",
    ),
)
"""Synthetic 5-function ABI for parametric-on-ABI tests. The two
hypothetical extra functions document the "future N-function contract"
shape D9 talks about — not currently registered in
``_ABI_BY_RUNTIME_VERSION``."""


__all__: list[str] = [
    "SYNTHETIC_5_ABI",
    "V1_ABI",
    "make_contract",
]
