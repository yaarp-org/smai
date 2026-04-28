"""Fixtures + builders for Task 2.B3 (harness builder + technique implementer).

Per the workspace's pytest --import-mode=importlib + ``conftest.py``
discovery layout: every package's ``tests/`` is on ``sys.path`` once,
so module names must be unique across packages. ``_b3_fakes`` avoids
colliding with ``_agent_fakes`` (B1), ``_b2_fakes`` (B2), the bedrock
plugin's ``_fakes``, and the orchestrator's ``_helpers``.

This module ships:

* :func:`make_harness_contract` — a minimal :class:`HarnessContract`
  with a known ``envelope.content_hash`` (frozen via
  :func:`smai_core.freeze_with_hash`) so the manifest tool's
  parent-contract-hash check has something to compare against.
* :func:`make_technique_contract` — same for non-baseline entries.
* :func:`make_baseline_technique_contract` — ``technique_id=None`` /
  ``is_baseline=True`` for additive-baseline tests.
* :func:`make_substitutive_baseline_technique_contract` — substitutive
  baseline with a real ``technique_id``.
* :func:`make_minimal_manifest` — builds a frozen
  :class:`HarnessAPIManifest` matching a hand-rolled harness directory,
  with ``content_hash``, ``harness_version_hash``, and
  ``parent_harness_contract_hash`` populated correctly.
"""

from __future__ import annotations

from smai_core import (
    Factor,
    HarnessContract,
    HarnessContractBody,
    PaperFidelityAnchor,
    ProposalFidelityAnchor,
    TechniqueContract,
    TechniqueContractBody,
    freeze_with_hash,
)
from smai_core.artifacts._envelope import ArtifactEnvelope
from smai_runtime import (
    MANIFEST_SCHEMA_VERSION,
    RUNTIME_TEMPLATE_VERSION,
    HarnessAPIManifest,
    HarnessExtensionPoint,
    compute_harness_version_hash,
    freeze_manifest,
)


def make_harness_contract(
    *,
    factor_type: str = "additive",
    factor_name: str = "augmentation",
    parent_experiment_id: str = "exp-1",
    parent_experiment_hash: str = "exp-hash",
    no_go_zones: list[str] | None = None,
) -> HarnessContract:
    """Build a frozen :class:`HarnessContract` with a populated content_hash."""
    envelope = ArtifactEnvelope(
        artifact_kind="harness_contract",
        artifact_id="hc-1",
        schema_version=1,
        compiler_version="0.0.0-test",
        parent_experiment_id=parent_experiment_id,
        registry_hashes={},
        surface_map={},
    )
    body = HarnessContractBody(
        parent_experiment_hash=parent_experiment_hash,
        factor=Factor(
            name=factor_name,
            type=factor_type,  # type: ignore[arg-type]
            description="test factor",
        ),
        seeds=[42, 1337],
        fixed_variables=[],
        required_metrics=[],
        optional_telemetry=[],
        no_go_zones=no_go_zones or ["experiment.py", "techniques/__init__.py"],
    )
    contract = HarnessContract(envelope=envelope, body=body)
    return freeze_with_hash(contract)


def make_technique_contract(
    *,
    parent_harness_contract_hash: str,
    entry_id: str = "entry-1",
    technique_id: str | None = "tq-1",
    is_baseline: bool = False,
    standard: bool = False,
    fidelity_anchor_kind: str | None = "proposal",
) -> TechniqueContract:
    """Build a frozen :class:`TechniqueContract`."""
    envelope = ArtifactEnvelope(
        artifact_kind="technique_contract",
        artifact_id=f"tc-{entry_id}",
        schema_version=1,
        compiler_version="0.0.0-test",
        parent_experiment_id="exp-1",
        registry_hashes={},
        surface_map={},
    )
    anchor: PaperFidelityAnchor | ProposalFidelityAnchor | None = None
    if fidelity_anchor_kind == "proposal":
        anchor = ProposalFidelityAnchor(
            proposal_id="prop-1",
        )
    elif fidelity_anchor_kind == "paper":
        anchor = PaperFidelityAnchor(
            doi="10.1234/test",
            arxiv_id="2401.12345",
        )
    body = TechniqueContractBody(
        entry_id=entry_id,
        parent_experiment_id="exp-1",
        parent_experiment_hash="exp-hash",
        parent_harness_contract_hash=parent_harness_contract_hash,
        technique_id=technique_id,
        technique_params=None,
        level_value=None,
        is_baseline=is_baseline,
        fidelity_anchor=anchor,
        standard=standard,
    )
    contract = TechniqueContract(envelope=envelope, body=body)
    return freeze_with_hash(contract)


def make_additive_baseline_technique_contract(
    *,
    parent_harness_contract_hash: str,
) -> TechniqueContract:
    """Build the additive-baseline contract (technique_id=None)."""
    return make_technique_contract(
        parent_harness_contract_hash=parent_harness_contract_hash,
        entry_id="entry-baseline",
        technique_id=None,
        is_baseline=True,
        standard=False,
        fidelity_anchor_kind=None,
    )


def make_substitutive_baseline_technique_contract(
    *,
    parent_harness_contract_hash: str,
) -> TechniqueContract:
    """Build a substitutive-baseline contract (technique_id is non-null)."""
    return make_technique_contract(
        parent_harness_contract_hash=parent_harness_contract_hash,
        entry_id="entry-vgg",
        technique_id="tq-vgg",
        is_baseline=True,
        standard=True,
        fidelity_anchor_kind=None,
    )


# Sample harness file contents the agent might write. Tests use these to
# produce a manifest whose ``harness_version_hash`` matches what
# ``compute_harness_version_hash`` would produce over the resulting
# directory.
SAMPLE_HARNESS_FILES: dict[str, bytes] = {
    "__init__.py": b"# placeholder harness package\n",
    "trainer.py": (
        b"from smai_runtime import HarnessComponents\n"
        b"def build_harness(config):\n"
        b"    return HarnessComponents(\n"
        b"        train_transforms=[],\n"
        b"        model_factory=lambda: None,\n"
        b"        default_loss=lambda *a, **k: None,\n"
        b"        training_config={},\n"
        b"        callbacks=[],\n"
        b"    )\n"
        b"def run_training_loop(components, config, seed):\n"
        b"    return object()\n"
        b"def evaluate(model, components, config):\n"
        b"    return {}\n"
    ),
}


def make_minimal_manifest(
    *,
    parent_harness_contract_hash: str,
    harness_files: dict[str, bytes] | None = None,
    factor_type: str = "additive",
) -> HarnessAPIManifest:
    """Build a frozen :class:`HarnessAPIManifest` for a sample harness.

    For ``additive`` factors the manifest declares an optional
    extension point (matches §9.1 — additive baselines must run the
    harness as-is). For ``substitutive`` factors it declares a
    mandatory ``model_wrapper`` extension point (§9.2).
    """
    files = harness_files if harness_files is not None else SAMPLE_HARNESS_FILES
    harness_version_hash = compute_harness_version_hash(files)

    if factor_type == "additive":
        ext_points = [
            HarnessExtensionPoint(
                key="train_transforms",
                type_signature="list[Callable[[Tensor], Tensor]]",
                purpose="optional training-set transforms",
                optional=True,
                integration_pattern="append",
            ),
        ]
    else:
        ext_points = [
            HarnessExtensionPoint(
                key="model_wrapper",
                type_signature="Callable[[nn.Module], nn.Module]",
                purpose="mandatory architecture slot",
                optional=False,
                integration_pattern="wrap",
            ),
        ]

    manifest = HarnessAPIManifest(
        extension_points=ext_points,
        integration_pattern_summary=("test fixture — additive optional / substitutive mandatory"),
        harness_version_hash=harness_version_hash,
        parent_harness_contract_hash=parent_harness_contract_hash,
        manifest_schema_version=MANIFEST_SCHEMA_VERSION,
        runtime_template_version=RUNTIME_TEMPLATE_VERSION,
    )
    return freeze_manifest(manifest)


__all__ = [
    "SAMPLE_HARNESS_FILES",
    "make_additive_baseline_technique_contract",
    "make_harness_contract",
    "make_minimal_manifest",
    "make_substitutive_baseline_technique_contract",
    "make_technique_contract",
]
