"""Host-side tests for ``make_dispatch_technique_implementation_sandboxed``.

Round-22 dogfood (2026-05-27) surfaced a missing-runtime-templates bug
in :func:`_materialize_technique_implementer_workspace`: the sandbox's
in-loop ``ValidationStep`` runs ``python experiment.py --mode validation``,
which requires the fixed templates ``experiment.py`` and
``techniques/__init__.py`` at the workspace root. The harness_builder
materializer calls :func:`create_workspace_skeleton` +
:func:`write_template_files` before staging contract artifacts; Step 7's
technique_implementer port omitted them. The test below locks in the
fix so a future refactor doesn't drop the calls again.

Reusing :mod:`_harness_builder_sandboxed_fixtures` per the existing
sandboxed-dispatch test pattern; the harness contract / manifest shapes
those fixtures emit are valid for the technique_implementer materializer
as well (it loads them via the same artifact-store contract keys).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from smai_agents.agents.technique_implementer import (
    DEFAULT_HARNESS_CONTRACT_KEY_TEMPLATE,
    DEFAULT_HARNESS_MANIFEST_KEY_TEMPLATE,
    DEFAULT_TECHNIQUE_GROUNDING_KEY_TEMPLATE,
    _materialize_technique_implementer_workspace,  # pyright: ignore[reportPrivateUsage]
)
from smai_core.plugins import ArtifactStore

from ._harness_builder_sandboxed_fixtures import (
    _RecordingArtifactStore,  # pyright: ignore[reportPrivateUsage]
    make_contract,
    make_technique_contract,
)


@pytest.mark.asyncio
async def test_materializer_stages_runtime_fixed_templates(tmp_path: Path) -> None:
    """``_materialize_technique_implementer_workspace`` must drop the fixed
    runtime templates (``experiment.py`` + ``techniques/__init__.py``)
    into the workspace root.

    Round-22 finding: missing template staging caused the sandbox's
    in-loop ValidationStep to fail with ``experiment.py does not exist``
    on the very first dispatch, burning the diagnose-retry budget on a
    failure the agent had no writable surface to fix (no-go-zone).
    """
    artifact_store = _RecordingArtifactStore()
    cg_id = "cg-tech-001"
    entry_id = "entry-cutout-treatment"

    harness_contract = make_contract()
    technique_contract = make_technique_contract()
    await artifact_store.put(
        DEFAULT_HARNESS_CONTRACT_KEY_TEMPLATE.format(cg_id=cg_id),
        harness_contract.model_dump_json().encode(),
    )
    # Minimal valid manifest payload (validation-mode shape with empty
    # extension_points; the materializer treats it as opaque bytes).
    await artifact_store.put(
        DEFAULT_HARNESS_MANIFEST_KEY_TEMPLATE.format(cg_id=cg_id),
        (
            b'{"extension_points": [], "integration_pattern_summary": "stub", '
            b'"harness_version_hash": "", "parent_harness_contract_hash": "", '
            b'"manifest_schema_version": 1, "runtime_template_version": "1.0.0"}'
        ),
    )

    workspace_path = tmp_path / entry_id
    await _materialize_technique_implementer_workspace(
        workspace_path=workspace_path,
        cg_id=cg_id,
        entry_id=entry_id,
        # The fake intentionally implements only the surface the
        # materializer touches (get / put / list); cast to ArtifactStore
        # so pyright doesn't insist on capabilities / url_for at the
        # call site (the harness_builder sandboxed tests dodge this by
        # going through a stub DispatchContext typed as Any).
        artifact_store=cast(ArtifactStore, cast(Any, artifact_store)),
        technique_contract=technique_contract,
        harness_contract_key=DEFAULT_HARNESS_CONTRACT_KEY_TEMPLATE.format(cg_id=cg_id),
        manifest_key=DEFAULT_HARNESS_MANIFEST_KEY_TEMPLATE.format(cg_id=cg_id),
        grounding_key=DEFAULT_TECHNIQUE_GROUNDING_KEY_TEMPLATE.format(
            cg_id=cg_id, entry_id=entry_id
        ),
    )

    assert (workspace_path / "experiment.py").is_file()
    assert (workspace_path / "techniques" / "__init__.py").is_file()
    # Per-entry artifacts the materializer was already wiring (regression
    # surface for a future refactor that might move them around).
    assert (workspace_path / "contracts" / "technique_contract.json").is_file()
    assert (workspace_path / "contracts" / "harness_contract.json").is_file()
    assert (workspace_path / "harness_api_manifest.json").is_file()
