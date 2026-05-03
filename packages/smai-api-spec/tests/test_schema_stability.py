"""JSON-Schema snapshot tests.

For every public Pydantic model, snapshot ``Model.model_json_schema()`` to
``tests/snapshots/<module>__<ClassName>.json`` and assert the in-memory
schema matches. Any change to a model's wire shape — even apparently
cosmetic field renames — produces a snapshot diff and forces a deliberate
accept.

This is the contract-stability mechanism that lets Yaarp v2's API depend
on this package without coordinating every release. A breaking change to
the wire format requires an explicit snapshot regeneration and a semver
bump per ``packages/smai-api-spec/README.md``.

Snapshot regeneration: set ``SMAI_API_SPEC_UPDATE_SNAPSHOTS=1`` and
re-run::

    SMAI_API_SPEC_UPDATE_SNAPSHOTS=1 \
      uv run pytest packages/smai-api-spec/tests/test_schema_stability.py

Then inspect the diff under ``tests/snapshots/``, decide whether the
shape change is intentional, and commit if so.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel
from smai_api_spec import (
    AgentStatusResponse,
    ApproveProposalResponse,
    ArtifactKeysResponse,
    CGStatusResponse,
    CGSummary,
    ComparisonGroupDetailResponse,
    CompiledArtifacts,
    CompileExperimentRequest,
    CompileExperimentResponse,
    CreatedCGRef,
    EntryAgentStatus,
    EntryDetailResponse,
    EntrySummary,
    EntryWithRuns,
    EvaluationResultResponse,
    HarnessAgentStatus,
    PaginationParams,
    PaperDetailResponse,
    PaperSubmissionResponse,
    PaperSummary,
    PluginInfo,
    PluginVerifyResult,
    PromotePartialResponse,
    ProposalDetailResponse,
    ProposalSubmissionResponse,
    ProposalSummary,
    RecentActivityItem,
    RejectProposalRequest,
    RejectProposalResponse,
    RunDetailResponse,
    RunSummary,
    StateChangeEvent,
    SubmitExperimentRequest,
    SubmitExperimentResponse,
    SubmitPaperRequest,
    SubmitProposalRequest,
    SummaryCounts,
    SystemConfigResponse,
    SystemDashboardResponse,
    SystemHealthResponse,
    SystemMigrateStatusResponse,
    SystemPluginsResponse,
    SystemVerifyRequest,
    SystemVerifyResponse,
    SystemVersionResponse,
    TechniqueRefSummary,
    WorkerHeartbeatEvent,
)
from smai_api_spec.comparison_groups import ListArtifactsParams, ListCGsParams
from smai_api_spec.errors import APIError, ErrorEnvelope, ValidationIssue
from smai_api_spec.papers import ListPapersParams
from smai_api_spec.proposals import ListProposalsParams
from smai_api_spec.runs import ListRunsParams

# All public models the snapshot suite asserts on. Extending the set is a
# deliberate act — adding a new model means committing a new snapshot
# alongside it.
SNAPSHOTTED_MODELS: tuple[type[BaseModel], ...] = (
    # Common
    PaginationParams,
    # Errors
    APIError,
    ErrorEnvelope,
    ValidationIssue,
    # Events
    StateChangeEvent,
    WorkerHeartbeatEvent,
    # Proposals
    SubmitProposalRequest,
    ProposalSubmissionResponse,
    ProposalSummary,
    ProposalDetailResponse,
    ApproveProposalResponse,
    RejectProposalRequest,
    RejectProposalResponse,
    ListProposalsParams,
    # Papers
    SubmitPaperRequest,
    PaperSubmissionResponse,
    PaperSummary,
    PaperDetailResponse,
    PromotePartialResponse,
    TechniqueRefSummary,
    ListPapersParams,
    # Experiments
    CompileExperimentRequest,
    CompileExperimentResponse,
    CompiledArtifacts,
    SubmitExperimentRequest,
    SubmitExperimentResponse,
    CreatedCGRef,
    # Comparison groups
    CGSummary,
    ComparisonGroupDetailResponse,
    CGStatusResponse,
    AgentStatusResponse,
    HarnessAgentStatus,
    EntryAgentStatus,
    EvaluationResultResponse,
    ArtifactKeysResponse,
    ListArtifactsParams,
    ListCGsParams,
    # Entries
    EntrySummary,
    EntryWithRuns,
    EntryDetailResponse,
    # Runs
    RunSummary,
    RunDetailResponse,
    ListRunsParams,
    # System
    SystemVersionResponse,
    SystemConfigResponse,
    SystemPluginsResponse,
    SystemVerifyRequest,
    SystemVerifyResponse,
    SystemDashboardResponse,
    SystemMigrateStatusResponse,
    SystemHealthResponse,
    PluginInfo,
    PluginVerifyResult,
    SummaryCounts,
    RecentActivityItem,
)

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
UPDATE_ENV_VAR = "SMAI_API_SPEC_UPDATE_SNAPSHOTS"


def _snapshot_path(model: type[BaseModel]) -> Path:
    """Snapshot filename: ``<module-leaf>__<ClassName>.json``.

    Embedding the module leaf in the filename keeps two same-named
    classes in different modules from colliding (defensive — none today,
    but the suite should be robust to future additions).
    """
    module_leaf = model.__module__.rsplit(".", 1)[-1]
    return SNAPSHOT_DIR / f"{module_leaf}__{model.__name__}.json"


def _canonical_schema_text(model: type[BaseModel]) -> str:
    """Pretty-printed, key-sorted JSON form of the model's JSON Schema.

    Sort keys + ``indent=2`` so diffs render line-by-line and snapshot
    bytes are stable across Python runs (Pydantic does not promise key
    ordering inside its schema dicts; sort defensively).
    """
    schema = model.model_json_schema()
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def _update_requested() -> bool:
    return os.environ.get(UPDATE_ENV_VAR, "").lower() in {"1", "true", "yes"}


@pytest.mark.parametrize("model", SNAPSHOTTED_MODELS, ids=lambda m: cast(type, m).__name__)
def test_schema_matches_snapshot(model: type[BaseModel]) -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_path = _snapshot_path(model)
    expected_text = _canonical_schema_text(model)

    if _update_requested():
        snapshot_path.write_text(expected_text, encoding="utf-8")
        return

    if not snapshot_path.exists():
        pytest.fail(
            f"snapshot {snapshot_path.relative_to(SNAPSHOT_DIR.parent)} is missing. "
            f"Generate it with `{UPDATE_ENV_VAR}=1 uv run pytest <this-file>`, "
            f"then commit the file."
        )

    actual_text = snapshot_path.read_text(encoding="utf-8")
    assert actual_text == expected_text, (
        f"JSON Schema for {model.__name__} drifted from snapshot at "
        f"{snapshot_path.relative_to(SNAPSHOT_DIR.parent)}. "
        f"If intentional, regenerate via "
        f"`{UPDATE_ENV_VAR}=1 uv run pytest <this-file>` and bump the "
        f"package semver per the discipline in the README."
    )
