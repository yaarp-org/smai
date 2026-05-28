"""Per-major-model round-trip tests for ``smai-api-spec``.

Each test constructs a model instance with realistic values, dumps it to
JSON, loads it back, and asserts equality. Catches obvious wire-shape
breakage (default-value drift, alias config errors, missing fields) at
the smallest unit. The deeper "did the JSON Schema change?" check lives
in :mod:`tests.test_schema_stability`.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
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
    CursorPage,
    EntryAgentStatus,
    EntryDetailResponse,
    EntrySummary,
    EntryWithRuns,
    EvaluationResultResponse,
    HarnessAgentStatus,
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
from smai_api_spec.errors import APIError, ErrorEnvelope, ValidationIssue

# All datetimes use this anchor — fixed instant so JSON snapshots and
# round-trip diffs are deterministic.
NOW = datetime(2026, 5, 3, 12, 0, 0, tzinfo=UTC)


def _roundtrip(model: object) -> None:
    """Dump → JSON → load → equality. ``model`` must be a Pydantic instance."""
    dumped = model.model_dump_json(by_alias=True)  # type: ignore[attr-defined]
    reloaded = type(model).model_validate_json(dumped)  # type: ignore[attr-defined]
    assert reloaded == model


# === Errors =================================================================


def test_error_envelope_roundtrip() -> None:
    env = ErrorEnvelope(
        error=APIError(
            code="VALIDATION_ERROR",
            message="payload failed validation",
            issues=[
                ValidationIssue(loc=["body", "submission_kind"], msg="bad value", type="literal"),
            ],
        )
    )
    _roundtrip(env)


def test_error_envelope_retryable() -> None:
    env = ErrorEnvelope(
        error=APIError(
            code="LLM_UPSTREAM",
            message="bedrock throttled",
            retryable=True,
        )
    )
    _roundtrip(env)


# === Pagination =============================================================


def test_cursor_page_of_proposal_summaries_roundtrip() -> None:
    page: CursorPage[ProposalSummary] = CursorPage[ProposalSummary](
        items=[
            ProposalSummary(
                id="prop_001",
                state="proposal_submitted",
                submission_kind="novel_technique",
                submitted_by="user@example.com",
                created_at=NOW,
                updated_at=NOW,
                version=0,
            )
        ],
        next_cursor="opaque-cursor-blob",
        count=1,
    )
    _roundtrip(page)


# === Events =================================================================


def test_state_change_event_aliases() -> None:
    """``from`` / ``to`` are field aliases — must serialize via the JSON form.

    Construction uses :meth:`BaseModel.model_validate` against the wire-
    shape dict because Pyright resolves Pydantic field aliases as the
    canonical kwarg name (``from`` is a reserved keyword and unusable as
    a kwarg literal); ``populate_by_name=True`` lets ``model_validate``
    accept either form, exercising the same code path the API will use
    when deserializing inbound payloads.
    """
    payload_in: dict[str, object] = {
        "kind": "comparison_group",
        "id": "cg_abc",
        "from": "implementing",
        "to": "implemented",
        "ts": NOW,
    }
    ev = StateChangeEvent.model_validate(payload_in)
    payload_out = ev.model_dump(by_alias=True)
    assert payload_out["from"] == "implementing"
    assert payload_out["to"] == "implemented"
    assert "from_state" not in payload_out
    assert "to_state" not in payload_out
    # JSON form parses back:
    reloaded = StateChangeEvent.model_validate_json(ev.model_dump_json(by_alias=True))
    assert reloaded == ev
    # And ``populate_by_name`` lets the Python attribute names work too
    # (this is what server-side construction will use):
    by_python_name = StateChangeEvent.model_validate(
        {
            "kind": "comparison_group",
            "id": "cg_abc",
            "from_state": "implementing",
            "to_state": "implemented",
            "ts": NOW,
        }
    )
    assert by_python_name == ev


def test_worker_heartbeat_roundtrip() -> None:
    _roundtrip(WorkerHeartbeatEvent(cycle_id=42, cycles_processed=42, ts=NOW))


# === Proposals ==============================================================


def test_submit_proposal_request_novel_technique_dict() -> None:
    req = SubmitProposalRequest(
        submission_kind="novel_technique",
        technique_description={"name": "warmup", "params": {"steps": 1000}},
    )
    _roundtrip(req)


def test_submit_proposal_request_reproduce_paper() -> None:
    req = SubmitProposalRequest(
        submission_kind="reproduce_paper",
        reproduce_paper_arxiv_id="2401.12345",
    )
    _roundtrip(req)


def test_submit_proposal_request_zero_descriptions_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        SubmitProposalRequest(submission_kind="novel_technique")
    assert "got none" in str(excinfo.value)


def test_submit_proposal_request_two_descriptions_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        SubmitProposalRequest(
            submission_kind="reproduce_paper",
            technique_description={"name": "x"},
            reproduce_paper_arxiv_id="2401.12345",
        )
    assert "exactly one" in str(excinfo.value)


def test_submit_proposal_request_kind_mismatch_rejected() -> None:
    """``submission_kind="reproduce_paper"`` requires ``reproduce_paper_arxiv_id``."""
    with pytest.raises(ValidationError):
        SubmitProposalRequest(
            submission_kind="reproduce_paper",
            technique_description={"name": "x"},
        )


def test_submit_proposal_request_arxiv_with_novel_kind_rejected() -> None:
    """``reproduce_paper_arxiv_id`` is only valid with ``submission_kind="reproduce_paper"``."""
    with pytest.raises(ValidationError):
        SubmitProposalRequest(
            submission_kind="novel_technique",
            reproduce_paper_arxiv_id="2401.12345",
        )


def test_proposal_submission_response_roundtrip() -> None:
    resp = ProposalSubmissionResponse(
        id="prop_001",
        state="proposal_submitted",
        submission_kind="novel_technique",
        technique_description_artifact_key="proposals/prop_001/technique_description.json",
        reproduce_paper_arxiv_id=None,
        submitted_by=None,
        created_at=NOW,
    )
    _roundtrip(resp)


def test_proposal_detail_response_roundtrip() -> None:
    resp = ProposalDetailResponse(
        id="prop_001",
        submitted_by=None,
        submission_kind="novel_technique",
        state="registered",
        technique_description_artifact_key="proposals/prop_001/technique_description.json",
        reproduce_paper_arxiv_id=None,
        design_plan_artifact_key="proposals/prop_001/design_plan.json",
        error_context_artifact_key=None,
        design_attempt=1,
        registration_attempt=1,
        user_decision="approved",
        user_decided_at=NOW,
        registered_cg_ids=["cg_001", "cg_002"],
        created_at=NOW,
        updated_at=NOW,
        version=4,
    )
    _roundtrip(resp)


def test_approve_proposal_response_roundtrip() -> None:
    _roundtrip(
        ApproveProposalResponse(
            id="prop_001",
            state="registered",
            cg_ids=["cg_001"],
            user_decided_at=NOW,
        )
    )


def test_reject_proposal_request_optional_reason() -> None:
    _roundtrip(RejectProposalRequest())
    _roundtrip(RejectProposalRequest(reason="not aligned with current quarter goals"))


def test_reject_proposal_response_roundtrip() -> None:
    _roundtrip(
        RejectProposalResponse(id="prop_001", state="rejected", user_decided_at=NOW),
    )


# === Papers =================================================================


def test_submit_paper_request_roundtrip() -> None:
    _roundtrip(SubmitPaperRequest(arxiv_id="2401.12345"))


def test_paper_submission_response_roundtrip() -> None:
    _roundtrip(
        PaperSubmissionResponse(arxiv_id="2401.12345", state="submitted", created_at=NOW),
    )


def test_paper_summary_roundtrip() -> None:
    _roundtrip(
        PaperSummary(
            arxiv_id="2401.12345",
            state="registered",
            title="Adaptive Schemas in Stochastic Optimization",
            created_at=NOW,
            updated_at=NOW,
            version=2,
        )
    )


def test_paper_detail_response_roundtrip() -> None:
    resp = PaperDetailResponse(
        arxiv_id="2401.12345",
        state="registered",
        title="Adaptive Schemas in Stochastic Optimization",
        authors=["A. Author", "B. Author"],
        abstract="We propose...",
        published_date=NOW,
        categories=["cs.LG"],
        latex_bundle_artifact_key="papers/2401.12345/source/bundle.tar.gz",
        expanded_tex_artifact_key="papers/2401.12345/expanded.tex",
        extracted_text_artifact_key="papers/2401.12345/text.txt",
        figures_artifact_key="papers/2401.12345/figures.json",
        screen_result_decision="accept",
        screen_result_reason="reproducible scope",
        technique_buffer_artifact_key="papers/2401.12345/draft_techniques.json",
        error_context_artifact_key=None,
        planning_attempt=1,
        screening_attempt=1,
        registration_attempt=1,
        technique_refs=[
            TechniqueRefSummary(
                technique_id="tech_xyz",
                name="adaptive_step",
                description="adaptive step-size schedule",
            )
        ],
        created_at=NOW,
        updated_at=NOW,
        version=5,
    )
    _roundtrip(resp)


def test_promote_partial_response_roundtrip() -> None:
    _roundtrip(PromotePartialResponse(arxiv_id="2401.12345", state="submitted"))


# === Experiments ============================================================


def test_compile_experiment_request_response_roundtrip() -> None:
    req = CompileExperimentRequest(definition_text="experiment:\n  name: stub\n")
    _roundtrip(req)
    resp = CompileExperimentResponse(
        compilations=[
            CompiledArtifacts(
                cg_id="cg_001",
                experiment_plan={"id": "ep_001"},
                harness_contract={"id": "hc_001"},
                technique_contracts=[{"id": "tc_001"}],
                validation_config={"id": "vc_001"},
            )
        ]
    )
    _roundtrip(resp)


def test_submit_experiment_request_response_roundtrip() -> None:
    _roundtrip(SubmitExperimentRequest(definition_text="experiment:\n  name: stub\n"))
    _roundtrip(
        SubmitExperimentResponse(
            cgs=[CreatedCGRef(cg_id="cg_001", state="draft")],
        ),
    )


# === Comparison groups + entries + runs =====================================


def test_cg_summary_roundtrip() -> None:
    _roundtrip(
        CGSummary(
            id="cg_001",
            proposal_id="prop_001",
            state="implementing",
            is_terminal=False,
            created_at=NOW,
            updated_at=NOW,
            version=3,
        )
    )


def test_cg_status_response_roundtrip() -> None:
    _roundtrip(
        CGStatusResponse(id="cg_001", state="running", is_terminal=False, updated_at=NOW),
    )


def test_run_summary_and_detail_roundtrip() -> None:
    summary = RunSummary(
        id="run_001",
        cg_id="cg_001",
        entry_id="entry_001",
        seed=42,
        state="succeeded",
        duration_seconds=37.5,
        raw_metrics_artifact_key="comparison-groups/cg_001/runs/entry_001/42/metrics.json",
        started_at=NOW,
        completed_at=NOW,
        failure_reason=None,
        run_attempt=1,
        updated_at=NOW,
    )
    _roundtrip(summary)
    detail = RunDetailResponse(
        id="run_001",
        cg_id="cg_001",
        entry_id="entry_001",
        seed=42,
        state="succeeded",
        duration_seconds=37.5,
        raw_metrics_artifact_key="comparison-groups/cg_001/runs/entry_001/42/metrics.json",
        started_at=NOW,
        completed_at=NOW,
        failure_reason=None,
        run_attempt=1,
        created_at=NOW,
        updated_at=NOW,
        version=4,
    )
    _roundtrip(detail)


def test_entry_summary_with_runs_and_detail_roundtrip() -> None:
    run_summary = RunSummary(
        id="run_001",
        cg_id="cg_001",
        entry_id="entry_001",
        seed=42,
        state="succeeded",
        duration_seconds=37.5,
        raw_metrics_artifact_key=None,
        started_at=NOW,
        completed_at=NOW,
        failure_reason=None,
        run_attempt=1,
        updated_at=NOW,
    )
    _roundtrip(
        EntrySummary(
            id="entry_001",
            cg_id="cg_001",
            technique_id="tech_xyz",
            is_baseline=False,
            state="implemented",
            technique_contract_hash=None,
            implementation_attempt=1,
            created_at=NOW,
            updated_at=NOW,
        )
    )
    entry_with_runs = EntryWithRuns(
        id="entry_001",
        cg_id="cg_001",
        technique_id="tech_xyz",
        is_baseline=False,
        state="implemented",
        technique_contract_hash=None,
        harness_api_manifest_hash=None,
        implementation_attempt=1,
        runs=[run_summary],
        created_at=NOW,
        updated_at=NOW,
        version=2,
    )
    _roundtrip(entry_with_runs)
    _roundtrip(
        EntryDetailResponse(
            id="entry_001",
            cg_id="cg_001",
            technique_id="tech_xyz",
            is_baseline=False,
            state="implemented",
            technique_contract_hash=None,
            harness_api_manifest_hash=None,
            implementation_attempt=1,
            runs=[run_summary],
            created_at=NOW,
            updated_at=NOW,
            version=2,
        )
    )


def test_comparison_group_detail_response_roundtrip() -> None:
    run_summary = RunSummary(
        id="run_001",
        cg_id="cg_001",
        entry_id="entry_001",
        seed=42,
        state="succeeded",
        duration_seconds=37.5,
        raw_metrics_artifact_key=None,
        started_at=NOW,
        completed_at=NOW,
        failure_reason=None,
        run_attempt=1,
        updated_at=NOW,
    )
    detail = ComparisonGroupDetailResponse(
        id="cg_001",
        proposal_id="prop_001",
        factor_model_id=None,
        state="complete",
        is_terminal=True,
        experiment_definition_id="ed_001",
        experiment_plan_hash=None,
        harness_contract_hash=None,
        validation_config_hash=None,
        code_review_attempt=1,
        code_review_result_hash=None,
        entries=[
            EntryWithRuns(
                id="entry_001",
                cg_id="cg_001",
                technique_id="tech_xyz",
                is_baseline=False,
                state="implemented",
                technique_contract_hash=None,
                harness_api_manifest_hash=None,
                implementation_attempt=1,
                runs=[run_summary],
                created_at=NOW,
                updated_at=NOW,
                version=2,
            )
        ],
        created_at=NOW,
        updated_at=NOW,
        version=8,
    )
    _roundtrip(detail)


def test_agent_status_response_roundtrip() -> None:
    _roundtrip(
        AgentStatusResponse(
            harness=HarnessAgentStatus(
                role="harness_builder",
                turn_count=4,
                monotonic_timestamp=12.5,
                wall_clock_utc=NOW,
                usage_total={"input_tokens": 10, "output_tokens": 2},
                last_tool_call="write_file",
                tool_errors_fired=0,
                attempt_index=None,
                supervisor_aborted=False,
            ),
            entries={
                "entry_001": EntryAgentStatus(
                    technique_id="tech_xyz",
                    status={"role": "technique_implementer", "turn_count": 6},
                )
            },
        )
    )


def test_evaluation_result_response_roundtrip() -> None:
    _roundtrip(
        EvaluationResultResponse(
            cg_id="cg_001",
            verdict="effect_supported",
            raw_metrics={"baseline": [0.91, 0.92], "treatment": [0.93, 0.95]},
            per_entry={"entry_001": {"verdict": "improvement"}},
            contextual_evaluation=None,
            artifact_key="comparison-groups/cg_001/evaluation_result.json",
        )
    )


def test_artifact_keys_response_roundtrip() -> None:
    _roundtrip(
        ArtifactKeysResponse(
            keys=[
                "comparison-groups/cg_001/harness/contract.json",
                "comparison-groups/cg_001/evaluation_result.json",
            ]
        )
    )


# === System =================================================================


def test_system_version_response_roundtrip() -> None:
    _roundtrip(
        SystemVersionResponse(
            smai_cli="0.1.0",
            smai_core="0.1.0",
            smai_api_spec="0.1.0",
            plugins={"smai-llm-bedrock": "0.1.0"},
        )
    )


def test_system_config_response_roundtrip() -> None:
    _roundtrip(
        SystemConfigResponse(
            config={
                "plugins": {"llm_provider": "bedrock"},
                "engine": {"poll_interval_seconds": 1.0},
            }
        )
    )


def test_system_plugins_response_roundtrip() -> None:
    _roundtrip(
        SystemPluginsResponse(
            llm_providers=[
                PluginInfo(
                    name="bedrock",
                    distribution="smai-llm-bedrock",
                    version="0.1.0",
                    selected=True,
                )
            ],
            metadata_stores=[],
            artifact_stores=[],
            computes=[],
        )
    )


def test_system_verify_request_and_response_roundtrip() -> None:
    _roundtrip(SystemVerifyRequest())
    _roundtrip(SystemVerifyRequest(plugins=["llm_provider", "metadata_store"]))
    _roundtrip(
        SystemVerifyResponse(
            llm_provider=PluginVerifyResult(ok=True, reason="ok", latency_ms=42.0),
            metadata_store=PluginVerifyResult(ok=True, reason="ok", latency_ms=8.0),
            artifact_store=PluginVerifyResult(ok=True, reason="ok", latency_ms=15.0),
            compute=PluginVerifyResult(
                ok=False,
                reason="ConnectionError: backend unreachable",
                latency_ms=None,
            ),
            overall_ok=False,
        )
    )


def test_system_dashboard_response_roundtrip() -> None:
    _roundtrip(
        SystemDashboardResponse(
            counts=SummaryCounts(
                proposals_in_flight=3,
                cgs_in_flight=2,
                runs_in_flight=4,
                papers_in_flight=1,
            ),
            recent_activity=[
                RecentActivityItem(kind="comparison_group", id="cg_001", state="running", ts=NOW)
            ],
        )
    )


def test_system_migrate_status_response_roundtrip() -> None:
    _roundtrip(
        SystemMigrateStatusResponse(
            at_head=False,
            head_revision="0042_user_schema",
            current="0041_initial",
        )
    )


def test_system_health_response_roundtrip() -> None:
    _roundtrip(SystemHealthResponse(status="ok"))


# === extra="forbid" enforcement (sanity check) ==============================


def test_extra_fields_rejected() -> None:
    """``model_config = ConfigDict(extra="forbid")`` is on every model — proves it."""
    with pytest.raises(ValidationError):
        SubmitProposalRequest.model_validate(
            {
                "submission_kind": "novel_technique",
                "technique_description": {"name": "x"},
                "unexpected_field": "should be rejected",
            }
        )
