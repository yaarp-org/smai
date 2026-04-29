"""Shared fixtures for Task 3.E1 spec tests (proposal pipeline).

Per the brief's test-fixture filename hygiene note (workspace
``--import-mode=importlib`` puts every package's tests/ on sys.path —
cross-package fixture filenames must not collide). ``_e1_*`` prefix
keeps these distinct from ``_specs_fakes`` (Task 2.C4) and
``_e3_fakes`` (Task 3.E3).

Builders shipped here:

* :func:`build_finalized_buffer_payload` — a dict shaped like a
  finalized :class:`smai_agents.PlannerBuffer` JSON, ready to write to
  ArtifactStore as the planner-side output. Used by spec tests that
  bypass the agent loop and pre-stage a buffer.
* :func:`make_planner_responses_for_finalize` — a list of canned
  :class:`ModelResponse`s the StubLlmProvider replays to drive the
  planner agent through a happy-path finalize sequence (one
  set_classification + N create_technique + N draft_comparison +
  set_conditions + draft_assertion + finalize_plan + finish).
* :func:`make_planner_session_runner` — a session-runner callable
  matching the :data:`smai_agents.agents.planner.SessionRunner` shape
  for tests that want to skip the loop entirely and inject a fake
  outcome + buffer mutation.
* :func:`make_proposal_record` — :class:`ProposalRecord` builder.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from smai_core.plugins import ModelResponse
from smai_orchestrator.entities.tracking import ProposalRecord


def make_proposal_record(
    *,
    proposal_id: str,
    state: str = "proposal_submitted",
    submission_kind: str = "novel_technique",
) -> ProposalRecord:
    now = datetime.now(UTC)
    return ProposalRecord(
        id=proposal_id,
        submitted_by=None,
        submission_kind=submission_kind,  # type: ignore[arg-type]
        technique_description_artifact_key=None,
        reproduce_paper_arxiv_id=None,
        state=state,  # type: ignore[arg-type]
        version=0,
        created_at=now,
        updated_at=now,
    )


def build_finalized_buffer_payload(
    *,
    proposal_id: str,
    cg_drafts: list[dict[str, Any]],
    finalized: bool = True,
) -> dict[str, Any]:
    """Build a JSON-shaped buffer payload.

    Each entry in ``cg_drafts`` should be a mapping with at least
    ``id``, ``factor_dimension``, ``factor_type`` keys; optional keys
    fill in with sensible defaults.
    """
    techniques: dict[str, dict[str, Any]] = {}
    cgs: list[dict[str, Any]] = []

    for draft in cg_drafts:
        cg_id = draft["id"]
        factor_dim = draft["factor_dimension"]
        factor_type = draft["factor_type"]
        # Build entries: 1 baseline + 1 treatment per CG. For additive
        # CGs the baseline has technique_symbolic_name=None per DEC-013.
        treatment_symbol = f"{cg_id}-tech-treatment"
        techniques[treatment_symbol] = {
            "symbolic_name": treatment_symbol,
            "name": f"{cg_id} treatment",
            "description": f"treatment for {cg_id}",
            "category": factor_dim,
            "compatible_factor_types": [factor_type],
            "standard": False,
            "fidelity_anchor": {"kind": "proposal", "proposal_id": proposal_id},
            "affects_extension_points": [
                "train_transforms" if factor_type == "additive" else "model_wrapper"
            ],
        }
        baseline_entry: dict[str, Any] = {
            "id": f"{cg_id}-entry-baseline",
            "is_baseline": True,
            "level": {
                "factor": factor_dim,
                "name": "baseline",
                "technique_symbolic_name": None if factor_type == "additive" else treatment_symbol,
            },
        }
        treatment_entry: dict[str, Any] = {
            "id": f"{cg_id}-entry-treatment",
            "is_baseline": False,
            "level": {
                "factor": factor_dim,
                "name": "treatment",
                "technique_symbolic_name": treatment_symbol,
            },
        }
        cgs.append(
            {
                "id": cg_id,
                "hypothesis": draft.get("hypothesis", "test hypothesis"),
                "factor_dimension": factor_dim,
                "factor_type": factor_type,
                "factor_description": "test factor",
                "controlled_conditions": draft.get(
                    "controlled_conditions",
                    {
                        "dataset": {"name": "cifar10", "split": "train", "version": "v1"},
                        "optimization": {"optimizer": "sgd", "lr": 0.1},
                        "seeds": [1, 2, 3],
                    },
                ),
                "entries": [baseline_entry, treatment_entry],
                "validation": draft.get(
                    "validation",
                    {
                        "metric": {"ref": "accuracy", "kind": "atomic"},
                        "direction": "higher_is_better",
                        "aggregation_method": "mean",
                        "comparison_rule": "compare_to_baseline",
                        "threshold": 0.01,
                        "seed_count_required": 3,
                    },
                ),
            }
        )

    return {
        "proposal_id": proposal_id,
        "paper_arxiv_id": None,
        "techniques": techniques,
        "comparison_groups": cgs,
        "classification": {
            "factor_dimension": cg_drafts[0]["factor_dimension"],
            "factor_type": cg_drafts[0]["factor_type"],
            "rationale": "test classification",
        },
        "follow_ups": [],
        "finalized": finalized,
    }


def make_planner_responses_for_finalize(
    *,
    proposal_id: str,
    cg_drafts: list[dict[str, Any]],
) -> list[ModelResponse]:
    """Build the canned LLM-response sequence that drives the planner
    through a complete finalize.

    For each draft CG: ``draft_create_technique`` (treatment) +
    ``draft_comparison`` + ``set_conditions`` + ``draft_assertion``.
    Finally one ``set_classification`` (issued first), one
    ``finalize_plan``, and one ``finish``.
    """
    from _agent_helpers import model_response  # type: ignore[import-not-found]  # noqa: PLC0415

    responses: list[ModelResponse] = [
        model_response(
            tool_uses=[
                (
                    "tu-set-classification",
                    "set_classification",
                    {
                        "factor_dimension": cg_drafts[0]["factor_dimension"],
                        "factor_type": cg_drafts[0]["factor_type"],
                        "rationale": "test classification",
                    },
                ),
            ],
            stop_reason="tool_use",
        ),
    ]

    for draft in cg_drafts:
        cg_id = draft["id"]
        factor_dim = draft["factor_dimension"]
        factor_type = draft["factor_type"]
        treatment_symbol = f"{cg_id}-tech-treatment"

        # draft_create_technique (treatment).
        responses.append(
            model_response(
                tool_uses=[
                    (
                        f"tu-create-{cg_id}",
                        "draft_create_technique",
                        {
                            "symbolic_name": treatment_symbol,
                            "name": f"{cg_id} treatment",
                            "description": f"treatment for {cg_id}",
                            "category": factor_dim,
                            "compatible_factor_types": [factor_type],
                            "standard": False,
                            "fidelity_anchor": {
                                "kind": "proposal",
                                "proposal_id": proposal_id,
                            },
                            "affects_extension_points": [
                                "train_transforms" if factor_type == "additive" else "model_wrapper"
                            ],
                        },
                    ),
                ],
                stop_reason="tool_use",
            )
        )
        # draft_comparison.
        responses.append(
            model_response(
                tool_uses=[
                    (
                        f"tu-comparison-{cg_id}",
                        "draft_comparison",
                        {
                            "id": cg_id,
                            "hypothesis": draft.get("hypothesis", "test hypothesis"),
                            "factor_dimension": factor_dim,
                            "factor_type": factor_type,
                            "factor_description": "test factor",
                            "entries": [
                                {
                                    "id": f"{cg_id}-entry-baseline",
                                    "is_baseline": True,
                                    "level": {
                                        "factor": factor_dim,
                                        "name": "baseline",
                                        "technique_symbolic_name": None
                                        if factor_type == "additive"
                                        else treatment_symbol,
                                    },
                                },
                                {
                                    "id": f"{cg_id}-entry-treatment",
                                    "is_baseline": False,
                                    "level": {
                                        "factor": factor_dim,
                                        "name": "treatment",
                                        "technique_symbolic_name": treatment_symbol,
                                    },
                                },
                            ],
                        },
                    ),
                ],
                stop_reason="tool_use",
            )
        )
        # set_conditions.
        responses.append(
            model_response(
                tool_uses=[
                    (
                        f"tu-conditions-{cg_id}",
                        "set_conditions",
                        {
                            "cg_id": cg_id,
                            "conditions": {
                                "dataset": {
                                    "name": "cifar10",
                                    "split": "train",
                                    "version": "v1",
                                },
                                "optimization": {"optimizer": "sgd", "lr": 0.1},
                                "seeds": [1, 2, 3],
                            },
                        },
                    ),
                ],
                stop_reason="tool_use",
            )
        )
        # draft_assertion.
        responses.append(
            model_response(
                tool_uses=[
                    (
                        f"tu-assertion-{cg_id}",
                        "draft_assertion",
                        {
                            "cg_id": cg_id,
                            "validation": {
                                "metric": {"ref": "accuracy", "kind": "atomic"},
                                "direction": "higher_is_better",
                                "aggregation_method": "mean",
                                "comparison_rule": "compare_to_baseline",
                                "threshold": 0.01,
                                "seed_count_required": 3,
                            },
                        },
                    ),
                ],
                stop_reason="tool_use",
            )
        )

    # finalize_plan + finish.
    responses.extend(
        [
            model_response(
                tool_uses=[("tu-finalize", "finalize_plan", {})],
                stop_reason="tool_use",
            ),
            model_response(
                tool_uses=[
                    (
                        "tu-finish",
                        "finish",
                        {"success": True, "summary": "planner finalized"},
                    ),
                ],
                stop_reason="tool_use",
            ),
        ]
    )
    return responses


def make_planner_session_runner(*, finalized: bool = True):  # type: ignore[no-untyped-def]
    """Test-only :data:`SessionRunner` that bypasses the agent loop.

    Returns a session-runner callable that, when invoked, immediately
    returns a synthetic :class:`AgentOutcome` without driving the loop.
    Useful for spec-level tests that want to assert on the dispatch
    handler's pre/post behavior without exercising the full multi-turn
    machinery.

    NB: this stub does NOT mutate the planner buffer or persist
    artifacts — tests that want a finalized buffer at a known
    artifact key should use :func:`build_finalized_buffer_payload` and
    pre-write the artifact directly.
    """
    from smai_agents.loop import AgentOutcome  # noqa: PLC0415
    from smai_core.plugins import TokenUsage  # noqa: PLC0415

    async def _runner(_session: Any) -> AgentOutcome:
        del _session
        return AgentOutcome(
            kind="finished",
            turn_count=0,
            usage_total=TokenUsage(input_tokens=0, output_tokens=0),
            finish_summary=None,
            finish_success=finalized,
        )

    return _runner


__all__ = [
    "build_finalized_buffer_payload",
    "make_planner_responses_for_finalize",
    "make_planner_session_runner",
    "make_proposal_record",
]
