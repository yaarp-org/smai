"""Pydantic round-trip + invariant tests for the single-call agent schemas.

Per ``04-agents.md`` §2.4 / §2.5 (schema definitions) and DEC-018
(no-silent-fallback discipline). The :class:`CodeReviewResult` invariant
— ``overall_pass`` true iff zero ``critical`` findings — is the v1
load-bearing example: the silently-coerced ``overall_pass`` was the
exact bug that motivated DEC-018; we encode it as a validator so the
gate-rule caller does not have to recompute it.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError
from smai_inline_agents import (
    CodeReviewResult,
    ContextualVerdict,
    EntryRanking,
    Finding,
)


def test_finding_round_trip() -> None:
    finding = Finding(
        severity="critical",
        target_id="entry-1",
        target_kind="entry",
        summary="missing softmax",
        detail="The classifier head returns raw logits; cross-entropy expects probabilities.",
        suggested_fix="apply nn.LogSoftmax(dim=-1) before NLLLoss",
    )
    revived = Finding.model_validate_json(finding.model_dump_json())
    assert revived == finding


def test_finding_severity_locked_to_v1_set() -> None:
    with pytest.raises(PydanticValidationError):
        Finding.model_validate(
            {
                "severity": "blocker",  # not in the locked enum
                "target_id": "entry-1",
                "target_kind": "entry",
                "summary": "x",
                "detail": "y",
            }
        )


def test_finding_target_kind_locked_to_v1_set() -> None:
    with pytest.raises(PydanticValidationError):
        Finding.model_validate(
            {
                "severity": "info",
                "target_id": "x",
                "target_kind": "module",  # not in the locked enum
                "summary": "x",
                "detail": "y",
            }
        )


def test_code_review_result_pass_with_no_critical_findings() -> None:
    result = CodeReviewResult(
        findings=[
            Finding(
                severity="warning",
                target_id="entry-1",
                target_kind="entry",
                summary="non-blocking lint nit",
                detail="prefer torch.nn.functional.relu over torch.relu",
            ),
        ],
        overall_pass=True,
    )
    assert result.overall_pass is True
    revived = CodeReviewResult.model_validate_json(result.model_dump_json())
    assert revived == result


def test_code_review_result_fail_with_critical_finding() -> None:
    result = CodeReviewResult(
        findings=[
            Finding(
                severity="critical",
                target_id="entry-1",
                target_kind="entry",
                summary="returns wrong shape",
                detail="apply() returns a list; harness expects a callable",
            ),
        ],
        overall_pass=False,
    )
    assert result.overall_pass is False


def test_code_review_result_validator_rejects_inconsistent_pass() -> None:
    """DEC-018: silent ``overall_pass=True`` despite a critical finding is the
    exact failure mode the structured-output discipline exists to prevent."""
    with pytest.raises(PydanticValidationError):
        CodeReviewResult(
            findings=[
                Finding(
                    severity="critical",
                    target_id="entry-1",
                    target_kind="entry",
                    summary="critical bug",
                    detail="x",
                ),
            ],
            overall_pass=True,
        )


def test_code_review_result_validator_rejects_inconsistent_fail() -> None:
    """The mirror invariant: ``overall_pass=False`` is loud when there are
    no critical findings — the LLM has not committed to a real verdict."""
    with pytest.raises(PydanticValidationError):
        CodeReviewResult(
            findings=[
                Finding(
                    severity="warning",
                    target_id="entry-1",
                    target_kind="entry",
                    summary="x",
                    detail="y",
                ),
            ],
            overall_pass=False,
        )


def test_code_review_result_round_trip_preserves_invariant() -> None:
    """The validator runs on revival too — a tampered JSON cannot be revived."""
    payload = (
        '{"findings": [{"severity": "critical", "target_id": "e", '
        '"target_kind": "entry", "summary": "x", "detail": "y"}], '
        '"overall_pass": true}'
    )
    with pytest.raises(PydanticValidationError):
        CodeReviewResult.model_validate_json(payload)


def test_contextual_verdict_round_trip() -> None:
    verdict = ContextualVerdict(
        overall_verdict="promising",
        summary="treatment beat baseline by 5% with consistent seed-level results",
        rankings=[
            EntryRanking(entry_id="t1", rank=1, rationale="highest mean accuracy"),
            EntryRanking(entry_id="b", rank=2, rationale="baseline reference"),
        ],
        insights=["all three seeds agreed in direction"],
        limitations=["only three seeds; CIs are wide"],
        suggested_follow_ups=["replicate with five seeds to tighten CIs"],
    )
    revived = ContextualVerdict.model_validate_json(verdict.model_dump_json())
    assert revived == verdict


def test_contextual_verdict_locked_overall_verdict_enum() -> None:
    with pytest.raises(PydanticValidationError):
        ContextualVerdict.model_validate(
            {
                "overall_verdict": "weak_pass",  # not in v1 set
                "summary": "x",
                "rankings": [],
                "insights": [],
                "limitations": [],
                "suggested_follow_ups": [],
            }
        )


def test_contextual_verdict_admits_empty_lists() -> None:
    """Inconclusive runs may legitimately have no rankings or insights to
    surface; the schema does not require non-empty lists."""
    v = ContextualVerdict(
        overall_verdict="inconclusive",
        summary="too few seeds completed; cannot interpret",
        rankings=[],
        insights=[],
        limitations=["seed_count_required=5; only 1 seed completed"],
        suggested_follow_ups=["re-run after diagnosing the run-time crashes"],
    )
    assert v.overall_verdict == "inconclusive"
