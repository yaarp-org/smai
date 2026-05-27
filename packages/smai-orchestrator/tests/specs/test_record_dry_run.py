"""Tests for the planner dry-run record-shape check (round-9 fix B).

Unit tests against
:func:`smai_orchestrator.specs._record_dry_run.dry_run_record_construction`.
End-to-end behavior — the planner's ``finalize_plan`` rejecting a buffer
whose projected records would fail Pydantic — lives in the planner test
suite (``packages/smai-inline-agents/tests/test_planner.py``); this file covers
the helper in isolation so unit-level regressions on either layer
(methodology projection vs record dry-run) surface independently.
"""

from __future__ import annotations

from typing import Any

import pytest
from smai_orchestrator.specs._record_dry_run import dry_run_record_construction
from smai_orchestrator.specs.proposal import _generate_cg_id, make_default_cg_id


def _build_valid_buffer(draft_cg_id: str = "cg-1") -> dict[str, Any]:
    """Build a well-shaped raw planner-buffer dict for the dry-run input."""
    return {
        "proposal_id": "prop-1",
        "techniques": {},
        "comparison_groups": [
            {
                "id": draft_cg_id,
                "factor_dimension": "augmentation",
                "factor_type": "additive",
                "factor_description": "d",
                "controlled_conditions": {
                    "dataset": {"name": "MNIST"},
                    "optimization": {"optimizer": "sgd", "lr": 0.1},
                    "seeds": [0, 1],
                },
                "entries": [
                    {
                        "id": "entry-baseline",
                        "is_baseline": True,
                        "level": {
                            "factor": "augmentation",
                            "name": "absent",
                            "technique_symbolic_name": None,
                        },
                    },
                    {
                        "id": "entry-treatment",
                        "is_baseline": False,
                        "level": {
                            "factor": "augmentation",
                            "name": "cutout",
                            "technique_symbolic_name": "tech-cutout",
                        },
                    },
                ],
                "validation": {
                    "metric": {"kind": "atomic", "ref": "accuracy"},
                    "direction": "higher_is_better",
                    "aggregation_method": "mean",
                    "comparison_rule": "compare_to_baseline",
                    "threshold": 0.0,
                    "seed_count_required": 1,
                },
            }
        ],
        "finalized": True,
    }


def test_dry_run_passes_for_well_shaped_buffer() -> None:
    """Happy path: every projected record validates → no errors."""
    cg_id = _generate_cg_id()
    buffer = _build_valid_buffer()
    errors = dry_run_record_construction(
        buffer=buffer,
        proposal_id="prop-1",
        cg_id_for=lambda p, d: cg_id,
    )
    assert errors == [], f"expected no errors; got {errors}"


def test_dry_run_rejects_entry_id_with_whitespace() -> None:
    """An entry id containing a space passes the methodology layer
    (``Entry.id`` has no format constraint) but fails EntryRecord's
    ``validate_id_format`` — the dry-run surfaces it as a field-path error
    so the planner re-prompt can self-correct."""
    cg_id = _generate_cg_id()
    buffer = _build_valid_buffer()
    buffer["comparison_groups"][0]["entries"][0]["id"] = "entry with space"

    errors = dry_run_record_construction(
        buffer=buffer,
        proposal_id="prop-1",
        cg_id_for=lambda p, d: cg_id,
    )
    assert errors, "expected at least one error for whitespace-laced entry id"
    joined = " | ".join(errors)
    assert "entry with space" in joined
    assert "entries" in joined


def test_dry_run_uses_supplied_cg_id_for_resolver() -> None:
    """The dry-run honors the caller-supplied id resolver — confirming the
    helper uses the SAME id-construction path the real registration handler
    will use."""
    captured: list[tuple[str, str]] = []

    def resolver(proposal_id: str, draft_cg_id: str) -> str:
        captured.append((proposal_id, draft_cg_id))
        return f"resolved-{proposal_id}-{draft_cg_id}"

    buffer = _build_valid_buffer(draft_cg_id="cg-draft-id")
    errors = dry_run_record_construction(
        buffer=buffer,
        proposal_id="prop-x",
        cg_id_for=resolver,
    )
    assert errors == []
    assert captured == [("prop-x", "cg-draft-id")]


def test_dry_run_default_resolver_produces_ulid_shaped_ids() -> None:
    """When the dry-run runs the default ``make_default_cg_id``, it
    generates fresh ULID-shaped ids per call — passing the 64-char cap
    even when the draft id is long."""
    long_draft = "cg-batchnorm-vs-dropout-mlp-fashionmnist-deep-comparison-id"
    proposal_id = "prop-with-long-id-12345-suffix-abc"
    buffer = _build_valid_buffer(draft_cg_id=long_draft)
    # Sanity: the legacy ``f"{proposal}--{draft}"`` shape would exceed 64.
    assert len(f"{proposal_id}--{long_draft}") > 64
    errors = dry_run_record_construction(
        buffer=buffer,
        proposal_id=proposal_id,
        cg_id_for=make_default_cg_id,
    )
    assert errors == [], f"default resolver should yield ULID-shaped ids; got {errors}"


@pytest.mark.parametrize("no_arg", [None])
def test_dry_run_does_not_touch_metadata_store(no_arg: Any) -> None:
    """The dry-run is pure-Python — confirming there's no I/O dependency."""
    del no_arg
    buffer = _build_valid_buffer()
    # No store argument, no fixture — just confirm the call runs to
    # completion synchronously without raising.
    dry_run_record_construction(
        buffer=buffer,
        proposal_id="prop-no-store",
        cg_id_for=make_default_cg_id,
    )
