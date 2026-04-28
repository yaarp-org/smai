"""§5.10 — Pipeline / sequence rejection (advisory v1).

Per DEC-031 sub-decision #10: pipeline / sequence factors are out of scope
for v1. Experimenters model them as flat substitutive factors over named
recipes. These advisory rules forward-look at v1→v2 evolution by flagging
substitutive factors whose level names parse as multi-component combinations
or sequence-style names; they NEVER block compilation.

The two rules live in this dedicated module rather than in Category A so the
"these are forward-looking advisories, not factor-structure errors"
distinction is visible in the file layout.
"""

from __future__ import annotations

from smai_core.entities.experiment import ExperimentDefinition
from smai_core.entities.registries import Registries
from smai_core.entities.validation_report import ValidationError

_PIPELINE_PARTITION_CHARS = (".", "+", "-")
_SEQUENCE_MARKERS = ("_then_", "->", "__then__")


def factor_suspected_pipeline_encoding(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``factor.suspected_pipeline_encoding`` (advisory).

    Substitutive factor with > 8 levels whose level names parse as
    multi-component combinations (dots, plus signs, hyphens partitioning
    into 2+ segments) hint at a flat encoding of a pipeline factor that may
    fit a vN ``pipeline`` factor type better. Pure heuristic; never blocks.
    """
    findings: list[ValidationError] = []
    if not experiment.factors:
        return findings
    factor = experiment.factors[0]
    if factor.type != "substitutive" or len(experiment.entries) <= 8:
        return findings
    multi_component = sum(
        1
        for entry in experiment.entries
        if any(
            ch in entry.level.name and len(entry.level.name.split(ch)) >= 2
            for ch in _PIPELINE_PARTITION_CHARS
        )
    )
    if multi_component >= 2:
        findings.append(
            ValidationError(
                code="factor.suspected_pipeline_encoding",
                message=(
                    f"Substitutive factor '{factor.name}' has {len(experiment.entries)} "
                    f"levels and {multi_component} level names look like multi-component "
                    f"combinations (dots, '+', '-'). v1 admits this; vN may offer a "
                    f"first-class pipeline factor type."
                ),
                location=f"factors[name={factor.name}]",
                severity="advisory",
            )
        )
    return findings


def factor_suspected_sequence_encoding(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``factor.suspected_sequence_encoding`` (advisory).

    Mirrors ``factor.suspected_pipeline_encoding`` for sequence-style names
    (``stage1_then_stage2``, ``pretrain_then_finetune``).
    """
    findings: list[ValidationError] = []
    if not experiment.factors:
        return findings
    factor = experiment.factors[0]
    if factor.type != "substitutive":
        return findings
    sequence_like = [
        entry.id
        for entry in experiment.entries
        if any(marker in entry.level.name for marker in _SEQUENCE_MARKERS)
    ]
    if sequence_like:
        findings.append(
            ValidationError(
                code="factor.suspected_sequence_encoding",
                message=(
                    f"Substitutive factor '{factor.name}' has level names that look "
                    f"like ordered stages on entries {sequence_like!r}. v1 admits this; "
                    f"vN may offer a first-class sequence factor type."
                ),
                location=f"factors[name={factor.name}]",
                severity="advisory",
            )
        )
    return findings


PIPELINE_REJECTION_RULES = (
    factor_suspected_pipeline_encoding,
    factor_suspected_sequence_encoding,
)
"""§5.10 pipeline / sequence rejection rules; advisory only, never block."""
