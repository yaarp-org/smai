"""Types and aliases used across the verification module.

Per ``designs/smai/02-dsl-and-contracts.md`` §5. The Pass-2 verifier composes
~35–50 rule callables; each callable has the signature
``(experiment, registries) -> list[ValidationError]``. The verifier collects
the findings, partitions by severity into a :class:`ValidationReport`, and
either returns a :class:`VerifiedExperimentDefinition` (no errors) or raises
:class:`VerificationError` (one or more error-severity findings).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from smai_core.entities.experiment import ExperimentDefinition
from smai_core.entities.registries import Registries
from smai_core.entities.validation_report import (
    RuleSeverity,
    ValidationError,
    ValidationReport,
    VerificationFinding,
)

__all__ = [
    "RuleCallable",
    "RuleSeverity",
    "ValidationError",
    "ValidationReport",
    "VerificationError",
    "VerificationFinding",
]


RuleCallable: TypeAlias = Callable[[ExperimentDefinition, Registries], list[ValidationError]]
"""Signature for a Pass-2 verification rule.

A rule receives the (Pydantic-validated) ``ExperimentDefinition`` and the
loaded ``Registries`` bundle, and returns a flat list of findings. An empty
list means the rule passed. Rules MUST be pure functions (§5.13)."""


class VerificationError(Exception):
    """Raised by ``verify(...)`` when one or more error-severity rules fired.

    The full :class:`ValidationReport` (including warnings and advisories) is
    attached so callers can surface or replay the entire result. The string
    form lists ``code``s for each error finding to make exception-driven
    debugging readable.
    """

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        codes = ", ".join(err.code for err in report.errors) or "<no codes>"
        super().__init__(f"verification failed with {len(report.errors)} error(s): {codes}")
