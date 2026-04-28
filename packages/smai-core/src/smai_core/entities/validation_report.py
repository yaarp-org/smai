"""``ValidationError`` and ``ValidationReport`` — Pass-2 finding shapes.

Per ``designs/smai/02-dsl-and-contracts.md`` §5.12. ``ValidationError`` is the
single finding type used by both factor-type plugins (§3) and Pass-2
verification rules (§5); ``ValidationReport`` is the success/error/warning/
advisory bundle returned by the compiler's ``verify(...)`` entry point.

The canonical doc names this type ``ValidationError``, not
``VerificationFinding``. ``VerificationFinding`` is exposed as a type alias so
specs / agents that prefer the more descriptive name resolve to the same
class.

``severity`` defaults to ``"error"`` so factor-type plugins (which only emit
error-class checks per §3.2 / §3.3) can construct findings with just
``code``/``message``/``location``; Pass-2 rules pass severity explicitly.
"""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict

RuleSeverity: TypeAlias = Literal["error", "warning", "advisory"]
"""The three severity tiers per §5: ``error`` blocks compilation; ``warning``
proceeds with a recorded warning; ``advisory`` proceeds quietly (verbose-only)."""


class ValidationError(BaseModel):
    """One methodology validation finding.

    The ``code`` field is the central key for documentation lookup and agent
    re-prompting; plugins use the ``<plugin_name>.<rule>`` namespacing
    convention (§3.6) and Pass-2 rules use ``<category>.<rule>`` (§5.2–§5.10).
    """

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    location: str
    severity: RuleSeverity = "error"
    suggested_fix: str | None = None
    related_entity: dict[str, Any] | None = None


VerificationFinding: TypeAlias = ValidationError
"""Alias for :class:`ValidationError`.

The canonical name in §5.12 is ``ValidationError``; this alias is provided so
docs / agents that use the more descriptive ``VerificationFinding`` resolve
to the same class without forcing a rename across factor-type plugins (§3).
"""


class ValidationReport(BaseModel):
    """The Pass-2 verifier's structured output.

    Per §5.12: ``success`` is True iff no ``error``-severity findings fired.
    ``errors`` / ``warnings`` / ``advisories`` partition the finding list by
    severity and preserve insertion order so the report is deterministic per
    §5.13.
    """

    model_config = ConfigDict(extra="forbid")

    success: bool
    errors: list[ValidationError] = []
    warnings: list[ValidationError] = []
    advisories: list[ValidationError] = []
