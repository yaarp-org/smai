"""``ValidationError`` — methodology validation finding shape.

Per ``designs/smai/02-dsl-and-contracts.md`` §5.12. Returned by factor-type
plugins (§3) and verification rules (§5). Severity defaults to ``"error"`` so
plugins (which emit error-class checks per §3.2 / §3.3) can construct
findings with just ``code``/``message``/``location``; the broader Pass-2
verifier (Task 1.5) will pass severity explicitly for warnings/advisories.

``ValidationReport`` is intentionally not defined here yet — Task 1.5 owns
the report-shape and the verifier entry point.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class ValidationError(BaseModel):
    """One methodology validation finding.

    The ``code`` field is the central key for documentation lookup and agent
    re-prompting; plugins use the ``<plugin_name>.<rule>`` namespacing
    convention (§3.6).
    """

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    location: str | None = None
    severity: Literal["error", "warning", "advisory"] = "error"
    suggested_fix: str | None = None
    related_entity: dict[str, Any] | None = None
