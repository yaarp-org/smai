""":class:`CodeReviewResult` — code reviewer's locked output schema.

Per ``04-agents.md`` §2.4 (verbatim shape) and §6 (single-call structured
output via ``tool_use``). Consumed by:

* :func:`smai_inline_agents.agents.code_reviewer.run_code_review` — the role-shaped
  wrapper that hands this to :func:`smai_inline_agents.structured_call` as
  ``output_schema``.
* Task 2.C4's ``implemented → running`` gate-rule body, which advances
  the CG iff ``overall_pass=True`` per DEC-016. On
  ``overall_pass=False`` the engine routes critical findings back
  through implementation per §11's retry-with-feedback flow.

The ``overall_pass`` invariant — true iff zero critical findings — is
enforced by a Pydantic validator so the gate-rule caller does not have
to recompute it. v1's reasoning carries forward: silently accepting
inconsistent ``overall_pass=True`` with a ``critical`` finding is the
same family of bug as DEC-018's "silent fallback to passed" failure
mode.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator


class Finding(BaseModel):
    """One reviewer finding — severity-tagged, target-routed.

    Per §2.4 verbatim. ``target_id`` is either an ``entry_id`` or a
    ``technique_id`` (the LLM picks whichever felt more natural in
    context); ``target_kind`` disambiguates. The gate-rule body
    resolves both — looking up ``technique_id → entry_id`` via
    ``MetadataStore`` for ``target_kind='technique'`` findings — and
    routes critical findings accordingly per §11's
    "Finding-to-entry resolution" paragraph.
    """

    model_config = ConfigDict(extra="forbid")

    severity: Literal["critical", "warning", "info"]
    target_id: str
    target_kind: Literal["entry", "technique"]
    summary: str
    detail: str
    suggested_fix: str | None = None


class CodeReviewResult(BaseModel):
    """Locked code-review verdict per §2.4.

    ``overall_pass`` is the boolean the gate rule consumes; the
    invariant it encodes is "no critical findings outstanding."
    Validator check below: ``overall_pass`` must equal
    ``not any(f.severity == "critical" for f in findings)``.

    Why an invariant rather than a computed property: the schema is
    serialized into a Bedrock ``tool_use`` input that the model produces
    — making ``overall_pass`` a real field lets the LLM commit to an
    explicit verdict (and the validator catches the inconsistency loud
    rather than letting a divergent claim slip through). DEC-018's
    no-silent-fallback discipline: an inconsistent ``overall_pass``
    flag is loud, not coerced.
    """

    model_config = ConfigDict(extra="forbid")

    findings: list[Finding]
    overall_pass: bool

    @model_validator(mode="after")
    def _overall_pass_matches_critical_findings(self) -> Self:
        has_critical = any(f.severity == "critical" for f in self.findings)
        expected = not has_critical
        if self.overall_pass != expected:
            raise ValueError(
                f"overall_pass={self.overall_pass!r} is inconsistent with "
                f"findings: {sum(1 for f in self.findings if f.severity == 'critical')} "
                f"critical finding(s) present (expected overall_pass={expected!r})"
            )
        return self


__all__ = ["CodeReviewResult", "Finding"]
