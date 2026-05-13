"""Dry-run record-shape validation for the planner (round-9 fix B).

The planner's ``finalize_plan`` calls
:func:`smai_core.project_buffer_to_documents` to surface methodology-layer
Pydantic errors (e.g. ``controlled_conditions.dataset`` shape) back to the
agent so it self-corrects via the round-6 re-prompt loop. That check is
*methodology-layer* (smai-core); the orchestrator-layer record constructors
(:class:`ComparisonGroupRecord`, :class:`EntryRecord`) have their own
Pydantic validators (id-format, content-hash format) that the methodology
projection can't see — so a buffer can pass the methodology check and still
fail registration on a record-level validator.

This helper closes that gap. Given the raw planner-buffer dict (the same
shape the real registration handler reads from ArtifactStore), it walks
the ``comparison_groups`` and ``entries`` exactly like
:func:`smai_orchestrator.specs.proposal._register_buffer` does and
constructs in-memory :class:`ComparisonGroupRecord` + :class:`EntryRecord`
instances using the SAME id-construction code path — but WITHOUT touching
:class:`MetadataStore` or :class:`ArtifactStore`. Any Pydantic validation
failure is returned as a structured error string for the planner.

Layering: this module lives in smai-orchestrator because the record
classes do (per ``01-data-model.md`` §5.1 / DEC-029). smai-agents'
``_finalize_plan_handler`` lazy-imports it, matching the existing
``from smai_orchestrator.engine.types import DispatchOutcome`` lazy-import
shape in :mod:`smai_agents.agents.planner`. smai-core stays clean of
pipeline imports.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, cast

from pydantic import ValidationError

from smai_orchestrator.entities.tracking import ComparisonGroupRecord, EntryRecord

# Placeholder content-hash used by the dry-run check. The real registration
# handler fills these from the compiler's :class:`ContractArtifactSet`
# (sha256 of the canonical-JSON envelope); the dry-run doesn't run the
# compiler so we feed a 64-char lowercase-hex string that satisfies the
# content-hash validator. The only thing we're really exercising here is
# the id-format + structural validators on every other field.
_PLACEHOLDER_CONTENT_HASH = "0" * 64


def dry_run_record_construction(
    *,
    buffer: dict[str, Any],
    proposal_id: str,
    cg_id_for: Callable[[str, str], str],
) -> list[str]:
    """Try to construct CG + Entry records for each draft; return error strings.

    Parameters mirror :func:`smai_orchestrator.specs.proposal._register_buffer`'s
    inputs so the dry-run and the real registration surface identical
    record-level errors. ``cg_id_for`` resolves
    ``(proposal_id, draft_cg_id) → cg_id`` (the real registration's
    resolver, post-A typically a fresh ULID-shaped id); ``buffer`` is the
    raw dict shape persisted as ``design_plan.json``.

    Returns an empty list when every record passes Pydantic. On failure,
    each entry is a one-line string keyed by the offending field path —
    same shape as
    :func:`smai_core.project_buffer_to_documents`'s error list, so the
    planner can join them and forward to the agent.
    """
    errors: list[str] = []
    now = datetime.now(UTC)
    raw_drafts = cast(list[dict[str, Any]], buffer.get("comparison_groups") or [])
    for draft in raw_drafts:
        draft_cg_id = cast(str, draft.get("id") or "draft-cg")
        cg_id = cg_id_for(proposal_id, draft_cg_id)
        try:
            ComparisonGroupRecord(
                id=cg_id,
                proposal_id=proposal_id,
                factor_model_id=None,
                symbolic_id=draft_cg_id,
                experiment_definition_id=cg_id,
                experiment_plan_hash=_PLACEHOLDER_CONTENT_HASH,
                harness_contract_hash=_PLACEHOLDER_CONTENT_HASH,
                validation_config_hash=_PLACEHOLDER_CONTENT_HASH,
                state="draft",
                created_at=now,
                updated_at=now,
            )
        except (ValidationError, ValueError, TypeError) as exc:
            errors.append(f"comparison_groups[{draft_cg_id!r}]: {_format_error(exc)}")

        raw_entries = cast(list[dict[str, Any]], draft.get("entries") or [])
        for raw_entry in raw_entries:
            entry_id = cast(str, raw_entry.get("id") or "entry")
            is_baseline = bool(raw_entry.get("is_baseline", False))
            raw_level = cast(dict[str, Any], raw_entry.get("level") or {})
            symbolic = cast(str | None, raw_level.get("technique_symbolic_name"))
            # Mirror :func:`smai_core.draft_cg_to_experiment_document`'s
            # symbolic-name → technique_id rule: the symbolic name IS the
            # technique_id when no external ref is found, which is the
            # surface the EntryRecord validator sees post-registration.
            technique_id = symbolic
            is_additive_baseline = is_baseline and technique_id is None
            try:
                EntryRecord(
                    id=entry_id,
                    cg_id=cg_id,
                    technique_id=technique_id,
                    is_baseline=is_baseline,
                    entry_id=entry_id,
                    technique_contract_hash=_PLACEHOLDER_CONTENT_HASH,
                    state="implemented" if is_additive_baseline else "pending",
                    created_at=now,
                    updated_at=now,
                )
            except (ValidationError, ValueError, TypeError) as exc:
                errors.append(
                    f"comparison_groups[{draft_cg_id!r}].entries[{entry_id!r}]: "
                    f"{_format_error(exc)}"
                )
    return errors


def _format_error(exc: Exception) -> str:
    """Render a record-construction error with field paths.

    Same shape as :func:`smai_core.dsl.buffer_projection._format_projection_error`
    so the agent gets consistent error messages whether the failure is
    methodology-layer or record-layer.
    """
    if isinstance(exc, ValidationError):
        parts: list[str] = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", ()))
            msg = err.get("msg", "")
            input_value = err.get("input")
            value_hint = (
                f" (got {type(input_value).__name__} {input_value!r})"
                if input_value is not None
                else ""
            )
            parts.append(f"{loc}: {msg}{value_hint}")
        return "; ".join(parts) if parts else str(exc)
    return f"{type(exc).__name__}: {exc}"


__all__ = ["dry_run_record_construction"]
