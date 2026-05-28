"""Planner-buffer → DSL document projection.

The planner agent's buffer is a free-form JSON shape; the registration
step in the proposal pipeline-spec must convert that shape into typed
:class:`ExperimentDocument` instances (and :class:`TechniqueRef`s) before
the methodology compiler runs.

Pre-round-8 the projection lived inline in
``smai_orchestrator.specs.proposal`` — which meant Pydantic shape errors
only surfaced at registration time (after human approval), wedging the
proposal in ``designed`` if the bug was on the projection side (e.g. a
bare-string ``controlled_conditions.dataset`` instead of the typed
``dict[str, str]`` :class:`ControlledConditions` requires). Lifting the
helpers into smai-core lets the planner's ``finalize_plan`` tool run the
same projection at draft time and surface errors back to the agent for
in-loop self-correction, so a buffer that can't be registered is never
finalized.

The helpers accept the JSON-friendly dict shape (matching the persisted
``design_plan.json`` artifact). Planner-side callers dump their in-memory
Pydantic buffer via ``buffer.model_dump(mode="python")`` before calling.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, cast

from pydantic import ValidationError

from smai_core.dsl.document import DslDocumentAdapter, ExperimentDocument
from smai_core.entities.technique import (
    ContextKind,
    FidelityAnchorAdapter,
    PaperFidelityAnchor,
    ProposalFidelityAnchor,
    ReviewerAttestedFidelityAnchor,
    TechniqueRef,
    anchor_implied_context_kind,
)


def draft_technique_to_ref(
    *,
    symbolic_name: str,
    raw: dict[str, Any],
    proposal_id: str,
    paper_arxiv_id: str | None,
) -> TechniqueRef:
    """Project a draft technique (raw dict from the planner buffer) into a
    :class:`TechniqueRef`.

    The buffer's symbolic name becomes the ref's stable ``id``. When
    ``standard=False`` and no anchor is set, defaults to a
    :class:`ProposalFidelityAnchor` pointing at this proposal id (per
    DEC-032); when ``paper_arxiv_id`` is set, defaults to a
    :class:`PaperFidelityAnchor` instead.
    """
    standard = bool(raw.get("standard", False))
    anchor_raw = cast(dict[str, Any] | None, raw.get("fidelity_anchor"))
    anchor: PaperFidelityAnchor | ProposalFidelityAnchor | ReviewerAttestedFidelityAnchor | None
    anchor = None
    if anchor_raw is not None:
        anchor = FidelityAnchorAdapter.validate_python(anchor_raw)
    if anchor is None and not standard:
        if paper_arxiv_id is not None:
            anchor = PaperFidelityAnchor(arxiv_id=paper_arxiv_id, doi=f"arxiv:{paper_arxiv_id}")
        else:
            anchor = ProposalFidelityAnchor(proposal_id=proposal_id)

    compatible_factor_types_raw = cast(
        list[str], raw.get("compatible_factor_types") or ["additive"]
    )
    # Set ``context_kind`` explicitly per ``upstream_requirements §1``
    # mapping. If the buffer carries one we trust it (the planner's
    # finalize-time projection check round-trips through Pydantic and
    # the :class:`TechniqueRef` validator rejects any anchor / standard
    # disagreement). Otherwise derive it from anchor / standard here so
    # the projection helper keeps working for planner-buffer payloads
    # that don't yet author the field — :func:`anchor_implied_context_kind`
    # returns the canonical mapping; non-standard anchor-less buffers
    # fall back to ``proposal`` because the projection just defaulted
    # the anchor to :class:`ProposalFidelityAnchor` above.
    raw_context_kind = cast(ContextKind | None, raw.get("context_kind"))
    context_kind: ContextKind = (
        raw_context_kind
        if raw_context_kind is not None
        else (anchor_implied_context_kind(anchor, standard) or "proposal")
    )
    return TechniqueRef(
        id=symbolic_name,
        name=cast(str, raw.get("name", symbolic_name)),
        description=cast(str, raw.get("description", "")),
        category=cast(str, raw.get("category", "uncategorized")),
        compatible_factor_types=[_validate_factor_type(t) for t in compatible_factor_types_raw],
        standard=standard,
        fidelity_anchor=anchor,
        affects_extension_points=cast(list[str], raw.get("affects_extension_points") or []),
        implies_controlled=cast(list[str], raw.get("implies_controlled") or []),
        parameter_schema=cast(dict[str, Any] | None, raw.get("parameter_schema")),
        context_kind=context_kind,
    )


def _validate_factor_type(value: str) -> Literal["additive", "substitutive"]:
    if value == "additive":
        return "additive"
    if value == "substitutive":
        return "substitutive"
    raise ValueError(f"unknown factor_type {value!r}")


def draft_cg_to_experiment_document(
    *,
    draft: dict[str, Any],
    cg_id: str,
    technique_refs_by_symbol: dict[str, TechniqueRef],
) -> ExperimentDocument:
    """Project a draft CG (raw dict) into an :class:`ExperimentDocument`.

    Builds the document-shaped raw dict and runs it through
    :data:`DslDocumentAdapter` with ``context={"smai_mode": "dsl"}`` so
    inner gates (e.g. :class:`ComparisonRule.baseline_entry_id` being
    compiler-filled) accept the partially-specified shape.

    Raises :class:`KeyError` / :class:`ValueError` / :class:`TypeError` /
    :class:`pydantic.ValidationError` on malformed input; the
    :func:`project_buffer_to_documents` wrapper catches and converts these
    into structured error strings for the planner.
    """
    factor_dim = cast(str, draft["factor_dimension"])
    factor_type = cast(str, draft["factor_type"])
    factor_description = cast(str, draft.get("factor_description", ""))
    if factor_type not in {"additive", "substitutive"}:
        raise ValueError(f"factor_type {factor_type!r} is not additive/substitutive")

    raw_entries = cast(list[dict[str, Any]], draft.get("entries") or [])
    entry_dicts: list[dict[str, Any]] = []
    for raw_entry in raw_entries:
        raw_level = cast(dict[str, Any], raw_entry["level"])
        symbolic = cast(str | None, raw_level.get("technique_symbolic_name"))
        technique_id: str | None = None
        if symbolic is not None:
            ref = technique_refs_by_symbol.get(symbolic)
            technique_id = ref.id if ref is not None else symbolic
        level_dict: dict[str, Any] = {
            "factor": factor_dim,
            "name": cast(str, raw_level.get("name", "level")),
        }
        if raw_level.get("description") is not None:
            level_dict["description"] = raw_level["description"]
        if technique_id is not None:
            level_dict["technique_id"] = technique_id
        if raw_level.get("technique_params") is not None:
            level_dict["technique_params"] = raw_level["technique_params"]
        entry_dicts.append(
            {
                "id": cast(str, raw_entry["id"]),
                "is_baseline": bool(raw_entry["is_baseline"]),
                "level": level_dict,
            }
        )

    raw_validation = cast(dict[str, Any] | None, draft.get("validation"))
    if raw_validation is None:
        raise ValueError(f"CG {cg_id!r} draft has no validation")
    metric_raw = cast(dict[str, Any], raw_validation["metric"])
    if "ref" not in metric_raw:
        raise ValueError("validation.metric missing 'ref' key")

    if metric_raw.get("kind", "atomic") == "parametric":
        metric_dict: dict[str, Any] = {
            "kind": "parametric",
            "family": cast(str, metric_raw["ref"]),
            "parameters": cast(dict[str, str | int | float], metric_raw.get("parameters", {})),
        }
    else:
        metric_dict = {"kind": "atomic", "ref": cast(str, metric_raw["ref"])}

    comparison_rule_value = cast(str, raw_validation.get("comparison_rule", "compare_to_baseline"))
    if comparison_rule_value not in {"compare_to_baseline", "compare_to_target"}:
        raise ValueError(f"unknown comparison_rule {comparison_rule_value!r}")
    direction_value = cast(str, raw_validation.get("direction", "higher_is_better"))
    if direction_value not in {"higher_is_better", "lower_is_better"}:
        raise ValueError(f"unknown direction {direction_value!r}")
    aggregation_method_value = cast(str, raw_validation.get("aggregation_method", "mean"))
    if aggregation_method_value not in {"mean", "median"}:
        raise ValueError(f"unknown aggregation_method {aggregation_method_value!r}")

    comparison_dict: dict[str, Any] = {
        "rule": comparison_rule_value,
        "threshold": float(raw_validation.get("threshold", 0.0)),
    }
    if raw_validation.get("target_value") is not None:
        comparison_dict["target_value"] = raw_validation["target_value"]

    validation_dict: dict[str, Any] = {
        "metric": metric_dict,
        "direction": direction_value,
        "aggregation": {"method": aggregation_method_value},
        "comparison": comparison_dict,
        "seed_count_required": int(raw_validation.get("seed_count_required", 1)),
    }
    if raw_validation.get("rationale") is not None:
        validation_dict["rationale"] = raw_validation["rationale"]

    document_dict: dict[str, Any] = {
        "kind": "experiment",
        "experiment": {
            "id": cg_id,
            "hypothesis": cast(str, draft.get("hypothesis", "")),
            "factors": [
                {
                    "name": factor_dim,
                    "type": factor_type,
                    "description": factor_description,
                }
            ],
            "controlled_conditions": cast(dict[str, Any], draft.get("controlled_conditions") or {}),
            "entries": entry_dicts,
            "validation": validation_dict,
        },
    }

    document = DslDocumentAdapter.validate_python(document_dict, context={"smai_mode": "dsl"})
    if not isinstance(document, ExperimentDocument):
        raise ValueError(
            f"buffer CG {cg_id!r} projected to {type(document).__name__}, "
            "expected ExperimentDocument"
        )
    return document


def project_buffer_to_documents(
    *,
    buffer: dict[str, Any],
    proposal_id: str,
    cg_id_for: Callable[[str, str], str] | None = None,
) -> tuple[list[ExperimentDocument], dict[str, TechniqueRef], list[str]]:
    """Project a finalized planner-buffer dict into typed documents.

    Returns ``(documents, technique_refs_by_symbol, errors)``. When
    ``errors`` is empty the buffer was projected cleanly; when non-empty
    each entry is a human-readable description of one projection failure
    (prefixed by the field path).

    Parameters mirror :func:`smai_orchestrator.specs.proposal._register_buffer`'s
    inputs so the registration handler and the planner's
    ``finalize_plan`` tool surface identical errors. The optional
    ``cg_id_for`` callable resolves ``(proposal_id, draft_cg_id) → cg_id``;
    when ``None``, the planner-side stable default ``"<proposal>--<draft>"``
    is used (matching the registration handler's default).
    """
    errors: list[str] = []
    techniques_raw = cast(dict[str, dict[str, Any]], buffer.get("techniques") or {})
    technique_refs_by_symbol: dict[str, TechniqueRef] = {}
    paper_arxiv_id = cast(str | None, buffer.get("paper_arxiv_id"))
    for symbolic_name, raw in techniques_raw.items():
        try:
            technique_refs_by_symbol[symbolic_name] = draft_technique_to_ref(
                symbolic_name=symbolic_name,
                raw=raw,
                proposal_id=proposal_id,
                paper_arxiv_id=paper_arxiv_id,
            )
        except (ValueError, TypeError, ValidationError) as exc:
            errors.append(f"techniques[{symbolic_name!r}]: {_format_projection_error(exc)}")

    raw_drafts = cast(list[dict[str, Any]], buffer.get("comparison_groups") or [])
    if not raw_drafts and not errors:
        errors.append("buffer has no comparison_groups; nothing to register")

    documents: list[ExperimentDocument] = []
    for draft in raw_drafts:
        draft_cg_id = cast(str, draft.get("id") or "draft-cg")
        if cg_id_for is not None:
            cg_id = cg_id_for(proposal_id, draft_cg_id)
        else:
            cg_id = f"{proposal_id}--{draft_cg_id}"
        try:
            document = draft_cg_to_experiment_document(
                draft=draft,
                cg_id=cg_id,
                technique_refs_by_symbol=technique_refs_by_symbol,
            )
        except (KeyError, ValueError, TypeError, ValidationError) as exc:
            errors.append(f"comparison_groups[{draft_cg_id!r}]: {_format_projection_error(exc)}")
            continue
        documents.append(document)

    return documents, technique_refs_by_symbol, errors


def _format_projection_error(exc: Exception) -> str:
    """Render a projection error as a one-paragraph string with field paths.

    For :class:`pydantic.ValidationError`, lists each error as
    ``<loc>: <msg> (got <type>)`` joined with ``; ``. For other exceptions,
    returns ``<TypeName>: <message>``.
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


__all__ = [
    "draft_cg_to_experiment_document",
    "draft_technique_to_ref",
    "project_buffer_to_documents",
]
