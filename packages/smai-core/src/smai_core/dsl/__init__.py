"""DSL surface: top-level document wrappers and parsing helpers.

Per ``designs/smai/02-dsl-and-contracts.md`` §2. The module exposes the
discriminated-union adapter that converges both authoring paths (programmatic
Pydantic construction and YAML/JSON file parsing) on the same compiler input.
"""

from smai_core.dsl.buffer_projection import (
    draft_cg_to_experiment_document,
    draft_technique_to_ref,
    project_buffer_to_documents,
)
from smai_core.dsl.document import (
    DslDocument,
    DslDocumentAdapter,
    ExperimentDocument,
    FactorModelDocument,
    load_dsl_document_from_json,
)

__all__ = [
    "DslDocument",
    "DslDocumentAdapter",
    "ExperimentDocument",
    "FactorModelDocument",
    "draft_cg_to_experiment_document",
    "draft_technique_to_ref",
    "load_dsl_document_from_json",
    "project_buffer_to_documents",
]
