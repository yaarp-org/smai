"""DSL document wrappers and the discriminated union.

Per ``designs/smai/02-dsl-and-contracts.md`` §2.2: a top-level DSL document is
either an ``ExperimentDocument`` (one CG) or a ``FactorModelDocument`` (a
Factor Model grouping multiple CGs per DEC-014). Pydantic's discriminated
union dispatches on the ``kind`` field.

Conditional-required-field gating (§2.5) lives on inner models (currently only
``ComparisonRule.baseline_entry_id``) and reads ``ValidationInfo.context`` to
decide which fields are required at which authoring stage. Callers select the
DSL gate by passing ``context={"smai_mode": "dsl"}`` to
``DslDocumentAdapter.validate_python``; the artifact gate (default) fires when
the compiler re-validates emitted artifacts. The context propagates to inner
models automatically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from smai_core.entities import ExperimentDefinition, FactorModel


class ExperimentDocument(BaseModel):
    """Top-level wrapper for a standalone CG experiment.

    The discriminator value is ``kind="experiment"``. The inner
    ``ExperimentDefinition`` carries the full CG spec (factor, controlled
    conditions, entries, validation criteria) per ``01-data-model.md`` §3.7.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["experiment"] = "experiment"
    experiment: ExperimentDefinition


class FactorModelDocument(BaseModel):
    """Top-level wrapper for a Factor Model grouping multiple CGs.

    The discriminator value is ``kind="factor_model"``. ``factor_model``
    carries the grouping metadata (research question, shared conditions,
    ``comparison_group_ids``) and ``experiments`` carries the comprising CG
    definitions. Each comprising CG must reference this model via
    ``ExperimentDefinition.factor_model_id``; the Pass-2 verifier (Task 1.5)
    enforces bidirectional consistency.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["factor_model"] = "factor_model"
    factor_model: FactorModel
    experiments: list[ExperimentDefinition]


DslDocument = Annotated[
    ExperimentDocument | FactorModelDocument,
    Field(discriminator="kind"),
]
"""Top-level DSL document discriminated union (§2.2).

Pydantic resolves the variant from the ``kind`` field at validation time;
no JSON Schema ``oneOf`` is invoked in v1.
"""

DslDocumentAdapter: TypeAdapter[DslDocument] = TypeAdapter(DslDocument)
"""``TypeAdapter`` for the document union.

Canonical authoring path (per §2.1)::

    DslDocumentAdapter.validate_python(
        yaml.safe_load(text),
        context={"smai_mode": "dsl"},
    )

YAML loading is caller-side — ``smai-core`` does not import pyyaml because
that would violate the runtime dep allowlist (DEC-029). The ``smai_mode``
context propagates to inner models so ``ComparisonRule`` (and any future
compiler-fill fields) gates correctly.
"""


def load_dsl_document_from_json(source: str | Path) -> DslDocument:
    """Parse and validate a JSON DSL document.

    A ``Path`` is read from disk; a ``str`` is treated as JSON text directly.
    Validation runs with ``smai_mode="dsl"`` context so DSL gates fire on
    compiler-fill fields (currently ``ComparisonRule.baseline_entry_id`` —
    see §2.5).

    YAML has no equivalent helper here — ``smai-core``'s runtime dep
    allowlist (Pydantic + jsonschema + stdlib, per DEC-029) excludes pyyaml.
    Callers parse YAML themselves and feed the resulting dict to
    ``DslDocumentAdapter.validate_python``::

        DslDocumentAdapter.validate_python(
            yaml.safe_load(open(path).read()),
            context={"smai_mode": "dsl"},
        )
    """
    text = source.read_text(encoding="utf-8") if isinstance(source, Path) else source
    payload = json.loads(text)
    return DslDocumentAdapter.validate_python(payload, context={"smai_mode": "dsl"})
