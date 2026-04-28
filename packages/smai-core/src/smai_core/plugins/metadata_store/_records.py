"""Forward-only stubs for pipeline-tracking entity records and state aliases.

Per ``07-plugin-interfaces.md`` §3.1: the :class:`MetadataStore` Protocol
references pipeline-tracking entity types (:class:`ComparisonGroupRecord`,
:class:`EntryRecord`, :class:`RunRecord`, :class:`ProposalRecord`,
:class:`PaperRecord`) and their state aliases. The concrete Pydantic
models live in pipeline packages per DEC-029 — the methodology layer
never sees them — but the Protocol module needs the names so its
signatures type-check.

§3.1 explicitly leaves the choice of "stub vs. ``TYPE_CHECKING`` import
from a pipeline package" as a Session-C open question (§12 OQ9). Two
constraints make stubs the only viable v1 choice:

1. ``01-data-model.md`` §5 is still a stub — the Pydantic shapes for the
   record types haven't been authored yet. There is nothing concrete to
   import.
2. ``tools/check_deps.py`` walks every AST import node in
   ``packages/smai-core/src/`` and forbids pipeline-package imports,
   including those guarded by ``if TYPE_CHECKING:``. Any
   ``from smai_orchestrator import ...`` would fail rule #2 of DEC-029
   regardless of guard.

So we ship empty Pydantic-model stubs here, configured with
``extra="allow"`` so plugin / engine code can still construct and
round-trip them with arbitrary fields. When pipeline-tracking entities
ship in ``01-data-model.md`` §5, these stubs are replaced (the public
import path ``smai_core.plugins.MetadataStore`` doesn't change; consumers
re-bind via fresh re-exports).

The ``*State`` aliases are typed as ``str`` per §5.4 ("this Protocol
carries them as opaque type parameters"). The exact ``Literal`` value
sets are owned by ``03-state-machine.md`` (CG / Entry / Run sub-spec)
and ``08-novel-technique-pipeline.md`` §2.4 / §5.2 (Proposal / Paper);
state names appear in those docs as forward references for orientation
only — the Protocol queries treat them as opaque labels.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class _PipelineRecordStub(BaseModel):
    """Common base for the forward-only pipeline-tracking entity stubs.

    ``extra="allow"`` so plugin / engine code can round-trip the records
    with their full (yet-to-be-specified) field set without the stub
    rejecting unknown fields. The empty body documents that *no field
    set is committed at the methodology layer* — the shape is owned by
    pipeline packages per DEC-029.
    """

    model_config = ConfigDict(extra="allow")


class ComparisonGroupRecord(_PipelineRecordStub):
    """Pipeline-tracking record for a ComparisonGroup. Forward-only stub.

    Concrete shape owned by ``01-data-model.md`` §5 (Session C).
    """


class EntryRecord(_PipelineRecordStub):
    """Pipeline-tracking record for an Entry. Forward-only stub.

    Concrete shape owned by ``01-data-model.md`` §5 (Session C).
    """


class RunRecord(_PipelineRecordStub):
    """Pipeline-tracking record for a single ``(entry, seed)`` run.
    Forward-only stub.

    Concrete shape owned by ``01-data-model.md`` §5 (Session C).
    """


class ProposalRecord(_PipelineRecordStub):
    """Pipeline-tracking record for a novel-technique proposal.
    Forward-only stub.

    Per DEC-032, proposals are the primary input path in v2; the
    Protocol surfaces ``ProposalRecord`` symmetric with
    :class:`PaperRecord`. Concrete shape owned by ``01-data-model.md``
    §5 (Session C) and ``08-novel-technique-pipeline.md``.
    """


class PaperRecord(_PipelineRecordStub):
    """Pipeline-tracking record for a paper-ingestion target.
    Forward-only stub.

    Concrete shape owned by ``01-data-model.md`` §5 (Session C) and
    ``08-novel-technique-pipeline.md`` §5.
    """


# State aliases. Per §5.4 — "this Protocol carries them as opaque type
# parameters." Resolving each to ``str`` matches the spec's "opaque label"
# framing. The canonical state-Literal value sets are owned by
# ``03-state-machine.md`` (CG / Entry / Run) and
# ``08-novel-technique-pipeline.md`` §2.4 / §5.2 (Proposal / Paper); when
# those docs settle, the aliases here tighten to the corresponding
# ``Literal[...]`` types without breaking consumer call sites.
CGState = str
EntryState = str
RunState = str
ProposalState = str
PaperState = str
