""":class:`MetadataStore` Protocol package — pipeline-tracking entity CRUD,
conditional state transitions, scheduling queries, leasing, in-flight counts.

Per ``designs/smai/07-plugin-interfaces.md`` §5, DEC-028 (four interfaces, not
five), DEC-029 (Protocol definitions in ``smai-core``), DEC-030
(SQL-shaped only), DEC-035 (Session-C settlements: cursor pagination,
lease semantics decoupled from entity-state CAS, caller-resolves
pool→states for in-flight counting), and Task 1.10 (real pipeline-tracking
record types in ``smai-orchestrator`` replace the Task 1.8 stubs).

Module layout:

* ``_capabilities.py`` — :class:`MetadataStoreCapabilities` (§5.5).
* ``_errors.py`` — error hierarchy (:class:`MetadataStoreError`,
  :class:`ConflictError`, :class:`LeaseLostError`).
* ``_protocol.py`` — :class:`MetadataStore` and :class:`Transaction`
  Protocols.

Pipeline-tracking record types (``ComparisonGroupRecord``, ``EntryRecord``,
``RunRecord``, ``ProposalRecord``, ``PaperRecord``, ``FactorModelRecord``)
and their ``*State`` Literal aliases live in
``smai_orchestrator.entities.tracking`` per Task 1.10 / ``01-data-model.md``
§5.1 — pipeline-layer records, not methodology entities. The Protocol
references them under ``TYPE_CHECKING`` so smai-core stays free of a
runtime dependency on smai-orchestrator (DEC-029). They are re-exported
from this module under ``TYPE_CHECKING`` only, so tooling that does
``from smai_core.plugins.metadata_store import ComparisonGroupRecord``
for typing convenience continues to work in pyright / mypy.
"""

from typing import TYPE_CHECKING

from smai_core.plugins.metadata_store._capabilities import MetadataStoreCapabilities
from smai_core.plugins.metadata_store._errors import (
    ConflictError,
    LeaseLostError,
    MetadataStoreError,
)
from smai_core.plugins.metadata_store._protocol import MetadataStore, Transaction

if TYPE_CHECKING:
    # Re-exports for typing convenience. Runtime construction of these
    # records happens through ``from smai_orchestrator.entities.tracking
    # import ...`` directly. ``tools/check_deps.py`` exempts
    # TYPE_CHECKING-guarded imports from rule 2.
    from smai_orchestrator.entities.tracking import (
        CGState,
        ComparisonGroupRecord,
        EntryRecord,
        EntryState,
        FactorModelRecord,
        PaperRecord,
        PaperState,
        ProposalRecord,
        ProposalState,
        RunRecord,
        RunState,
    )

__all__ = [
    "CGState",
    "ComparisonGroupRecord",
    "ConflictError",
    "EntryRecord",
    "EntryState",
    "FactorModelRecord",
    "LeaseLostError",
    "MetadataStore",
    "MetadataStoreCapabilities",
    "MetadataStoreError",
    "PaperRecord",
    "PaperState",
    "ProposalRecord",
    "ProposalState",
    "RunRecord",
    "RunState",
    "Transaction",
]
