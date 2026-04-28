""":class:`MetadataStore` Protocol package — pipeline-tracking entity CRUD,
conditional state transitions, scheduling queries, leasing, in-flight counts.

Per ``designs/smai/07-plugin-interfaces.md`` §5, DEC-028 (four interfaces, not
five), DEC-029 (Protocol definitions in ``smai-core``), DEC-030
(SQL-shaped only), and DEC-035 (Session-C settlements: cursor pagination,
lease semantics decoupled from entity-state CAS, caller-resolves
pool→states for in-flight counting).

Module layout:

* ``_records.py`` — forward stubs for the pipeline-tracking entity types
  and ``*State`` aliases (§3.1; the concrete shapes live in pipeline
  packages per DEC-029, but the Protocol surface needs the names).
* ``_capabilities.py`` — :class:`MetadataStoreCapabilities` (§5.5).
* ``_errors.py`` — error hierarchy (:class:`MetadataStoreError`,
  :class:`ConflictError`, :class:`LeaseLostError`).
* ``_protocol.py`` — :class:`MetadataStore` and :class:`Transaction`
  Protocols.
"""

from smai_core.plugins.metadata_store._capabilities import MetadataStoreCapabilities
from smai_core.plugins.metadata_store._errors import (
    ConflictError,
    LeaseLostError,
    MetadataStoreError,
)
from smai_core.plugins.metadata_store._protocol import MetadataStore, Transaction
from smai_core.plugins.metadata_store._records import (
    CGState,
    ComparisonGroupRecord,
    EntryRecord,
    EntryState,
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
