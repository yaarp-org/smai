""":class:`MetadataStore` error hierarchy.

Per ``designs/smai/07-plugin-interfaces.md`` §5.4 (:class:`ConflictError`)
and §5.6.7 (:class:`LeaseLostError`); DEC-035 sub-decision #2 (``nonce``
as canonical gate; lease ops decoupled from entity-state CAS).

The base :class:`MetadataStoreError` exists so plugin authors and
consumers can ``except`` against the whole family in fall-through paths
without enumerating every subclass.
"""

from __future__ import annotations


class MetadataStoreError(Exception):
    """Base class for all :class:`MetadataStore` errors."""


class ConflictError(MetadataStoreError):
    """Raised by ``transition_*_state`` when the row's current ``version``
    does not match ``expected_version`` (§5.4).

    The orchestrator treats this as "someone else handled it" and moves
    on — same semantics as v1's DDB ``ConditionExpression`` failure
    (``designs/yaarp/dao_design.md`` §3.4).
    """

    def __init__(
        self,
        entity_type: str,
        entity_id: str,
        expected_version: int,
        actual_version: int,
    ) -> None:
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(
            f"{entity_type} {entity_id!r}: expected version {expected_version}, "
            f"actual version {actual_version}",
        )


class LeaseLostError(MetadataStoreError):
    """Raised by :meth:`MetadataStore.extend_lease` when the lease has
    expired and been reacquired by another worker (§5.6.7).

    The dispatch handler MUST abort cleanly when it sees this — the
    entity is no longer the handler's to mutate.
    """

    def __init__(self, entity_kind: str, entity_id: str) -> None:
        self.entity_kind = entity_kind
        self.entity_id = entity_id
        super().__init__(f"lease lost on {entity_kind} {entity_id!r}")
