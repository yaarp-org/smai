"""Adapter that routes engine calls through entity-kind-specific
:class:`MetadataStore` methods.

The Protocol exposes per-kind methods (``get_cg`` / ``transition_cg_state``,
``get_entry`` / ``transition_entry_state``, etc.) per
``07-plugin-interfaces.md`` §5; the engine is entity-kind-agnostic per
``05-orchestrator.md`` §1.1. This module bridges the two.

The cast at the boundary is deliberate: ``MetadataStore.transition_*_state``
methods accept ``target_state: CGState`` / ``target_state: EntryState``
etc. (``Literal[...]`` types per `01` §5), but the engine treats every
state name as an opaque string. Casting to ``Any`` at the Protocol
boundary is the right operational shape — the Pyright strict scope keeps
the engine internals honest, but the per-kind ``Literal`` discipline is
the plugin's job, not the engine's.
"""

from __future__ import annotations

from typing import Any, cast

from smai_core.plugins import EntityKind, MetadataStore

from smai_orchestrator.entities.tracking import (
    ComparisonGroupRecord,
    EntryRecord,
    PaperRecord,
    ProposalRecord,
    RunRecord,
)

# Union of state-driven record types. ``FactorModelRecord`` is excluded
# per DEC-031 #5 (degenerate lifecycle; not state-driven).
StateDrivenRecord = ComparisonGroupRecord | EntryRecord | RunRecord | ProposalRecord | PaperRecord


def entity_id_for(record: StateDrivenRecord) -> str:
    """Extract the primary-key string from a state-driven record.

    Most records use ``id``; :class:`PaperRecord` uses ``arxiv_id`` per
    `01` §5.7 / `07` §5.3 (``get_paper(arxiv_id)``). The engine treats
    the value as opaque; this helper centralizes the per-kind branch
    so engine code stays kind-agnostic.
    """
    if isinstance(record, PaperRecord):
        return record.arxiv_id
    return record.id


async def get_entity(
    store: MetadataStore,
    kind: EntityKind,
    entity_id: str,
) -> StateDrivenRecord | None:
    """Read an entity record from the store, dispatching per kind.

    Returns ``None`` if the entity does not exist.
    """
    match kind:
        case "cg":
            return await store.get_cg(entity_id)
        case "entry":
            return await store.get_entry(entity_id)
        case "run":
            return await store.get_run(entity_id)
        case "proposal":
            return await store.get_proposal(entity_id)
        case "paper":
            return await store.get_paper(entity_id)


async def transition_state(
    store: MetadataStore,
    kind: EntityKind,
    entity_id: str,
    expected_version: int,
    target_state: str,
    fields: dict[str, object],
) -> StateDrivenRecord:
    """CAS-transition an entity to ``target_state``, dispatching per kind.

    Raises :class:`smai_core.plugins.metadata_store.ConflictError` if
    the row's current ``version`` does not equal ``expected_version``.

    The cast at the call site is deliberate — see module docstring.
    The plugin treats ``target_state`` as ``Literal[...]`` per `07` §5;
    the engine treats it as an opaque string per `05` §1.1.
    """
    state = cast(Any, target_state)
    match kind:
        case "cg":
            return await store.transition_cg_state(entity_id, expected_version, state, **fields)
        case "entry":
            return await store.transition_entry_state(entity_id, expected_version, state, **fields)
        case "run":
            return await store.transition_run_state(entity_id, expected_version, state, **fields)
        case "proposal":
            return await store.transition_proposal_state(
                entity_id, expected_version, state, **fields
            )
        case "paper":
            return await store.transition_paper_state(entity_id, expected_version, state, **fields)


__all__ = ["StateDrivenRecord", "entity_id_for", "get_entity", "transition_state"]
