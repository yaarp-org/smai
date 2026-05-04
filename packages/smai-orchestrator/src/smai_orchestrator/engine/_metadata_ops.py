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

Per Task 4.K2: this module is also the canonical fire-on-transition
seam (``12-ui-process.md`` §6.4 RESOLVED 2026-05-03 — engine-wraps).
:func:`transition_state` accepts an optional ``event_channel`` +
``from_state``; on successful CAS it calls
:meth:`EventChannel.fire_transition` if ``from_state != target_state``
(handle-only writes that re-target the same state are not transitions
and produce no event).
"""

from __future__ import annotations

import logging
from typing import Any, cast

from smai_core.plugins import EntityKind, MetadataStore
from smai_events import EventChannel, EventEntityKind

from smai_orchestrator.entities.tracking import (
    ComparisonGroupRecord,
    EntryRecord,
    PaperRecord,
    ProposalRecord,
    RunRecord,
)

_log = logging.getLogger(__name__)


# Map between the engine's internal kind discriminator (`smai_core.plugins.EntityKind`,
# ``Literal["cg", "entry", "run", "proposal", "paper"]``) and the API
# spec's outward-facing kind (``EventEntityKind``,
# ``Literal["proposal", "paper", "comparison_group", "entry", "run"]``).
# "cg" → "comparison_group" is the only renaming; the rest are identity.
_KIND_TO_EVENT_KIND: dict[EntityKind, EventEntityKind] = {
    "cg": "comparison_group",
    "entry": "entry",
    "run": "run",
    "proposal": "proposal",
    "paper": "paper",
}


def event_kind_for(kind: EntityKind) -> EventEntityKind:
    """Translate the engine's internal kind into the API-spec event kind.

    Per Task 4.K2 plumbing: the engine uses ``"cg"`` while the SSE wire
    payload uses ``"comparison_group"`` (per ``StateChangeEvent.kind``
    in :mod:`smai_api_spec.events`). This helper is the single
    translation point so the engine's call sites stay kind-agnostic.
    """
    return _KIND_TO_EVENT_KIND[kind]


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


async def transition_state(  # noqa: PLR0913
    store: MetadataStore,
    kind: EntityKind,
    entity_id: str,
    expected_version: int,
    target_state: str,
    fields: dict[str, object],
    *,
    event_channel: EventChannel | None = None,
    from_state: str | None = None,
) -> StateDrivenRecord:
    """CAS-transition an entity to ``target_state``, dispatching per kind.

    Raises :class:`smai_core.plugins.metadata_store.ConflictError` if
    the row's current ``version`` does not equal ``expected_version``.

    The cast at the call site is deliberate — see module docstring.
    The plugin treats ``target_state`` as ``Literal[...]`` per `07` §5;
    the engine treats it as an opaque string per `05` §1.1.

    Per Task 4.K2: when ``event_channel`` and ``from_state`` are both
    supplied AND ``from_state != target_state``, the engine fires
    :meth:`EventChannel.fire_transition` after the CAS succeeds. Same-
    state writes (``run_dispatch`` step 3 records the job handle on the
    in-progress state without changing it) intentionally produce no
    event. Failures inside :meth:`fire_transition` are logged-and-
    swallowed so a broken event channel cannot break the engine — the
    persisted state already reflects the transition.
    """
    state = cast(Any, target_state)
    match kind:
        case "cg":
            record = await store.transition_cg_state(entity_id, expected_version, state, **fields)
        case "entry":
            record = await store.transition_entry_state(
                entity_id, expected_version, state, **fields
            )
        case "run":
            record = await store.transition_run_state(entity_id, expected_version, state, **fields)
        case "proposal":
            record = await store.transition_proposal_state(
                entity_id, expected_version, state, **fields
            )
        case "paper":
            record = await store.transition_paper_state(
                entity_id, expected_version, state, **fields
            )

    if event_channel is not None and from_state is not None and from_state != target_state:
        try:
            await event_channel.fire_transition(
                kind=event_kind_for(kind),
                id=entity_id,
                from_state=from_state,
                to_state=target_state,
            )
        except Exception:  # noqa: BLE001 — event channel must not break the engine
            _log.exception(
                "event_channel.fire_transition raised for %s/%s %s→%s; ignoring",
                kind,
                entity_id,
                from_state,
                target_state,
            )

    return record


__all__ = [
    "StateDrivenRecord",
    "entity_id_for",
    "event_kind_for",
    "get_entity",
    "transition_state",
]
