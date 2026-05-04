"""Event-broker primitives for the SMAI v2 live-updates channel.

Per ``designs/smai/12-ui-process.md`` §6 / DEC-037 — Task 4.K2 lands
this sibling-of-``smai-core`` package so the engine (publisher) does
not need to depend on ``smai-api`` (consumer). Task 4.K3 ships the
cross-process ``PgNotifyEventChannel`` against the same Protocol.

Public surface:

* :class:`EventChannel` — the Protocol the engine fires against.
* :class:`NullEventChannel` — no-op default; safe ``EngineConfig``
  default value.
* :class:`InProcessEventChannel` — Case A implementation that writes
  through to a local :class:`EventBroker`.
* :class:`EventBroker` — the in-process pub/sub primitive both
  in-process and Postgres-LISTEN publishers feed into; the SSE handler
  drains via :meth:`EventBroker.subscribe`.
* :class:`EnvelopedEvent`, :data:`OVERFLOW_SENTINEL` — the items the
  broker emits to subscribers.
"""

from __future__ import annotations

from smai_events._broker import (
    OVERFLOW_SENTINEL,
    BrokerEvent,
    EnvelopedEvent,
    EventBroker,
    SubscriberItem,
)
from smai_events._channel import (
    EventChannel,
    EventEntityKind,
    NullEventChannel,
)
from smai_events._in_process import InProcessEventChannel

__all__ = [
    "BrokerEvent",
    "EnvelopedEvent",
    "EventBroker",
    "EventChannel",
    "EventEntityKind",
    "InProcessEventChannel",
    "NullEventChannel",
    "OVERFLOW_SENTINEL",
    "SubscriberItem",
]
