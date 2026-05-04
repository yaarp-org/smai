# smai-events

In-process event-broker primitives for the SMAI v2 live-updates channel.

Per `designs/smai/12-ui-process.md` §6 and the §12 OQ1 resolution
(2026-05-03): the engine fires state-machine transitions to a process-
local `EventBroker`; the API process's SSE handler subscribes to the
broker and drains events to connected SPA clients. This package owns
the abstraction so the orchestrator (publisher) does not need to depend
on `smai-api` (consumer).

## Surface

`from smai_events import EventBroker, EventChannel, InProcessEventChannel, NullEventChannel`

* **`EventChannel`** — the Protocol the engine fires against. Two
  methods: `fire_transition(...)` and `fire_heartbeat(...)`. Both are
  `async`; both are no-args at the schema level apart from event-shape
  fields.
* **`NullEventChannel`** — no-op fallback. The default
  `EngineConfig.event_channel`; ensures every existing test that does
  not care about events keeps passing.
* **`EventBroker`** — anyio memory-stream-based pub/sub with a per-
  subscriber bounded buffer and an in-memory ring buffer for
  Last-Event-ID replay. Process-local; fan-out is `O(subscribers)` per
  publish.
* **`InProcessEventChannel`** — `EventChannel` implementation that
  writes through to an `EventBroker`. Used in the in-band Runtime
  (Case A per `12` §6.2). Cross-process Case B (Postgres
  LISTEN/NOTIFY) ships in Task 4.K3 against the same `EventChannel`
  Protocol.

## Wire format note

The broker carries `StateChangeEvent` / `WorkerHeartbeatEvent` payloads
from `smai_api_spec.events`. SSE wire encoding (the `id:` / `event:` /
`data:` lines, `text/event-stream` Content-Type, the `:keepalive`
heartbeat comment, the `refetch_all` overflow sentinel) lives in the
`smai-api` SSE route, not here — this package is transport-agnostic so
a hypothetical alternative consumer (a websocket adapter, an in-memory
test inspector) can subscribe to the same broker.
