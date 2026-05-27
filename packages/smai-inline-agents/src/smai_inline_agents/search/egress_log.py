"""Egress audit-log writer for the planner search-tool surface.

Per ``planner_refactor/design_notes/search_tool_surface.md`` §8 and
``architectural_decisions.md`` §10 (trust boundary -- every fetched
URL and search query is logged for audit).

Storage: ArtifactStore, alongside the planner's conversation trace.
Path: ``proposals/{proposal_id}/planner/search_log.jsonl`` (per
implementation_plan §1; the design-note prose uses ``egress_log.jsonl``
under the proposal root, the plan locks the path under ``planner/``).

Format: JSONL, one event per line, schema locked at
:class:`~smai_inline_agents.search.types.EgressEvent`.

Implementation:

* Events buffer in-memory.
* Every event flush appends to the in-memory buffer with a Pydantic
  ``model_dump_json()`` (single-line, no embedded newlines).
* The dispatcher calls :meth:`EgressLogger.close` on session exit;
  ``close()`` serializes the buffered events into a single ``put()``
  call on the ArtifactStore.
* The writer is buffered (not streaming) because (a) ArtifactStore
  Protocol is put-only, no append, and (b) sessions are short (~30s -
  several minutes); losing buffered events on a worker crash is
  acceptable per the audit-trail-not-prevention posture in §10.

Disabled mode: when ``log_queries=False`` in the config, the logger
becomes a no-op (writes are dropped, ``close()`` does not call
``put()``). Default is ``True`` per the design note.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Literal, Protocol, runtime_checkable

from smai_inline_agents.search.types import EgressEvent

_log = logging.getLogger(__name__)


@runtime_checkable
class ArtifactStoreLike(Protocol):
    """Minimal duck-typed ArtifactStore surface used by the logger.

    We don't import :class:`smai_core.plugins.ArtifactStore` directly
    because (a) the logger only needs ``put``, and (b) typing this as a
    structural Protocol keeps the test-fake surface tiny.
    """

    async def put(self, key: str, data: bytes, content_type: str | None = None) -> None: ...


class EgressLogger:
    """JSONL writer for the egress audit log.

    Constructed via :meth:`open`. Buffered in-memory; flushes the full
    JSONL blob to ArtifactStore once on :meth:`close` per the
    Protocol's put-only semantics.
    """

    def __init__(
        self,
        *,
        session_id: str,
        proposal_id: str | None,
        artifact_store: ArtifactStoreLike | None,
        enabled: bool,
    ) -> None:
        self._session_id = session_id
        self._proposal_id = proposal_id
        self._artifact_store = artifact_store
        self._enabled = enabled
        self._events: list[EgressEvent] = []
        self._closed = False
        self._lock = asyncio.Lock()

    @classmethod
    def open(
        cls,
        session_id: str,
        *,
        proposal_id: str | None = None,
        artifact_store: ArtifactStoreLike | None = None,
        enabled: bool = True,
    ) -> EgressLogger:
        """Construct a logger for one planner session.

        ``proposal_id`` may be ``None`` in unit-test / smoke contexts;
        when ``None``, :meth:`close` falls back to a ``sessions/{id}/``
        artifact key so the log is still recoverable.

        ``artifact_store`` may be ``None`` for unit-test contexts that
        only exercise the buffered event list via :attr:`events`; in
        that case :meth:`close` skips the put.
        """
        return cls(
            session_id=session_id,
            proposal_id=proposal_id,
            artifact_store=artifact_store,
            enabled=enabled,
        )

    @property
    def events(self) -> list[EgressEvent]:
        """Read-only accessor for tests + introspection. Returns a copy
        of the buffered events."""
        return list(self._events)

    @property
    def artifact_key(self) -> str:
        """Where :meth:`close` will write the JSONL blob.

        Honors the design-note path layout: under the proposal root if
        a proposal id is known; under ``sessions/`` otherwise (for unit
        / smoke contexts).
        """
        if self._proposal_id is not None:
            return f"proposals/{self._proposal_id}/planner/search_log.jsonl"
        return f"sessions/{self._session_id}/planner/search_log.jsonl"

    async def log(
        self,
        *,
        type: Literal["web_search", "academic_search", "fetch_url"],
        provider: str,
        query: str | None = None,
        url: str | None = None,
        max_results: int | None = None,
        result_count: int | None = None,
        status: int | None = None,
        content_type: str | None = None,
        duration_ms: int,
        error: str | None = None,
        providers_succeeded: list[str] | None = None,
        providers_failed: list[str] | None = None,
        fallback_reason: str | None = None,
    ) -> None:
        """Append one event to the buffer. No-op when disabled.

        ``type`` shadows the built-in but matches the design note's
        schema field name exactly; the trade-off is intentional. The
        Pydantic alias surface keeps this readable.
        """
        if not self._enabled:
            return
        if self._closed:
            raise RuntimeError("EgressLogger.log called after close")
        event = EgressEvent(
            ts=datetime.now(UTC),
            type=type,
            session_id=self._session_id,
            provider=provider,
            query=query,
            url=url,
            max_results=max_results,
            result_count=result_count,
            status=status,
            content_type=content_type,
            duration_ms=duration_ms,
            error=error,
            providers_succeeded=providers_succeeded,
            providers_failed=providers_failed,
            fallback_reason=fallback_reason,
        )
        async with self._lock:
            self._events.append(event)

    async def close(self) -> None:
        """Flush the buffered events to ArtifactStore as a JSONL blob.

        Idempotent. When ``log_queries`` is disabled, the put is skipped
        entirely (no zero-byte artifact written).
        """
        if self._closed:
            return
        self._closed = True
        if not self._enabled or self._artifact_store is None:
            return
        if not self._events:
            return
        payload = "\n".join(e.model_dump_json() for e in self._events) + "\n"
        try:
            await self._artifact_store.put(
                self.artifact_key,
                payload.encode("utf-8"),
                content_type="application/x-ndjson",
            )
        except Exception:  # noqa: BLE001 - audit log must not break the planner
            _log.exception(
                "failed to flush egress log to ArtifactStore at %s",
                self.artifact_key,
            )


__all__ = [
    "ArtifactStoreLike",
    "EgressLogger",
]
