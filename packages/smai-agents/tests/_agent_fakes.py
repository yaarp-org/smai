"""In-process fakes shared across smai-agents tests.

* :class:`StubLlmProvider` — deterministic :class:`LlmProvider` whose
  responses come from a queue; lets tests drive the loop turn-by-turn.
* :class:`StubArtifactStore` — minimal :class:`ArtifactStore` whose
  body is a dict; supports ``put``/``get``/``exists``/``delete``/
  ``list``/``url_for`` so :func:`smai_agents.between_turn.write_status`
  and :func:`smai_agents.between_turn.maybe_check_supervisor_nudge`
  exercise round-trip behavior.
"""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator
from typing import Literal

from smai_core.plugins import (
    ArtifactNotFound,
    ArtifactStoreCapabilities,
    CacheConfig,
    LlmCapabilities,
    ModelResponse,
    NormalizedMessage,
    ToolDefinition,
)


class StubLlmProvider:
    """A queue-driven :class:`LlmProvider` for unit tests.

    The provider is constructed with a list of canned :class:`ModelResponse`
    instances. ``call`` pops the next one off the queue, records the
    arguments it was called with, and returns the response. When the
    queue is empty, ``call`` raises :class:`AssertionError` (the test
    set up the wrong number of responses).
    """

    def __init__(
        self,
        responses: list[ModelResponse],
        *,
        capabilities: LlmCapabilities | None = None,
    ) -> None:
        self.name = "stub"
        self.capabilities = capabilities or LlmCapabilities(
            supports_caching=True,
            context_window=200_000,
            max_output_tokens=4_096,
            supports_tool_use=True,
            model_id="stub:test",
        )
        self._responses: deque[ModelResponse] = deque(responses)
        self.calls: list[dict[str, object]] = []

    async def call(
        self,
        system: str,
        messages: list[NormalizedMessage],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        cache_config: CacheConfig | None = None,
    ) -> ModelResponse:
        if not self._responses:
            raise AssertionError(
                "StubLlmProvider: ran out of canned responses (extra .call())"
            )
        # Snapshot the args so the test can assert on them after.
        self.calls.append(
            {
                "system": system,
                "messages": [m.model_dump() for m in messages],
                "tools": [t.model_dump() for t in tools] if tools else None,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "cache_config": cache_config.model_dump() if cache_config else None,
            }
        )
        return self._responses.popleft()


class StubArtifactStore:
    """In-memory :class:`ArtifactStore` for unit tests."""

    def __init__(self) -> None:
        self.name = "stub-artifacts"
        self.capabilities = ArtifactStoreCapabilities(
            supports_presigned_urls=False,
            max_object_size_bytes=None,
        )
        self._data: dict[str, bytes] = {}
        self.put_calls: list[tuple[str, bytes, str | None]] = []
        self.delete_calls: list[str] = []

    async def put(
        self, key: str, data: bytes, content_type: str | None = None
    ) -> None:
        self._data[key] = data
        self.put_calls.append((key, data, content_type))

    async def get(self, key: str) -> bytes:
        if key not in self._data:
            raise ArtifactNotFound(key)
        return self._data[key]

    async def exists(self, key: str) -> bool:
        return key in self._data

    async def list(self, prefix: str) -> AsyncIterator[str]:
        keys = sorted(k for k in self._data if k.startswith(prefix))

        async def _gen() -> AsyncIterator[str]:
            for key in keys:
                yield key

        return _gen()

    async def delete(self, key: str) -> None:
        self.delete_calls.append(key)
        self._data.pop(key, None)

    async def url_for(
        self,
        key: str,
        expires_in: int = 3600,
        method: Literal["GET", "PUT"] = "GET",
    ) -> str:
        del expires_in, method
        return f"stub://{key}"


__all__ = ["StubArtifactStore", "StubLlmProvider"]
