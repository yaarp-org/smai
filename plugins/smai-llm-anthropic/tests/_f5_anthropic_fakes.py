"""In-process Anthropic client fake — the conformance test seam.

A duck-typed fake that returns canned ``messages.create`` results from
an in-memory queue. Functionally equivalent to a recorded VCR cassette:
deterministic, no live network call, and the canned outcomes are
explicit Python literals (no binary cassette files to refresh on
SDK-side schema changes).

The fake implements only the surface :class:`AnthropicProvider`
actually calls: ``client.messages.create(**kwargs) -> Message-like``.
``messages._conformance_queue`` is the hook by which
:meth:`AnthropicProvider._conformance_inject_fault` stages exceptions /
responses for upcoming calls.
"""

from __future__ import annotations

from collections import deque
from typing import Any


class _FakeMessages:
    """The ``client.messages`` namespace with ``create``."""

    def __init__(self, default_response: dict[str, Any] | None = None) -> None:
        self._conformance_queue: deque[Any] = deque()
        self._default_response: dict[str, Any] = default_response or _DEFAULT_OK
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self._conformance_queue:
            outcome = self._conformance_queue.popleft()
            if isinstance(outcome, BaseException):
                raise outcome
            if isinstance(outcome, dict):
                return outcome
            raise TypeError(
                f"FakeAnthropicClient queued an unsupported outcome: {type(outcome).__name__}"
            )
        return self._default_response


class FakeAnthropicClient:
    """Stand-in for :class:`anthropic.AsyncAnthropic` in tests.

    Each ``messages.create`` call dequeues the next staged outcome:

    * an :class:`Exception` instance — raised
    * a :class:`dict` — returned

    When the queue is empty, the configured ``default_response`` is
    returned. The captured ``messages.calls`` list lets tests assert
    request shape (cache markers, tool definitions, message ordering).
    """

    def __init__(self, *, default_response: dict[str, Any] | None = None) -> None:
        self.messages = _FakeMessages(default_response=default_response)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.messages.calls


_DEFAULT_OK: dict[str, Any] = {
    "id": "msg_default",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "fixture-ok"}],
    "stop_reason": "end_turn",
    "usage": {
        "input_tokens": 5,
        "output_tokens": 2,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    },
}


__all__ = ["FakeAnthropicClient"]
