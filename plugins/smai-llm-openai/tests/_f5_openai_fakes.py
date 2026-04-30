"""In-process OpenAI client fake — the conformance test seam.

Duck-typed fake exposing ``client.chat.completions.create(**kwargs)``
that returns canned responses from an in-memory queue. Functionally
equivalent to a recorded VCR cassette.

``client.chat.completions._conformance_queue`` is the hook by which
:meth:`OpenAIProvider._conformance_inject_fault` stages exceptions /
responses for upcoming calls.
"""

from __future__ import annotations

from collections import deque
from typing import Any


class _FakeCompletions:
    """The ``client.chat.completions`` namespace with ``create``."""

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
                f"FakeOpenAIClient queued an unsupported outcome: {type(outcome).__name__}"
            )
        return self._default_response


class _FakeChat:
    """The ``client.chat`` namespace exposing ``completions``."""

    def __init__(self, default_response: dict[str, Any] | None = None) -> None:
        self.completions = _FakeCompletions(default_response=default_response)


class FakeOpenAIClient:
    """Stand-in for :class:`openai.AsyncOpenAI` in tests.

    Each ``chat.completions.create`` call dequeues the next staged
    outcome (Exception → raised, dict → returned). The captured
    ``chat.completions.calls`` list lets tests assert request shape.
    """

    def __init__(self, *, default_response: dict[str, Any] | None = None) -> None:
        self.chat = _FakeChat(default_response=default_response)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.chat.completions.calls


_DEFAULT_OK: dict[str, Any] = {
    "id": "chatcmpl_default",
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "fixture-ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 5,
        "completion_tokens": 2,
        "total_tokens": 7,
    },
}


__all__ = ["FakeOpenAIClient"]
