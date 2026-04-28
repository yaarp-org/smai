"""In-process Bedrock client fake — the "VCR-or-equivalent" test seam.

The Phase 2 task brief allows VCR-style recorded fixtures *or
equivalent*. A botocore-shaped fake that returns canned ``.converse``
results from an in-memory queue is functionally equivalent to a VCR
cassette here: tests stay deterministic, CI never makes a live AWS
call, and the recorded outcomes are explicit Python literals (no
binary cassette files to refresh on AWS-side schema changes).

The fake implements only the surface :class:`BedrockProvider` actually
calls: ``.converse(**kwargs) -> dict``. ``_conformance_queue`` is the
hook by which :meth:`BedrockProvider._conformance_inject_fault` stages
exceptions / responses for upcoming calls.
"""

from __future__ import annotations

from collections import deque
from typing import Any


class FakeBedrockClient:
    """Stand-in for the boto3 ``bedrock-runtime`` client in tests.

    Each ``.converse`` call dequeues the next staged outcome:

    * an :class:`Exception` instance — raised
    * a :class:`dict` — returned

    When the queue is empty, the configured ``default_response`` is
    returned (sentinel "every call succeeds with this canned shape").
    The captured ``calls`` list lets tests assert request shape (cache
    points, tool config, message ordering).
    """

    def __init__(self, *, default_response: dict[str, Any] | None = None) -> None:
        self._conformance_queue: deque[Any] = deque()
        self._default_response: dict[str, Any] = default_response or _DEFAULT_OK
        self.calls: list[dict[str, Any]] = []

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self._conformance_queue:
            outcome = self._conformance_queue.popleft()
            if isinstance(outcome, BaseException):
                raise outcome
            if isinstance(outcome, dict):
                return outcome
            raise TypeError(
                f"FakeBedrockClient queued an unsupported outcome: {type(outcome).__name__}"
            )
        return self._default_response


_DEFAULT_OK: dict[str, Any] = {
    "output": {
        "message": {
            "role": "assistant",
            "content": [{"text": "fixture-ok"}],
        }
    },
    "stopReason": "end_turn",
    "usage": {
        "inputTokens": 5,
        "outputTokens": 2,
        "cacheReadInputTokens": 0,
        "cacheWriteInputTokens": 0,
    },
}


__all__ = ["FakeBedrockClient"]
