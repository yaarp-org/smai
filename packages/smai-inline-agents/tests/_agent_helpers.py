"""Test helpers — :class:`ModelResponse` builders for the stub provider."""

from __future__ import annotations

from typing import Any

from smai_core.plugins import (
    ModelResponse,
    NormalizedMessage,
    StopReason,
    TextContent,
    TokenUsage,
    ToolUseContent,
)


def model_response(
    *,
    text: str | None = None,
    tool_uses: list[tuple[str, str, dict[str, Any]]] | None = None,
    stop_reason: StopReason = "end_turn",
    input_tokens: int = 1,
    output_tokens: int = 1,
) -> ModelResponse:
    """Build a :class:`ModelResponse` for the stub provider.

    ``tool_uses`` is a list of ``(tool_use_id, tool_name, input_dict)``
    triples; matching the ``ToolUseContent`` shape lets tests drive the
    loop's tool-execution path with a single canned response.
    """
    content: list[TextContent | ToolUseContent] = []
    if text is not None:
        content.append(TextContent(text=text))
    if tool_uses is not None:
        for tu_id, tu_name, tu_input in tool_uses:
            content.append(
                ToolUseContent(
                    id=tu_id,
                    name=tu_name,
                    input=dict(tu_input),
                )
            )
    return ModelResponse(
        message=NormalizedMessage(role="assistant", content=list(content)),
        stop_reason=stop_reason,
        usage=TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


__all__ = ["model_response"]
