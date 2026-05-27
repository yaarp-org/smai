"""Single-call structured output via ``tool_use``.

Per ``04-agents.md`` §6 and DEC-018. For single-call agents (screener,
enricher, code reviewer, supervisor, contextual evaluator) where the
output is pure structured data, the call requests structured output
through a synthetic ``tool_use`` channel — *not* by parsing JSON out of
free-text. The provider validates structured output against the tool's
input schema; the caller never has to fence-strip, JSON.parse-with-fallback,
or pattern-match braces.

DEC-018's discipline:

1. **No silent fallbacks.** If the model returns text instead of a
   tool call, retry once with an explicit instruction. If the second
   attempt also fails, raise :class:`StructuredCallFailed` with full
   context — ever.
2. **Retry with visibility.** A ``logger.warning`` surfaces every retry.
   Chronic retries on a particular role mean the prompt isn't
   reliably eliciting tool use, which is a config issue, not a
   substrate issue.
3. **One implementation, many consumers.** Code reviewer, contextual
   evaluator, supervisor, screener, enricher all consume this without
   modification — :data:`output_schema` and :data:`tool_name` are the
   only role-specific knobs.

Cache defaults match §5 final paragraph: ``cache_static_prefix=True``,
``cache_initial_message=True``, ``rolling_cache_count=0`` (no rolling
on a one-message conversation).
"""

from __future__ import annotations

import logging
from typing import TypeVar

from pydantic import BaseModel, ValidationError
from smai_core.plugins import (
    CacheConfig,
    LlmProvider,
    NormalizedMessage,
    TextContent,
    ToolDefinition,
    ToolUseContent,
)

from smai_inline_agents.cache import SINGLE_CALL_CACHE_CONFIG

logger = logging.getLogger(__name__)


# Per §6: "If both attempts fail (text instead of tool call), throw
# with full context." Deliberately a subclass of :class:`Exception`,
# not :class:`smai_core.plugins.LlmProviderError` — this is a
# pipeline-layer parsing/contract failure, not a provider transport
# failure.
class StructuredCallFailed(Exception):
    """Raised when both structured-call attempts fail to elicit a tool_use.

    Carries the full failure context so the caller can log a debug
    artifact (the orchestrator dispatch handler that invoked us writes
    this to ``ArtifactStore`` for runbook follow-up).
    """

    def __init__(
        self,
        *,
        tool_name: str,
        attempt_count: int,
        last_assistant_text: str | None,
        last_validation_error: str | None,
    ) -> None:
        self.tool_name = tool_name
        self.attempt_count = attempt_count
        self.last_assistant_text = last_assistant_text
        self.last_validation_error = last_validation_error
        super().__init__(
            f"structured_call(tool={tool_name!r}) failed after {attempt_count} attempts: "
            f"validation_error={last_validation_error!r}, "
            f"last_text={(last_assistant_text or '')[:200]!r}"
        )


_RETRY_INSTRUCTION = (
    "You returned text instead of calling the {tool_name} tool. "
    "Respond by invoking the {tool_name} tool with the structured payload — "
    "do not include any text outside the tool call."
)


_OutputSchemaT = TypeVar("_OutputSchemaT", bound=BaseModel)


async def structured_call(
    *,
    llm: LlmProvider,
    system: str,
    user_message: str,
    output_schema: type[_OutputSchemaT],
    tool_name: str,
    tool_description: str,
    max_tokens: int = 4096,
    cache_config: CacheConfig | None = None,
) -> _OutputSchemaT:
    """Single-call structured output via a synthetic ``tool_use``.

    Per §6 (verbatim signature). Behavior:

    1. Call the LLM with the synthetic tool. The model is expected to
       respond with ``stop_reason='tool_use'`` and a single ``tool_use``
       block whose input parses against ``output_schema``.
    2. If the model returned text instead of a tool call (or the
       payload failed schema validation), retry once with an explicit
       instruction to use the tool. Log a warning per DEC-018's
       "retry with visibility" pattern.
    3. If the second attempt also fails, raise
       :class:`StructuredCallFailed` with full context. No silent
       fallback. Ever.
    """
    if cache_config is None:
        cache_config = SINGLE_CALL_CACHE_CONFIG

    tool_def = ToolDefinition(
        name=tool_name,
        description=tool_description,
        input_schema=output_schema.model_json_schema(),
    )

    messages: list[NormalizedMessage] = [
        NormalizedMessage(role="user", content=[TextContent(text=user_message)]),
    ]

    parsed, last_text, last_validation_error = await _attempt_call(
        llm=llm,
        system=system,
        messages=messages,
        tool_def=tool_def,
        output_schema=output_schema,
        max_tokens=max_tokens,
        cache_config=cache_config,
    )
    if parsed is not None:
        return parsed

    logger.warning(
        "structured_call: tool=%s attempt=1 returned %s; retrying with "
        "explicit instruction (raw=%r)",
        tool_name,
        "text-only" if last_validation_error is None else "schema-invalid payload",
        (last_text or "")[:500],
    )

    # Per §6 step 2: append the assistant's prior response (if any) and a
    # user-side reminder, then retry once.
    if last_text is not None:
        messages.append(
            NormalizedMessage(
                role="assistant",
                content=[TextContent(text=last_text)],
            )
        )
    messages.append(
        NormalizedMessage(
            role="user",
            content=[TextContent(text=_RETRY_INSTRUCTION.format(tool_name=tool_name))],
        )
    )

    parsed, last_text, last_validation_error = await _attempt_call(
        llm=llm,
        system=system,
        messages=messages,
        tool_def=tool_def,
        output_schema=output_schema,
        max_tokens=max_tokens,
        cache_config=cache_config,
    )
    if parsed is not None:
        return parsed

    raise StructuredCallFailed(
        tool_name=tool_name,
        attempt_count=2,
        last_assistant_text=last_text,
        last_validation_error=last_validation_error,
    )


async def _attempt_call(
    *,
    llm: LlmProvider,
    system: str,
    messages: list[NormalizedMessage],
    tool_def: ToolDefinition,
    output_schema: type[_OutputSchemaT],
    max_tokens: int,
    cache_config: CacheConfig,
) -> tuple[_OutputSchemaT | None, str | None, str | None]:
    """Run one structured-call attempt.

    Returns ``(parsed, assistant_text, validation_error)``. Exactly one
    of ``parsed`` / ``(assistant_text, validation_error)`` is informative
    for the caller's retry decision.
    """
    response = await llm.call(
        system=system,
        messages=messages,
        tools=[tool_def],
        max_tokens=max_tokens,
        cache_config=cache_config,
    )

    # Look for a tool_use block whose name matches our synthetic tool.
    text_buffer: list[str] = []
    for block in response.message.content:
        if isinstance(block, ToolUseContent) and block.name == tool_def.name:
            try:
                parsed = output_schema.model_validate(block.input)
            except ValidationError as exc:
                # DEC-018 case: tool was used but the payload failed
                # schema validation. Surface as a "schema-invalid"
                # retry trigger; the assistant_text captures any
                # narrative that came along with the bad payload.
                return None, _join_text(text_buffer), str(exc)
            return parsed, None, None
        if isinstance(block, TextContent):
            text_buffer.append(block.text)

    # No tool_use found — return whatever text the model produced so
    # the caller can log it.
    return None, _join_text(text_buffer), None


def _join_text(parts: list[str]) -> str | None:
    if not parts:
        return None
    return "\n".join(parts)


__all__ = ["StructuredCallFailed", "structured_call"]
