"""Static capability metadata per Bedrock model.

Per ``07-plugin-interfaces.md`` §4.4 and the implementation_plan §3.3 Task
2.A1 brief: each :class:`smai_core.plugins.LlmProvider` instance carries
its own :class:`LlmCapabilities`. v1's global ``MODEL_REGISTRY`` collapses
into per-instance metadata; per-task model selection (v1's
``getModelForTask``) is an agent-config concern owned by Task 2.B1.

The table below is intentionally narrow — it covers the Bedrock-catalog
models the agent fleet actually selects from per ``04-agents.md`` §4
(Claude tiers + GLM + Qwen + Nova). Unknown model IDs fall through to a
conservative default (no caching, 128k context, 4k output) and surface a
log line so a downstream consumer can register their own capabilities at
construction time via the ``capabilities`` constructor kwarg if the
default is wrong.
"""

from __future__ import annotations

from smai_core.plugins import LlmCapabilities

# Per the task brief: "supports_caching=True for caching-eligible Bedrock
# models (Claude tiers, Nova, Qwen, GLM all cache fine via cachePoint)."
# Diverges from v1's ``MODEL_REGISTRY`` (which only enabled caching for
# Claude / Nova) — task spec is authoritative.
_BEDROCK_CAPABILITIES: dict[str, LlmCapabilities] = {
    # Claude — full Bedrock Converse + cachePoint support. The catalog
    # inference-profile ID for Opus 4.7 is ``us.anthropic.claude-opus-4-7``
    # (no ``-v1`` suffix); ``supports_caching=True`` is assumed by parity
    # with Opus 4.6 (which caches via ``cachePoint``) — re-verify once the
    # profile is granted in the target account.
    "us.anthropic.claude-opus-4-7": LlmCapabilities(
        supports_caching=True,
        context_window=200_000,
        max_output_tokens=16_384,
        supports_tool_use=True,
        model_id="us.anthropic.claude-opus-4-7",
    ),
    "us.anthropic.claude-opus-4-6-v1": LlmCapabilities(
        supports_caching=True,
        context_window=200_000,
        max_output_tokens=16_384,
        supports_tool_use=True,
        model_id="us.anthropic.claude-opus-4-6-v1",
    ),
    "us.anthropic.claude-sonnet-4-6": LlmCapabilities(
        supports_caching=True,
        context_window=200_000,
        max_output_tokens=16_384,
        supports_tool_use=True,
        model_id="us.anthropic.claude-sonnet-4-6",
    ),
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": LlmCapabilities(
        supports_caching=True,
        context_window=200_000,
        max_output_tokens=16_384,
        supports_tool_use=True,
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
    ),
    # Nova — Amazon's first-party tier with cachePoint support
    "amazon.nova-pro-v1:0": LlmCapabilities(
        supports_caching=True,
        context_window=300_000,
        max_output_tokens=5_000,
        supports_tool_use=True,
        model_id="amazon.nova-pro-v1:0",
    ),
    # GLM — task brief: caches fine via cachePoint
    "zai.glm-5": LlmCapabilities(
        supports_caching=True,
        context_window=200_000,
        max_output_tokens=128_000,
        supports_tool_use=True,
        model_id="zai.glm-5",
    ),
    # Qwen — task brief: caches fine via cachePoint
    "qwen.qwen3-coder-480b-a35b-v1:0": LlmCapabilities(
        supports_caching=True,
        context_window=128_000,
        max_output_tokens=16_384,
        supports_tool_use=True,
        model_id="qwen.qwen3-coder-480b-a35b-v1:0",
    ),
    "qwen.qwen3-coder-next": LlmCapabilities(
        supports_caching=True,
        context_window=128_000,
        max_output_tokens=16_384,
        supports_tool_use=True,
        model_id="qwen.qwen3-coder-next",
    ),
}


def lookup_capabilities(model_id: str) -> LlmCapabilities:
    """Return :class:`LlmCapabilities` for ``model_id``.

    Falls through to a conservative default when the ID is unknown — the
    caller can override by passing ``capabilities=...`` to the
    :class:`BedrockProvider` constructor directly.
    """
    entry = _BEDROCK_CAPABILITIES.get(model_id)
    if entry is not None:
        return entry
    return LlmCapabilities(
        supports_caching=False,
        context_window=128_000,
        max_output_tokens=4_096,
        supports_tool_use=True,
        model_id=model_id,
    )


__all__ = ["lookup_capabilities"]
