"""PydanticAI :class:`Agent` factory bound to a Pydantic ``output_type``.

Per pydantic_ai_spike.md §6, the scoped adoption is
``Agent + tool decorators + output_type``. Sub-PR C1 lands the body-
generation steps with no tools and an ``output_type`` discriminated
against the D7a / D7b schemas.

Bedrock prompt-caching wires per-step via :class:`BedrockCacheSettings`
+ :class:`BedrockModelSettings.bedrock_cache_*` per
notes/pydantic_ai_bedrock_caching.md. Sub-PR D replaces sub-PR C1's
opportunistic boolean gate with a structured per-step configuration
matrix — call sites pass a per-step :class:`BedrockCacheSettings`
to express different TTLs / on-off mixes per workflow step (diagnose
typically benefits from longer-TTL tool-definition cache than body
generation, etc.).
"""

from __future__ import annotations

from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockModelSettings

OutputT = TypeVar("OutputT", bound=BaseModel)

# Sentinel for the "no Bedrock caching settings wired" branch. PydanticAI
# accepts a falsy / absent ``model_settings`` cleanly; we pass settings
# only when the model is Bedrock-shaped so OpenAI / Anthropic paths
# remain unimpacted.
_BEDROCK_PROVIDER_NAME = "bedrock"

# TTL Literal — matches PydanticAI 1.102.0's accepted shape for the
# Bedrock cache flags (``bool | Literal['5m', '1h']``). The ``"off"``
# spelling lets a per-step config opt out cleanly without re-typing
# ``False``; the factory maps it to PydanticAI's omit-the-flag branch.
BedrockCacheTtl = Literal["off", "5m", "1h"]


class BedrockCacheSettings(BaseModel):
    """Per-step Bedrock prompt-cache configuration.

    Sub-PR D replaces sub-PR C1's opportunistic boolean gate with this
    structured shape. Each field maps to PydanticAI 1.102.0's
    :class:`BedrockModelSettings` Bedrock-only flags (per
    notes/pydantic_ai_bedrock_caching.md):

    * :attr:`cache_instructions` — caches the system prompt + static
      instructions parts. ``"5m"`` is the default per the spike report
      (covers a typical agent-loop turn-set well; ``"1h"`` only useful
      for very long-running sessions).
    * :attr:`cache_tool_definitions` — caches the tool schema block.
      Tool definitions are static across turns within a session;
      ``"5m"`` covers the entire workflow's diagnose-step retry budget
      cleanly.
    * :attr:`cache_messages` — rolling cache markers on the message
      history. Disabled by default because the rolling-cache pattern
      requires per-turn :class:`CachePoint` insertion in addition to
      this flag; the sandbox does not yet drive that. Wire later when
      the multi-cycle review-feedback loop ships (architectural hedge
      §12 item 4).

    Inline-role roles (planner, code_reviewer, etc.) use the host-side
    ``smai-agents`` cache.py shape; they do not consult this Pydantic
    model.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    cache_instructions: BedrockCacheTtl = "5m"
    cache_tool_definitions: BedrockCacheTtl = "5m"
    cache_messages: BedrockCacheTtl = "off"


# Default settings for body-generation / baseline / diagnose step calls.
# Matches sub-PR C1's opportunistic-mode behavior verbatim
# (``5m`` instructions + tool_definitions) so the sub-PR D refactor is
# behaviorally identical at the default; the structured shape is what
# enables per-step variation when a future cost-tuning pass wants it.
DEFAULT_BEDROCK_CACHE_SETTINGS = BedrockCacheSettings()

# Explicit "no caching" sentinel — handy for tests that need to
# bypass the cache flags + for the controllable opt-out shape.
NO_BEDROCK_CACHE = BedrockCacheSettings(
    cache_instructions="off",
    cache_tool_definitions="off",
    cache_messages="off",
)


def build_agent(
    *,
    provider: str,
    model_id: str,
    output_type: type[OutputT],
    system_prompt: str,
    bedrock_cache_settings: BedrockCacheSettings | None = None,
) -> Agent[None, OutputT]:
    """Construct a PydanticAI :class:`Agent` bound to ``output_type``.

    ``provider`` + ``model_id`` form the PydanticAI model spec string
    (e.g., ``"bedrock:us.anthropic.claude-sonnet-4-6"``). The string form
    routes through PydanticAI's ``infer_model``; no provider SDK is
    instantiated here (the sandbox image carries the SDKs, and
    PydanticAI lazy-loads the right client at run time).

    Body-generation steps register no tools per architectural_decisions
    §12 #1 (bundle-completeness in schema). This factory exposes no
    ``tools=`` parameter for that reason; sub-PR C2's diagnose step adds
    the controlled ``read_file`` escape hatch via :meth:`Agent.tool_plain`
    after the Agent is constructed.

    ``bedrock_cache_settings`` carries the per-step Bedrock prompt-cache
    configuration (sub-PR D thread 3 replacing sub-PR C1's opportunistic
    bool). ``None`` is treated as :data:`DEFAULT_BEDROCK_CACHE_SETTINGS`.
    Non-Bedrock providers ignore the settings (cost-driven; non-Bedrock
    providers have different caching contracts).
    """
    settings = (
        bedrock_cache_settings
        if bedrock_cache_settings is not None
        else DEFAULT_BEDROCK_CACHE_SETTINGS
    )
    model_spec = f"{provider}:{model_id}"

    if provider == _BEDROCK_PROVIDER_NAME and not _settings_are_all_off(settings):
        return Agent(
            model=model_spec,
            output_type=output_type,
            system_prompt=system_prompt,
            model_settings=_to_bedrock_model_settings(settings),
        )

    return Agent(
        model=model_spec,
        output_type=output_type,
        system_prompt=system_prompt,
    )


def _settings_are_all_off(settings: BedrockCacheSettings) -> bool:
    """Return ``True`` if every TTL field is the ``"off"`` opt-out sentinel."""
    return (
        settings.cache_instructions == "off"
        and settings.cache_tool_definitions == "off"
        and settings.cache_messages == "off"
    )


def _to_bedrock_model_settings(
    settings: BedrockCacheSettings,
) -> BedrockModelSettings:
    """Project :class:`BedrockCacheSettings` to PydanticAI's
    :class:`BedrockModelSettings`, omitting fields whose TTL is
    ``"off"`` (PydanticAI's absent-flag branch is the "no cache"
    behavior; ``False`` is also accepted but ``omit-the-flag`` is the
    documented idiom)."""
    kwargs: dict[str, str] = {}
    if settings.cache_instructions != "off":
        kwargs["bedrock_cache_instructions"] = settings.cache_instructions
    if settings.cache_tool_definitions != "off":
        kwargs["bedrock_cache_tool_definitions"] = settings.cache_tool_definitions
    if settings.cache_messages != "off":
        kwargs["bedrock_cache_messages"] = settings.cache_messages
    return BedrockModelSettings(**kwargs)  # type: ignore[arg-type]


__all__ = [
    "DEFAULT_BEDROCK_CACHE_SETTINGS",
    "NO_BEDROCK_CACHE",
    "BedrockCacheSettings",
    "BedrockCacheTtl",
    "build_agent",
]
