"""PydanticAI :class:`Agent` factory bound to a Pydantic ``output_type``.

Per pydantic_ai_spike.md §6, the scoped adoption is
``Agent + tool decorators + output_type``. Sub-PR C1 lands the body-
generation steps with no tools and an ``output_type`` discriminated
against the D7a / D7b schemas.

Bedrock prompt-caching wires opportunistically via
:class:`BedrockModelSettings.bedrock_cache_*` per notes/pydantic_ai_bedrock_caching.md.
The settings are passed when the resolved model id is a Bedrock model;
sub-PR D handles the full per-step caching configuration matrix.
"""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockModelSettings

OutputT = TypeVar("OutputT", bound=BaseModel)

# Sentinel for the "no Bedrock caching settings wired" branch. PydanticAI
# accepts a falsy / absent ``model_settings`` cleanly; we pass settings
# only when the model is Bedrock-shaped so OpenAI / Anthropic paths
# remain unimpacted.
_BEDROCK_PROVIDER_NAME = "bedrock"


def build_agent(
    *,
    provider: str,
    model_id: str,
    output_type: type[OutputT],
    system_prompt: str,
    enable_bedrock_caching: bool = True,
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
    the controlled ``read_file`` escape hatch with a dedicated factory.

    ``enable_bedrock_caching`` opportunistically wires Bedrock prompt
    caching when the resolved provider is Bedrock. Off-by-default for
    other providers (cost-driven; non-Bedrock providers have different
    caching contracts).
    """
    model_spec = f"{provider}:{model_id}"

    if enable_bedrock_caching and provider == _BEDROCK_PROVIDER_NAME:
        return Agent(
            model=model_spec,
            output_type=output_type,
            system_prompt=system_prompt,
            model_settings=BedrockModelSettings(
                bedrock_cache_instructions="5m",
                bedrock_cache_tool_definitions="5m",
            ),
        )

    return Agent(
        model=model_spec,
        output_type=output_type,
        system_prompt=system_prompt,
    )


__all__ = ["build_agent"]
