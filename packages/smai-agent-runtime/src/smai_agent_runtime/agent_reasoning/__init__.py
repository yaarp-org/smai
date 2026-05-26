"""PydanticAI Agent construction + per-step model selection.

Per architectural_decisions.md §3, this module is sandbox-side: it lives
in the agent-runtime container image alongside the mini-orchestrator,
and is consumed by the per-step body-generation handlers in
:mod:`smai_agent_runtime.harness_builder`.

Two pieces:

* :func:`get_model_for_step` — per-step model resolution per D3's six-layer
  precedence chain (env > config > defaults), parameterized so the
  harness-builder mini-orchestrator can pick the model for each step at
  dispatch time without depending on the host orchestrator's
  ``EngineConfig`` shape.
* :func:`build_agent` — minimal PydanticAI :class:`Agent` factory bound
  to a Pydantic ``output_type``. Opportunistically wires Bedrock
  prompt-caching via :class:`BedrockModelSettings` when the resolved
  model id is a Bedrock model. Sub-PR D handles the full per-step caching
  configuration matrix.
"""

from __future__ import annotations

from smai_agent_runtime.agent_reasoning.agents import build_agent
from smai_agent_runtime.agent_reasoning.model_selection import (
    DEFAULT_BEDROCK_PROVIDER,
    SANDBOXED_ROLE_DEFAULTS,
    ModelSelectionError,
    SandboxedRole,
    get_model_for_step,
)

__all__ = [
    "DEFAULT_BEDROCK_PROVIDER",
    "ModelSelectionError",
    "SANDBOXED_ROLE_DEFAULTS",
    "SandboxedRole",
    "build_agent",
    "get_model_for_step",
]
