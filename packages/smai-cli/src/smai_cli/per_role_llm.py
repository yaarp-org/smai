"""Per-role :class:`LlmProvider` resolution for the agent fleet.

Per ``designs/smai/04-agents.md`` §4 / DEC-022, and the carry-forward
from Task 2.C3's status note: per-role per-``model_id`` divergence
lives in the CLI, not in :mod:`smai_orchestrator.runtime.instantiate`
(which would force ``smai-orchestrator → smai-agents`` runtime
coupling).

Resolution path:

1. For each :data:`smai_agents.model_selection.TaskRole`, call
   :func:`smai_agents.model_selection.get_model_for_task` to resolve a
   ``(provider_name, model_id)`` tuple. This honors:
   - explicit ``overrides`` (set by future system-config UIs);
   - ``SMAI_MODEL_<ROLE_UPPER>`` env vars;
   - the per-role :data:`TASK_DEFAULTS` table.
2. Group roles by their resolved tuple. One :class:`LlmProvider`
   instance per **unique** tuple — sibling roles that resolve to the
   same ``(provider, model_id)`` share an instance.
3. Wave-1 v1 ships the Bedrock plugin only; the per-tuple constructor
   instantiates :class:`smai_llm_bedrock.BedrockProvider` directly,
   reading the base ``llm_provider_config`` (region, etc.) from the
   :class:`PluginSelection`. Other providers (Anthropic / OpenAI) land
   alongside their plugin packages in Phase 3 (Task 3.F5); the CLI
   raises :class:`UnsupportedLlmProvider` if one of them is requested
   in Phase 2.
4. Return the per-role map for handoff via
   :class:`PluginOverrides.llm_providers` to
   :func:`smai_orchestrator.runtime.instantiate.instantiate_plugins`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from smai_agents.model_selection import TaskRole, get_model_for_task
from smai_core.plugins import LlmProvider
from smai_orchestrator import DEFAULT_TASK_ROLES, PluginSelection

if TYPE_CHECKING:
    pass


class UnsupportedLlmProvider(RuntimeError):
    """Raised when a per-role :func:`get_model_for_task` resolves to a
    provider name that has no Phase-2 plugin shipped.

    Phase-2 ships ``bedrock`` only; ``anthropic`` / ``openai`` land in
    Phase 3 alongside their plugin packages.
    """


def build_per_role_llm_providers(
    selection: PluginSelection,
    *,
    env: Mapping[str, str] | None = None,
    overrides: dict[TaskRole, tuple[str, str]] | None = None,
) -> dict[str, LlmProvider]:
    """Build the per-role :class:`LlmProvider` map.

    For every :data:`TaskRole` in :data:`DEFAULT_TASK_ROLES`,
    ``get_model_for_task(role, env=env)`` resolves a ``(provider,
    model_id)`` tuple. Roles that resolve to the same tuple share a
    single :class:`LlmProvider` instance.

    The base ``selection.llm_provider_config`` (e.g., ``region``)
    parameterizes every constructed instance — only the per-task
    ``model_id`` varies between instances.

    Returns the role-name keyed dict suitable for hand-off via
    :class:`PluginOverrides.llm_providers`. Note the dict keys are
    ``str`` (not :data:`TaskRole`) to match the orchestrator's
    surface — :data:`DEFAULT_TASK_ROLES` is the synced literal set.
    """
    env_map: Mapping[str, str] = env if env is not None else os.environ
    base_kwargs: dict[str, Any] = dict(selection.llm_provider_config)
    # ``model_id`` is the per-instance parameter — strip from base
    # kwargs so each tuple-keyed instance can supply its own.
    base_kwargs.pop("model_id", None)

    per_tuple_cache: dict[tuple[str, str], LlmProvider] = {}
    out: dict[str, LlmProvider] = {}
    for role_str in DEFAULT_TASK_ROLES:
        role = cast(TaskRole, role_str)
        provider_name, model_id = get_model_for_task(role, env=dict(env_map), overrides=overrides)
        cache_key = (provider_name, model_id)
        if cache_key not in per_tuple_cache:
            per_tuple_cache[cache_key] = _construct_provider(
                provider_name=provider_name,
                model_id=model_id,
                base_kwargs=base_kwargs,
            )
        out[role_str] = per_tuple_cache[cache_key]
    return out


def _construct_provider(
    *, provider_name: str, model_id: str, base_kwargs: dict[str, Any]
) -> LlmProvider:
    """Construct one :class:`LlmProvider` for the given ``(provider, model_id)`` tuple.

    Phase 2 supports ``bedrock`` only; other providers raise
    :class:`UnsupportedLlmProvider` so the CLI surface fails-fast
    rather than mid-agent-loop. Other plugin types land in Phase 3
    Task 3.F5 alongside the :class:`AnthropicProvider` /
    :class:`OpenAIProvider` plugins.
    """
    if provider_name == "bedrock":
        from smai_llm_bedrock import BedrockProvider  # noqa: PLC0415 — lazy import

        kwargs = dict(base_kwargs)
        kwargs["model_id"] = model_id
        if "region" not in kwargs:
            kwargs["region"] = "us-east-1"
        return BedrockProvider(**kwargs)
    raise UnsupportedLlmProvider(
        f"per-role LLM provider {provider_name!r} (model_id={model_id!r}) is not "
        f"shipped in Phase 2; only 'bedrock' is supported (anthropic / openai land "
        f"in Task 3.F5)"
    )


__all__ = [
    "UnsupportedLlmProvider",
    "build_per_role_llm_providers",
]
