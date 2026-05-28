"""Per-task model selection for the agent fleet.

Per ``04-agents.md`` §4 and DEC-020 / DEC-022. Lifts v1's
``getModelForTask()`` to Python; the function returns a
``(provider_name, model_id)`` tuple. The orchestrator's dispatch
handler (Task 2.C4) is the caller — it constructs one
:class:`smai_core.plugins.LlmProvider` per per-task model and passes
the resulting ``dict[TaskRole, LlmProvider]`` to
:class:`smai_inline_agents.loop.AgentSession` (so the loop never instantiates
plugins itself; plugin instantiation lives upstream).

Resolution order, per §4 (round-7 reordering — env is highest so an
operator's shell override always wins over a checked-in config file):

1. ``SMAI_MODEL_<ROLE_UPPER>`` env var, format ``"<provider>:<model_id>"``.
   The v2 shape; v1's env-var convention was ``MODEL_<ROLE>``. Equivalent
   to the nested ``SMAI_ENGINE__ROLE_MODELS__<ROLE>`` form which lands in
   the config layer (see layer 2).

2. ``overrides[role]`` — the config-supplied per-role override map. The CLI
   builds this from ``RuntimeConfig.engine.role_models`` (each entry is a
   bare model id; the provider is the configured ``llm_provider``) — and a
   future ``MetadataStore``-backed system-config UI (DEC-022) can feed the
   same arg. This function takes the resolved override map directly, so it
   is not bound to a specific config-loading pathway.

3. Per-role default from :data:`TASK_DEFAULTS`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Literal

# Per §4 verbatim — eight roles. Five fleet roles plus the supervisor
# plus the two paper-ingestion-internal roles whose model selection
# still flows through this resolver per §4's TaskRole comment.
#
# planner_refactor Step 3 adds ``ingestion``: the SciReplicate-shaped
# Paper Agent (smai_inline_agents.ingestion) resolves its PydanticAI
# model via ``role_models["ingestion"]`` per
# planner_refactor/architectural_decisions.md §9 / ingestion_subagent.md §1.
TaskRole = Literal[
    "planner",
    "harness_builder",
    "technique_implementer",
    "code_reviewer",
    "contextual_evaluator",
    "supervisor",
    "screener",
    "enricher",
    "ingestion",
]


# Per §4 verbatim. Concrete model IDs are deployment-config and updated
# as model generations turn over; this table fixes the per-role tier
# shape (Opus for the heavy reasoning roles, Sonnet for the bounded
# single-call roles) per DEC-005's Bedrock-first stance and §4's
# "preserve v1's 'Opus for everything that matters' pattern" comment.
TASK_DEFAULTS: dict[TaskRole, tuple[str, str]] = {
    "planner": ("bedrock", "us.anthropic.claude-opus-4-6-v1"),
    "harness_builder": ("bedrock", "us.anthropic.claude-opus-4-6-v1"),
    "technique_implementer": ("bedrock", "us.anthropic.claude-opus-4-6-v1"),
    "code_reviewer": ("bedrock", "us.anthropic.claude-opus-4-6-v1"),
    "contextual_evaluator": ("bedrock", "us.anthropic.claude-sonnet-4-6"),
    "supervisor": ("bedrock", "us.anthropic.claude-sonnet-4-6"),
    "screener": ("bedrock", "us.anthropic.claude-sonnet-4-6"),
    "enricher": ("bedrock", "us.anthropic.claude-sonnet-4-6"),
    # Paper-extraction subagent (planner_refactor Step 3). Sonnet tier:
    # the SciReplicate retrieval loop is bounded extraction, not the
    # heavy open-ended reasoning the Opus roles do.
    "ingestion": ("bedrock", "us.anthropic.claude-sonnet-4-6"),
}


_ENV_VAR_PREFIX = "SMAI_MODEL_"


class ModelSelectionError(ValueError):
    """Raised when an env-var override is malformed.

    Per §4 the override format is ``"<provider>:<model_id>"``. An empty
    provider, empty model id, or missing ``:`` are all malformed; we
    surface a typed error so the caller can distinguish the
    "configuration is broken" path from the "no override set" path.
    """


def get_model_for_task(
    role: TaskRole,
    *,
    step: str | None = None,
    overrides: Mapping[TaskRole, tuple[str, str] | dict[str, tuple[str, str]]] | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    """Resolve ``(provider_name, model_id)`` for ``(role[, step])``.

    Resolution order (D3-extended; round-7 reordering — env wins over config):

    1. ``env[SMAI_MODEL_<ROLE>__<STEP>]`` (if ``step`` given) parsed as
       ``"<provider>:<model_id>"``. Most specific layer.
    2. ``env[SMAI_MODEL_<ROLE_UPPER>]`` parsed as ``"<provider>:<model_id>"``.
       Defaults to :data:`os.environ`.
    3. ``overrides[role][step]`` if ``overrides[role]`` is a step-keyed
       dict and ``step`` matches an entry.
    4. ``overrides[role]["_default"]`` if ``overrides[role]`` is a
       step-keyed dict (D3 ``"_default"`` sentinel; role-level fallback
       distinct from :data:`TASK_DEFAULTS`).
    5. ``overrides[role]`` if it is a bare ``(provider, model_id)``
       tuple (round-7 flat shape).
    6. :data:`TASK_DEFAULTS` ``[role]``.

    Inline-role callers (planner, code_reviewer, etc.) typically pass
    ``step=None``; the resolver still walks the nested form's
    ``"_default"`` sentinel before falling to :data:`TASK_DEFAULTS`.
    """
    env_map = env if env is not None else os.environ
    if step is not None:
        step_env_value = env_map.get(_step_env_var_for(role, step))
        if step_env_value is not None:
            return _parse_env_value(role, step_env_value)

    env_value = env_map.get(_env_var_for(role))
    if env_value is not None:
        return _parse_env_value(role, env_value)

    if overrides is not None:
        explicit = overrides.get(role)
        if isinstance(explicit, dict):
            if step is not None:
                step_entry = explicit.get(step)
                if step_entry is not None:
                    return step_entry
            default_entry = explicit.get("_default")
            if default_entry is not None:
                return default_entry
        elif explicit is not None:
            return explicit

    return TASK_DEFAULTS[role]


def _env_var_for(role: TaskRole) -> str:
    return f"{_ENV_VAR_PREFIX}{role.upper()}"


def _step_env_var_for(role: TaskRole, step: str) -> str:
    return f"{_ENV_VAR_PREFIX}{role.upper()}__{step.upper()}"


def _parse_env_value(role: TaskRole, raw: str) -> tuple[str, str]:
    if ":" not in raw:
        raise ModelSelectionError(f"{_env_var_for(role)} must be 'provider:model_id'; got {raw!r}")
    provider, model_id = raw.split(":", 1)
    provider = provider.strip()
    model_id = model_id.strip()
    if not provider or not model_id:
        raise ModelSelectionError(f"{_env_var_for(role)} must be 'provider:model_id'; got {raw!r}")
    return provider, model_id


__all__ = [
    "ModelSelectionError",
    "TASK_DEFAULTS",
    "TaskRole",
    "get_model_for_task",
]
