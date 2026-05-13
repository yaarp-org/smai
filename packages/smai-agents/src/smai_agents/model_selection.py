"""Per-task model selection for the agent fleet.

Per ``04-agents.md`` §4 and DEC-020 / DEC-022. Lifts v1's
``getModelForTask()`` to Python; the function returns a
``(provider_name, model_id)`` tuple. The orchestrator's dispatch
handler (Task 2.C4) is the caller — it constructs one
:class:`smai_core.plugins.LlmProvider` per per-task model and passes
the resulting ``dict[TaskRole, LlmProvider]`` to
:class:`smai_agents.loop.AgentSession` (so the loop never instantiates
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
from typing import Literal

# Per §4 verbatim — eight roles. Five fleet roles plus the supervisor
# plus the two paper-ingestion-internal roles whose model selection
# still flows through this resolver per §4's TaskRole comment.
TaskRole = Literal[
    "planner",
    "harness_builder",
    "technique_implementer",
    "code_reviewer",
    "contextual_evaluator",
    "supervisor",
    "screener",
    "enricher",
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
    overrides: dict[TaskRole, tuple[str, str]] | None = None,
    env: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Resolve ``(provider_name, model_id)`` for ``role``.

    Resolution order (round-7 reordering — env wins over config):

    1. ``env[SMAI_MODEL_<ROLE_UPPER>]`` parsed as ``"<provider>:<model_id>"``.
       Defaults to :data:`os.environ`.
    2. ``overrides[role]`` — the config-supplied per-role override map
       (``RuntimeConfig.engine.role_models`` once the CLI has folded in
       the configured provider name; or a future system-config UI).
    3. :data:`TASK_DEFAULTS` ``[role]``.
    """
    env_map = env if env is not None else os.environ
    env_value = env_map.get(_env_var_for(role))
    if env_value is not None:
        return _parse_env_value(role, env_value)

    if overrides is not None:
        explicit = overrides.get(role)
        if explicit is not None:
            return explicit

    return TASK_DEFAULTS[role]


def _env_var_for(role: TaskRole) -> str:
    return f"{_ENV_VAR_PREFIX}{role.upper()}"


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
