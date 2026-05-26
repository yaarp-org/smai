"""Per-step model resolution per D3 (extends round-7's per-role shape).

Six-layer precedence (most specific wins):

1. ``SMAI_MODEL_<ROLE>__<STEP>`` env var (format ``"<provider>:<model_id>"``).
2. ``SMAI_MODEL_<ROLE>`` env var (round-7 shortcut; same format).
3. ``overrides[role][step]`` (nested form's per-step entry).
4. ``overrides[role]["_default"]`` (nested form's role-level fallback).
5. ``overrides[role]`` (flat form, round-7 shape).
6. :data:`SANDBOXED_ROLE_DEFAULTS` ``[role]``.

The orchestrator-side host wiring that widens
``EngineConfig.role_models`` to nested-map shape and passes it through
to the sandbox via env is out of scope for sub-PR C1 (sub-PR D handles
the full host plumbing). What lands here is the sandbox-side consult:
given an ``overrides`` map (constructed by the dispatcher and passed
in) and the process env, resolve the model for the active step.

Restricted to the two sandboxed roles per architectural_decisions §6
(planner / supervisor / code_reviewer / contextual_evaluator stay
inline and do not consult this module).
"""

from __future__ import annotations

import os
from typing import Literal

SandboxedRole = Literal["harness_builder", "technique_implementer"]
"""The two roles whose mini-orchestrator dispatches per-step agents.

Inline roles (planner / supervisor / code_reviewer / contextual_evaluator)
do not consult this resolver — their model selection happens host-side
in the orchestrator worker via the legacy
``smai_agents.model_selection`` path.
"""


SANDBOXED_ROLE_DEFAULTS: dict[SandboxedRole, tuple[str, str]] = {
    "harness_builder": ("bedrock", "us.anthropic.claude-opus-4-6-v1"),
    "technique_implementer": ("bedrock", "us.anthropic.claude-opus-4-6-v1"),
}
"""Per-role fallback ``(provider, model_id)`` for the sandboxed fleet.

Mirrors the relevant subset of ``smai_agents.model_selection.TASK_DEFAULTS``
without importing it (smai-agent-runtime is sandbox-side and must not
depend on smai-agents per architectural_decisions §3). When the host
orchestrator widens to D3's nested-map shape, these defaults are the
post-precedence-chain floor.
"""


DEFAULT_BEDROCK_PROVIDER: str = "bedrock"
"""Provider-name string the resolved model is tagged with when only a
bare model id appears in the env (the env-shortcut uses
``"<provider>:<model_id>"``, but a future config layer may hand a bare
model id whose provider is the deployment's configured ``llm_provider``).
The mini-orchestrator currently grounds against Bedrock; widening to
multi-provider routing is sub-PR D's surface."""


_ENV_VAR_PREFIX = "SMAI_MODEL_"


class ModelSelectionError(ValueError):
    """Raised when an env-var override is malformed.

    The override format is ``"<provider>:<model_id>"``. An empty provider,
    empty model id, or missing ``:`` are all malformed; we surface a
    typed error so the caller can distinguish "configuration is broken"
    from "no override set".
    """


# Override map shape:
#   {role: model_id_str} (flat / round-7)
#   {role: {step | "_default": model_id_str}} (nested / D3)
# The mini-orchestrator dispatcher walks both shapes via the precedence
# chain below.
RoleOverride = str | dict[str, str]
OverrideMap = dict[str, RoleOverride]


def get_model_for_step(
    role: SandboxedRole,
    step: str,
    *,
    overrides: OverrideMap | None = None,
    env: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Resolve ``(provider_name, model_id)`` for ``(role, step)``.

    Precedence order is the six-layer chain documented in the module
    docstring. The function never raises ``KeyError``: every layer is
    optional and the role-default at the bottom of
    :data:`SANDBOXED_ROLE_DEFAULTS` is guaranteed for every
    :data:`SandboxedRole`.

    :raises ModelSelectionError: if an env override is malformed.
    """
    env_map = env if env is not None else os.environ

    step_env = env_map.get(_step_env_var(role, step))
    if step_env is not None:
        return _parse_env_value(role, step, step_env)

    role_env = env_map.get(_role_env_var(role))
    if role_env is not None:
        return _parse_env_value(role, None, role_env)

    if overrides is not None:
        role_override = overrides.get(role)
        if isinstance(role_override, dict):
            step_entry = role_override.get(step)
            if isinstance(step_entry, str):
                return (DEFAULT_BEDROCK_PROVIDER, step_entry)
            default_entry = role_override.get("_default")
            if isinstance(default_entry, str):
                return (DEFAULT_BEDROCK_PROVIDER, default_entry)
        elif isinstance(role_override, str):
            return (DEFAULT_BEDROCK_PROVIDER, role_override)

    return SANDBOXED_ROLE_DEFAULTS[role]


def _step_env_var(role: SandboxedRole, step: str) -> str:
    return f"{_ENV_VAR_PREFIX}{role.upper()}__{step.upper()}"


def _role_env_var(role: SandboxedRole) -> str:
    return f"{_ENV_VAR_PREFIX}{role.upper()}"


def _parse_env_value(role: SandboxedRole, step: str | None, raw: str) -> tuple[str, str]:
    if ":" not in raw:
        var_name = _step_env_var(role, step) if step is not None else _role_env_var(role)
        raise ModelSelectionError(f"{var_name} must be 'provider:model_id'; got {raw!r}")
    provider, model_id = raw.split(":", 1)
    provider = provider.strip()
    model_id = model_id.strip()
    if not provider or not model_id:
        var_name = _step_env_var(role, step) if step is not None else _role_env_var(role)
        raise ModelSelectionError(f"{var_name} must be 'provider:model_id'; got {raw!r}")
    return provider, model_id


__all__ = [
    "DEFAULT_BEDROCK_PROVIDER",
    "ModelSelectionError",
    "OverrideMap",
    "RoleOverride",
    "SANDBOXED_ROLE_DEFAULTS",
    "SandboxedRole",
    "get_model_for_step",
]
