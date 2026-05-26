"""Minimal per-step prompt loader for the sandbox-side mini-orchestrator.

Reads one YAML file (no base / variant / override composition; per-step
self-containment per D8a) and exposes:

* ``system_prompt`` — passed to :func:`smai_agent_runtime.agent_reasoning.build_agent`
  as the agent's system prompt.
* ``initial_user_message_template`` — a Jinja2 template the caller
  renders against a per-step input bundle to produce the first user
  message.

Strict-undefined rendering surfaces a missing variable as a clear error
rather than silently empty text — load-bearing for catching bundle-shape
drift early (per the D7a / D7b sub-spike testing pattern).

Process-local caching is intentionally absent. Sub-PR C1 prompt loads
are per-step (a handful per CG) and the YAMLs are small; caching adds
test brittleness for negligible runtime win.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import Any, cast

import yaml
from jinja2 import Environment, StrictUndefined


@dataclass(frozen=True)
class StepPrompt:
    """Loaded per-step prompt data.

    Mirrors the relevant subset of D8a's ``PromptLayer`` shape from
    ``smai_agents.prompts.schema.PromptLayer``: sub-PR C1 only consumes
    ``system_prompt`` and ``initial_user_message_template``, so the
    other fields (``tools``, ``structured_output_tool``, ``notes``) are
    deliberately not surfaced — the agent factory does not need them
    (output type comes from the call site; body-generation steps
    register no tools by construction).
    """

    name: str
    system_prompt: str
    initial_user_message_template: str


_PROMPT_PACKAGE = "smai_agent_runtime.prompts"


def load_step_prompt(role: str, step_filename: str) -> StepPrompt:
    """Load one per-step prompt YAML from the package's prompts tree.

    ``role`` selects the role subdirectory (e.g., ``"harness_builder"``);
    ``step_filename`` is the bare YAML filename (e.g.,
    ``"step_2_fill_init_py.yaml"``). Both arguments are validated as
    package-resource path segments to keep the on-disk shape audit-able
    from a directory listing.
    """
    resource_pkg = f"{_PROMPT_PACKAGE}.{role}"
    raw = resources.files(resource_pkg).joinpath(step_filename).read_text()
    loaded: Any = yaml.safe_load(raw)
    if not isinstance(loaded, dict):
        raise ValueError(
            f"step prompt {step_filename!r} for role {role!r} did not parse to a mapping"
        )
    parsed = cast(dict[str, Any], loaded)
    name = parsed.get("name")
    system_prompt = parsed.get("system_prompt")
    template = parsed.get("initial_user_message_template")
    if not isinstance(name, str) or not name:
        raise ValueError(f"step prompt {step_filename!r} missing required 'name' field")
    if not isinstance(system_prompt, str) or not system_prompt:
        raise ValueError(f"step prompt {step_filename!r} missing required 'system_prompt' field")
    if not isinstance(template, str) or not template:
        raise ValueError(
            f"step prompt {step_filename!r} missing required 'initial_user_message_template' field"
        )
    return StepPrompt(
        name=name,
        system_prompt=system_prompt,
        initial_user_message_template=template,
    )


def render_user_message(template: str, variables: dict[str, Any]) -> str:
    """Render a Jinja2 template against ``variables`` with StrictUndefined.

    A missing variable surfaces as :class:`jinja2.UndefinedError`. The
    mini-orchestrator's per-step dispatcher catches and surfaces that as
    a workflow-level bundle error rather than letting the agent receive
    a partially-rendered prompt.
    """
    env = Environment(undefined=StrictUndefined, autoescape=False)
    return env.from_string(template).render(**variables)


__all__ = ["StepPrompt", "load_step_prompt", "render_user_message"]
