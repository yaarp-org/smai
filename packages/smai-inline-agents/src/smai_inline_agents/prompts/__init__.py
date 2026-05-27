"""Prompt configuration as first-class surface.

Per ``04-agents.md`` §10. Hybrid: YAML on disk, Pydantic in process.

* :class:`PromptLayer` / :class:`ToolBinding` /
  :class:`StructuredOutputTool` — the on-disk YAML schema.
* :class:`PromptConfig` — the composed result of base + variant +
  override merging, the in-process surface the agent loop consumes.
* :func:`load_prompt_config` — assembles a :class:`PromptConfig` for
  ``(role, variant_name, overrides_dir)`` with a process-local cache.
* :func:`render_initial_user_message` — Jinja2-renders
  ``initial_user_message_template`` with task-specific variables.
* :func:`clear_prompt_config_cache` — test helper.

Composition: **base → variant → override**, "later wins" — except for
``tools`` where a non-null layer **replaces** the prior list (per §10.2,
appending is a footgun). Layer chain is logged on the composed
:class:`PromptConfig.layer_chain`.
"""

from __future__ import annotations

from smai_inline_agents.prompts.loader import (
    PromptConfigError,
    PromptConfigNotFound,
    PromptConfigValidationError,
    clear_prompt_config_cache,
    load_prompt_config,
    render_initial_user_message,
)
from smai_inline_agents.prompts.schema import (
    PromptConfig,
    PromptLayer,
    PromptLayerKind,
    StructuredOutputTool,
    ToolBinding,
)

__all__ = [
    "PromptConfig",
    "PromptConfigError",
    "PromptConfigNotFound",
    "PromptConfigValidationError",
    "PromptLayer",
    "PromptLayerKind",
    "StructuredOutputTool",
    "ToolBinding",
    "clear_prompt_config_cache",
    "load_prompt_config",
    "render_initial_user_message",
]
