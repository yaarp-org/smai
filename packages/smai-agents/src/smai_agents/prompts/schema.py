"""Pydantic schema for prompt configuration.

Per ``04-agents.md`` §10.1. The on-disk YAML files validate against
:class:`PromptLayer`; the composed in-process result is
:class:`PromptConfig`.

Layer kinds (per §10.2):

* ``base`` — the shipped default for the role (in
  ``smai_agents/prompts/<role>/base.yaml``). Every role has exactly one
  base. Updating the base is a code-shipped change.
* ``variant`` — a deployment-selected alternative for the role
  (``smai_agents/prompts/<role>/variants/<name>.yaml``). Multiple
  variants per role coexist; deployment config picks one. Variant slots
  per §2.5 / DEC-032: planner has ``novel_technique`` and
  ``paper_ingestion``; contextual evaluator has ``compare_to_baseline``
  and ``compare_to_target``.
* ``override`` — per-deployment final-word layer
  (``<overrides_dir>/<role>.yaml``). Exists for prompt iteration
  without touching shipped code.

Merge semantics for ``tools`` (§10.2): a non-null layer **replaces**
the prior list — appending would be a footgun. To add one tool, the
variant copies the base list and adds the entry. The ``enabled: bool``
field gives a softer alternative for the common "disable a tool from
the base" case.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from smai_agents.model_selection import TaskRole

PromptLayerKind = Literal["base", "variant", "override"]


class ToolBinding(BaseModel):
    """References a tool registered in the agent's tool registry.

    Per §10.1: ``enabled: bool = True`` lets variants A/B-test
    "removing tool X improves reasoning" without redefining the full
    tool list. The agent loop is expected to filter out
    ``enabled=False`` bindings at session-construction time.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description=(
            "Tool name as registered in the ToolRegistry "
            "(e.g., 'read_file', 'run_experiment')."
        ),
    )
    enabled: bool = Field(
        default=True,
        description=(
            "Whether the tool is active. False means the binding stays "
            "in the variant config (so a future change can re-enable "
            "it without re-listing) but the loop does not register the "
            "tool with the LlmProvider."
        ),
    )


class StructuredOutputTool(BaseModel):
    """Synthetic-tool description for single-call agents (§6).

    Multi-turn agents leave this null on every layer; single-call
    agents (code reviewer, contextual evaluator, supervisor, screener,
    enricher) set it on their base layer.

    The schema is referenced by Python import path (e.g.,
    ``"smai_agents.schemas.code_review:CodeReviewResult"``) — resolved
    at load time. v1's ``loadPrompt`` did similar runtime resolution
    via the registry pattern (DEC-006); v2 keeps the indirection so
    schema modules don't have to be imported at YAML-validation time.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description=(
            "Tool name passed to the LlmProvider as the synthetic tool "
            "(e.g., 'submit_review')."
        ),
    )
    description: str = Field(
        ...,
        description="Tool description — surfaced to the LLM as the synthetic tool's description.",
    )
    schema_module: str = Field(
        ...,
        description=(
            "Python import path for the Pydantic output schema, "
            "in the form 'pkg.module:ClassName' "
            "(e.g., 'smai_agents.schemas.code_review:CodeReviewResult'). "
            "Resolved at load time."
        ),
    )


class PromptLayer(BaseModel):
    """One layer of a prompt config (§10.1 verbatim).

    Every YAML file under ``smai_agents/prompts/`` deserializes to one
    of these. All content fields are optional; a layer that doesn't
    set a field leaves it inherited from the prior layer per §10.2's
    "later wins" merge.

    The base layer for every role MUST set ``system_prompt`` and
    ``initial_user_message_template`` (validated at composition time
    via :class:`PromptConfig`). Variants and overrides may leave them
    null to inherit.
    """

    model_config = ConfigDict(extra="forbid")

    layer: PromptLayerKind = Field(
        ...,
        description="Layer kind — base | variant | override.",
    )
    name: str = Field(
        ...,
        description=(
            "Human-readable layer id (e.g., 'harness_builder/base', "
            "'planner/novel_technique'). Surfaced in "
            "PromptConfig.layer_chain for auditability."
        ),
    )
    role: TaskRole = Field(
        ...,
        description="The TaskRole this layer is for. Must match the directory name.",
    )

    system_prompt: str | None = None
    initial_user_message_template: str | None = None
    tools: list[ToolBinding] | None = None
    structured_output_tool: StructuredOutputTool | None = None
    notes: str | None = Field(
        default=None,
        description="Free-form notes for human reviewers. Not consumed by the loop.",
    )


class PromptConfig(BaseModel):
    """Composed result of base + variant + override merging (§10.1 verbatim).

    The in-process surface the agent loop consumes. Constructed by
    :func:`smai_agents.prompts.load_prompt_config`; never written to
    disk. ``layer_chain`` is logged at session start (§3.4) so a debug
    review of "what prompt was actually run" never has to
    reverse-engineer the merge.
    """

    model_config = ConfigDict(extra="forbid")

    role: TaskRole
    system_prompt: str
    initial_user_message_template: str
    tools: list[ToolBinding]
    structured_output_tool: StructuredOutputTool | None
    layer_chain: list[str] = Field(
        ...,
        description=(
            "Resolved layer names, in application order. "
            "E.g., ['harness_builder/base', 'harness_builder/v3_lint_first', "
            "'tenant_acme/harness_builder']."
        ),
    )


__all__ = [
    "PromptConfig",
    "PromptLayer",
    "PromptLayerKind",
    "StructuredOutputTool",
    "ToolBinding",
]
