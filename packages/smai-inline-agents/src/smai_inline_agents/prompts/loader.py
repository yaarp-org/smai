"""Loader and Jinja2 renderer for prompt configs.

Per ``04-agents.md`` §10. The loader:

1. Reads the base YAML for the role from the package-resource
   directory ``smai_inline_agents/prompts/<role>/base.yaml``.
2. If ``variant_name`` is provided, reads the variant YAML from
   ``smai_inline_agents/prompts/<role>/variants/<variant_name>.yaml``.
3. If ``overrides_dir`` is provided and ``<overrides_dir>/<role>.yaml``
   exists, reads it as the override layer.
4. Validates each YAML against :class:`PromptLayer`.
5. Merges per §10.2 semantics ("later wins" on every field, except
   ``tools`` where a non-null layer **replaces** the prior list).
6. Validates the composed result against :class:`PromptConfig`.

Caching (§10.3): a process-local cache keyed by
``(role, variant_name, overrides_dir, overrides_dir_mtime)``. The
overrides-dir mtime is included so an interactive ``smai dev`` session
that edits an override YAML does not see stale config — v1's
``dev/`` scratch space is the analogous use case.

Jinja2 rendering (§10.1): the
``initial_user_message_template`` is a Jinja2 template; the agent loop
calls :func:`render_initial_user_message` at session start with
task-specific variables (CG id, paper extract path, contract artifact
ids, etc.). Strict-undefined mode is used so a missing variable surfaces
as a clear error rather than silently empty text.
"""

from __future__ import annotations

import threading
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined, TemplateSyntaxError, UndefinedError
from pydantic import ValidationError

from smai_inline_agents.model_selection import TaskRole
from smai_inline_agents.prompts.schema import (
    PromptConfig,
    PromptLayer,
    PromptLayerKind,
    StructuredOutputTool,
    ToolBinding,
)

# Package-resource path for the shipped base + variant YAMLs. The
# directory layout is:
#
#   smai_inline_agents/prompts/<role>/base.yaml
#   smai_inline_agents/prompts/<role>/variants/<variant_name>.yaml
#
_PROMPT_PACKAGE = "smai_inline_agents.prompts"


# =============================================================================
# Errors
# =============================================================================


class PromptConfigError(Exception):
    """Base class for prompt-config loader errors."""


class PromptConfigNotFound(PromptConfigError):
    """Raised when a base YAML, variant YAML, or override YAML is missing."""

    def __init__(self, *, role: TaskRole, layer: PromptLayerKind, location: str) -> None:
        self.role = role
        self.layer = layer
        self.location = location
        super().__init__(
            f"prompt config not found: role={role!r} layer={layer!r} location={location!r}"
        )


class PromptConfigValidationError(PromptConfigError):
    """Raised when a YAML file fails Pydantic validation, or when the
    composed :class:`PromptConfig` is missing required fields."""

    def __init__(self, *, location: str, detail: str) -> None:
        self.location = location
        self.detail = detail
        super().__init__(f"prompt config invalid at {location!r}: {detail}")


# =============================================================================
# Cache (process-local, keyed by (role, variant_name, overrides_dir, mtime))
# =============================================================================


_CacheKey = tuple[TaskRole, str | None, str | None, float | None]
_cache: dict[_CacheKey, PromptConfig] = {}
_cache_lock = threading.Lock()


def clear_prompt_config_cache() -> None:
    """Clear the process-local prompt-config cache.

    Used by tests and by ``smai dev`` reload paths.
    """
    with _cache_lock:
        _cache.clear()


# =============================================================================
# Loader
# =============================================================================


def load_prompt_config(
    role: TaskRole,
    *,
    variant_name: str | None = None,
    overrides_dir: Path | str | None = None,
) -> PromptConfig:
    """Compose the prompt config for ``role`` (§10.1 / §10.3).

    Args:
        role: The :data:`TaskRole` for which to load config.
        variant_name: If non-null, loads
            ``smai_inline_agents/prompts/<role>/variants/<variant_name>.yaml``
            as the variant layer.
        overrides_dir: If non-null and ``<overrides_dir>/<role>.yaml``
            exists, loads it as the override layer.

    Returns:
        Composed :class:`PromptConfig`.

    Raises:
        PromptConfigNotFound: base YAML missing, variant requested but
            not present, or override path is a malformed file.
        PromptConfigValidationError: YAML failed schema validation, or
            the composed config is missing required fields.
    """
    overrides_dir_path = Path(overrides_dir) if overrides_dir is not None else None
    overrides_mtime = _overrides_dir_mtime(overrides_dir_path, role)
    cache_key: _CacheKey = (
        role,
        variant_name,
        str(overrides_dir_path) if overrides_dir_path is not None else None,
        overrides_mtime,
    )

    with _cache_lock:
        cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    layers: list[tuple[str, PromptLayer]] = []

    # --- Base layer (always required) ---
    base_loc, base_layer = _load_base_layer(role)
    layers.append((base_loc, base_layer))

    # --- Variant layer (optional) ---
    if variant_name is not None:
        variant_loc, variant_layer = _load_variant_layer(role, variant_name)
        layers.append((variant_loc, variant_layer))

    # --- Override layer (optional) ---
    if overrides_dir_path is not None:
        override = _load_override_layer(role, overrides_dir_path)
        if override is not None:
            override_loc, override_layer = override
            layers.append((override_loc, override_layer))

    composed = _compose(role=role, layers=layers)

    with _cache_lock:
        _cache[cache_key] = composed
    return composed


# =============================================================================
# Layer-loading helpers
# =============================================================================


def _load_base_layer(role: TaskRole) -> tuple[str, PromptLayer]:
    """Read and validate the base YAML for ``role``."""
    package = f"{_PROMPT_PACKAGE}.{role}"
    try:
        files = resources.files(package)
    except (ModuleNotFoundError, FileNotFoundError) as exc:
        raise PromptConfigNotFound(
            role=role,
            layer="base",
            location=f"{package}/base.yaml",
        ) from exc

    base_resource = files / "base.yaml"
    if not _resource_is_file(base_resource):
        raise PromptConfigNotFound(
            role=role,
            layer="base",
            location=f"{package}/base.yaml",
        )
    location = f"{package}/base.yaml"
    raw = base_resource.read_text(encoding="utf-8")
    parsed = _parse_yaml(raw, location=location)
    layer = _validate_layer(parsed, location=location, expected_kind="base")
    if layer.role != role:
        raise PromptConfigValidationError(
            location=location,
            detail=f"role field {layer.role!r} does not match directory {role!r}",
        )
    return location, layer


def _load_variant_layer(
    role: TaskRole,
    variant_name: str,
) -> tuple[str, PromptLayer]:
    """Read and validate the variant YAML for ``(role, variant_name)``."""
    package = f"{_PROMPT_PACKAGE}.{role}.variants"
    try:
        files = resources.files(package)
    except (ModuleNotFoundError, FileNotFoundError) as exc:
        raise PromptConfigNotFound(
            role=role,
            layer="variant",
            location=f"{package}/{variant_name}.yaml",
        ) from exc

    variant_resource = files / f"{variant_name}.yaml"
    if not _resource_is_file(variant_resource):
        raise PromptConfigNotFound(
            role=role,
            layer="variant",
            location=f"{package}/{variant_name}.yaml",
        )
    location = f"{package}/{variant_name}.yaml"
    raw = variant_resource.read_text(encoding="utf-8")
    parsed = _parse_yaml(raw, location=location)
    layer = _validate_layer(parsed, location=location, expected_kind="variant")
    if layer.role != role:
        raise PromptConfigValidationError(
            location=location,
            detail=f"role field {layer.role!r} does not match directory {role!r}",
        )
    return location, layer


def _load_override_layer(
    role: TaskRole,
    overrides_dir: Path,
) -> tuple[str, PromptLayer] | None:
    """Read and validate ``<overrides_dir>/<role>.yaml`` if present."""
    candidate = overrides_dir / f"{role}.yaml"
    if not candidate.is_file():
        return None
    location = str(candidate)
    raw = candidate.read_text(encoding="utf-8")
    parsed = _parse_yaml(raw, location=location)
    layer = _validate_layer(parsed, location=location, expected_kind="override")
    if layer.role != role:
        raise PromptConfigValidationError(
            location=location,
            detail=f"role field {layer.role!r} does not match override target {role!r}",
        )
    return location, layer


def _overrides_dir_mtime(
    overrides_dir: Path | None,
    role: TaskRole,
) -> float | None:
    """Return the mtime of ``<overrides_dir>/<role>.yaml`` if it exists.

    Threaded into the cache key so an edit to an override YAML
    invalidates the cache entry on the next load.
    """
    if overrides_dir is None:
        return None
    candidate = overrides_dir / f"{role}.yaml"
    try:
        return candidate.stat().st_mtime
    except FileNotFoundError:
        # No override file present — track ``None`` so a future write
        # to the file produces a different cache key.
        return None


def _resource_is_file(traversable: Traversable) -> bool:
    """importlib.resources Traversable doesn't expose .exists() consistently."""
    try:
        return traversable.is_file()
    except (FileNotFoundError, NotADirectoryError):
        return False


def _parse_yaml(raw: str, *, location: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise PromptConfigValidationError(
            location=location,
            detail=f"YAML parse failed: {exc}",
        ) from exc


def _validate_layer(
    parsed: Any,
    *,
    location: str,
    expected_kind: PromptLayerKind,
) -> PromptLayer:
    if not isinstance(parsed, dict):
        raise PromptConfigValidationError(
            location=location,
            detail=f"YAML root must be a mapping, got {type(parsed).__name__}",
        )
    try:
        layer = PromptLayer.model_validate(parsed)
    except ValidationError as exc:
        raise PromptConfigValidationError(
            location=location,
            detail=str(exc),
        ) from exc
    if layer.layer != expected_kind:
        raise PromptConfigValidationError(
            location=location,
            detail=(f"layer kind {layer.layer!r} does not match expected {expected_kind!r}"),
        )
    return layer


# =============================================================================
# Composition (§10.2)
# =============================================================================


def _compose(
    *,
    role: TaskRole,
    layers: list[tuple[str, PromptLayer]],
) -> PromptConfig:
    """Merge ``layers`` into a :class:`PromptConfig`.

    Per §10.2:

    * ``system_prompt`` / ``initial_user_message_template`` /
      ``structured_output_tool`` — later non-null wins.
    * ``tools`` — later non-null **replaces** (no append).
    * ``layer_chain`` — accumulates layer names in application order.
    """
    system_prompt: str | None = None
    initial_user_message_template: str | None = None
    tools: list[ToolBinding] | None = None
    structured_output_tool: StructuredOutputTool | None = None
    layer_chain: list[str] = []

    for location, layer in layers:
        layer_chain.append(layer.name)
        del location  # location is part of layer.name in shipped configs
        if layer.system_prompt is not None:
            system_prompt = layer.system_prompt
        if layer.initial_user_message_template is not None:
            initial_user_message_template = layer.initial_user_message_template
        if layer.tools is not None:
            tools = list(layer.tools)
        if layer.structured_output_tool is not None:
            structured_output_tool = layer.structured_output_tool

    base_location = layers[0][0] if layers else "<no layers>"

    if system_prompt is None:
        raise PromptConfigValidationError(
            location=base_location,
            detail="composed PromptConfig is missing system_prompt",
        )
    if initial_user_message_template is None:
        raise PromptConfigValidationError(
            location=base_location,
            detail=("composed PromptConfig is missing initial_user_message_template"),
        )

    return PromptConfig(
        role=role,
        system_prompt=system_prompt,
        initial_user_message_template=initial_user_message_template,
        tools=tools or [],
        structured_output_tool=structured_output_tool,
        layer_chain=layer_chain,
    )


# =============================================================================
# Jinja2 rendering (§10.1)
# =============================================================================


# Strict-undefined mode so a missing variable surfaces as a clear
# error. Trim/lstrip blocks default to off (matches the YAML's
# whitespace expectations).
_jinja_env = Environment(
    undefined=StrictUndefined,
    autoescape=False,
    keep_trailing_newline=True,
)


def render_initial_user_message(
    config: PromptConfig,
    /,
    **variables: Any,
) -> str:
    """Render :attr:`PromptConfig.initial_user_message_template` with ``variables``.

    Per §10.1: the agent loop renders the template at session start
    with task-specific variables (CG id, paper extract path, contract
    artifact ids, etc.). Strict-undefined surfaces missing variables
    as :class:`PromptConfigValidationError`.
    """
    try:
        template = _jinja_env.from_string(config.initial_user_message_template)
    except TemplateSyntaxError as exc:
        raise PromptConfigValidationError(
            location=f"<role={config.role}>/initial_user_message_template",
            detail=f"Jinja2 syntax error: {exc}",
        ) from exc
    try:
        return template.render(**variables)
    except UndefinedError as exc:
        raise PromptConfigValidationError(
            location=f"<role={config.role}>/initial_user_message_template",
            detail=f"missing template variable: {exc}",
        ) from exc


__all__ = [
    "PromptConfigError",
    "PromptConfigNotFound",
    "PromptConfigValidationError",
    "clear_prompt_config_cache",
    "load_prompt_config",
    "render_initial_user_message",
]
