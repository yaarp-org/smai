"""``integrate_technique_output`` — splice technique output into the harness.

Per ``10-runtime-and-templates.md`` §3.5. For each extension-point key declared
on the manifest:

  - If ``technique_output`` provides a value, splice it into the corresponding
    field of the components bundle following the manifest's
    ``integration_pattern``.
  - If absent and ``optional=True``, leave the harness's default in place.
  - If absent and ``optional=False``, the run was already aborted by the
    type-check (§6.1); we never see this branch.

The closed v1 mapping from manifest keys to component fields and admissible
patterns lives in :mod:`smai_runtime.components`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from smai_runtime.components import (
    ADMISSIBLE_PATTERNS_FOR_KEY,
    COMPONENT_FIELD_FOR_KEY,
    HarnessComponents,
)
from smai_runtime.errors import TechniqueOutputContractError
from smai_runtime.manifest import HarnessAPIManifest, IntegrationPattern


def integrate_technique_output(
    harness_components: HarnessComponents,
    technique_output: dict[str, Any],
    manifest: HarnessAPIManifest,
) -> HarnessComponents:
    """Return a new :class:`HarnessComponents` with technique values spliced in.

    The check has already passed by the time this function runs (the
    template orders the type check before this call); but we re-validate
    that every manifest key is in the closed v1 mapping (§8.4) and the
    pattern is admissible — a manifest declaring an unknown key or an
    out-of-set pattern is a tooling error, not an agent error.
    """
    updates: dict[str, Any] = {}

    for ep in manifest.extension_points:
        if ep.key not in technique_output:
            continue
        if ep.key not in COMPONENT_FIELD_FOR_KEY:
            raise TechniqueOutputContractError(
                code="unknown_key",
                message=(
                    f"manifest declares extension key {ep.key!r} which is not "
                    f"in the closed v1 component-field mapping"
                ),
                extension_point_key=ep.key,
                expected_signature=ep.type_signature,
                actual_value_repr=None,
                manifest_hash=manifest.content_hash,
            )
        field = COMPONENT_FIELD_FOR_KEY[ep.key]
        admissible = ADMISSIBLE_PATTERNS_FOR_KEY.get(ep.key, frozenset())
        if ep.integration_pattern not in admissible:
            raise TechniqueOutputContractError(
                code="type_signature_mismatch",
                message=(
                    f"manifest pattern {ep.integration_pattern!r} for key "
                    f"{ep.key!r} is not in the v1 admissible set "
                    f"{sorted(admissible)}"
                ),
                extension_point_key=ep.key,
                expected_signature=ep.type_signature,
                actual_value_repr=None,
                manifest_hash=manifest.content_hash,
            )

        current = getattr(harness_components, field)
        new_value = _apply_pattern(ep.integration_pattern, current, technique_output[ep.key])
        updates[field] = new_value

    return harness_components.model_copy(update=updates)


def _apply_pattern(
    pattern: IntegrationPattern,
    current: Any,
    technique_value: Any,
) -> Any:
    """Closed-set integration patterns (§3.5)."""
    if pattern == "replace":
        return technique_value

    if pattern == "append":
        if current is None:
            return list(technique_value)
        if isinstance(technique_value, list):
            return list(cast("list[Any]", current)) + list(cast("list[Any]", technique_value))
        # tolerated: a single-callable append.
        return list(cast("list[Any]", current)) + [technique_value]

    if pattern == "wrap":
        # technique_value is a wrapper: Callable[[Component], Component].
        # If the harness's current factory is a zero-arg builder, wrap its
        # output by composing wrapper ∘ factory.
        if not callable(technique_value):
            raise TypeError("`wrap` pattern requires technique value to be callable")
        if current is None:
            return technique_value
        wrapper = cast("Callable[[Any], Any]", technique_value)
        inner = cast("Callable[..., Any]", current)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return wrapper(inner(*args, **kwargs))

        return wrapped

    if pattern == "override_dict":
        if not isinstance(technique_value, dict):
            raise TypeError("`override_dict` pattern requires a dict technique value")
        merged: dict[str, Any] = {}
        if isinstance(current, dict):
            merged.update(cast("dict[str, Any]", current))
        merged.update(cast("dict[str, Any]", technique_value))
        return merged

    raise ValueError(f"unknown integration pattern: {pattern!r}")
