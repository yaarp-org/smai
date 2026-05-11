"""``ControlledConditions`` — the ceteris paribus surface.

Per ``designs/smai/01-data-model.md`` §3.5 and §4.4. The plugin-extensibility
hot path: declared core fields plus ``extra="allow"`` for arbitrary
domain-specific top-level fields, with a ``has_field``/``get_field`` helper
that looks across all three structural locations uniformly.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from smai_core.entities.compute_requirements import ComputeRequirements


class ControlledConditions(BaseModel):
    """Everything held constant across entries within a CG.

    The schema declares the typed core (``dataset``, ``optimization``,
    ``seeds``, ``compute``) plus an explicit ``additional_fixed`` bucket;
    ``extra="allow"`` admits arbitrary additional top-level fields
    (``architecture``, ``pruning_method``, ``activation_function``, etc.)
    without forcing every domain-specific value under
    ``additional_fixed``. ``has_field`` / ``get_field`` provide a single
    source of truth across declared fields, extras, and
    ``additional_fixed``.

    ``compute`` is the per-experiment :class:`ComputeRequirements`
    block (`02-dsl-and-contracts.md` §7.4 / round-3 friction (C)).
    Defaults preserve pre-Phase-1 behavior (``gpu=True``) so existing
    YAMLs round-trip unchanged.
    """

    model_config = ConfigDict(extra="allow")

    dataset: dict[str, str]
    optimization: dict[str, str | int | float]
    seeds: list[int]
    compute: ComputeRequirements = Field(default_factory=ComputeRequirements)
    additional_fixed: dict[str, str | int | float | bool] | None = None

    def has_field(self, name: str) -> bool:
        """True if ``name`` is present as a declared field, an extra, or in additional_fixed."""
        if name in type(self).model_fields:
            value = getattr(self, name, None)
            if value is not None and name != "additional_fixed":
                return True
        extras = self.model_extra or {}
        if name in extras:
            return True
        if self.additional_fixed and name in self.additional_fixed:
            return True
        return False

    def get_field(self, name: str) -> Any | None:
        """Companion to ``has_field``; returns the value or ``None``."""
        if name in type(self).model_fields:
            value = getattr(self, name, None)
            if value is not None and name != "additional_fixed":
                return value
        extras = self.model_extra or {}
        if name in extras:
            return extras[name]
        if self.additional_fixed and name in self.additional_fixed:
            return self.additional_fixed[name]
        return None
