"""Loader for the bundled v1 metric registry JSON.

Reads ``metric_registry_v1.json`` from the package's bundled data via
``importlib.resources`` (so it works regardless of how ``smai-core`` is
installed — editable, wheel, zipapp). Validates against the
:class:`MetricRegistry` / :class:`AtomicMetricEntry` / :class:`ParametricFamily`
Pydantic models from ``01-data-model.md`` §3.8.1 and enforces the
registry-policy invariant (no ``=`` or ``__`` in canonical names, family
names, or string-form parameter values).

The loader is memoized (``functools.cache``) so repeated calls are cheap and
share one validated registry instance.
"""

from __future__ import annotations

import json
from functools import cache
from importlib.resources import files
from typing import Any, cast

from smai_core.entities.metric import (
    AtomicMetricEntry,
    MetricRegistry,
    ParametricFamily,
)

METRIC_REGISTRY_RESOURCE_NAME: str = "metric_registry_v1.json"
"""File name of the bundled JSON, under ``smai_core.data``."""

METRIC_REGISTRY_SCHEMA_VERSION: str = "v1"
"""Expected ``schema_version`` value at the top of the JSON."""

_FORBIDDEN_REGISTRY_TOKENS: tuple[str, ...] = ("=", "__")
"""Substrings forbidden in canonical names, family names, and string-form
parameter values per ``01-data-model.md`` §3.8.1 (so the
``metric_ref_to_runtime_key`` projection in ``02-dsl-and-contracts.md`` §6.4
round-trips deterministically)."""


class MetricRegistryDataError(ValueError):
    """Raised when the bundled metric registry JSON fails structural checks
    (schema version mismatch, duplicate names, registry-policy invariant
    violation, or Pydantic-rejected entry shape)."""


def _validate_registry_token(value: str, *, kind: str, where: str) -> None:
    for forbidden in _FORBIDDEN_REGISTRY_TOKENS:
        if forbidden in value:
            raise MetricRegistryDataError(
                f"{kind} {value!r} at {where} contains forbidden token "
                f"{forbidden!r} (registry-policy invariant — see 01-data-model.md §3.8.1)"
            )


def _check_parametric_value_invariant(family: ParametricFamily) -> None:
    seen = family.parameter_values_seen
    where = f"parametric.{family.family_name}.parameter_values_seen"
    if isinstance(seen, list):
        values: list[str | int | float] = seen
        for v in values:
            if isinstance(v, str):
                _validate_registry_token(v, kind="parameter value", where=where)
        return
    # dict[str, list[...]] — multi-parameter
    for param_name, vals in seen.items():
        _validate_registry_token(param_name, kind="parameter name", where=where)
        for v in vals:
            if isinstance(v, str):
                _validate_registry_token(
                    v, kind="parameter value", where=f"{where}[{param_name!r}]"
                )


def _index_atomic(
    rows: list[dict[str, Any]],
) -> dict[str, AtomicMetricEntry]:
    out: dict[str, AtomicMetricEntry] = {}
    for row in rows:
        entry = AtomicMetricEntry.model_validate(row)
        if entry.canonical_name in out:
            raise MetricRegistryDataError(
                f"duplicate atomic canonical_name {entry.canonical_name!r} in bundled registry"
            )
        _validate_registry_token(
            entry.canonical_name,
            kind="canonical_name",
            where=f"atomic[{entry.canonical_name!r}]",
        )
        out[entry.canonical_name] = entry
    return out


def _index_parametric(
    rows: list[dict[str, Any]],
) -> dict[str, ParametricFamily]:
    out: dict[str, ParametricFamily] = {}
    for row in rows:
        family = ParametricFamily.model_validate(row)
        if family.family_name in out:
            raise MetricRegistryDataError(
                f"duplicate parametric family_name {family.family_name!r} in bundled registry"
            )
        _validate_registry_token(
            family.family_name,
            kind="family_name",
            where=f"parametric[{family.family_name!r}]",
        )
        # Every parametric family must declare at least one observed value.
        seen = family.parameter_values_seen
        if isinstance(seen, list):
            empty = len(seen) == 0
        else:
            empty = (not seen) or any(len(v) == 0 for v in seen.values())
        if empty:
            raise MetricRegistryDataError(
                f"parametric family {family.family_name!r} has empty "
                f"parameter_values_seen — every family must list >=1 observed value"
            )
        _check_parametric_value_invariant(family)
        out[family.family_name] = family
    return out


@cache
def load_metric_registry() -> MetricRegistry:
    """Load and validate the bundled v1 metric registry.

    Reads ``smai_core/data/metric_registry_v1.json``, validates its structure
    against the Pydantic models, enforces the registry-policy invariant, and
    returns a frozen-by-convention :class:`MetricRegistry` instance. Memoized
    via :func:`functools.cache` so repeated calls share one validated instance
    (the registry is read-only after load per ``02-dsl-and-contracts.md``
    §4.2).
    """
    raw_text = (files("smai_core.data") / METRIC_REGISTRY_RESOURCE_NAME).read_text(encoding="utf-8")
    parsed: object = json.loads(raw_text)
    if not isinstance(parsed, dict):
        raise MetricRegistryDataError(
            f"bundled metric registry must be a JSON object; got {type(parsed).__name__}"
        )
    payload = cast(dict[str, Any], parsed)
    schema_version = payload.get("schema_version")
    if schema_version != METRIC_REGISTRY_SCHEMA_VERSION:
        raise MetricRegistryDataError(
            f"bundled metric registry schema_version={schema_version!r}, expected "
            f"{METRIC_REGISTRY_SCHEMA_VERSION!r}"
        )
    atomic_raw = payload.get("atomic")
    parametric_raw = payload.get("parametric")
    if not isinstance(atomic_raw, list):
        raise MetricRegistryDataError("bundled registry's 'atomic' field must be a list")
    if not isinstance(parametric_raw, list):
        raise MetricRegistryDataError("bundled registry's 'parametric' field must be a list")
    return MetricRegistry(
        atomic=_index_atomic(cast(list[dict[str, Any]], atomic_raw)),
        parametric=_index_parametric(cast(list[dict[str, Any]], parametric_raw)),
    )
