"""Bundled static data + loaders for the v1 methodology layer.

Per ``designs/smai/01-data-model.md`` §3.8.1 and ``02-dsl-and-contracts.md``
§4 / §6: the closed v1 metric registry ships as a static JSON file inside
``smai-core``. JSON (not YAML) so the loader needs only ``json`` from the
standard library — ``smai-core``'s runtime dep allowlist (DEC-029) does not
admit pyyaml.

The on-disk shape is a list of entries; the loader maps each list to a
``dict`` keyed by ``canonical_name`` / ``family_name`` (the
:class:`MetricRegistry` Pydantic shape), validates, and enforces the
registry-policy invariant (no ``=`` or ``__`` in canonical names, family
names, or string-form parameter values per ``01-data-model.md`` §3.8.1).
"""

from smai_core.data._metric_registry_loader import (
    METRIC_REGISTRY_RESOURCE_NAME,
    METRIC_REGISTRY_SCHEMA_VERSION,
    MetricRegistryDataError,
    load_metric_registry,
)

__all__ = [
    "METRIC_REGISTRY_RESOURCE_NAME",
    "METRIC_REGISTRY_SCHEMA_VERSION",
    "MetricRegistryDataError",
    "load_metric_registry",
]
