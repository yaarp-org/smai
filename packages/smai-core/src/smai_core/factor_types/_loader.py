"""Entry-point discovery for factor-type plugins.

Per ``designs/smai/02-dsl-and-contracts.md`` §3.5 and DEC-026 (dbt-adapter
pattern). Reads ``importlib.metadata.entry_points(group="smai.factor_types")``,
loads each entry, validates Protocol conformance via ``isinstance`` against
the ``runtime_checkable`` Protocol, and indexes by ``plugin.name``. Name
collisions are a startup error.
"""

from __future__ import annotations

from importlib.metadata import entry_points

from smai_core.factor_types._protocol import FactorTypePlugin

ENTRY_POINT_GROUP: str = "smai.factor_types"


class FactorTypePluginError(RuntimeError):
    """Raised on plugin discovery / conformance / collision failures."""


def load_builtin_factor_type_plugins() -> dict[str, FactorTypePlugin]:
    """Discover registered factor-type plugins via entry points.

    Returns a dict keyed by ``plugin.name``. Includes the in-tree built-ins
    (``additive``, ``substitutive``) plus any third-party plugins installed
    in the same environment that register under ``smai.factor_types``.

    Raises ``FactorTypePluginError`` if a discovered object does not satisfy
    the ``FactorTypePlugin`` Protocol or if two plugins claim the same
    ``name``.
    """
    discovered = entry_points(group=ENTRY_POINT_GROUP)
    plugins: dict[str, FactorTypePlugin] = {}

    for entry in discovered:
        loaded = entry.load()
        if not isinstance(loaded, FactorTypePlugin):
            raise FactorTypePluginError(
                f"Entry point {entry.name}={entry.value!r} did not yield a "
                f"FactorTypePlugin (got {type(loaded).__name__})."
            )
        # The plugin's self-declared name (loaded.name) is the registry key,
        # not the entry-point key (entry.name) — so a third-party plugin
        # mis-registered as `additive = ...:plugin` whose .name is "additive"
        # cannot coexist with the built-in.
        plugin_name = loaded.name
        if plugin_name in plugins:
            raise FactorTypePluginError(
                f"Duplicate factor-type plugin name {plugin_name!r}: "
                f"{entry.value!r} collides with a previously loaded plugin."
            )
        plugins[plugin_name] = loaded

    return plugins
