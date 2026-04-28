"""Factor-type plugin contract and built-in plugins.

Per ``designs/smai/02-dsl-and-contracts.md`` §3 and DEC-031 sub-decision #3.
The Protocol lives here (rather than under ``smai_core.entities``) so that
the entity layer can import it without forming a cycle: ``_protocol.py``
holds its references to ``ExperimentDefinition`` / ``Registries`` /
``ValidationError`` under ``TYPE_CHECKING``.
"""

from smai_core.factor_types._loader import (
    ENTRY_POINT_GROUP,
    FactorTypePluginError,
    load_builtin_factor_type_plugins,
)
from smai_core.factor_types._protocol import FactorTypePlugin
from smai_core.factor_types.additive import AdditivePlugin
from smai_core.factor_types.substitutive import SubstitutivePlugin

__all__ = [
    "ENTRY_POINT_GROUP",
    "AdditivePlugin",
    "FactorTypePlugin",
    "FactorTypePluginError",
    "SubstitutivePlugin",
    "load_builtin_factor_type_plugins",
]
