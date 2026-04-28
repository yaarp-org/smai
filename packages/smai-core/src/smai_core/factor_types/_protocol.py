"""``FactorTypePlugin`` Protocol — methodology-layer extension seam.

Per ``designs/smai/02-dsl-and-contracts.md`` §3.1, DEC-026 (entry-point
discovery, dbt-adapter pattern), and DEC-031 sub-decision #3 (flat factor
types in v1; vN may register additional types via entry points without
forking ``smai-core``).

This module sits inside ``smai_core.factor_types`` rather than
``smai_core.entities`` so that ``entities/registries.py`` can import the
Protocol cleanly: ``_protocol.py`` keeps its references to
``ExperimentDefinition``, ``Registries``, and ``ValidationError`` under
``TYPE_CHECKING``, breaking what would otherwise be an import cycle
(registries → factor_types → entities/registries).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from smai_core.entities.experiment import ExperimentDefinition
    from smai_core.entities.registries import Registries
    from smai_core.entities.validation_report import ValidationError


@runtime_checkable
class FactorTypePlugin(Protocol):
    """A factor type — e.g., ``additive`` or ``substitutive``.

    Plugins register via Python entry points under the ``smai.factor_types``
    group::

        [project.entry-points."smai.factor_types"]
        additive = "smai_core.factor_types.additive:plugin"
        substitutive = "smai_core.factor_types.substitutive:plugin"

    smai-core ships the v1 set (additive, substitutive). External packages
    can register additional factor types in vN without changing smai-core;
    the compiler discovers them at startup. Collisions on ``name`` are a
    startup error.
    """

    name: str
    """Canonical short name. Matches the value of ``Factor.type``. Must be
    unique across all loaded plugins."""

    description: str
    """Human-readable explanation. Surfaces in error messages and docs."""

    def validate(
        self,
        experiment: ExperimentDefinition,
        registries: Registries,
    ) -> list[ValidationError]:
        """Single-method validation — emits the plugin's full check list.

        Called once per CG during Pass 2. Returns a flat list of
        ``ValidationError``s (empty list = pass). Plugins emit factor-level,
        entry-level, and controlled-conditions-level checks from this one
        function (the v1 narrowing of implementation-guidance contracts).

        ``registries`` carries ``technique_registry``, ``metric_registry``,
        and ``factor_type_plugins`` for cross-references inside the plugin's
        check list.
        """
        ...
