""":func:`compile_experiment` — public Tier B entry point.

Verify + emit. Accepts either a single ``ExperimentDocument`` (returns one
:class:`ContractArtifactSet`) or a ``FactorModelDocument`` (returns one set
per child experiment, keyed by experiment id).
"""

from __future__ import annotations

from typing import overload

from smai_core.artifacts._emit import (
    emit_artifacts,
    emit_factor_model_artifacts,
)
from smai_core.artifacts._set import ContractArtifactSet
from smai_core.dsl.document import (
    DslDocument,
    ExperimentDocument,
    FactorModelDocument,
)
from smai_core.entities.registries import Registries
from smai_core.verification._verify import verify


@overload
def compile_experiment(
    document: ExperimentDocument, registries: Registries
) -> ContractArtifactSet: ...


@overload
def compile_experiment(
    document: FactorModelDocument, registries: Registries
) -> dict[str, ContractArtifactSet]: ...


def compile_experiment(
    document: DslDocument, registries: Registries
) -> ContractArtifactSet | dict[str, ContractArtifactSet]:
    """Verify the DSL document and emit its contract artifacts.

    For an :class:`ExperimentDocument` returns a single
    :class:`ContractArtifactSet`. For a :class:`FactorModelDocument` returns
    a ``dict[experiment_id, ContractArtifactSet]`` (per §8.4 — the
    FactorModel itself emits no artifact).

    Raises :class:`smai_core.verification.VerificationError` if the document
    fails verification. Warnings produced at emission time
    (cost-tag dedup, etc.) are surfaced via the lower-level
    :func:`emit_artifacts` / :func:`emit_factor_model_artifacts` entry
    points; ``compile_experiment`` is the lossy convenience wrapper that
    returns just the artifact set(s).
    """
    if isinstance(document, FactorModelDocument):
        verified_list = verify(document, registries)
        sets, _report = emit_factor_model_artifacts(document, verified_list, registries)
        return sets
    verified_one = verify(document, registries)
    artifact_set, _report = emit_artifacts(verified_one, registries)
    return artifact_set
