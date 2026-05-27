"""``HarnessComponents`` — the Pydantic-typed bundle the harness exposes.

Per ``10-runtime-and-templates.md`` §8.1 / §8.4. The harness builder
constructs an instance and exposes it via ``build_harness(config)``; the
integrator splices technique outputs into it. Field set is closed in v1
to match the closed v1 set of manifest extension-point keys (§8.4).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict

from smai_runtime.manifest import IntegrationPattern


class HarnessComponents(BaseModel):
    """Closed-set v1 components bundle (§8.1).

    Field types are left intentionally loose (``Callable`` rather than
    ``Callable[[nn.Module], nn.Module]``) — the manifest's
    ``type_signature`` is the load-bearing typed surface; this Pydantic
    model is the integration substrate.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    train_transforms: list[Callable[..., Any]] = []
    model_factory: Callable[..., Any] | None = None
    default_loss: Callable[..., Any] | None = None
    training_config: dict[str, Any] = {}
    callbacks: list[Callable[..., Any]] = []


# Closed v1 mapping from manifest extension-point keys to HarnessComponents
# fields and their admissible integration patterns (§8.4). The integrator
# uses this table; no manifest may declare a key outside its domain.
COMPONENT_FIELD_FOR_KEY: dict[str, str] = {
    "train_transforms": "train_transforms",
    "model_wrapper": "model_factory",
    "loss_fn": "default_loss",
    "training_overrides": "training_config",
    "callbacks": "callbacks",
}

ADMISSIBLE_PATTERNS_FOR_KEY: dict[str, frozenset[IntegrationPattern]] = {
    "train_transforms": frozenset({"append", "replace"}),
    "model_wrapper": frozenset({"wrap", "replace"}),
    "loss_fn": frozenset({"replace"}),
    "training_overrides": frozenset({"override_dict"}),
    "callbacks": frozenset({"append", "replace"}),
}

# Per-key safest default integration pattern for the validation-mode
# stub manifest the runner synthesizes when no real manifest is staged
# (round 23 sub-PR R1; see :func:`smai_runtime.runner._load_contracts_by_mode`
# and the round-20 in-code comment its successor replaces). "Safest" means
# the splice produces a sensible result when the baseline returns the key
# with its natural-typed value and the harness's matching component is at
# its default (empty list / None / empty dict). Two-option keys pick the
# pattern that preserves the harness's default rather than overwriting it:
# ``append`` for list-typed slots, ``wrap`` for the model factory. Single-
# option keys (loss_fn, training_overrides) have one admissible pattern by
# construction.
#
# Round-22 dogfood (project_round22_real_llm_dogfood.md, Wall #2) hit the
# previous ``extension_points=[]`` stub's blind spot: the agent's baseline
# returning a non-empty dict was rejected as ``unknown_key`` because the
# stub declared zero keys. The closed v1 mapping makes a richer stub
# derivable without any host-side or per-contract plumbing.
DEFAULT_VALIDATION_STUB_PATTERN_FOR_KEY: dict[str, IntegrationPattern] = {
    "train_transforms": "append",
    "model_wrapper": "wrap",
    "loss_fn": "replace",
    "training_overrides": "override_dict",
    "callbacks": "append",
}

# Per-key type signatures the validation-mode stub manifest declares.
# Match :class:`HarnessComponents`' field types so the integrator's
# splice (which mutates the components bundle in place) preserves type
# integrity. The stub's ``check_technique_output`` still catches values
# of the wrong shape (e.g. a baseline returning a non-list for
# ``train_transforms``).
DEFAULT_VALIDATION_STUB_TYPE_SIGNATURE_FOR_KEY: dict[str, str] = {
    "train_transforms": "list[Callable[..., Any]]",
    "model_wrapper": "Callable[..., Any]",
    "loss_fn": "Callable[..., Any]",
    "training_overrides": "dict[str, Any]",
    "callbacks": "list[Callable[..., Any]]",
}
