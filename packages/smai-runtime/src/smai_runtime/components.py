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
