"""``Factor``, ``Level``, ``Entry`` — the experimental-design core.

Per ``designs/smai/01-data-model.md`` §3.3 and §3.4. Levels are inline on
``Entry`` (no separate ``LevelRef``); ``Factor`` has no ``levels[]`` list per
§4.1. ``Level.technique_id`` is nullable per DEC-013 (additive baselines).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from smai_core.entities.numeric import NumericValue
from smai_core.entities.technique import TechniqueParams


class Factor(BaseModel):
    """The dimension being varied in a CG.

    Two factor types in v1 per DEC-012; the ``Literal`` is locked to keep v1
    narrow. Adding factor types in vN (per DEC-031 sub-decision #3) is an
    additive change to the type annotation.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    type: Literal["additive", "substitutive"]
    description: str


class Level(BaseModel):
    """A specific value of a factor; always a leaf in v1, always inline on Entry.

    Combinations:

    * Substitutive level — ``technique_id`` is non-null.
    * Additive baseline — ``technique_id`` is null (DEC-013; harness extension
      point defaults to no-op).
    * Additive non-baseline — ``technique_id`` is non-null.
    * Ordinal/continuous level — ``value`` is non-null (level sits on a
      numeric axis).
    * Configured technique — ``technique_params`` carries this entry's
      configuration of the technique.

    ``factor`` names the enclosing factor; ``entry.level_factor_matches`` (Task
    1.5 verification) checks consistency with the CG's factor name.
    """

    model_config = ConfigDict(extra="forbid")

    factor: str
    name: str
    description: str | None = None
    technique_id: str | None = None
    technique_params: TechniqueParams | None = None
    value: NumericValue | None = None


class Entry(BaseModel):
    """One arm of a CG — a specific level + the parameters used to instantiate it.

    ``is_baseline`` is explicit on every entry (per §4.6). The compiler verifies
    count and consistency at Pass-2 (Task 1.5).
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    is_baseline: bool
    level: Level
