"""``NumericValue`` — continuous/ordinal value primitive on Levels.

Per ``designs/smai/01-data-model.md`` §3.1.1 and DEC-031 sub-decision #6.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class NumericValue(BaseModel):
    """Optional metadata on a Level for points on a continuous/ordinal axis.

    The compiler uses ``value`` to admit verification rules (range membership,
    ordinal monotonicity) and to enable trend observation in the verdict
    context. ``min``/``max`` consistency and value-in-range checks are Pass-2
    verification rules (Task 1.5), not entity-level invariants.
    """

    model_config = ConfigDict(extra="forbid")

    value: float | int
    kind: Literal["continuous", "ordinal"]
    unit: str | None = None
    min: float | None = None
    max: float | None = None
