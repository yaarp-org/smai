"""Pass-2 verification — methodology-layer compiler stage.

Per ``designs/smai/02-dsl-and-contracts.md`` §5. Composition layout:

- ``_types``: rule signature, finding type, and the
  :class:`VerificationError` exception.
- ``category_a..h_*``: one file per §5 category, exposing
  ``CATEGORY_*_RULES`` tuples of rule callables.
- ``category_pipeline_rejection``: §5.10 advisory rules.
- ``_verify``: the deterministic ``verify(...)`` entry point.

Public surface is re-exported via ``smai_core``; consumers should import from
the top-level package rather than from this subpackage's internals.
"""

from smai_core.verification._types import (
    RuleCallable,
    VerificationError,
)
from smai_core.verification._verify import (
    FACTOR_MODEL_RULES,
    PER_CG_RULES,
    verify,
    verify_experiment,
    verify_to_report,
)
from smai_core.verification.category_f_technique_compatibility import (
    TECHNIQUE_CATEGORY_V1_CLOSED_SET,
)

__all__ = [
    "FACTOR_MODEL_RULES",
    "PER_CG_RULES",
    "RuleCallable",
    "TECHNIQUE_CATEGORY_V1_CLOSED_SET",
    "VerificationError",
    "verify",
    "verify_experiment",
    "verify_to_report",
]
