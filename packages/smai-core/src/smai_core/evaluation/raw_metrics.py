"""``RawMetrics`` and friends — the evaluator's input contract.

Per ``designs/smai/06-mechanical-evaluation.md`` §2. The shape an outer
system (Tier A pipeline runtime, or Tier B integrator on its own
infrastructure) must produce before invoking :func:`evaluate`. Derivable
in full from ``HarnessContract.body.required_metrics`` and
``HarnessContract.body.optional_telemetry`` via
:func:`smai_core.metric_ref_to_runtime_key`.

The :class:`SeedRunOutcome` shape is deliberate per §2.1: ``completed: bool``
plus a ``failure_reason`` string disambiguates "this run failed" from
"this seed wasn't reported yet" — silence would conflate the two.

The canonical-hash helper :func:`raw_metrics_canonical_hash` per §2.3
projects ``RawMetrics`` to a verdict-bearing subset (``failure_reason`` and
``optional`` telemetry stripped) and hashes via the same canonical-JSON +
sha256 path used by the contract envelopes (reused from
``smai_core.artifacts._canonical``).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from smai_core.artifacts._canonical import artifact_hash


class SeedRunOutcome(BaseModel):
    """Result of one ``(entry, seed)`` run (§2).

    ``completed=True`` iff the run finished and produced metric values; in
    that case ``required`` is a non-``None`` map from canonical *runtime
    key* (per ``smai_core.metric_ref_to_runtime_key``) to value. ``optional``
    carries telemetry keyed the same way; per-metric value type derives
    from the metric registry's ``input_shape`` (most are scalars; some
    richer — ``training_loss_curve`` is a list, ``peak_gpu_memory`` is an
    ``int``, etc.).

    ``failure_reason`` is populated iff ``completed=False`` (free-text from
    the runtime; consumed by ``verdict_context`` narrative grounding only —
    never the boolean verdict). Stripped from the canonical-hash form.
    """

    model_config = ConfigDict(extra="forbid")

    completed: bool
    required: dict[str, float | int] | None = None
    optional: dict[str, Any] | None = None
    failure_reason: str | None = None


class EntryMetrics(BaseModel):
    """All seeds' outcomes for a single entry (§2)."""

    model_config = ConfigDict(extra="forbid")

    entry_id: str
    seed_outcomes: dict[int, SeedRunOutcome]


class RawMetrics(BaseModel):
    """The full input to the evaluator (§2).

    ``by_entry`` keys are entry IDs; they MUST match the parent
    :class:`ExperimentPlan`'s entries plus
    ``ValidationConfig.body.comparison.baseline_entry_id``. The evaluator
    raises :class:`RawMetricsShapeError` (§7) if the keys do not align.
    """

    model_config = ConfigDict(extra="forbid")

    by_entry: dict[str, EntryMetrics]


_HASH_PRECISION: int = 9
"""Stable rounding precision for required-metric values in canonical form.

Per §2.3: float64 → 1e-9 stable rounding to absorb low-order numerical
noise without changing the verdict-bearing identity.
"""


def _canonical_required_payload(required: dict[str, float | int] | None) -> dict[str, float | int]:
    """Round floats in ``required`` to ``_HASH_PRECISION`` for stable hashing.

    Ints pass through unchanged. ``None`` becomes ``{}`` so the canonical
    form is shape-stable across completed/failed seeds.
    """
    if required is None:
        return {}
    out: dict[str, float | int] = {}
    for key, value in required.items():
        if isinstance(value, bool):
            out[key] = int(value)
        elif isinstance(value, int):
            out[key] = value
        else:
            out[key] = round(float(value), _HASH_PRECISION)
    return out


def raw_metrics_canonical_form(rm: RawMetrics) -> dict[str, Any]:
    """Return ``rm``'s verdict-bearing canonical form per §2.3.

    Excluded from the canonical form (verdict-irrelevant): ``failure_reason``
    (free text), ``optional`` telemetry (large, varying-shape; consumed by
    ``verdict_context`` only). Included with stable rounding: ``completed``
    plus the ``required`` numeric values per seed. Sorting is performed by
    the downstream ``canonical_json`` helper (lexicographic on dict keys).
    """
    by_entry: dict[str, Any] = {}
    for entry_id, em in rm.by_entry.items():
        seeds: dict[str, Any] = {}
        for seed, outcome in em.seed_outcomes.items():
            seeds[str(seed)] = {
                "completed": outcome.completed,
                "required": _canonical_required_payload(outcome.required),
            }
        by_entry[entry_id] = {"seed_outcomes": seeds}
    return {"by_entry": by_entry}


def raw_metrics_canonical_hash(rm: RawMetrics) -> str:
    """Sha256 hex digest of ``rm``'s canonical form (§2.3).

    Two semantically equivalent ``RawMetrics`` (same completed/failed
    pattern, same numeric values within precision) produce the same hash
    even if their ``failure_reason`` strings or ``optional`` telemetry
    differ. This is the hash recorded on :class:`Verdict.raw_metrics_hash`.
    """
    return artifact_hash(raw_metrics_canonical_form(rm))
