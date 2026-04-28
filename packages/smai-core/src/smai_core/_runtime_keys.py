"""Pure projection from a typed :class:`MetricRef` to its runtime string key.

Per ``02-dsl-and-contracts.md`` §6.4 / ``06-mechanical-evaluation.md`` §3.

Encoding:

* atomic     → ``ref.ref``                         (``"accuracy"``, ``"perplexity"``)
* parametric → ``f"{family}__{k1}={v1}__{k2}={v2}"`` with parameter keys
  sorted lexicographically and double-underscore separators
  (``"top_k_accuracy__k=5"``, ``"min_traj_error_at_k__error_kind=minADE__k=6"``).

Parameter values stringify via Python's primitive ``str()`` (ints decimal,
floats shortest round-trip, strings as their literal text — the registry
policy invariant guarantees no parameter value contains ``=`` or ``__`` so
the encoding round-trips deterministically).

Used by the contract emitter (Task 1.6) to dedup ``optional_telemetry`` and
will be reused by the evaluator (Task 1.7) when looking up values in
``SeedRunOutcome.required`` / ``SeedRunOutcome.optional``.
"""

from __future__ import annotations

from smai_core.entities.metric import AtomicMetricRef, MetricRef, ParametricMetricRef


def metric_ref_to_runtime_key(ref: MetricRef) -> str:
    """Return the canonical runtime string key for a :class:`MetricRef`.

    Pure, total over well-formed refs, deterministic. Same ref always
    produces byte-equal output regardless of the order parameters are
    declared in (parameter keys are sorted lexicographically before
    encoding).
    """
    if isinstance(ref, AtomicMetricRef):
        return ref.ref
    assert isinstance(ref, ParametricMetricRef)
    parts: list[str] = [ref.family]
    for key in sorted(ref.parameters):
        parts.append(f"{key}={ref.parameters[key]}")
    return "__".join(parts)
