"""Metric-emission contract — produces ``RawMetrics``-compatible output.

Per ``10-runtime-and-templates.md`` §10. The harness's ``evaluate()`` returns a
flat dict; this module serializes it to ``workspace/metrics.json`` after
validating that every required-metric runtime key (per
:func:`smai_core.metric_ref_to_runtime_key`) is present.

The shape this writes is what :class:`smai_core.SeedRunOutcome.required` is
populated from per §2 of ``06-mechanical-evaluation.md``; consumers
construct a :class:`smai_core.SeedRunOutcome` from this output via
:func:`build_seed_run_outcome`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smai_core import (
    HarnessContract,
    SeedRunOutcome,
    metric_ref_to_runtime_key,
)

from smai_runtime.errors import MetricsContractError

METRICS_FILENAME: str = "metrics.json"
"""§10.1 — runtime writes the verdict-bearing metrics here."""

VALIDATION_RESULTS_FILENAME: str = "validation_results.json"
"""§10.4 — structured validation output written during validation runs."""


def required_runtime_keys(harness_contract: HarnessContract) -> list[str]:
    """Project ``required_metrics`` to the canonical runtime keys (§10.1)."""
    return [metric_ref_to_runtime_key(ref) for ref in harness_contract.body.required_metrics]


def optional_runtime_keys(harness_contract: HarnessContract) -> list[str]:
    """Project ``optional_telemetry`` to the canonical runtime keys (§10.1)."""
    return [metric_ref_to_runtime_key(ref) for ref in harness_contract.body.optional_telemetry]


def validate_metrics_dict(
    metrics: dict[str, Any],
    harness_contract: HarnessContract,
) -> None:
    """Raise :class:`MetricsContractError` if any required key is absent (§10.1).

    Optional-telemetry presence is best-effort: missing keys are fine,
    present-but-unrecognized keys are tolerated (caller may filter via
    :func:`filter_to_known_keys`).
    """
    required = required_runtime_keys(harness_contract)
    missing = [key for key in required if key not in metrics]
    if missing:
        raise MetricsContractError(
            missing_keys=missing,
            present_keys=sorted(metrics),
        )


def filter_to_known_keys(
    metrics: dict[str, Any],
    harness_contract: HarnessContract,
) -> dict[str, Any]:
    """Drop keys that are neither required nor declared optional (§4 / §10.1).

    Per ``HarnessContract`` field ``optional_telemetry``: present-but-unrecognized
    keys are dropped with a warning. We elide the warning at this layer (the
    runtime emits a single structured-error surface for required-key failure
    only); call sites can add stderr warnings as needed.
    """
    known: set[str] = set(required_runtime_keys(harness_contract)) | set(
        optional_runtime_keys(harness_contract)
    )
    return {k: v for k, v in metrics.items() if k in known}


def write_metrics(
    workspace: Path,
    metrics: dict[str, Any],
    harness_contract: HarnessContract,
) -> Path:
    """Validate ``metrics`` against ``harness_contract`` and write them as JSON.

    Validation is mandatory: a missing required key is a runtime-mechanical
    failure (§10.1). Returned path is the absolute location of
    ``metrics.json`` for the caller's convenience.
    """
    validate_metrics_dict(metrics, harness_contract)
    target = workspace / METRICS_FILENAME
    target.write_text(json.dumps(metrics, sort_keys=True, indent=2))
    return target


def build_seed_run_outcome(
    metrics: dict[str, Any],
    harness_contract: HarnessContract,
    *,
    completed: bool = True,
    failure_reason: str | None = None,
) -> SeedRunOutcome:
    """Construct a :class:`smai_core.SeedRunOutcome` from a ``metrics.json``-shaped
    dict (§10.2).

    The split between ``required`` and ``optional`` is keyed off the contract;
    a key in ``metrics`` that maps to a required-metric runtime key lands in
    ``required``, optional-telemetry keys land in ``optional``, and any
    other keys are dropped (§10.1).
    """
    if not completed:
        return SeedRunOutcome(
            completed=False,
            required=None,
            optional=None,
            failure_reason=failure_reason,
        )

    required_keys = set(required_runtime_keys(harness_contract))
    optional_keys = set(optional_runtime_keys(harness_contract))

    required: dict[str, float | int] = {}
    optional: dict[str, Any] = {}
    for key, value in metrics.items():
        if key in required_keys:
            required[key] = _coerce_to_required_value(value)
        elif key in optional_keys:
            optional[key] = value

    return SeedRunOutcome(
        completed=True,
        required=required,
        optional=optional or None,
        failure_reason=None,
    )


def _coerce_to_required_value(value: Any) -> float | int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, str):
        return float(value)
    raise TypeError(
        f"required-metric value must be int|float|bool|str (got {type(value).__name__})"
    )
