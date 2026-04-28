"""Shared helpers for Task 1.7 mechanical-evaluator tests.

Constructs typical ``RawMetrics`` shapes for each worked-example fixture
without re-deriving them in every test module.
"""

from __future__ import annotations

from collections.abc import Sequence

from smai_core import (
    EntryMetrics,
    RawMetrics,
    SeedRunOutcome,
    ValidationConfig,
    metric_ref_to_runtime_key,
)


def metric_key_for(config: ValidationConfig) -> str:
    """The runtime key for ``config.body.metric``."""
    return metric_ref_to_runtime_key(config.body.metric)


def constant_seed_outcomes(
    seeds: Sequence[int], value: float, metric_key: str
) -> dict[int, SeedRunOutcome]:
    """Synthesize ``len(seeds)`` completed seeds at ``value`` for ``metric_key``."""
    return {seed: SeedRunOutcome(completed=True, required={metric_key: value}) for seed in seeds}


def varied_seed_outcomes(
    seed_to_value: dict[int, float], metric_key: str
) -> dict[int, SeedRunOutcome]:
    """Synthesize completed seeds with the given per-seed values."""
    return {
        seed: SeedRunOutcome(completed=True, required={metric_key: value})
        for seed, value in seed_to_value.items()
    }


def failed_seed_outcomes(
    seeds: Sequence[int], reason: str = "synthetic_failure"
) -> dict[int, SeedRunOutcome]:
    """Synthesize ``len(seeds)`` failed seeds with the given ``failure_reason``."""
    return {seed: SeedRunOutcome(completed=False, failure_reason=reason) for seed in seeds}


def merge_outcomes(
    *parts: dict[int, SeedRunOutcome],
) -> dict[int, SeedRunOutcome]:
    """Right-most wins on collision; convenience for mixed completed/failed."""
    out: dict[int, SeedRunOutcome] = {}
    for part in parts:
        out.update(part)
    return out


def make_raw_metrics(
    by_entry: dict[str, dict[int, SeedRunOutcome]],
) -> RawMetrics:
    """Wrap ``{entry_id: seed_outcomes}`` into a typed :class:`RawMetrics`."""
    return RawMetrics(
        by_entry={
            entry_id: EntryMetrics(entry_id=entry_id, seed_outcomes=seed_outcomes)
            for entry_id, seed_outcomes in by_entry.items()
        }
    )


def baseline_seeds() -> tuple[int, ...]:
    """The seed list shared by every fixture's ``controlled_conditions.seeds``."""
    return (42, 1337, 2024, 9999, 55)
