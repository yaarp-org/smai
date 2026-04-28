"""Comparison rule registry — ``compare_to_baseline`` / ``compare_to_target``.

Per ``designs/smai/06-mechanical-evaluation.md`` §6: closed v1 registry of
per-entry comparison rules. Each rule is keyed by its ``name`` and satisfies
the :class:`ComparisonFunction` Protocol below.

Both v1 rules share identical numerical behavior — they differ only in
*what* the second positional argument means semantically. ``compare_to_baseline``
takes the aggregated baseline-entry metric (resolved by the compiler from
``ValidationConfig.body.comparison.baseline_entry_id``);
``compare_to_target`` takes ``ValidationConfig.body.comparison.target_value``
directly. The shared mechanics:

- ``compute_delta(treatment, comparand, direction)``:
  raw delta is ``treatment - comparand``; sign-flipped when
  ``direction == "lower_is_better"`` so a positive delta always means
  "treatment improved over the comparand."
- ``passed_threshold(delta, threshold, direction)``:
  ``delta >= threshold``. Per ``02-dsl-and-contracts.md`` §5.6
  (``validation.threshold_sign_matches_direction``), thresholds must be
  non-negative; ``threshold == 0.0`` allows ties.

The ``direction`` argument is ``Literal["higher_is_better", "lower_is_better"]``
in practice — the registry's ``ambiguous`` direction is resolved to one of
the two by ``ValidationCriteria.direction`` upstream (see
``02-dsl-and-contracts.md`` §6.5).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ComparisonFunction(Protocol):
    """Protocol satisfied by every comparison rule in the v1 closed set."""

    name: str
    """Canonical name; the key the rule is registered under."""

    def compute_delta(
        self,
        treatment: float,
        baseline_or_target: float,
        direction: str,
    ) -> float:
        """Signed delta — positive means treatment improved on the comparand.

        ``baseline_or_target`` is the aggregated baseline metric for
        ``compare_to_baseline`` and the user-declared target for
        ``compare_to_target``. The parameter is named generically because the
        same Protocol services both v1 rules.
        """
        ...

    def passed_threshold(self, delta: float, threshold: float, direction: str) -> bool:
        """Verdict bit: did the delta clear the threshold for this direction?"""
        ...


class ComparisonRuleNotRegistered(KeyError):
    """Raised by :func:`get_comparison_rule` when ``name`` is not registered."""


def _signed_delta(treatment: float, comparand: float, direction: str) -> float:
    raw = treatment - comparand
    if direction == "higher_is_better":
        return raw
    if direction == "lower_is_better":
        return -raw
    raise ValueError(
        f"unsupported direction {direction!r}; expected 'higher_is_better' or 'lower_is_better'"
    )


def _passed(delta: float, threshold: float, direction: str) -> bool:
    # Validate `direction` so the rule fails loudly on a malformed call rather
    # than silently accepting any string. The threshold-sign rule lives in
    # Task 1.5; here we only enforce direction wellformedness.
    if direction not in ("higher_is_better", "lower_is_better"):
        raise ValueError(
            f"unsupported direction {direction!r}; expected 'higher_is_better' or 'lower_is_better'"
        )
    return delta >= threshold


class _CompareToBaseline:
    name: str = "compare_to_baseline"

    def compute_delta(
        self,
        treatment: float,
        baseline_or_target: float,
        direction: str,
    ) -> float:
        return _signed_delta(treatment, baseline_or_target, direction)

    def passed_threshold(self, delta: float, threshold: float, direction: str) -> bool:
        return _passed(delta, threshold, direction)


class _CompareToTarget:
    name: str = "compare_to_target"

    def compute_delta(
        self,
        treatment: float,
        baseline_or_target: float,
        direction: str,
    ) -> float:
        return _signed_delta(treatment, baseline_or_target, direction)

    def passed_threshold(self, delta: float, threshold: float, direction: str) -> bool:
        return _passed(delta, threshold, direction)


COMPARISON_RULE_REGISTRY: dict[str, ComparisonFunction] = {
    "compare_to_baseline": _CompareToBaseline(),
    "compare_to_target": _CompareToTarget(),
}
"""The v1 closed set, keyed by ``ComparisonFunction.name`` (matches
``ComparisonRule.rule`` from ``01-data-model.md`` §3.6)."""


def get_comparison_rule(name: str) -> ComparisonFunction:
    """Look up a comparison rule by name; raise on miss."""
    rule = COMPARISON_RULE_REGISTRY.get(name)
    if rule is None:
        raise ComparisonRuleNotRegistered(
            f"comparison rule {name!r} is not registered; v1 closed set is "
            f"{sorted(COMPARISON_RULE_REGISTRY)}"
        )
    return rule
