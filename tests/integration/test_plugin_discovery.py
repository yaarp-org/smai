"""Smoke test: every plugin's entry point is discoverable and loadable.

Phase 0 introduced this test to prove the ``importlib.metadata`` entry-point
mechanism wires up correctly before Phase 1 starts depending on it. As
plugins land their real implementations (Phase 2 onwards), the
"raises NotImplementedError on no-arg construction" assertion stops
holding — real plugins (e.g., :class:`smai_llm_bedrock.BedrockProvider`)
require constructor args (region / model_id), so a no-arg call now
raises :class:`TypeError` instead.

The test therefore asserts the discovery + loadability invariant only:
the entry-point name resolves to a callable class object. Per-plugin
constructor contracts and Protocol conformance are exercised by each
plugin's own conformance subclass under ``plugins/<name>/tests/``.
"""

from importlib.metadata import entry_points

import pytest

EXPECTED: dict[str, set[str]] = {
    "smai.llm_providers": {"bedrock", "anthropic", "openai"},
    "smai.metadata_stores": {"sqlite", "postgres"},
    "smai.artifact_stores": {"localfs", "s3"},
    "smai.computes": {"localgpu", "modal", "runpod"},
}


@pytest.mark.parametrize(("group", "expected"), EXPECTED.items())
def test_entry_points_discovered(group: str, expected: set[str]) -> None:
    discovered = {ep.name for ep in entry_points(group=group)}
    missing = expected - discovered
    assert not missing, f"Missing in {group}: {missing}"


@pytest.mark.parametrize("group", EXPECTED.keys())
def test_entry_points_loadable(group: str) -> None:
    for ep in entry_points(group=group):
        if ep.name not in EXPECTED[group]:
            continue
        cls = ep.load()
        assert isinstance(cls, type), f"{group}:{ep.name} entry point did not resolve to a class"
