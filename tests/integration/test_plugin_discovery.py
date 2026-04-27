"""Smoke test: every plugin's entry point is discoverable and loadable.

Phase 0 stub. The classes raise NotImplementedError on construction by design;
this test only proves the discovery mechanism (importlib.metadata entry points)
works end-to-end before Phase 1 starts using it.
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
        with pytest.raises(NotImplementedError):
            cls()
