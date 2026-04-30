"""Entry-point discovery + import smoke tests."""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points

from smai_core.plugins import LlmProvider
from smai_llm_anthropic import AnthropicProvider


def test_module_imports() -> None:
    module = importlib.import_module("smai_llm_anthropic")
    assert module.AnthropicProvider is AnthropicProvider


def test_anthropic_provider_runtime_checkable_against_protocol() -> None:
    """`runtime_checkable` requires attribute presence; the class itself
    must satisfy the LlmProvider Protocol shape (without needing a real
    Anthropic API key)."""
    from _f5_anthropic_fakes import FakeAnthropicClient  # type: ignore[import-not-found]

    instance = AnthropicProvider(
        model_id="claude-opus-4-7",
        anthropic_client=FakeAnthropicClient(),
    )
    assert isinstance(instance, LlmProvider)


def test_entry_point_advertises_anthropic_provider() -> None:
    eps = entry_points(group="smai.llm_providers")
    matching = [ep for ep in eps if ep.name == "anthropic"]
    assert matching, "anthropic entry point not registered under smai.llm_providers"
    loaded = matching[0].load()
    assert loaded is AnthropicProvider
