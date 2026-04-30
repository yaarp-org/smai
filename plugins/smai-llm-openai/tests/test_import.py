"""Entry-point discovery + import smoke tests."""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points

from smai_core.plugins import LlmProvider
from smai_llm_openai import OpenAIProvider


def test_module_imports() -> None:
    module = importlib.import_module("smai_llm_openai")
    assert module.OpenAIProvider is OpenAIProvider


def test_openai_provider_runtime_checkable_against_protocol() -> None:
    """`runtime_checkable` requires attribute presence; the class itself
    must satisfy the LlmProvider Protocol shape (without needing a real
    OpenAI API key)."""
    from _f5_openai_fakes import FakeOpenAIClient  # type: ignore[import-not-found]

    instance = OpenAIProvider(
        model_id="gpt-4o",
        openai_client=FakeOpenAIClient(),
    )
    assert isinstance(instance, LlmProvider)


def test_entry_point_advertises_openai_provider() -> None:
    eps = entry_points(group="smai.llm_providers")
    matching = [ep for ep in eps if ep.name == "openai"]
    assert matching, "openai entry point not registered under smai.llm_providers"
    loaded = matching[0].load()
    assert loaded is OpenAIProvider
