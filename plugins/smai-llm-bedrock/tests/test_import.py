"""Entry-point discovery + import smoke tests."""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points

from smai_core.plugins import LlmProvider
from smai_llm_bedrock import BedrockProvider


def test_module_imports() -> None:
    module = importlib.import_module("smai_llm_bedrock")
    assert module.BedrockProvider is BedrockProvider


def test_bedrock_provider_runtime_checkable_against_protocol() -> None:
    """`runtime_checkable` requires attribute presence; the class
    itself must satisfy the LlmProvider Protocol shape (without
    needing instantiation, which would touch boto3 + AWS).
    """
    # Construct against an in-memory fake so we don't reach for AWS.
    from _bedrock_fakes import FakeBedrockClient  # type: ignore[import-not-found]

    instance = BedrockProvider(
        region="us-east-1",
        model_id="us.anthropic.claude-opus-4-6-v1",
        bedrock_client=FakeBedrockClient(),
    )
    assert isinstance(instance, LlmProvider)


def test_entry_point_advertises_bedrock_provider() -> None:
    eps = entry_points(group="smai.llm_providers")
    matching = [ep for ep in eps if ep.name == "bedrock"]
    assert matching, "bedrock entry point not registered under smai.llm_providers"
    loaded = matching[0].load()
    assert loaded is BedrockProvider
