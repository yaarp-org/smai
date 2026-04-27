import importlib


def test_module_imports():
    module = importlib.import_module("smai_llm_bedrock")
    assert module is not None
