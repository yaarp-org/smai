import importlib


def test_module_imports():
    module = importlib.import_module("smai_inline_agents")
    assert module is not None
