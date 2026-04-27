import importlib


def test_module_imports():
    module = importlib.import_module("smai_orchestrator")
    assert module is not None
