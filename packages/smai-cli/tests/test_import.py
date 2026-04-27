import importlib


def test_module_imports():
    module = importlib.import_module("smai_cli")
    assert module is not None
