import importlib


def test_module_imports():
    module = importlib.import_module("smai_artifacts_localfs")
    assert module is not None
