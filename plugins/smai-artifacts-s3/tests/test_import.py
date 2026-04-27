import importlib


def test_module_imports():
    module = importlib.import_module("smai_artifacts_s3")
    assert module is not None
