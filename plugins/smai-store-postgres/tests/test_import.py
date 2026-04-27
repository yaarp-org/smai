import importlib


def test_module_imports():
    module = importlib.import_module("smai_store_postgres")
    assert module is not None
