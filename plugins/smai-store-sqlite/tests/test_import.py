import importlib


def test_module_imports():
    module = importlib.import_module("smai_store_sqlite")
    assert module is not None
