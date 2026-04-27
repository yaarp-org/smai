import importlib


def test_module_imports():
    module = importlib.import_module("smai_compute_localgpu")
    assert module is not None
