"""Pytest fixtures for the smai-events tests.

Mounts the test directory on ``sys.path`` so the shared
``_4_k2_fixtures`` module is importable without a ``packages.``
qualifier (matches the workspace's pytest ``--import-mode=importlib``
+ ``conftest.py``-based discovery convention).
"""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent / "tests"
sys.path.insert(0, str(_TESTS_DIR))
