"""Pytest fixtures + sys.path setup for the integration test tree.

The workspace runs pytest with ``--import-mode=importlib`` (top-level
``pyproject.toml``); module names must therefore be unique. The
integration tests use ``_e3_*`` prefixed fixture modules per the Task
3.E3 brief's filename-hygiene guidance — they don't collide with
sibling test trees, but they need their parent directory mounted on
``sys.path`` so the test modules can ``from _e3_fakes import ...``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_INTEGRATION_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_INTEGRATION_TESTS_DIR))
