"""Pytest configuration for smai-llm-bedrock tests.

Adds this tests/ directory to ``sys.path`` so per-test modules can
import the shared :mod:`_fakes` helper by name. Pytest's
``importlib`` mode (set on the workspace root) does not implicitly
extend ``sys.path``, so this conftest does so explicitly. Mirrors the
``smai-core/tests/conftest.py`` pattern.
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
