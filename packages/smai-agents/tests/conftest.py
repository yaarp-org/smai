"""Pytest configuration for smai-agents tests.

Adds this tests/ directory to ``sys.path`` so per-test modules can
import the shared :mod:`_agent_fakes` and :mod:`_helpers` modules by
name. Mirrors the pattern in ``smai-llm-bedrock/tests/conftest.py``.

The fakes module is named ``_agent_fakes`` (rather than ``_fakes``) to
avoid colliding with the bedrock plugin's ``tests/_fakes.py`` when the
two test suites run in the same pytest session — Python's module
cache resolves whichever directory comes first on ``sys.path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
