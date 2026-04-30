"""Pytest configuration for smai-llm-openai tests.

Adds this tests/ directory to ``sys.path`` so per-test modules can
import the shared :mod:`_fakes` helper by name. Mirrors the
``smai-llm-bedrock/tests/conftest.py`` pattern.
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
