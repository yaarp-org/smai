"""Pytest configuration for the still-in-smai-agents sandboxed-dispatcher tests.

Step 8 Wave 1 moved the inline-agents test fixtures + tests to
:mod:`smai-inline-agents.tests`. The few sandboxed-dispatcher tests
that remain here keep their own conftest so :mod:`_harness_builder_sandboxed_fixtures`
is importable by sibling tests under ``--import-mode=importlib``.
Wave 2 dissolves this directory entirely.
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
