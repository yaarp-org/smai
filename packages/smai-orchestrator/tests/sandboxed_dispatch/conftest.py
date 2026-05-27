"""Pytest configuration for the sandboxed-dispatcher tests.

Step 8 Wave 2 relocated these tests from ``smai-agents/tests/`` to
:mod:`smai_orchestrator.tests.sandboxed_dispatch`. The shared
:mod:`_harness_builder_sandboxed_fixtures` module needs to be on
``sys.path`` for the sibling test modules to import it by name under
``--import-mode=importlib``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
