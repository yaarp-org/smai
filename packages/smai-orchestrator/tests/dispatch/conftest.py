"""Pytest path setup for ``smai-orchestrator/tests/dispatch/`` (agent-refactor Step 2).

Mirrors ``tests/engine/conftest.py``: pytest's ``importlib`` import-mode
does not add the test directory to ``sys.path``, so sibling helper
modules (``_compute_dispatcher_fakes.py``) are not importable by their
bare name without this nudge.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
# ``tests/engine/_helpers.py`` carries ``FakeArtifactStore`` which the
# dispatch unit tests reuse. Stage that directory on sys.path so it's
# importable here too.
sys.path.insert(0, str(Path(__file__).parent.parent / "engine"))
