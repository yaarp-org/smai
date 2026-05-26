"""Test-suite shim: ensure `smai_agent_runtime` resolves under
``--import-mode=importlib`` whether or not the workspace has been
``uv sync``-installed, and put this ``tests/`` directory on
``sys.path`` so per-task fixture modules (``_<task>_<purpose>.py``)
can be imported by name. Mirrors the conftest pattern in sibling
packages.
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
