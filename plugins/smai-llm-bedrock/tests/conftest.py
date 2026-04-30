"""Pytest configuration for smai-llm-bedrock tests.

Adds this tests/ directory to ``sys.path`` so per-test modules can
import the shared :mod:`_bedrock_fakes` helper by name. Pytest's
``importlib`` mode (set on the workspace root) does not implicitly
extend ``sys.path``, so this conftest does so explicitly. Mirrors the
``smai-core/tests/conftest.py`` pattern. Per Task R4 fix #4: the
helper is task-prefixed (``_bedrock_fakes`` rather than the generic
``_fakes``) so ``sys.path``-augmenting conftests across sibling
plugins do not collide on a shared module name.
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
