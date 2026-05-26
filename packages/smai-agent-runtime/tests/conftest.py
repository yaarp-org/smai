"""Test-suite shim: ensure `smai_agent_runtime` resolves under
``--import-mode=importlib`` whether or not the workspace has been
``uv sync``-installed. Mirrors the conftest pattern in sibling packages
that need their own ``src/`` on ``sys.path`` for in-tree test runs.
"""

from __future__ import annotations
