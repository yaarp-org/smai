"""Pytest configuration for smai-compute-runpod tests.

Adds this tests/ directory to ``sys.path`` so per-test modules can
import the shared :mod:`_fakes` helper by name. Pytest's ``importlib``
import mode (set on the workspace root) does not implicitly extend
``sys.path``, so this conftest does so explicitly. Mirrors the
``smai-llm-bedrock/tests/conftest.py`` pattern.

Defines the ``fake_runpod_backend`` fixture used by the conformance
suite and the unit tests. The default test mode runs against an
in-process :class:`FakeRunPodBackend` that simulates RunPod's REST API
at the ``httpx`` transport layer — no network round-trip, no
credentials, deterministic. The opt-in real-RunPod lane is in
:mod:`test_real_runpod` and skips by default.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _runpod_fakes import FakeRunPodBackend  # noqa: E402
from smai_compute_runpod import RunPodCompute  # noqa: E402


@pytest.fixture
def fake_runpod_backend() -> Iterator[FakeRunPodBackend]:
    """Module-level fake backend; one per test."""
    yield FakeRunPodBackend()


@pytest.fixture
def runpod_compute(fake_runpod_backend: FakeRunPodBackend) -> RunPodCompute:
    """A :class:`RunPodCompute` wired to the fake backend."""
    transport = fake_runpod_backend.transport()
    client = httpx.AsyncClient(transport=transport, base_url="https://rest.runpod.io")
    return RunPodCompute(
        api_key="rpa_fake_test_key",
        api_base="https://rest.runpod.io/v1",
        client=client,
    )
