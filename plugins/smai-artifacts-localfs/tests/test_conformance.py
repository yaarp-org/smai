"""Conformance suite for :class:`LocalFsStore`.

Subclasses :class:`smai_core.plugins.conformance.ArtifactStoreConformance`
per Task 2.A3 — runs the universal contract suite against the local-fs
plugin instance.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from smai_artifacts_localfs import LocalFsStore
from smai_core.plugins.conformance import ArtifactStoreConformance


class TestLocalFsStoreConformance(ArtifactStoreConformance):
    @pytest.fixture
    def store(self, tmp_path: Path) -> LocalFsStore:
        return LocalFsStore(root=tmp_path / "artifacts")
