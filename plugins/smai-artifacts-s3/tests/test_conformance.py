"""Conformance suite for :class:`S3Store`.

Subclasses :class:`smai_core.plugins.conformance.ArtifactStoreConformance`
per Task 3.F2 — runs the universal contract suite against the S3
plugin instance, with ``moto`` standing in for real S3 so the suite
runs offline in CI and deterministically locally.
"""

from __future__ import annotations

import pytest
from smai_artifacts_s3 import S3Store
from smai_core.plugins.conformance import ArtifactStoreConformance


class TestS3StoreConformance(ArtifactStoreConformance):
    @pytest.fixture
    def store(self, s3_store: S3Store) -> S3Store:
        return s3_store
