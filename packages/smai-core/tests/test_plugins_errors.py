"""Error-hierarchy tests for plugin Protocol error classes.

Per Task 1.8 — for each Protocol's error hierarchy, assert subclass
relationships and that the base is catchable. Plugin authors and
consumers should be able to ``except <Base>`` and catch the whole
family.
"""

from __future__ import annotations

import pytest
from smai_core.plugins import (
    ArtifactNotFound,
    ArtifactStoreError,
    ArtifactTooLarge,
    ComputeError,
    ComputeUnavailable,
    ConflictError,
    JobHandle,
    JobImageInvalid,
    JobNotFound,
    LeaseLostError,
    LlmProviderAuthError,
    LlmProviderError,
    LlmProviderInvalidRequest,
    LlmProviderRateLimited,
    LlmProviderUnavailable,
    MetadataStoreError,
    PresignedUrlsUnsupported,
)

# ---------- LlmProvider hierarchy --------------------------------------------


def test_llm_provider_subclass_relationships() -> None:
    assert issubclass(LlmProviderRateLimited, LlmProviderError)
    assert issubclass(LlmProviderUnavailable, LlmProviderError)
    assert issubclass(LlmProviderInvalidRequest, LlmProviderError)
    assert issubclass(LlmProviderAuthError, LlmProviderError)
    assert issubclass(LlmProviderError, Exception)


def test_llm_provider_base_catches_all_subclasses() -> None:
    for cls in (
        LlmProviderRateLimited,
        LlmProviderUnavailable,
        LlmProviderInvalidRequest,
        LlmProviderAuthError,
    ):
        with pytest.raises(LlmProviderError):
            raise cls("boom")


# ---------- MetadataStore hierarchy ------------------------------------------


def test_metadata_store_subclass_relationships() -> None:
    assert issubclass(ConflictError, MetadataStoreError)
    assert issubclass(LeaseLostError, MetadataStoreError)
    assert issubclass(MetadataStoreError, Exception)


def test_metadata_store_base_catches_all_subclasses() -> None:
    with pytest.raises(MetadataStoreError):
        raise ConflictError("cg", "id-1", expected_version=2, actual_version=3)
    with pytest.raises(MetadataStoreError):
        raise LeaseLostError("cg", "id-1")


def test_conflict_error_carries_versions() -> None:
    err = ConflictError("entry", "e-99", expected_version=4, actual_version=5)
    assert err.entity_type == "entry"
    assert err.entity_id == "e-99"
    assert err.expected_version == 4
    assert err.actual_version == 5
    # message includes both versions for operator diagnostics
    assert "4" in str(err)
    assert "5" in str(err)


def test_lease_lost_error_carries_kind_and_id() -> None:
    err = LeaseLostError("run", "run-77")
    assert err.entity_kind == "run"
    assert err.entity_id == "run-77"


# ---------- ArtifactStore hierarchy ------------------------------------------


def test_artifact_store_subclass_relationships() -> None:
    assert issubclass(ArtifactNotFound, ArtifactStoreError)
    assert issubclass(PresignedUrlsUnsupported, ArtifactStoreError)
    assert issubclass(ArtifactTooLarge, ArtifactStoreError)
    assert issubclass(ArtifactStoreError, Exception)


def test_artifact_not_found_carries_key() -> None:
    err = ArtifactNotFound("comparison-groups/cg-1/missing.json")
    assert err.key == "comparison-groups/cg-1/missing.json"
    assert "missing" in str(err)


def test_artifact_too_large_carries_size_and_limit() -> None:
    err = ArtifactTooLarge(key="x", size=10_000_000_000, limit=5 * 1024**3)
    assert err.key == "x"
    assert err.size == 10_000_000_000
    assert err.limit == 5 * 1024**3


def test_artifact_store_base_catches_all_subclasses() -> None:
    with pytest.raises(ArtifactStoreError):
        raise ArtifactNotFound("k")
    with pytest.raises(ArtifactStoreError):
        raise PresignedUrlsUnsupported()
    with pytest.raises(ArtifactStoreError):
        raise ArtifactTooLarge("k", 1, 0)


# ---------- Compute hierarchy ------------------------------------------------


def test_compute_subclass_relationships() -> None:
    assert issubclass(JobNotFound, ComputeError)
    assert issubclass(JobImageInvalid, ComputeError)
    assert issubclass(ComputeUnavailable, ComputeError)
    assert issubclass(ComputeError, Exception)


def test_compute_base_catches_all_subclasses() -> None:
    handle = JobHandle(plugin="modal", handle="abc")
    with pytest.raises(ComputeError):
        raise JobNotFound(handle)
    with pytest.raises(ComputeError):
        raise JobImageInvalid("ecr/x:tag", "manifest not found")
    with pytest.raises(ComputeError):
        raise ComputeUnavailable("substrate down")


def test_job_not_found_carries_handle() -> None:
    handle = JobHandle(plugin="batch", handle="job-arn")
    err = JobNotFound(handle)
    assert err.handle is handle


def test_job_image_invalid_carries_image_and_reason() -> None:
    err = JobImageInvalid("ecr/foo:tag", "manifest unknown")
    assert err.image == "ecr/foo:tag"
    assert err.reason == "manifest unknown"
    assert "ecr/foo:tag" in str(err)
