"""Shared fixtures for the S3 plugin's test modules.

The default test mode runs against ``moto`` (an in-process S3 fake) —
no AWS credentials, no network round-trip, deterministic. The opt-in
real-AWS lane is in :mod:`test_real_aws` and skips by default.
"""

from __future__ import annotations

from collections.abc import Iterator

import boto3  # pyright: ignore[reportMissingTypeStubs]
import pytest
from botocore.config import Config  # pyright: ignore[reportMissingTypeStubs]
from moto import mock_aws  # pyright: ignore[reportMissingTypeStubs]
from smai_artifacts_s3 import S3Store

# Project canonical region per CLAUDE.md. SigV4 is forced via
# ``signature_version="s3v4"`` in the S3Store client builder regardless
# of region, so us-east-1 carries no signing-version risk.
TEST_REGION: str = "us-east-1"
TEST_BUCKET: str = "smai-conformance-bucket"


@pytest.fixture
def s3_store() -> Iterator[S3Store]:
    """Yield an :class:`S3Store` bound to a freshly-created moto bucket.

    The ``mock_aws`` context wraps the entire test, so any boto3 call
    made through the store (or directly) is intercepted. ``moto`` patches
    at the ``botocore`` endpoint layer, which works across the
    :func:`asyncio.to_thread` boundary the store uses internally — the
    patches are process-global, not thread-local.
    """
    with mock_aws():
        client = boto3.client(  # pyright: ignore[reportUnknownMemberType]
            "s3",
            region_name=TEST_REGION,
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
            config=Config(signature_version="s3v4"),
        )
        # ``us-east-1`` is S3's default region and rejects
        # ``CreateBucketConfiguration={"LocationConstraint": ...}`` — the
        # bucket is implicitly created in us-east-1 when no constraint
        # is passed. ``test_presigned_url.py`` documents the same quirk.
        client.create_bucket(Bucket=TEST_BUCKET)  # pyright: ignore[reportUnknownMemberType]
        yield S3Store(
            bucket=TEST_BUCKET,
            region=TEST_REGION,
            client=client,
            # A small ceiling lets the conformance suite's
            # ``test_max_object_size_honored`` allocate just past the
            # limit without the 5 GiB malloc that the production default
            # would force. The fixture keys (1 byte / 1 KiB / 1 MiB)
            # all fit comfortably.
            max_object_size_bytes=2 * 1024 * 1024,  # 2 MiB
        )
