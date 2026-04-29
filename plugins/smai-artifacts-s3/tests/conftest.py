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

# A region that requires SigV4 — picking ``us-west-2`` (rather than
# ``us-east-1``, the legacy default) keeps the moto-mocked tests aligned
# with how real-world buckets are provisioned today.
TEST_REGION: str = "us-west-2"
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
        client.create_bucket(  # pyright: ignore[reportUnknownMemberType]
            Bucket=TEST_BUCKET,
            CreateBucketConfiguration={"LocationConstraint": TEST_REGION},
        )
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
