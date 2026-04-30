"""Opt-in production-readiness check against real AWS S3.

Skips by default. Runs only when ``AWS_TEST_BUCKET`` is set in the
environment (and standard AWS credentials are picked up via boto3's
default chain — env vars, profile, IAM role).

Round-trips every Protocol method against a real bucket so a developer
with credentials can verify before reporting Task 3.F2 complete that
the plugin works end-to-end against the substrate it is meant to
target. Not part of the always-green CI lane (would leak credentials,
cost real money on the bucket, and depend on out-of-tree state) — see
the plugin's ``README.md`` for the recommended workflow.
"""

from __future__ import annotations

import os
import urllib.request
import uuid
from collections.abc import AsyncIterator

import pytest
from smai_artifacts_s3 import S3Store

AWS_TEST_BUCKET_ENV: str = "AWS_TEST_BUCKET"
AWS_TEST_REGION_ENV: str = "AWS_TEST_REGION"

pytestmark = [
    pytest.mark.credentialed,
    pytest.mark.skipif(
        AWS_TEST_BUCKET_ENV not in os.environ,
        reason=(
            f"Set ${AWS_TEST_BUCKET_ENV} (and AWS credentials via the boto3 "
            "default chain) to run real-AWS validation. See "
            "plugins/smai-artifacts-s3/README.md for setup."
        ),
    ),
]


@pytest.fixture
def real_store() -> S3Store:
    """Build an :class:`S3Store` against the real bucket named in the env."""
    bucket = os.environ[AWS_TEST_BUCKET_ENV]
    region = os.environ.get(AWS_TEST_REGION_ENV)
    # A unique per-run key prefix avoids interference if multiple
    # contributors point at the same shared bucket.
    prefix = f"smai-real-aws-test/{uuid.uuid4()}/"
    return S3Store(bucket=bucket, region=region, prefix=prefix)


async def test_real_aws_round_trip(real_store: S3Store) -> None:
    """Exercise put / exists / get / list / url_for / delete against
    real AWS S3."""
    key = "round-trip"
    payload = b"hello from real-AWS test"
    await real_store.put(key, payload, content_type="text/plain")
    try:
        assert await real_store.exists(key) is True
        assert await real_store.get(key) == payload

        # ``list`` walks the per-run prefix; the key we wrote should be
        # the only one.
        iterator: AsyncIterator[str] = await real_store.list("")
        seen = [k async for k in iterator]
        assert key in seen

        url = await real_store.url_for(key, expires_in=60)
        # Real AWS endpoint, real signature — fetches the bytes back.
        with urllib.request.urlopen(url) as response:  # noqa: S310 - signed URL
            assert response.read() == payload
    finally:
        await real_store.delete(key)
        # Idempotent — second delete does not raise.
        await real_store.delete(key)
