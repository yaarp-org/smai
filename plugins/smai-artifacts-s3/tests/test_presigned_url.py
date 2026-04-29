"""Presigned-URL semantics for :class:`S3Store`.

Beyond the universal :class:`ArtifactStoreConformance` suite (which
asserts only that ``url_for`` returns a string), this module verifies
the S3-specific shape of the presigned URL — SigV4
``X-Amz-Expires=<seconds>`` query parameter, the URL is fetchable over
HTTP, and the constructor's per-store default vs. the call-site
override interact as documented.

Network access pattern: the tests run against
:class:`moto.server.ThreadedMotoServer` (a real local HTTP server
backed by moto's S3 implementation), so the presigned URL is fetchable
end-to-end without leaving localhost. This is the only S3 test path
that needs a live HTTP endpoint — the conformance suite uses moto's
``mock_aws`` interception mode instead.
"""

from __future__ import annotations

import urllib.request
from collections.abc import Iterator
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

import boto3  # pyright: ignore[reportMissingTypeStubs]
import pytest
from botocore.config import Config  # pyright: ignore[reportMissingTypeStubs]
from moto.server import ThreadedMotoServer  # pyright: ignore[reportMissingTypeStubs]
from smai_artifacts_s3 import S3Store

PRESIGNED_REGION: str = "us-east-1"
PRESIGNED_BUCKET: str = "smai-presigned-test"


@pytest.fixture(scope="module")
def moto_server() -> Iterator[str]:
    """Spin up a local ``ThreadedMotoServer`` once per module.

    Module-scope amortizes the ~few-hundred-ms server start cost across
    every parametrized run in the file. Returns the endpoint URL the
    boto3 client should target.
    """
    server = ThreadedMotoServer(port=0)
    server.start()
    # ``ThreadedMotoServer`` does not expose the bound port via a public
    # API, so we reach into the underlying ``http.server`` instance.
    # ``server_address`` is a 2-tuple for AF_INET; ``cast`` narrows it
    # for pyright since the stub admits both AF_INET and AF_INET6 (the
    # 4-tuple shape is not a runtime concern here).
    underlying = cast("Any", server)._server
    raw_address = cast("tuple[str, int]", underlying.server_address)
    host, port = raw_address
    endpoint = f"http://{host}:{port}"
    try:
        yield endpoint
    finally:
        server.stop()


def _build_test_client(endpoint: str) -> Any:
    """Construct a boto3 ``s3`` client pointed at the moto server.

    SigV4 is forced so the presigned URL has the modern
    ``X-Amz-Expires`` query-parameter shape (rather than the legacy
    ``Expires=<unix-timestamp>`` that SigV2 emits).

    Returns ``Any`` to keep the call sites that reach into
    ``client.exceptions`` / ``client.create_bucket`` legible — boto3's
    clients are dynamically generated and not statically typed.
    """
    return cast("Any", boto3).client(
        "s3",
        endpoint_url=endpoint,
        region_name=PRESIGNED_REGION,
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
        config=Config(signature_version="s3v4"),
    )


@pytest.fixture
def store(moto_server: str) -> S3Store:
    """Build a fresh :class:`S3Store` against the module's moto server.

    Each test gets its own client (separate connection pool) but shares
    the underlying server backend — the bucket created here is reused
    across tests in the module.
    """
    client = _build_test_client(moto_server)
    # Idempotent ``create_bucket``: moto returns BucketAlreadyOwnedByYou
    # the second time around, which we swallow. ``us-east-1`` does not
    # take ``CreateBucketConfiguration`` per AWS's own quirk.
    try:
        client.create_bucket(Bucket=PRESIGNED_BUCKET)
    except client.exceptions.BucketAlreadyOwnedByYou:
        pass
    return S3Store(
        bucket=PRESIGNED_BUCKET,
        region=PRESIGNED_REGION,
        presigned_url_expiry_seconds=900,
        client=client,
    )


def _expires_query(url: str) -> int:
    """Pull ``X-Amz-Expires=<n>`` out of the presigned URL's query string."""
    qs = parse_qs(urlparse(url).query)
    expires = qs.get("X-Amz-Expires")
    assert expires, f"presigned URL missing X-Amz-Expires: {url!r}"
    return int(expires[0])


async def test_url_for_includes_x_amz_expires(store: S3Store) -> None:
    """The constructor's ``presigned_url_expiry_seconds`` shows up in the
    URL's ``X-Amz-Expires`` query parameter."""
    key = "presigned/default-expiry"
    await store.put(key, b"x")
    try:
        url = await store.url_for(key)
        assert _expires_query(url) == 900  # constructor default
    finally:
        await store.delete(key)


async def test_url_for_explicit_expires_in_overrides_default(store: S3Store) -> None:
    """``expires_in`` passed at the call site overrides the constructor's
    per-store default."""
    key = "presigned/override-expiry"
    await store.put(key, b"x")
    try:
        url = await store.url_for(key, expires_in=120)
        assert _expires_query(url) == 120
    finally:
        await store.delete(key)


async def test_url_for_fetches_object_bytes(store: S3Store) -> None:
    """Fetching the presigned URL over HTTP returns the artifact bytes.

    Exercises the full SigV4 round-trip against moto's HTTP server —
    if the URL signature is malformed, this fails with HTTP 403.
    """
    key = "presigned/fetchable"
    payload = b"fetched via presigned url"
    await store.put(key, payload)
    try:
        url = await store.url_for(key)
        # noqa: S310 - localhost moto server, test-only.
        with urllib.request.urlopen(url) as response:  # noqa: S310
            assert response.read() == payload
    finally:
        await store.delete(key)


async def test_url_for_method_put_distinct_from_get(store: S3Store) -> None:
    """``method="PUT"`` produces a distinct URL shape from the default GET.

    Both URLs target the same key but encode different signed
    operations; a GET-signed URL refused over PUT (and vice versa) is
    AWS's actual security boundary. We assert only on the URL contents
    here — moto's server enforces the boundary in
    :func:`test_url_for_fetches_object_bytes`.
    """
    key = "presigned/method-shape"
    await store.put(key, b"x")
    try:
        get_url = await store.url_for(key, method="GET")
        put_url = await store.url_for(key, method="PUT")
        # Different ``X-Amz-Signature`` (SigV4 binds to the verb).
        get_sig = parse_qs(urlparse(get_url).query)["X-Amz-Signature"][0]
        put_sig = parse_qs(urlparse(put_url).query)["X-Amz-Signature"][0]
        assert get_sig != put_sig
    finally:
        await store.delete(key)
