# smai-artifacts-s3

`ArtifactStore` plugin: BYO-bucket S3 implementation. Per
`07-plugin-interfaces.md` §6 and `implementation_plan.md` §3.4 Task
3.F2.

## Use

```python
from smai_artifacts_s3 import S3Store

store = S3Store(
    bucket="my-existing-bucket",
    region="us-east-1",
    prefix="prod/",                       # optional key namespacing
    presigned_url_expiry_seconds=900,     # 15-minute default
)

await store.put("foo/bar.bin", payload, content_type="application/octet-stream")
data = await store.get("foo/bar.bin")
url = await store.url_for("foo/bar.bin")  # presigned, 900-second expiry
```

The bucket must already exist; the plugin does not auto-create or
auto-discover buckets. Credentials come from the boto3 default chain
(`AWS_PROFILE`, `AWS_REGION`, IAM role, env vars).

## Tests

The plugin ships three test modes:

1. **`mock_aws` conformance** (`tests/test_conformance.py`) — runs the
   universal `ArtifactStoreConformance` suite against an in-process
   moto fake. No network, no credentials. This is the always-on lane.
2. **Threaded-moto presigned-URL tests** (`tests/test_presigned_url.py`)
   — spins up `moto.server.ThreadedMotoServer` (a real local HTTP
   endpoint) so the presigned URL can be fetched end-to-end. No
   credentials.
3. **Real-AWS round-trip** (`tests/test_real_aws.py`) — opt-in
   production-readiness check. Skipped unless `AWS_TEST_BUCKET` is in
   the environment. Run locally with credentials before declaring an
   S3-affecting change ready:

```sh
AWS_TEST_BUCKET=my-test-bucket AWS_TEST_REGION=us-east-1 \
    uv run pytest plugins/smai-artifacts-s3/tests/test_real_aws.py -v
```

The real-AWS lane uses a per-run UUID key prefix so concurrent runs
against a shared bucket do not interfere; cleanup is in the test's
`finally` block.
