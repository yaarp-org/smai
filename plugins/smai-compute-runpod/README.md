# smai-compute-runpod

`Compute` plugin: RunPod REST-API implementation. Pairs with
`smai-compute-localgpu` (single-host Docker reference) and
`smai-compute-modal` (Modal Sandboxes) as the third `Compute` plugin.
RunPod is the GPU-cloud option for operators who want a pay-per-pod
surface without running their own GPU host.

## Use

```python
import os
from smai_compute_runpod import RunPodCompute

os.environ["RUNPOD_API_KEY"] = "rpa_..."  # or set in your shell / systemd unit

compute = RunPodCompute(
    default_gpu_type="NVIDIA H100 80GB HBM3",   # optional override
    default_timeout_seconds=3600,
)

handle = await compute.submit(
    image="docker.io/your-org/your-image:latest",
    command=["python", "train.py", "--epochs=10"],
    env={"WANDB_API_KEY": "..."},
    gpu=True,
    timeout_seconds=7200,
    # plugin_options (kwargs):
    gpu_type="NVIDIA H100 80GB HBM3",   # per-call override
    gpu_count=2,                          # default 1
    container_disk_gb=20,                 # default 10
    cloud_type="SECURE",                  # or "COMMUNITY"
)

# Poll status, fetch logs, cancel ...
status = await compute.status(handle)
logs = await compute.logs(handle)
await compute.cancel(handle)
```

The API key comes from `RUNPOD_API_KEY`; pass `api_key=...` for tests.
Per-pod credentials never enter shell history.

## GPU dispatch table

The plugin ships a small dispatch table that maps SMAI's generic GPU
specs to RunPod GPU type ids. Operators who want a RunPod-specific
GPU id pass `plugin_options["gpu_type"]` directly.

| SMAI tier   | RunPod GPU type id           | Notes                              |
| ----------- | ---------------------------- | ---------------------------------- |
| `default`   | `NVIDIA RTX A4000`           | Cheapest small-batch tier.         |
| `small`     | `NVIDIA RTX A4000`           | Same as `default`.                 |
| `medium`    | `NVIDIA RTX A5000`           | Mid-tier, common availability.     |
| `large`     | `NVIDIA RTX A6000`           | 48 GB VRAM, professional tier.     |
| `a100`      | `NVIDIA A100 80GB PCIe`      | Datacenter tier; capacity-sensitive. |
| `h100`      | `NVIDIA H100 80GB HBM3`      | Most expensive tier.               |

The id strings are the canonical RunPod identifiers. You can pass any
RunPod-published id directly via `plugin_options["gpu_type"]` without
changing the dispatch table. The dispatch table itself is a SMAI-side
convenience; RunPod's catalog is large and grows over time.

## Choosing Pods vs Serverless

RunPod exposes two execution surfaces: **Pods** (long-lived containers
that boot from an image and accept arbitrary commands) and
**Serverless endpoints** (pre-deployed worker pools that accept job
payloads).

This plugin targets **Pods**. The Compute Protocol's contract is
"submit an image + command, get a handle back"; Pods map onto that
natively. Serverless requires the operator to build + deploy a
worker per workload shape, which moves substantial work from the
Compute layer back to the operator. SMAI's "operator chooses an image
and a command per dispatch" is materially closer to Pods semantics.

If your workload is high-volume (>>100 dispatches/day per shape) and
boot latency matters more than image flexibility, pre-deploy a
RunPod Serverless endpoint and write a separate, narrower
`Compute` plugin against `https://api.runpod.io/v2/{endpoint_id}/...`.
The Pods plugin and the (hypothetical) Serverless plugin can coexist
under different entry-point names without conflict.

## State mapping

The plugin translates RunPod's pod-status strings to SMAI's
[`JobState`](../../packages/smai-core/src/smai_core/plugins/compute.py)
literal:

| RunPod status                 | SMAI state             | Notes                                         |
| ----------------------------- | ---------------------- | --------------------------------------------- |
| `IN_QUEUE`, `INITIALIZING`, `STARTING`, `PENDING` | `submitted` | Pre-running phase.                            |
| `IN_PROGRESS`, `RUNNING`      | `running`              | Actively executing.                           |
| `COMPLETED` / `EXITED` (exit_code=0) | `succeeded`     | Pod ran to clean completion.                  |
| `COMPLETED` / `EXITED` (exit_code≠0) | `failed`        | Non-zero exit.                                |
| `COMPLETED` / `EXITED` (after `cancel()`) | `cancelled` | Plugin marks intent on the JobHandle metadata. |
| `TERMINATED`, `STOPPED`, `CANCELLED` (after `cancel()`) | `cancelled` | User cancellation propagates here.            |
| `TERMINATED`, `STOPPED`, `CANCELLED` (no cancel) | `failed` | Substrate-killed without user intent.         |
| `TIMED_OUT`                   | `timeout`              | Substrate enforced its own timeout.           |
| `FAILED`                      | `failed`               | Hard substrate failure.                       |

Caller-side timeout enforcement: the plugin also tracks elapsed
wall-clock since `submitted_at` in the JobHandle metadata; once
elapsed > `timeout_seconds` and the pod is non-terminal, the plugin
issues a DELETE and returns `state='timeout'`. Same shape as
`smai-compute-localgpu`'s `timeout_seconds` enforcement.

## Image validation

RunPod has no eager image-validation surface (no analog to
`docker pull`). The plugin therefore defers image validation to the
substrate: bad image references surface as `state='failed'` once the
pod tries to pull. The conformance suite's `test_invalid_image_raises`
test takes its "plugin defers image validation" branch and skips cleanly.

A future enhancement could add an opt-in eager check via Docker Registry's
`HEAD /v2/<name>/manifests/<tag>` from `submit()`; this is currently
deferred.

## Tests

The plugin ships three test modes:

1. **Mocked-HTTP conformance** (`tests/test_conformance.py`): runs the
   universal `ComputeConformance` suite against an in-process
   `_FakeRunPodBackend` wired in via `httpx.MockTransport`. No network,
   no credentials. This is the always-on lane.
2. **Plugin-internal unit tests** (`tests/test_unit.py`): covers
   translation tables and quirks the conformance suite does not exercise
   (status mapping edge cases, GPU dispatch, shell quoter, API-key
   env-var fallback).
3. **Real-RunPod round-trip** (`tests/test_real_runpod.py`): opt-in
   production-readiness check. Marked `@pytest.mark.credentialed` and
   `skipif`-on-env. Skipped unless `RUNPOD_API_KEY` is set. Run
   locally with credentials before declaring a RunPod-affecting
   change ready:

```sh
RUNPOD_API_KEY=rpa_... \
    uv run pytest plugins/smai-compute-runpod/tests/test_real_runpod.py -v
```

The real-RunPod lane uses `python:3.12-slim` and the cheapest GPU
tier; per-run cost should be a fraction of a cent. Override via
`RUNPOD_TEST_GPU_TYPE` / `RUNPOD_TEST_IMAGE` for ops with different
account access.
