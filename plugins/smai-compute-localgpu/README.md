# smai-compute-localgpu

Local-Docker reference implementation of the `Compute` plugin. The default
substrate for `smai dev`. Substrate is **Docker**; compatible OCI runtimes
(Podman, Apple `container`, containerd via `nerdctl`) that alias the `docker`
binary on `$PATH` work without code changes.

## Install

This plugin ships in the `smai` workspace. From a development checkout:

```bash
uv sync
```

For published wheels (post-M4):

```bash
pip install smai-compute-localgpu
```

## Build the reference images

SMAI does not publish prebuilt images. Build the reference Dockerfiles yourself before first use.

### Agent image (CPU, ~500 MB)

Used when `LocalGpuCompute.submit(..., gpu=False)` for agent-side code execution (harness builder, technique implementer, code reviewer, contextual evaluator).

```bash
docker build -t smai-agent:dev \
    -f plugins/smai-compute-localgpu/dockerfiles/agent.Dockerfile .
```

### Runtime image (GPU, ~5–8 GB)

Used when `LocalGpuCompute.submit(..., gpu=True)` for GPU experiment seed runs.

```bash
docker build -t smai-runtime:dev \
    -f plugins/smai-compute-localgpu/dockerfiles/runtime.Dockerfile .
```

If `LocalGpuCompute.submit` is called with one of these tags and the image isn't built, the plugin raises `JobImageInvalid` with the exact `docker build` command embedded in the error.

## GPU prerequisites (Linux / WSL2 only)

GPU passthrough requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) on a Linux host (or WSL2 with GPU support enabled in Docker Desktop). On `gpu=True` job dispatch the plugin runs `nvidia-smi` as a preflight check; a missing toolkit raises `ComputeUnavailable`.

## Mac caveat

Docker Desktop on macOS / Apple Silicon **cannot pass GPU through**. On Darwin:

* The agent image (`gpu=False`) works fine.
* `submit(..., gpu=True)` raises `ComputeUnavailable` immediately with a pointer to `smai-compute-modal` / `smai-compute-runpod` for GPU experiment runs.
* The plugin's `capabilities.supports_gpu` is reported as `False` on Darwin.

This is a Docker / macOS limitation, not a SMAI policy choice. The plugin fails fast rather than letting a `gpu=True` job silently fall back to CPU mode and produce wrong-looking metrics.

## Usage

```python
from smai_compute_localgpu import LocalGpuCompute

compute = LocalGpuCompute()  # uses smai-agent:dev / smai-runtime:dev defaults

handle = await compute.submit(
    image="smai-agent:dev",
    command=["python", "agent_loop.py"],
    env={"SMAI_TASK": "harness_build"},
    timeout_seconds=3600,
    workspace="/path/to/workspace",  # bind-mounted at /workspace
)

while True:
    status = await compute.status(handle)
    if status.state in {"succeeded", "failed", "cancelled", "timeout"}:
        break
    await asyncio.sleep(1)

print(await compute.logs(handle))
```

Tier A integrators (the in-tree `smai` CLI / hosted backend) instantiate the plugin via the `smai.computes` entry-point group; Tier B integrators import `LocalGpuCompute` directly.

## Cleanup

The plugin does NOT pass `--rm` to `docker run`, so containers persist after exit and `status` / `logs` keep working. Stopped containers carry the `smai-localgpu=1` label so you can prune them with:

```bash
docker container prune --filter label=smai-localgpu=1
```

## Conformance

The 8-method `ComputeConformance` contract suite from `smai-core` runs against `LocalGpuCompute` whenever a Docker daemon is reachable; on hosts without Docker (CI, dev machines without Docker installed) the suite skips cleanly.

Run with:

```bash
pytest plugins/smai-compute-localgpu/tests/test_conformance.py
```

## Linting the Dockerfiles

The two reference Dockerfiles are checked with [hadolint](https://github.com/hadolint/hadolint) in CI. Run locally with:

```bash
hadolint plugins/smai-compute-localgpu/dockerfiles/*.Dockerfile
```

## Out of scope

* Multi-host GPU pools (single-host only).
* Cost accounting.
* Spot vs on-demand selection.
* Trust boundaries / IAM (the local plugin runs as the invoking user; no credential surface).
