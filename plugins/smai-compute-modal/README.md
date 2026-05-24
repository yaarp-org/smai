# smai-compute-modal

`Compute` plugin: Modal Sandboxes implementation. Production GPU-cloud
substrate using the Modal Sandbox-per-job pattern (vs Modal Functions),
which allows agent-generated experiment code materialized at runtime to
be submitted without pre-deployment.

## What it does

Each `submit()` call creates a fresh Modal Sandbox running the
specified `image` + `command` + `env`; `status()` polls it; `logs()`
reads the Sandbox's stdout / stderr; `cancel()` terminates it. The
The Sandbox-per-job pattern was chosen because agent-generated
`experiment.py` is materialized at runtime rather than pre-deployed, so
Modal Functions (which require a pre-deployed worker) do not fit.

`ModalCompute.capabilities`:

| Field | Value | Why |
|---|---|---|
| `supports_gpu` | `True` | Modal's GPU inventory covers T4 / L4 / A100 / H100. |
| `max_timeout_seconds` | `86400` (24h) | Modal's hard Sandbox cap. Surfaced so the engine's lease loop respects it. |
| `supports_log_streaming` | `False` | Logs are returned after termination; no live `tail -f`. |

## Configuration

```python
from smai_compute_modal import ModalCompute

compute = ModalCompute()
# or with non-default Modal app namespace + GPU type:
compute = ModalCompute(
    app_name="my-deployment",
    default_gpu_type="A100",
)
```

### Authentication

The plugin reads Modal credentials from the SDK's default chain:

| Env var | Required? | Source |
|---|---|---|
| `MODAL_TOKEN_ID` | yes (or `~/.modal.toml`) | `modal token new` |
| `MODAL_TOKEN_SECRET` | yes (or `~/.modal.toml`) | `modal token new` |

The constructor does **not** accept token args; credentials never enter
shell history.

### Image publication

`submit(image=..., ...)` expects a substrate-resolvable Docker tag. The
operator publishes the image to a registry Modal can pull from (Docker Hub,
GHCR, public ECR, etc.) **before first use**. The plugin does not
auto-build or auto-push images; image distribution is the operator's
responsibility.

A typical pipeline:

```bash
docker build -t ghcr.io/your-org/smai-runtime:dev \
  -f plugins/smai-compute-localgpu/dockerfiles/runtime.Dockerfile .
docker push ghcr.io/your-org/smai-runtime:dev

# then in your engine config:
# Compute = ModalCompute(...)
# image="ghcr.io/your-org/smai-runtime:dev"
```

## `plugin_options`

`submit()` accepts plugin-specific options via `**plugin_options`:

| Key | Type | Default | Effect |
|---|---|---|---|
| `gpu_type` | `str` | `"T4"` (constructor `default_gpu_type`) | Modal GPU spec: `"T4"` / `"L4"` / `"A100"` / `"A100-80GB"` / `"H100"` etc. Only consulted when `gpu=True`. |
| `cpu` | `float \| None` | `None` | Modal `cpu` request. |
| `memory_mb` | `int \| None` | `None` | Modal `memory` request, in MiB. |

## Job state mapping

| Modal poll result | Plugin state | Notes |
|---|---|---|
| `None` (Sandbox running) | `running` | The plugin enforces a defensive timeout: if `elapsed > timeout_seconds`, calls `terminate()` and returns `timeout`. |
| Exit code `0` | `succeeded` | |
| Non-zero exit + `cancel_requested` flag in handle metadata | `cancelled` | Set by `cancel()`. |
| Non-zero exit + elapsed ≥ `timeout_seconds` | `timeout` | Modal's substrate-side timeout fired. |
| Non-zero exit (other) | `failed` | |

`cancel_requested` lives in `JobHandle.metadata` because Modal's exit
code alone is ambiguous: terminate, timeout, and a genuine non-zero
crash all surface as non-zero return codes.

## Tests

The plugin's always-on test suite runs against an in-process Modal SDK
fake (`tests/_fakes.py`) that spawns real OS subprocesses for each
"Sandbox," so the conformance fixtures actually execute the way they
would on a real Sandbox. No Modal credentials required:

```bash
uv run pytest plugins/smai-compute-modal/
```

### Credentialed lane (local-manual only)

`tests/test_real_modal.py` exercises the real Modal substrate. Per Task
3.G3's no-credentials-in-CI convention, this file:

* is marked `@pytest.mark.credentialed`,
* skips cleanly if `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` are absent,
* is **never run on CI** (local-manual only).

To run before merging a change:

```bash
export MODAL_TOKEN_ID=...
export MODAL_TOKEN_SECRET=...
uv run pytest plugins/smai-compute-modal/tests/test_real_modal.py -v -m credentialed
```

## Out of scope (v1)

* **Multi-app routing** (single `app_name` per `ModalCompute` instance).
* **Auto image build / publication** (operator responsibility).
* **Modal Function deployment** (vs Sandbox-per-job). Sandboxes are used
  because agent-generated code is materialized at runtime, not pre-deployed.
* **Modal Volumes / dataset caching**. The `**plugin_options` surface stays
  open for a future `volumes` plugin option.
* **Cost accounting**. Modal's billing surface is not yet exposed through the
  `JobStatus` shape.
* **OIDC AWS federation**.
