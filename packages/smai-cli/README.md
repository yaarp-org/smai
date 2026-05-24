# smai-cli

The `smai` command-line tool: 16 verbs, config layering, `RuntimeConfig`,
and the in-band `Runtime`. For a project overview see the
[main repo README](../../README.md).

---

## CLI verbs

| Verb | Purpose |
|---|---|
| `smai dev` | Boot the laptop deployment (SQLite + LocalFs + LocalGpu + Bedrock; in-band worker; `poll_interval=10s`; headless). `--reset` wipes `state.db`, `artifacts/`, and `workspaces/` under `$SMAI_HOME` before booting (recovery hatch for a record wedged in a non-terminal state); `--yes`/`-y` skips the confirmation prompt. |
| `smai start` | Boot the production worker (out-of-band; explicit plugin selections required; refuses incomplete config or stale schema). `--worker-id` pins the lease identity (default: `hostname-pid-uuid8`). |
| `smai ui` | Boot the API + SPA process. Auto-detects `--with-worker` / `--no-worker` from plugin shape (sqlite+localfs: on; anything else: off). `--host`, `--port`, `--worker-id`, `--reload` (uvicorn dev auto-reload, do not use in production). |
| `smai run <yaml>` | Compile + register a CG; optional `--watch` polls until terminal. `--techniques FILE` (repeatable) upserts hand-authored `TechniqueRef`s into the store before compiling. |
| `smai submit-proposal <desc>` | Primary input verb (DEC-032). Submit a novel-technique description inline, via `--description-file FILE`, or pass `-` to read from stdin. `--reproduce-paper <arxiv-id>` submits a reproduce-paper proposal instead. |
| `smai approve-proposal <id>` | Human gate at `designed`. Atomically registers 1-N CGs in `draft`. |
| `smai reject-proposal <id>` | Reject a proposal at `designed`. Optional `--reason` free text. |
| `smai ingest <arxiv-id>` | Supporting input verb. Fetch, parse, screen, plan, and register paper-derived `TechniqueRef`s. Does not produce CGs; follow with `smai submit-proposal --reproduce-paper`. `--promote-partial` transitions an existing `partial` paper to `submitted`. |
| `smai status <id>` | Read pipeline-tracking state for a CG, proposal, or paper id (probed in that order). Surfaces state, attempt counters, `last_error`, elapsed, and a live agent-status snippet. `--watch` polls until terminal (CG ids only). `--poll-interval SECS` (default 5.0). `--format json` for machine-readable output. |
| `smai compile <yaml>` | Methodology only: emit the four contract artifacts to `--out DIR` or stdout as JSON. Never touches `MetadataStore` or `Compute`. Supply `--techniques FILE` for any technique the experiment references. |
| `smai migrate` | Apply Alembic schema migrations against the configured `MetadataStore`. `--check` exits 0/1 on head/stale. `--dry-run` renders SQL without executing. `--prune` runs the retention sweep. `--upgrade-to tenant_aware` applies the opt-in tenant-id schema branch. |
| `smai verify` | Plugin-ping pre-flight: structured PASS/FAIL per plugin (LLM 1-token completion, store count query, artifact HEAD, compute no-op status). Always-on container-image config check included. `--probe-image` submits a real no-op job per runtime image (costs money; opt-in). `--format json`. |
| `smai init` | Scaffold `smai.yaml` and a sample `experiment.yaml` in a target directory. `--force` overwrites existing files. |
| `smai plugins` | List discovered plugins per entry-point namespace and the currently selected one. `--format json`. |
| `smai version` | Print versions of `smai-cli`, `smai-core`, `smai-orchestrator`, `smai-agents`, `smai-runtime`, and the four default reference plugins. `--format json`. |
| `smai serve` | **Deprecated in v2.** One-line stderr warning on every invocation; behavior unchanged. Use `smai ui --no-worker` instead. Source-tree removal scheduled for v2.1. |

---

## Configuration

### File search order

`smai` looks for a config file in this order; the first match wins and
absent paths are silently skipped:

1. `--config PATH` flag
2. `$SMAI_CONFIG` env var
3. `./smai.yaml` in the current working directory
4. `$XDG_CONFIG_HOME/smai/config.yaml` (falls back to `~/.config/smai/config.yaml` when `XDG_CONFIG_HOME` is unset)

### Layering precedence

Sources are merged field-granularly (not document-level); later sources
win per field:

```
in-code defaults  <  smai.yaml  <  SMAI_* env vars  <  CLI flags
```

`smai dev` defaults: SQLite + LocalFs + LocalGpu + Bedrock, `poll_interval=10s`.
`smai start` defaults: nothing (all four plugin selections are required).

### `SMAI_*` env-var convention

Every `RuntimeConfig` field is settable via `SMAI_<FIELD>`. Use double
underscores to separate nesting levels:

```bash
SMAI_ENGINE__POLL_INTERVAL_SECONDS=30
SMAI_PLUGINS__METADATA_STORE_CONFIG__URI="postgresql+asyncpg://user:pw@host/smai"
SMAI_ENGINE__WORKER_COUNT=2
```

Three vars are consumed outside the layering pipeline and are not
forwarded to `RuntimeConfig`:

| Var | Effect |
|---|---|
| `SMAI_CONFIG` | Path to a `smai.yaml`; alternative to `--config`. |
| `SMAI_HOME` | Base directory for `smai dev`'s managed paths (default `~/.smai`). |
| `SMAI_ARTIFACTS_ROOT` | Fallback root for the `localfs` artifact store when no `root` key is set in `artifact_store_config`. |

### Annotated `smai.yaml` skeleton

```yaml
engine:
  poll_interval_seconds: 30        # 10 under smai dev defaults
  worker_count: 1                  # >1 turns on leasing; requires a lease-capable store
  fair_scheduling: "off"           # "off" | "rr" | "weighted"
  # runtime_image / runtime_cpu_image: container images for experiment seed runs.
  # Defaults: smai-runtime:dev / smai-runtime-cpu:dev (build locally; not published by SMAI).
  # The dispatcher picks per the CG's compute.gpu flag.
  # runtime_image: ghcr.io/your-org/smai-runtime:prod
  # runtime_cpu_image: ghcr.io/your-org/smai-runtime-cpu:prod
  role_models: {}                  # per-role agent-model overrides; see below

plugins:
  llm_provider:   bedrock          # bedrock | anthropic | openai
  metadata_store: sqlite           # sqlite | postgres
  artifact_store: localfs          # localfs | s3
  compute:        localgpu         # localgpu | modal | runpod

  # Each *_config block is splatted as **kwargs into the plugin constructor.
  # A wrong key fails at boot with "got an unexpected keyword argument".
  llm_provider_config:
    region: us-east-1
    model_id: us.anthropic.claude-opus-4-6-v1
  metadata_store_config: {}        # see "Where state lives" below
  artifact_store_config: {}        # localfs default: $SMAI_ARTIFACTS_ROOT or ~/.smai/artifacts
  compute_config: {}               # localgpu default: smai-runtime:dev / smai-agent:dev images

# smai ui only. The entire block is optional.
api:
  host: 127.0.0.1
  port: 8000
  with_worker: auto                # auto: on for sqlite+localfs, off otherwise
  auth: { enabled: false }         # opt-in bearer-token mode
```

### Per-plugin `*_config` keys

The value of each `*_config` block is splatted as `**kwargs` into the
plugin constructor. Keys are exactly the constructor parameters:

| Plugin | `*_config` keys (defaults in parens) | Credentials |
|---|---|---|
| `bedrock` | `region` (`us-east-1`), `model_id` | AWS default credential chain + Bedrock model access granted in the console for that model/region |
| `anthropic` | `model_id` (`claude-opus-4-7`) | `ANTHROPIC_API_KEY` |
| `openai` | `model_id` | `OPENAI_API_KEY` (required at construction; even `smai verify` fails without it) |
| `sqlite` | `uri` (`sqlite+aiosqlite:///:memory:`); a SQLAlchemy URL; absolute path needs four slashes: `sqlite+aiosqlite:////home/me/.smai/state.db` | none |
| `postgres` | `uri`, `use_advisory_locks` (`True`), `tenant_aware` (`False`), `fair_scheduling` (`off`), `fair_scheduling_weights`, `engine_kwargs` (e.g. `{pool_size: 10}`) | URL embeds credentials |
| `localfs` | `root` (`$SMAI_ARTIFACTS_ROOT` or `~/.smai/artifacts`) | none |
| `s3` | `bucket` (required), `region`, `prefix` (`""`), `presigned_url_expiry_seconds`, `max_object_size_bytes` | AWS default credential chain; bucket must exist |
| `localgpu` | `agent_image` (`smai-agent:dev`), `runtime_image` (`smai-runtime:dev`), `runtime_cpu_image` (`smai-runtime-cpu:dev`), `workspace` | Docker running locally; build all three reference images yourself (SMAI does not publish them). On Linux the workspace bind-mount must be writable by uid 1000. |
| `modal` | `app_name` (`smai`), `default_gpu_type` (`T4`), `max_timeout_seconds` (`86400`) | `~/.modal.toml` or `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` |
| `runpod` | `api_base`, `default_gpu_type` (`NVIDIA RTX A4000`), `default_timeout_seconds` (`3600`), `max_timeout_seconds`, `default_container_disk_gb` (`10`) | `RUNPOD_API_KEY` |

Per-submit knobs (a job's `gpu_type`, `cpu`, `memory_mb` on Modal; `workspace`
on localgpu) are job options the engine passes per dispatch, not constructor
keys, so they do not go in `compute_config`.

### Per-role agent models

`plugins.llm_provider_config.model_id` is the provider's base model, not
what the agent fleet runs. Each role has its own model (Opus tier for
`planner`, `harness_builder`, `technique_implementer`, `code_reviewer`;
Sonnet tier for `contextual_evaluator`, `supervisor`, `screener`,
`enricher`). Override per role under `engine.role_models`:

```yaml
engine:
  role_models:
    planner: us.anthropic.claude-opus-4-6-v1
    code_reviewer: us.anthropic.claude-sonnet-4-6
```

Precedence (highest first):

1. `SMAI_MODEL_<ROLE>` env var (e.g. `SMAI_MODEL_PLANNER=bedrock:us.anthropic.claude-opus-4-6-v1`). The `provider:model_id` form is the only way to route a single role to a different provider than the base.
2. The equivalent nested form: `SMAI_ENGINE__ROLE_MODELS__PLANNER`.
3. `engine.role_models` in `smai.yaml`.
4. Built-in per-role defaults.

Cross-provider routing for a role via `engine.role_models` (yaml) is not
supported; use the env var for that.

### Logging

`smai dev` and `smai start` default to `WARNING`-level logging. Flags and
env var:

| Flag | Effect |
|---|---|
| `--verbose` / `-v` | Raise worker and engine to `INFO` (per-dispatch, per-transition, per-cycle lines) |
| `-vv` | `DEBUG` |
| `SMAI_LOG_LEVEL=INFO` | Honored when no `-v` flag is present; `-v` wins over the env var |

---

## Where state lives

`smai dev` and `smai ui` provision `~/.smai/` for you and inject the
paths into the config, so a bare `metadata_store_config: {}` works under
those verbs:

```
$SMAI_HOME/           (default: ~/.smai/)
  state.db            SQLite metadata store
  artifacts/          LocalFs artifact store root
  workspaces/         per-CG agent workspaces
```

`smai migrate`, `smai verify`, and `smai start` use the `smai.yaml` value
verbatim. An empty `metadata_store_config: {}` resolves to the sqlite
plugin's in-memory default, which makes `smai migrate` a silent no-op
(it reports success, then the database evaporates on exit). For those
verbs, set an explicit URI:

```yaml
plugins:
  metadata_store_config:
    uri: "sqlite+aiosqlite:////absolute/path/state.db"   # four slashes for absolute path
    # or for production:
    # uri: "postgresql+asyncpg://user:pw@host:5432/smai"
```

The sqlite plugin expands a leading `~`, so
`sqlite+aiosqlite:///~/.smai/state.db` also works.

---

## The `techniques.json` format

Both `smai compile --techniques` and `smai run --techniques` accept a
JSON file containing `TechniqueRef` objects. The file may be a JSON list
or an object keyed by id:

```json
[
  {
    "id": "tech_cutout",
    "name": "Cutout",
    "description": "Cutout regularization: random square patches masked out of training images.",
    "category": "augmentation",
    "compatible_factor_types": ["additive"],
    "affects_extension_points": ["train_transforms"],
    "standard": true
  }
]
```

Required fields: `id`, `name`, `description`, `category`,
`compatible_factor_types` (`additive` / `substitutive`),
`affects_extension_points`. Optional: `standard` (default `false`; when
`false`, a `fidelity_anchor` is required), `fidelity_anchor`,
`implies_controlled`, `parameter_schema`. Unknown fields are rejected.

The `--techniques` flag is repeatable; multiple files merge. A technique
id appearing in more than one file is an error. For `smai run`, each
`TechniqueRef` is upserted into the `MetadataStore` before compilation,
so verification passes; the upsert is idempotent.

---

## Production deployment

For systemd / supervisord / launchd unit examples, connection-pool
sizing, structured-logging patterns, bearer-token mode mechanics, and the
container-image build-and-push runbook, see
[`OPERATIONS.md`](OPERATIONS.md) in this directory.
