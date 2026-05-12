# SMAI v2 — operations notes for `smai start`

This file ships brief operational guidance for running `smai start` in
production. It is intentionally not a full deployment guide — that's
3.I1 (M4 gate) territory. The goal here is to give an operator the
minimum viable deployment recipes for the three common Linux/macOS
service supervisors plus the recommended connection-pool sizing and
log-handling patterns.

Per `09-cli.md` §6 / DEC-024: `smai start` is the production-mode
worker entrypoint. It expects to run as a long-lived service against
a Postgres `MetadataStore` (or single-VM SQLite for `worker_count=1`
deployments), an `ArtifactStore` (LocalFs for single-host or S3 BYO
for distributed), a `Compute` substrate (Modal Sandboxes / RunPod /
LocalGpu), and a `LlmProvider` (Bedrock / Anthropic / OpenAI).

## 1. systemd unit example

```ini
# /etc/systemd/system/smai-worker.service
[Unit]
Description=SMAI v2 production worker
After=network-online.target postgresql.service
Wants=network-online.target

[Service]
Type=simple
User=smai
Group=smai
WorkingDirectory=/var/lib/smai
Environment=SMAI_CONFIG=/etc/smai/smai.yaml
Environment=SMAI_WORKER_ID=%H-%i
# AWS / Bedrock credentials via the standard chain (instance profile,
# ~/.aws/credentials, etc.). Anthropic / OpenAI plugins read from
# their own env vars (ANTHROPIC_API_KEY, OPENAI_API_KEY).
ExecStartPre=/usr/local/bin/smai migrate --check -c /etc/smai/smai.yaml
ExecStart=/usr/local/bin/smai start -c /etc/smai/smai.yaml
Restart=on-failure
RestartSec=5s
KillMode=mixed
KillSignal=SIGTERM
TimeoutStopSec=30s
StandardOutput=journal
StandardError=journal
SyslogIdentifier=smai-worker

[Install]
WantedBy=multi-user.target
```

Key choices:

- `KillSignal=SIGTERM`: `smai start` traps SIGTERM via
  `loop.add_signal_handler` and drains gracefully.
- `TimeoutStopSec=30s`: gives the in-flight worker cycle time to
  finish and release any held leases via `MetadataStore.release_lease`
  before systemd escalates to SIGKILL. The default lease lifetime is
  120s (`EngineConfig.lease_seconds`); tune `TimeoutStopSec` if the
  deployment uses a non-default value.
- `ExecStartPre=smai migrate --check`: belt-and-suspenders against a
  stale schema. `smai start` runs the same check internally via
  `_check_schema_at_head`; the systemd-level check fails fast before
  the worker process even forks.
- `Restart=on-failure` + `RestartSec=5s`: crash-loop with a 5s
  backoff. Combine with systemd's `StartLimitIntervalSec=60s` /
  `StartLimitBurst=3` if the deployment wants to escalate to a
  sysadmin alert after persistent failures.
- `SMAI_WORKER_ID=%H-%i`: systemd substitutes `%H` (hostname) +
  `%i` (instance id from `smai-worker@N.service` template syntax)
  for stable lease-holder identity per DEC-035 #2.

## 2. supervisor.d config example

```ini
; /etc/supervisor/conf.d/smai-worker.conf
[program:smai-worker]
command=/usr/local/bin/smai start -c /etc/smai/smai.yaml
directory=/var/lib/smai
user=smai
autostart=true
autorestart=true
stopsignal=TERM
stopwaitsecs=30
environment=SMAI_CONFIG="/etc/smai/smai.yaml",SMAI_WORKER_ID="%(host_node_name)s-%(process_num)s"
stdout_logfile=/var/log/smai/worker.log
stderr_logfile=/var/log/smai/worker.err.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=10
```

`stopsignal=TERM` + `stopwaitsecs=30` mirror the systemd recipe. The
log files are rotated by supervisor itself; for journald-aware
deployments, prefer the systemd unit above.

## 3. launchd plist example (macOS)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!-- /Library/LaunchDaemons/com.smai.worker.plist -->
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
                       "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>com.smai.worker</string>
    <key>ProgramArguments</key>
    <array>
      <string>/usr/local/bin/smai</string>
      <string>start</string>
      <string>-c</string>
      <string>/usr/local/etc/smai/smai.yaml</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
      <key>SMAI_CONFIG</key>
      <string>/usr/local/etc/smai/smai.yaml</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>UserName</key>
    <string>smai</string>
    <key>StandardOutPath</key>
    <string>/usr/local/var/log/smai/worker.log</string>
    <key>StandardErrorPath</key>
    <string>/usr/local/var/log/smai/worker.err.log</string>
    <key>ExitTimeOut</key>
    <integer>30</integer>
  </dict>
</plist>
```

`launchd` sends SIGTERM by default; `ExitTimeOut=30` mirrors the
systemd / supervisor recipes.

## 4. Connection-pool sizing

The Postgres `MetadataStore` plugin (`smai-store-postgres`) and the
SQLite plugin (`smai-store-sqlite`) both hold a single
`sqlalchemy.ext.asyncio.AsyncEngine`. SQLAlchemy's defaults for
`AsyncEngine`:

- `pool_size=5` (default)
- `max_overflow=10` (default)
- `pool_timeout=30s` (default)
- `pool_pre_ping=False` (default — the engine does NOT issue a
  health-check before each checkout)

For a single-worker `smai start` deployment these defaults are
sufficient — the worker loop issues serial scheduling queries +
per-entity transitions. For multi-worker deployments
(`EngineConfig.worker_count > 1`):

- Bump `pool_size` to roughly `worker_count + 5` to leave headroom
  for the lease-extension heartbeat task that runs alongside the
  main dispatch task.
- Set `pool_pre_ping=True` if the deployment is long-lived and
  Postgres has aggressive idle-connection timeouts (RDS
  `idle_in_transaction_session_timeout`, PgBouncer `idle_timeout`).
  The pre-ping cost is one round-trip per checkout; cheap relative
  to the dispatch work but not free.
- Tune `pool_timeout` based on the worker's poll cadence:
  `EngineConfig.poll_interval_seconds * 2` is a defensible
  upper bound — if a checkout takes longer than that, the pool is
  saturated and the worker won't make forward progress regardless.

These knobs aren't currently exposed through `RuntimeConfig`; they're
plugin-internal (per `07-plugin-interfaces.md` §5.7 — hand-rolled-vs-
SQLAlchemy resolution). To override, fork the plugin or open a PR
adding a config field.

### 4.1 LISTEN connection budget for `smai ui --no-worker`

Each `smai ui --no-worker` process that targets a Postgres
`MetadataStore` holds **one dedicated asyncpg connection** for the
`LISTEN smai_events` task (per Task 4.K3 / `12-ui-process.md` §6.3).
This connection sits outside the SQLAlchemy pool because `LISTEN`
requires a connection to be parked on the channel, which the pool
cannot satisfy.

When sizing Postgres `max_connections`, budget:

- `worker_count` connections for each `smai start` worker's
  SQLAlchemy pool (per §4 above), plus headroom.
- `+1` connection per `smai ui --no-worker` process for its dedicated
  `LISTEN` connection, plus that process's own SQLAlchemy pool for
  serving `/api/v1/...` reads.

Multiple `smai ui` processes can run against the same Postgres; each
holds its own `LISTEN` connection and Postgres fans out the `NOTIFY`
payload to all of them, so the budget scales linearly with the number
of API processes.

## 5. Log handling

> **No checkpoint persistence.** A killed or restarted worker re-runs the
> in-flight agent loop from turn 0. The orchestrator ships only an
> in-memory checkpointer backend, and no checkpointer is instantiated by
> `smai dev` or `smai start`; there is no `checkpoints` table. The
> recovery primitive is orphan-grace reset, which re-dispatches the whole
> step rather than resuming it. Size `orphan_grace_seconds` (and
> `lease_seconds`) with that in mind: a worker that dies mid-dispatch
> burns whatever LLM / GPU spend the step had accrued, and the re-dispatch
> starts over. A SQL-backed checkpointer is post-M5 backlog.

`smai start` uses Python's standard `logging` module — no structured
loggers (per `09` §8 — no log-format commitments in v1). Default log
level is `WARNING`; for production, surface INFO via the
`SMAI_LOG_LEVEL` env var or by passing
`logging.config.dictConfig(...)` from a wrapper script.

The four plugins each emit logs through their own `logger = logging.getLogger(__name__)`; capture with a single root handler:

```python
import logging.config

logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "structured": {
            "format": "%(asctime)s %(levelname)s %(name)s: %(message)s",
        },
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "structured",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["stdout"],
    },
})
```

For systemd / supervisor, prefer stdout — the supervisor captures
stdout into journald / log files. For containerized deployments
(ECS / K8s), stdout is the standard log seam.

Per `09` §8: secret values matching `*_api_key` / `*_password` /
`*_secret` field names are masked at log-emit time by the CLI's
secret-masking convention. A future hardening would scrub
`logging.LogRecord.msg` at the root handler level; for now the
plugins are responsible for not constructing log messages containing
their own secrets.

## 6. `smai ui --no-worker` companion deployment

Per `designs/smai/12-ui-process.md` §10.3, the recommended self-hosted
production shape (DEC-027) is:

- N `smai start` worker processes against shared Postgres + S3 (this
  file's §1–§3 cover the unit shapes; multi-worker by spawning N
  `smai start` services with distinct `--worker-id` values).
- One or more `smai ui --no-worker` processes for the API + SPA host,
  behind a TLS-terminating reverse proxy (nginx / caddy).

Notes specific to the API+SPA process:

- **Install extra.** Production hosts that serve the SPA install
  `pip install smai-api[ui]`; the `[ui]` extra pulls in `smai-ui`,
  whose wheel ships the built React bundle as package data. Headless
  deployments (API only, no SPA at `/`) install bare `smai-api` and
  the SPA mount degrades cleanly. Per Task 4.N1 +
  `12-ui-process.md` §8.2.
- **Pre-flights.** `smai ui --no-worker` runs soft pre-flights:
  it logs warnings on missing plugin slots but boots anyway, so
  existing data stays readable through the API even if a slot is
  misconfigured. Workers run their strict pre-flights independently.
- **`LISTEN` connection.** Each `smai ui` process holds its own
  asyncpg `LISTEN` connection separate from the SQLAlchemy pool. See
  §4.1 above for the connection-budget calculus.
- **Bind address.** `api.host` defaults to loopback (`127.0.0.1`).
  Production deployments put a reverse proxy on the public interface
  and let the API stay bound to loopback or the proxy's internal
  network. The `Host:` header allowlist (`11-api.md` §7.1) is enforced
  unconditionally; broaden the allowlist when the proxy fronts the
  API on a public hostname.

A minimal systemd unit for the API process:

```ini
[Unit]
Description=SMAI v2 UI host (API + SPA)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=smai
Group=smai
Environment=SMAI_CONFIG=/etc/smai/smai.yaml
Environment=SMAI_LOG_LEVEL=INFO
ExecStart=/usr/local/bin/smai ui --no-worker
Restart=on-failure
RestartSec=5
KillSignal=SIGTERM
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

Same restart / log-handling guidance as `smai start` applies. SSE
clients reconnect automatically on restart per `11-api.md` §8.3.

### 6.1 Optional bearer-token auth

The default API posture is no token: loopback bind plus `Host:` header
allowlist (`11-api.md` §7.1). For deployments that want layered auth
(particularly Case B remote-data, where the local API process holds
credentials for remote backends), bearer mode is opt-in via
`smai.yaml`:

```yaml
api:
  auth:
    enabled: true                              # default false
    token_path: ~/.smai/api-token              # 0600, auto-generated
```

Mechanics:

- **Token file.** On first `smai ui` launch with `auth.enabled: true`,
  the process generates a fresh token via `secrets.token_urlsafe(32)`
  and writes it to `token_path` mode `0o600`. Existing tokens are
  preserved across restarts so browser tabs keep working.
- **Multiple processes.** Each `smai ui` process generates its own
  token. If you run more than one, give each a distinct `token_path`
  and surface the right one to each operator group.
- **Programmatic clients.** CI scripts and `curl` invocations read the
  token file directly and set `Authorization: Bearer <token>`,
  symmetrical to how `gh` reads `~/.config/gh/hosts.yml`.
- **SPA bootstrap.** The SPA's `index.html` is served with the token
  interpolated as `<script>window.__SMAI_TOKEN__ = "...";</script>`;
  the SPA client reads the variable once at boot and includes it on
  every request. Per `13-frontend.md` §12.4, the token never lives in
  `localStorage` or in URLs.
- **Rotation.** A `smai auth rotate` verb is a post-M5 backlog item.
  Until it lands, rotation is a manual `rm` on the token file plus
  `smai ui` restart; browser tabs need a refresh after rotation.
- **SSE caveat.** SSE in bearer mode currently uses the native
  `EventSource`, which cannot set custom headers. The header-aware
  polyfill (`fetchEventSource`) is a post-M5 backlog item per
  `13-frontend.md` §7.3; until that lands, live-update events do not
  fire when bearer mode is enabled. REST reads + mutations work
  normally; the SPA falls back to TanStack Query's default refetch
  cadence.

### 6.2 Verifying a fresh deployment

The canonical end-to-end smoke is `tests/integration/test_smai_ui_e2e.py`
(per Task 4.N3): submit a proposal via `POST /api/v1/proposals`, drive
through the planner + approval + CG-execution states via SSE, and
fetch the resulting `evaluation_result.json` artifact via
`GET /api/v1/comparison-groups/{id}/artifacts/...`. The journey
completes in ~12s wall-clock on a stub-LLM + stub-Compute config.

To run against a deployment's actual config:

```bash
SMAI_CONFIG=/etc/smai/smai.yaml uv run pytest \
    tests/integration/test_smai_ui_e2e.py -v
```

The Case-A path (in-process worker, sqlite + localfs) is the canonical
exercise; the Case-B credentialed variant
(`test_full_user_journey_remote_data`) is currently a structural
placeholder per Task 4.N3's status note (body `pytest.skip`s pending
Postgres + S3 wiring; tracked in §8 backlog).

---

## 7. Plugin substrate notes

Per-plugin credential and environment requirements are tabulated in
`README.md` "Configuration" (the per-plugin `*_config` key table). A
few that bite in practice:

- **`localgpu` images + Linux bind-mount UID.** Build all three
  reference images (`smai-agent:dev`, `smai-runtime:dev`,
  `smai-runtime-cpu:dev`) from
  `plugins/smai-compute-localgpu/dockerfiles/` before deploying with
  `compute: localgpu` (SMAI publishes none). The run-record dispatcher
  selects `smai-runtime:dev` for GPU experiment runs and
  `smai-runtime-cpu:dev` for `controlled_conditions.compute.gpu: false`
  ones; override the names via `engine.runtime_image` /
  `engine.runtime_cpu_image`. All three images run as the non-root
  `smai` user (uid 1000), and `LocalGpuCompute` bind-mounts a per-CG
  workspace dir at `/workspace`; on Linux that dir must be writable by
  uid 1000 (`chmod -R a+rwX` the `workspace_root`, or run `smai` as uid
  1000), or an agent's writes into `/workspace` fail with a permission
  error. (macOS / Docker Desktop's virtiofs mount is writable
  regardless of host uid.)
- **Modal compute needs `python` (not just `python3`) on `PATH`.** The
  Modal SDK shells out to a bare `python` for some operations.
  `uv run smai start ...` puts one there, and so does a virtualenv with
  the usual `python` symlink, but a system that only has `python3` will
  fail with `python: command not found` from inside the SDK. If your
  service unit runs `smai` from a bare interpreter, either invoke it via
  `uv run` / an activated venv, or symlink `python` -> `python3` on the
  service account's `PATH`.
- **Bedrock needs model access granted, not just AWS credentials.** A
  working credential chain still gets `AccessDeniedException: <model> is
  not available for this account` until the model is enabled in the
  Bedrock console for that region. `smai verify` surfaces this with a
  hint pointing at `aws bedrock list-inference-profiles --region <r>`.

---

## 8. Cross-references

- `designs/smai/09-cli.md` §6 — `smai start` verb shape + required
  config + failure modes.
- `designs/smai/12-ui-process.md` §10.3 — `smai ui --no-worker`
  deployment recipe; capacity-planning notes for SSE keepalive +
  LISTEN connections.
- `designs/smai/05-orchestrator.md` §7.2 — self-hosted production
  deployment shape (Postgres / S3 / Modal canonical).
- `designs/smai/05-orchestrator.md` §3.5 — multi-worker leasing
  contract that the lease-holder-id surfaces to.
- DEC-024 — in-house orchestrator carries to production.
- DEC-027 — this is the OSS-side production shape; the hosted
  backend wraps it later.
- DEC-028 — engine config distinct from plugin selection.
- DEC-035 — multi-worker leasing primitive + lease-holder identity
  (the `--worker-id` flag's reason for being).

Production deployment docs (Kubernetes / ECS / Nomad manifests, CI /
CD pipelines, load-balancer config) ship with 3.I1; this file is the
minimum viable operational shape for the OSS deployment surface.
