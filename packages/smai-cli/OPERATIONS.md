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

## 5. Log handling

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

## 6. Cross-references

- `designs/smai/09-cli.md` §6 — `smai start` verb shape + required
  config + failure modes.
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
