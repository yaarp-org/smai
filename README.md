# SMAI

> Methodology-as-infrastructure for **comparative-claim ML research**.
> Definition in, verdict out — with the metric, the factor structure, and
> the verdict path locked at compile time.

SMAI is the experiment-execution stage that an outer auto-research pipeline
plugs into when the question is **comparative** ("does technique X beat
technique Y under matched conditions?"). It is *not* an idea generator,
not a writeup tool, and not a hill-climbing optimizer — peer territory
to those, with a different shape.

The unit of value is a small set of guarantees, not a feature list:

- **Compile-time verification** — confounded experiments fail to compile.
  Factor exclusivity, control completeness, metric well-formedness, and
  comparability of entries are mechanically checked.
- **Runtime determinism** — the path from raw multi-seed metrics to a
  boolean verdict has no LLM in it. The metric is fixed at compile time;
  the aggregation rule is fixed; the agent that wrote the implementation
  never sees the verdict criterion.
- **Contracts as the locked surface** — the DSL compiler emits four
  immutable JSON artifacts (`ExperimentPlan`, `HarnessContract`,
  `TechniqueContract[]`, `ValidationConfig`). Agents read; nothing
  rewrites them mid-flight.

This repo is the **OSS surface**: methodology layer, agent loops,
orchestrator, runtime, CLI, and the four reference plugin interfaces
with local + production implementations.

> **Status: private pre-release.** APIs may break without notice. The
> repo is private; PyPI distribution and a public docs site are gated
> on an explicit user decision (per `designs/smai/implementation_plan.md`
> §1.4). Until that gate is flipped, install from the local workspace
> via `uv sync`. Apache 2.0 licensed.

---

## Two integration patterns

SMAI has a two-layer stack with a deliberately asymmetric shape, and
two canonical integration patterns (per
`designs/smai/00-vision.md` §1.1):

| Pattern | What it gives you | What you import |
|---|---|---|
| **Tier A — deep delegation** | Hand SMAI a definition (or a description and let the planner draft one). Read back a verdict + artifact set. Full pipeline: agents, orchestrator, runtime, plugins. | `smai_cli.runtime.Runtime` (programmatic), or the `smai` CLI. |
| **Tier B — methodology as library** | Compile your own definitions, run experiments on your own infrastructure, call the evaluator for the verdict. The pipeline layer isn't involved. | `smai_core` only — `compile_experiment`, `evaluate`, the four contract types, the four plugin Protocols. |

The methodology layer is **atomic**: pulling out one piece breaks the
others' guarantees, so `smai-core` ships as a single coherent package
with the whole compile-then-evaluate pipeline. The pipeline layer is
**composable**: each piece is independently consumable. A "mixed"
pattern — using some pipeline components but not others — is supported
by the modularity but is *not* a documented happy path; soundness
guarantees still hold (they're enforced at the package boundary), but
operational guarantees from the pipeline's between-turn coordination
may degrade outside the canonical patterns.

---

## Repo layout

```
smai/
├── packages/
│   ├── smai-core/              # methodology layer (DEC-029):
│   │                           # data model, DSL + compiler, contract artifacts,
│   │                           # mechanical evaluator, four plugin Protocols.
│   │                           # Allowlisted deps: pydantic, jsonschema, stdlib.
│   ├── smai-runtime/           # harness/technique runtime + fixed templates.
│   ├── smai-agents/            # custom Bedrock-Converse-style agent loop, six
│   │                           # fleet roles (planner, harness builder, technique
│   │                           # implementer, code reviewer, contextual evaluator,
│   │                           # supervisor).
│   ├── smai-orchestrator/      # engine + pipeline-spec format + worker loop +
│   │                           # checkpointer; the four SMAI PipelineSpec instances
│   │                           # (CG execution, proposal, paper ingestion, RunRecord
│   │                           # sub-spec); Alembic migrations.
│   ├── smai-cli/               # CLI verbs, config layering, RuntimeConfig,
│   │                           # plugin instantiation, in-band Runtime.
│   └── smai/                   # umbrella package (per DEC-026 / 09 §9; eventual
│                               # `from smai import Runtime` re-export surface).
├── plugins/
│   ├── smai-llm-bedrock/       # LlmProvider — AWS Bedrock Converse (reference)
│   ├── smai-llm-anthropic/     # LlmProvider — Anthropic SDK
│   ├── smai-llm-openai/        # LlmProvider — OpenAI SDK
│   ├── smai-store-sqlite/      # MetadataStore — SQLite reference (smai dev default)
│   ├── smai-store-postgres/    # MetadataStore — Postgres production
│   ├── smai-artifacts-localfs/ # ArtifactStore — local filesystem reference
│   ├── smai-artifacts-s3/      # ArtifactStore — S3 (BYO bucket)
│   ├── smai-compute-localgpu/  # Compute — local Docker on host GPU
│   ├── smai-compute-modal/     # Compute — Modal Sandboxes
│   └── smai-compute-runpod/    # Compute — RunPod Pods API
├── tests/                      # cross-package integration + smoke tests
├── tools/check_deps.py         # methodology-atomicity dependency lint
├── pyproject.toml              # uv workspace root
└── LICENSE                     # Apache 2.0
```

---

## Quick start

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
git clone <repo-url> smai
cd smai
uv sync                         # install workspace + dev deps
uv run pytest                   # run the test suite (~1700 tests)
```

Boot the laptop demo:

```bash
# Configure AWS credentials in the default chain — the dev defaults use
# the smai-llm-bedrock plugin (claude-opus-4-7 in us-east-1). To use
# Anthropic or OpenAI directly, see "Configuration" below.
$ uv run smai dev
smai dev: worker running. workspace_root=/home/you/.smai/workspace
poll_interval=10s. Press Ctrl+C to stop.
```

`smai dev` boots the in-band degenerate-production deployment:
SQLite (`~/.smai/state.db`) + LocalFs (`~/.smai/artifacts`) + LocalGpu
(Docker) + Bedrock, single in-process worker, `poll_interval=10s` for
interactive feel. Ctrl+C drains gracefully.

In a second terminal, submit an experiment:

```bash
$ uv run smai run tests/fixtures/experiments/cutout_on_cifar10.yaml --watch
cg_01J7PA8K2X9F4ZB6QH4N0P1234
cg_01J7PA8K2X9F4ZB6QH4N0P1234: complete
```

`smai status <cg-id> --watch` does the same polling without re-submitting.

---

## Tier A — programmatic full pipeline

The CLI is a thin wrapper around the in-process service surface
(per `designs/smai/09-cli.md` §1.2). Programmatic Tier-A consumers
import `Runtime` directly:

```python
import asyncio

from smai_cli import Runtime, dev_defaults, load_runtime_config


async def main() -> None:
    runtime_config = load_runtime_config(defaults=dev_defaults())
    yaml_text = open("tests/fixtures/experiments/cutout_on_cifar10.yaml").read()

    async with Runtime.start_in_band(runtime_config) as runtime:
        # Submit — returns one or more CG IDs (a multi-factor experiment
        # may register N CGs; a single-factor returns one).
        cg_ids = await runtime.experiments.submit_text(yaml_text)
        for cg_id in cg_ids:
            snap = await runtime.status.wait_for_terminal(cg_id, timeout=None)
            print(f"{cg_id}: {snap.state}")


asyncio.run(main())
```

`Runtime.start_in_band(...)` is async-context-managed: plugins are
opened on enter and drained / closed on exit — no detached `Runtime`
instance to manage. The `run_worker=False` keyword skips the in-band
worker for one-shot submissions (the CLI's `smai run` uses this); the
default boots the worker.

The novel-technique input path uses `runtime.proposals` instead of
`runtime.experiments`. The submit + approve flow is two synchronous
calls around the planner's async work; a human gate sits between
`designed` and `registered`:

```python
async with Runtime.start_in_band(runtime_config) as runtime:
    submission = await runtime.proposals.submit(
        proposal_id="prop_my_cutout_idea",
        submission_kind="novel_technique",
        technique_description="Add Cutout augmentation to ResNet-50 on CIFAR-10 ...",
    )
    # Planner drafts the ExperimentDefinition into the `designed` state.
    # Use runtime.status / runtime.proposals.get to poll, then approve:
    record = await runtime.proposals.get(submission.proposal_id)
    if record.state == "designed":
        await runtime.proposals.approve(submission.proposal_id)
    # Approval registers 1..N CGs which then run through the CG pipeline.
```

For production deployment (out-of-band worker, Postgres + S3 + remote
compute), use `Runtime.start_worker(...)`. See
`packages/smai-cli/OPERATIONS.md` for systemd / supervisord / launchd
recipes plus connection-pool sizing and structured-logging patterns.

---

## Tier B — methodology as library

The methodology layer is atomic and pipeline-independent. The whole
compile-then-evaluate pipeline imports from `smai_core` and depends only
on `pydantic` + `jsonschema` + stdlib (per DEC-029; enforced by
`tools/check_deps.py`).

```python
import yaml

from smai_core import (
    compile_experiment,
    evaluate,
    load_default_registries,
    EntryMetrics,
    RawMetrics,
    SeedRunOutcome,
)
from smai_core.dsl import DslDocumentAdapter

# 1. Load and compile a YAML experiment definition.
doc_dict = yaml.safe_load(open("tests/fixtures/experiments/cutout_on_cifar10.yaml"))
doc = DslDocumentAdapter.validate_python(doc_dict)
registries = load_default_registries()
artifact_set = compile_experiment(doc, registries)

# 2. Run the experiment yourself on your own infrastructure. Produce
#    raw multi-seed metrics in the SMAI shape — the harness contract
#    pins the per-entry required metric keys.
raw = RawMetrics(
    by_entry={
        "baseline": EntryMetrics(seed_runs=[
            SeedRunOutcome(seed=0, required={"top_k_accuracy_k_1": 0.802}, optional={}),
            SeedRunOutcome(seed=1, required={"top_k_accuracy_k_1": 0.798}, optional={}),
            SeedRunOutcome(seed=2, required={"top_k_accuracy_k_1": 0.805}, optional={}),
        ]),
        "cutout": EntryMetrics(seed_runs=[
            SeedRunOutcome(seed=0, required={"top_k_accuracy_k_1": 0.823}, optional={}),
            SeedRunOutcome(seed=1, required={"top_k_accuracy_k_1": 0.819}, optional={}),
            SeedRunOutcome(seed=2, required={"top_k_accuracy_k_1": 0.826}, optional={}),
        ]),
    }
)

# 3. Evaluate. This is the locked verdict path — no LLM, no agent, no
#    metric-substitution. The result is byte-equal-deterministic across
#    runs.
result = evaluate(artifact_set.validation_config, raw)
print(result.verdict.result)        # "pass" | "fail" | "inconclusive"
print(result.verdict.summary)
```

`smai_core` ships an entry-point-discoverable factor-type plugin
contract (`smai.factor_types` namespace), so a Tier B integrator can
register additional factor types without forking. The two built-in
factor types are `additive` and `substitutive`.

---

## CLI verb summary

15 verbs, grouped by what they do:

| Verb | Purpose |
|---|---|
| `smai dev` | Boot the laptop deployment (SQLite + LocalFs + LocalGpu + Bedrock; in-band worker; tighter poll for interactive feel). |
| `smai start` | Boot the production deployment (out-of-band worker; explicit plugin selections required; refuses to boot on incomplete config or stale schema). |
| `smai serve` | Run the read-only HTTP dashboard against the configured plugins (no worker). |
| `smai run <experiment.yaml>` | Compile + register a CG; optional `--watch` polls until terminal. |
| `smai submit-proposal <description>` | **Primary input verb** (per DEC-032). Submit a novel-technique description; the planner drafts the `ExperimentDefinition`. |
| `smai approve-proposal <id>` / `smai reject-proposal <id>` | Human gate at `designed`; approval atomically registers 1–N CGs. |
| `smai ingest <arxiv-id>` | Supporting input verb. Fetch + parse + screen + plan + register paper-derived `TechniqueRef`s. Rarely needed in default workflows. |
| `smai status [<id>] [--watch]` | Read pipeline-tracking state from `MetadataStore`. |
| `smai compile <experiment.yaml>` | Methodology-only: emit the four contract artifacts to disk or stdout. Never touches `MetadataStore` or `Compute`. |
| `smai migrate` | Apply schema migrations (Alembic-backed, per `MetadataStore` plugin). `--check` / `--dry-run` / `--prune` modes. |
| `smai verify` | Plugin-ping pre-flight: structured PASS/FAIL per plugin via Protocol-level read-only methods. |
| `smai init` | Scaffold a `smai.yaml` with sensible defaults and inline comments. |
| `smai plugins` | List discovered plugins per interface and the currently-selected plugin. |
| `smai version` | Print versions of `smai`, `smai-core`, and currently-loaded plugin packages. |

Full reference: `designs/smai/09-cli.md` §1 (verb table + `Runtime`
mapping); `designs/smai/09-cli.md` §5 / §6 (`smai dev` / `smai start`
deployment shapes).

---

## Configuration

`smai dev` boots with no `smai.yaml` and no flags. Anything you set in a
`smai.yaml` overrides the in-code defaults; environment variables
override the file; CLI flags override env vars
(`designs/smai/09-cli.md` §2). Config layering:

1. In-code defaults (`dev_defaults()` for `smai dev`; explicit required
   for `smai start`).
2. `smai.yaml` (current directory or `--config`).
3. Environment variables (`SMAI_*`, double-underscore for nested fields:
   `SMAI_PLUGINS__METADATA_STORE_CONFIG__URI=postgres://...`).
4. CLI flags.

Canonical `smai.yaml` shape:

```yaml
# Engine
engine:
  poll_interval_seconds: 30        # 10 in dev defaults
  worker_count: 1                  # >1 implies leasing; production-only
  orphan_grace_seconds: 600
  lease_seconds: 120
  fair_scheduling: "off"           # "off" | "round_robin" | "weighted"

# Plugin selection
plugins:
  llm_provider: bedrock            # bedrock | anthropic | openai
  metadata_store: sqlite           # sqlite | postgres
  artifact_store: localfs          # localfs | s3
  compute: localgpu                # localgpu | modal | runpod

  llm_provider_config:
    region: us-east-1
    model_id: us.anthropic.claude-opus-4-7-v1
  metadata_store_config:
    path: ./.smai/state.db         # sqlite
    # uri: postgres://user:pw@host:5432/smai_prod   # postgres
  artifact_store_config:
    root: ./.smai/artifacts        # localfs
    # bucket: my-smai-artifacts    # s3
    # prefix: smai/                # s3
  compute_config:
    workdir: ./.smai/compute       # localgpu
    # gpu_type: T4                 # modal / runpod
```

The same `smai.yaml` works for `smai dev` and `smai start`; the
difference is which fields are defaulted vs required. `smai start`
refuses to boot if any plugin selection is missing
(`designs/smai/09-cli.md` §6.2).

The local↔production swap is a config change, not a code change
(per `oss_strategy.md` §10 settled-decision #3): swap `sqlite` →
`postgres`, `localfs` → `s3`, `localgpu` → `modal` / `runpod`, and
your engine, agent loops, pipeline-specs, and contract artifacts are
unchanged.

---

## Plugin matrix

Four pluggable boundaries, each with one or more reference
implementations bundled in this repo. The OSS package is fully
self-contained — every reference plugin is here.

| Interface | Plugin | What it is |
|---|---|---|
| `LlmProvider` | `smai-llm-bedrock` | Phase-2 reference. AWS Bedrock Converse. AWS credential chain + `model_id`. Supports caching via `cachePoint`. |
| | `smai-llm-anthropic` | Anthropic SDK adapter. `ANTHROPIC_API_KEY`. Supports caching via `cache_control: ephemeral`. |
| | `smai-llm-openai` | OpenAI SDK adapter. `OPENAI_API_KEY`. `supports_caching=False` (server-side caching is opaque to callers). |
| `MetadataStore` | `smai-store-sqlite` | `smai dev` default; single-file zero-config. SQLAlchemy 2.0 async Core + aiosqlite (per DEC-036). |
| | `smai-store-postgres` | Production self-host; same Core schema as SQLite via cross-plugin import. Advisory-lock fast path on lease acquisition. Opt-in `tenant_aware=True`. |
| `ArtifactStore` | `smai-artifacts-localfs` | `smai dev` default; root-rooted local filesystem; presigned URLs unsupported. |
| | `smai-artifacts-s3` | BYO bucket; SigV4 presigned URLs. `boto3` + `asyncio.to_thread`. |
| `Compute` | `smai-compute-localgpu` | `smai dev` default. Local Docker subprocess; ships agent + runtime Dockerfiles. |
| | `smai-compute-modal` | Modal Sandboxes; sync SDK + `asyncio.to_thread`. GPU type plumbed via `**plugin_options`. |
| | `smai-compute-runpod` | RunPod Pods API over raw `httpx`. Six-tier GPU dispatch table. |

Discovery is entry-point-based per DEC-026 (dbt-adapter pattern). The
namespaces are `smai.llm_providers`, `smai.metadata_stores`,
`smai.artifact_stores`, `smai.computes`. `smai plugins` walks all four
and prints what's discoverable.

A future plugin lives in its own subdirectory under `plugins/`, declares
its entry point in `pyproject.toml`, and subclasses the conformance test
base from `smai-core`. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the
plugin-author guide.

---

## Architecture overview

The two-layer split (per `designs/smai/00-vision.md` §3 and DEC-029):

- **Methodology layer** (`smai-core`) — the data model, the DSL +
  compiler that emits the four contract artifacts, the mechanical
  evaluator that turns raw multi-seed metrics into a verdict, and the
  four plugin Protocol definitions. No agents, no orchestrator, no
  network. Atomic — pulling out a piece breaks the others' guarantees.
- **Pipeline layer** — agents (six fleet roles), the orchestrator
  engine + pipeline-specs + worker loop + checkpointer, the runtime
  (harness/technique substrate, fixed templates, `HarnessAPIManifest`),
  and the CLI. Composable — package boundaries enforce independence.

The orchestrator runs **four pipeline-specs**:

1. **CG execution** — drives a `ComparisonGroupRecord` from `draft`
   through `implementing` → `running` → `evaluating` → `complete`. The
   inner state machine where harness-builder + technique-implementer +
   code-reviewer + contextual-evaluator agents land.
2. **Proposal pipeline** — primary input path per DEC-032. Drives a
   `ProposalRecord` from `submitted` through `designing` (planner runs)
   → `designed` (human gate) → `registered` (1–N CGs created on
   approval).
3. **Paper ingestion** — supporting utility per DEC-032. Drives a
   `PaperRecord` through fetch → screen → plan → register, populating
   the technique pool from arXiv.
4. **`RunRecord` sub-spec** — per-`RunRecord` state machine
   (`pending` → `submitted` → `succeeded`/`failed`/`inconclusive`),
   aggregated over by the CG-execution `running → evaluating` gate.

The full design corpus lives in `designs/smai/`; the cross-references
above use canonical paths (`designs/smai/00-vision.md` etc.) within this
repo.

---

## Documentation map

The design corpus is the authoritative reference. This README and
`CONTRIBUTING.md` are entry points; the design docs are the spec.

| Doc | What it covers |
|---|---|
| `designs/smai/00-vision.md` | Anchor framing: the integration patterns, the "contracts as the locked surface" insight, principles, system shape. Read first. |
| `designs/smai/01-data-model.md` | Methodology entities and pipeline-tracking record types. |
| `designs/smai/02-dsl-and-contracts.md` | DSL, the schema layer, the four contract artifacts, verification rules, factor-type plugin contract. |
| `designs/smai/03-state-machine.md` | Pipeline-spec state machines for CG execution, proposal, paper ingestion, `RunRecord` sub-spec. |
| `designs/smai/04-agents.md` | Agent loop, prompt-config surface, six fleet roles, per-task model selection. |
| `designs/smai/05-orchestrator.md` | Engine, pipeline-spec format, worker loop, checkpointer, leasing, fair scheduling. |
| `designs/smai/06-mechanical-evaluation.md` | `evaluate(...)`, `RawMetrics`, `RunCost`, the deterministic verdict path. |
| `designs/smai/07-plugin-interfaces.md` | The four plugin Protocols + conformance test discipline. **§8 is the cross-plugin conformance contract this README points at.** |
| `designs/smai/08-novel-technique-pipeline.md` | Proposal pipeline-spec + paper ingestion. |
| `designs/smai/09-cli.md` | CLI verb surface, config layering, `RuntimeConfig`, plugin instantiation flow. |
| `designs/smai/10-runtime-and-templates.md` | Harness/technique runtime, fixed templates, no-go-zone hash check, manifest-driven type check. |
| `designs/smai/oss_strategy.md` | OSS-vs-closed split, plugin architecture, license, settled decisions. |
| `packages/smai-cli/OPERATIONS.md` | systemd / supervisord / launchd recipes; connection-pool sizing; log handling for `smai start`. |
| `packages/smai-orchestrator/src/smai_orchestrator/migrations/MIGRATIONS.md` | Migration framework: rollback policy, retention defaults, adding a revision runbook. |

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for repo layout, local-dev
setup, the project conventions, and the plugin-author guide. The four
plugin Protocols ship parameterizable conformance test base classes
under `smai_core.plugins.conformance` — a new plugin subclasses one,
overrides a `make_<interface>()` factory, and inherits the contract
suite.

---

## License

Apache 2.0 — see [`LICENSE`](LICENSE). Per `oss_strategy.md` §6: the
SDK, DSL, agent loops, contracts, runtime, mechanical evaluator, and
reference plugin implementations all ship under Apache 2.0.
