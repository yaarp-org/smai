# SMAI

> Methodology-as-infrastructure for **comparative-claim ML research**.
> Definition in, verdict out — with the metric, the factor structure, and
> the verdict path locked at compile time.

SMAI is the experiment-execution stage that an outer auto-research
pipeline — an "AI scientist" loop — plugs into when the question is
**comparative** ("does technique X beat technique Y under matched
conditions?"). It is *not* an idea generator, not a writeup tool, and
not a hill-climbing optimizer — peer territory to those, with a
different shape.

### Why an AI scientist needs this

A long-horizon novel-research agent is a stack of LLM turns, each one
cheap to nudge in a direction that makes the result look better.
Without external structure, a few characteristic failure modes show up
reliably:

- **Metric drift** — the agent reframes what it's measuring mid-run
  when early results disappoint, so "we passed" ends up true under a
  metric the agent didn't set out to test.
- **Confounding** — two things change between baseline and treatment,
  and the agent attributes the delta to its preferred change.
- **Hill-climbing dressed up as comparison** — hyperparameters get
  ratcheted until a threshold clears, rather than to test the
  hypothesis. The implementer is its own judge.
- **Pipeline-encoding masquerading as comparison** — "we ran X then Y
  then Z" gets reported as if it were a controlled comparative
  experiment, even though no comparison ever happened.
- **Loss of provenance** — at hour six of an agent loop, the
  experiment being executed is no longer the one defined at hour one;
  intermediate turns silently rewrote it, and there's no immutable
  trail.

SMAI is the structural infrastructure that closes those off. The outer
AI scientist still does the open-ended work — proposing ideas,
deciding what to test next, writing things up. SMAI takes over the
moment the question becomes "did this beat that under matched
conditions?" and returns a verdict the AI scientist could not have
nudged.

The unit of value is a small set of guarantees, not a feature list:

- **Compile-time verification** — confounded experiments fail to
  compile. Factor exclusivity, control completeness, metric
  well-formedness, comparability of entries, and "this looks like a
  sequential pipeline pretending to be a comparison" are all
  mechanically checked before any code runs.
- **Runtime determinism** — the path from raw multi-seed metrics to a
  boolean verdict has no LLM in it. The metric is fixed at compile
  time; the aggregation rule is fixed; the agent that wrote the
  implementation never sees the verdict criterion. Self-grading is
  structurally impossible.
- **Contracts as the locked surface** — the DSL compiler emits four
  immutable JSON artifacts (`ExperimentPlan`, `HarnessContract`,
  `TechniqueContract[]`, `ValidationConfig`), each content-hashed and
  parented to its predecessor. Agents read them; nothing rewrites them
  mid-flight. The audit trail is reproducible by construction.

This repo is the **OSS surface**: methodology layer, agent loops,
orchestrator, runtime, CLI, and the four reference plugin interfaces
with local + production implementations.

> **Status: private pre-release.** APIs may break without notice. PyPI
> distribution and a public docs site are gated on an explicit user
> decision; until that gate is flipped, install from the local workspace
> via `uv sync`. Apache 2.0 licensed.

---

## How `smai-core` works

The methodology layer is one library: `smai-core`. It has no agents in
it, no network calls, and no orchestrator. It does four things, in
order: take a definition, verify it's answerable, emit immutable
contracts, and — once raw metrics exist — produce a verdict.

### 1. The research agent writes an experiment definition

The definition pins the comparative claim down to a verifiable shape:
*one factor* that varies across entries, *all the other variables*
held fixed, *the metric* that decides it, and *which entry is the
baseline*. The shape in YAML is roughly:

```yaml
kind: experiment
experiment:
  id: aug_cutout_vs_none_cifar10
  hypothesis: |
    Cutout augmentation improves CIFAR-10 top-1 accuracy over no
    augmentation, holding architecture and optimizer fixed.

  # The ONE thing that varies across entries.
  factors:
    - name: cutout_augmentation
      type: additive            # or substitutive

  # Everything else, pinned.
  controlled_conditions:
    dataset:      { name: cifar10, split: standard_50k_train_10k_test }
    architecture: { name: resnet50 }
    optimization: { optimizer: adamw, learning_rate: 0.001, ..., epochs: 100 }
    seeds:        [42, 1337, 2024, 9999, 55]

  # The entries being compared. Exactly one is the baseline.
  entries:
    - id: no_aug_baseline
      is_baseline: true
      level: { factor: cutout_augmentation, name: absent, technique_id: null }
    - id: cutout_treatment
      level:
        factor: cutout_augmentation
        name: present
        technique_id: tech_cutout
        technique_params: { patch_size: 16 }

  # The verdict path. Locked here, never rewritten downstream.
  validation:
    metric:      { kind: parametric, family: top_k_accuracy, parameters: { k: 1 } }
    direction:   higher_is_better
    aggregation: { method: mean }
    comparison:  { rule: compare_to_baseline, threshold: 0.003 }
    seed_count_required: 5
```

A multi-CG experiment — a *factor model* — groups several single-factor
comparisons under a shared research question; each runs and verifies
independently.

### 2. The compiler verifies the claim is *answerable*

Before any code runs, `compile_experiment(...)` checks the definition
against several dozen mechanical rules grouped into eight categories.
Confounded experiments fail at compile. What's caught:

- **Factor structure** — exactly one factor varies; the factor name is
  unique; the factor type is registered; entries cover at least two
  levels.
- **Entry vs factor compatibility** — every entry references the
  declared factor at a level the factor admits; entries are
  distinguishable (no two entries reduce to the same technique +
  params).
- **Controlled-conditions completeness** — required fields are present;
  seed count matches `seed_count_required`; seeds are unique.
- **Metric well-formedness** — the metric family is registered;
  required parameters are present (`k` for `top_k_accuracy`); the
  comparison threshold's sign agrees with `direction`; cost metrics
  live in `optional_telemetry`, not `required`.
- **Validation soundness** — the baseline entry resolves; the
  comparison rule is well-formed; `seed_count_required` is positive;
  any trend check is applicable.
- **Technique compatibility** — every referenced `technique_id` is
  registered; the technique's factor-type matches the factor; technique
  params validate against the technique's schema; numeric level values
  lie in declared ranges.
- **Pipeline-encoding heuristics** — flags definitions that look like a
  *sequence of steps* masquerading as a comparative experiment (a
  common shape mistake the compiler refuses to allow through).
- **Cross-CG factor-model rules** — for grouped experiments, shared
  conditions are consistent and factor names don't collide across CGs.

Each rule produces a structured violation with a stable code, e.g.
`metric.parametric_required_parameters_present` or
`validation.threshold_sign_matches_direction`. This is what "confounded
experiments fail to compile" cashes out as.

### 3. The compiler emits four immutable JSON artifacts

A successful compile produces an `ArtifactSet` of four contracts. Each
artifact carries an envelope with a content hash and the hash of its
parent — so any downstream consumer can prove what it's reading
against:

**`ExperimentPlan`** — substantively the same as the input definition,
plus the envelope. The faithful design-time record.

```json
{
  "envelope": { "hash": "...", "parent_hash": null, "type": "ExperimentPlan", ... },
  "body": { "hypothesis": "...", "factors": [...], "controlled_conditions": {...},
            "entries": [...], "validation": {...} }
}
```

**`HarnessContract`** — what the harness substrate must expose. The
factor is the lone variable the harness *doesn't* fix; everything else
is flattened out of `controlled_conditions` into `fixed_variables`.

```json
{
  "envelope": { "hash": "...", "parent_hash": "<ExperimentPlan hash>", ... },
  "body": {
    "factor": { "name": "cutout_augmentation", "type": "additive", ... },
    "seeds": [42, 1337, 2024, 9999, 55],
    "fixed_variables": [
      { "path": "dataset.name",                "value": "cifar10" },
      { "path": "architecture.name",           "value": "resnet50" },
      { "path": "optimization.learning_rate",  "value": 0.001 },
      "..."
    ],
    "required_metrics": [ { "kind": "parametric", "family": "top_k_accuracy", "parameters": { "k": 1 } } ],
    "optional_telemetry": [],
    "no_go_zones": [ "..." ]
  }
}
```

**`TechniqueContract[]`** — one per entry, each linked back to the
harness it must conform to.

```json
{
  "envelope": { "hash": "...", "parent_hash": "<HarnessContract hash>", ... },
  "body": {
    "entry_id": "cutout_treatment",
    "technique_id": "tech_cutout",
    "technique_params": { "patch_size": 16 },
    "is_baseline": false,
    "parent_experiment_hash": "...",
    "parent_harness_contract_hash": "..."
  }
}
```

**`ValidationConfig`** — the single input the verdict path takes
besides the raw metrics.

```json
{
  "envelope": { "hash": "...", "parent_hash": "<ExperimentPlan hash>", ... },
  "body": {
    "metric":      { "kind": "parametric", "family": "top_k_accuracy", "parameters": { "k": 1 } },
    "direction":   "higher_is_better",
    "aggregation": { "method": "mean" },
    "comparison":  { "rule": "compare_to_baseline", "threshold": 0.003,
                     "baseline_entry_id": "no_aug_baseline" },
    "seed_count_required": 5
  }
}
```

These are JSON. The pipeline serializes them to `ArtifactStore` and
agents read them as JSON; a Tier-B integrator can dump them with
`artifact_set.to_json()`. Anyone — including the code-reviewer agent —
who wants to know "what is this experiment really comparing, and how
will it be judged?" reads these artifacts, not the YAML.

### 4. `evaluate(...)` produces the verdict — mechanically

Once raw multi-seed metrics exist (whoever produced them), the verdict
is a pure function of two inputs:

```
evaluate(
  validation_config: ValidationConfig,   # locked at compile time
  raw: RawMetrics,                       # { entry_id -> [ {seed, required, optional}, ... ] }
) -> EvaluationResult                    # verdict: pass | fail | inconclusive
```

This call has no LLM in it, no agent, and no metric substitution. It:

1. Aggregates per-entry seed runs by the configured rule (`mean`, …).
2. Compares each non-baseline entry against the baseline by the
   configured rule (`compare_to_baseline`) and threshold.
3. Emits a verdict with the structured intermediates attached.

Two evaluations of the same `ValidationConfig` against the same
`RawMetrics` produce byte-equal results. This determinism is why "the
agent that wrote the implementation never sees the verdict criterion"
is meaningful: the criterion is the `ValidationConfig`; the
implementation only ever sees the `HarnessContract` and its own
`TechniqueContract`.

That's `smai-core` in full: definition → compile (verify + emit) →
evaluate. Everything else in this repo is the pipeline that automates
producing the raw metrics in between.

---

## How the pipeline layer works

`smai-core` defines *what* a sound comparative experiment looks like;
the pipeline layer is *how* you go from contract artifacts to raw
metrics without writing any of the harness or technique code yourself.
It has three components, packaged independently:

- **Agents** (`smai-agents`) — six fleet roles backed by a custom
  Bedrock-Converse-style agent loop. The **planner** drafts an
  `ExperimentDefinition` from a novel-technique description. The
  **harness builder** writes the harness substrate from the
  `HarnessContract`. The **technique implementer** fills in each
  `TechniqueContract`. The **code reviewer** reads the contracts and
  the diffs to flag drift. The **contextual evaluator** annotates the
  numeric verdict with discussion. The **supervisor** sits over multi-CG
  orchestration. Each role has its own per-task model selection.
- **Orchestrator** (`smai-orchestrator`) — a generic pipeline-spec
  engine: state machines + a worker loop with leasing + a checkpointer.
  The same engine drives four different specs: comparison-group
  execution (the inner state machine where harness and technique
  implementations are produced and reviewed), the proposal pipeline
  (primary input path, with a human gate at `designed`), paper
  ingestion (supporting utility), and a `RunRecord` sub-spec the CG
  loop inlines for each seed run. Multi-worker deployments use leasing
  + fair scheduling; single-worker dev mode skips the leasing
  fast-path.
- **Plugins** — the boundary between the pipeline and the outside
  world. Four interfaces, each a Python `Protocol`: `LlmProvider`
  (where agent calls go), `MetadataStore` (where pipeline-tracking
  records live), `ArtifactStore` (where the four JSON contracts and
  harness/technique code live), `Compute` (where seed runs execute).
  Reference implementations for local-dev and production are bundled
  in this repo.

The runtime substrate (`smai-runtime`) carries the fixed templates
and manifest-driven type checks the harness builder emits against — it
sits between the agents and `Compute`.

The split between `smai-core` and the pipeline is deliberate: the
methodology layer is *atomic* (pulling out a piece breaks the others'
guarantees), so it ships as one package with a tiny dep allowlist
(`pydantic` + `jsonschema` + stdlib). The pipeline is *composable* —
each piece is independently consumable, and a Tier-B integrator can
import `smai-core` alone and run the methodology layer against their
own infrastructure.

---

## Two integration patterns

The two-layer split surfaces as two canonical integration patterns:

| Pattern | What it gives you | What you import |
|---|---|---|
| **Tier A — deep delegation** | Hand SMAI a definition (or a description and let the planner draft one). Read back a verdict + artifact set. Full pipeline: agents, orchestrator, runtime, plugins. | `smai_cli.runtime.Runtime` (programmatic), or the `smai` CLI. |
| **Tier B — methodology as library** | Compile your own definitions, run experiments on your own infrastructure, call the evaluator for the verdict. The pipeline layer isn't involved. | `smai_core` only — `compile_experiment`, `evaluate`, the four contract types, the four plugin Protocols. |

A "mixed" pattern — using some pipeline components but not others — is
supported by the modularity but is *not* a documented happy path;
soundness guarantees still hold (they're enforced at the package
boundary), but operational guarantees from the pipeline's between-turn
coordination may degrade outside the canonical patterns.

---

## Repo layout

```
smai/
├── packages/
│   ├── smai-core/              # methodology layer: data model, DSL +
│   │                           # compiler, contract artifacts, mechanical
│   │                           # evaluator, four plugin Protocols.
│   │                           # Allowlisted deps: pydantic, jsonschema, stdlib.
│   ├── smai-runtime/           # harness/technique runtime + fixed templates.
│   ├── smai-agents/            # custom Bedrock-Converse-style agent loop, six
│   │                           # fleet roles (planner, harness builder, technique
│   │                           # implementer, code reviewer, contextual evaluator,
│   │                           # supervisor).
│   ├── smai-orchestrator/      # engine + pipeline-spec format + worker loop +
│   │                           # checkpointer; the four pipeline-spec instances
│   │                           # (CG execution, proposal, paper ingestion,
│   │                           # RunRecord sub-spec); Alembic migrations.
│   ├── smai-cli/               # CLI verbs, config layering, RuntimeConfig,
│   │                           # plugin instantiation, in-band Runtime.
│   └── smai/                   # umbrella package (currently a stub; eventual
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

The CLI is a thin wrapper around the in-process service surface.
Programmatic Tier-A consumers import `Runtime` directly:

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
compile-then-evaluate pipeline imports from `smai_core` and depends
only on `pydantic` + `jsonschema` + stdlib (enforced by
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
| `smai submit-proposal <description>` | Primary input verb. Submit a novel-technique description; the planner drafts the `ExperimentDefinition`. |
| `smai approve-proposal <id>` / `smai reject-proposal <id>` | Human gate at `designed`; approval atomically registers 1–N CGs. |
| `smai ingest <arxiv-id>` | Supporting input verb. Fetch + parse + screen + plan + register paper-derived `TechniqueRef`s. Rarely needed in default workflows. |
| `smai status [<id>] [--watch]` | Read pipeline-tracking state from `MetadataStore`. |
| `smai compile <experiment.yaml>` | Methodology-only: emit the four contract artifacts to disk or stdout. Never touches `MetadataStore` or `Compute`. |
| `smai migrate` | Apply schema migrations (Alembic-backed, per `MetadataStore` plugin). `--check` / `--dry-run` / `--prune` modes. |
| `smai verify` | Plugin-ping pre-flight: structured PASS/FAIL per plugin via Protocol-level read-only methods. |
| `smai init` | Scaffold a `smai.yaml` with sensible defaults and inline comments. |
| `smai plugins` | List discovered plugins per interface and the currently-selected plugin. |
| `smai version` | Print versions of `smai`, `smai-core`, and currently-loaded plugin packages. |

---

## Configuration

`smai dev` boots with no `smai.yaml` and no flags. Anything you set in
a `smai.yaml` overrides the in-code defaults; environment variables
override the file; CLI flags override env vars. Config layering:

1. In-code defaults (`dev_defaults()` for `smai dev`; explicit required
   for `smai start`).
2. `smai.yaml` (current directory or `--config`).
3. Environment variables (`SMAI_*`, double-underscore for nested
   fields: `SMAI_PLUGINS__METADATA_STORE_CONFIG__URI=postgres://...`).
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
refuses to boot if any plugin selection is missing.

The local↔production swap is a config change, not a code change: swap
`sqlite` → `postgres`, `localfs` → `s3`, `localgpu` → `modal` /
`runpod`, and your engine, agent loops, pipeline-specs, and contract
artifacts are unchanged.

---

## Plugin matrix

Four pluggable boundaries, each with one or more reference
implementations bundled in this repo. The OSS package is fully
self-contained — every reference plugin is here.

| Interface | Plugin | What it is |
|---|---|---|
| `LlmProvider` | `smai-llm-bedrock` | AWS Bedrock Converse. AWS credential chain + `model_id`. Supports caching via `cachePoint`. |
| | `smai-llm-anthropic` | Anthropic SDK adapter. `ANTHROPIC_API_KEY`. Supports caching via `cache_control: ephemeral`. |
| | `smai-llm-openai` | OpenAI SDK adapter. `OPENAI_API_KEY`. `supports_caching=False` (server-side caching is opaque to callers). |
| `MetadataStore` | `smai-store-sqlite` | `smai dev` default; single-file zero-config. SQLAlchemy 2.0 async Core + aiosqlite. |
| | `smai-store-postgres` | Production self-host; same Core schema as SQLite via cross-plugin import. Advisory-lock fast path on lease acquisition. Opt-in `tenant_aware=True`. |
| `ArtifactStore` | `smai-artifacts-localfs` | `smai dev` default; root-rooted local filesystem; presigned URLs unsupported. |
| | `smai-artifacts-s3` | BYO bucket; SigV4 presigned URLs. `boto3` + `asyncio.to_thread`. |
| `Compute` | `smai-compute-localgpu` | `smai dev` default. Local Docker subprocess; ships agent + runtime Dockerfiles. |
| | `smai-compute-modal` | Modal Sandboxes; sync SDK + `asyncio.to_thread`. GPU type plumbed via `**plugin_options`. |
| | `smai-compute-runpod` | RunPod Pods API over raw `httpx`. Six-tier GPU dispatch table. |

Discovery is entry-point-based (dbt-adapter pattern). The namespaces
are `smai.llm_providers`, `smai.metadata_stores`,
`smai.artifact_stores`, `smai.computes`. `smai plugins` walks all four
and prints what's discoverable.

A future plugin lives in its own subdirectory under `plugins/`,
declares its entry point in `pyproject.toml`, and subclasses the
conformance test base from `smai-core`. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the plugin-author guide.

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

Apache 2.0 — see [`LICENSE`](LICENSE). The SDK, DSL, agent loops,
contracts, runtime, mechanical evaluator, and reference plugin
implementations all ship under Apache 2.0.
