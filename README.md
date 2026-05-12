# SMAI

> Methodology-as-infrastructure for comparative-claim ML research.
> **Definition in, verdict out** — with the metric, the factor structure,
> and the verdict path locked at compile time.

SMAI is the experiment-execution stage that an outer auto-research
pipeline — an "AI scientist" loop — hands off to when its question turns
**comparative**: *does technique X beat technique Y under matched
conditions?* It is not an idea generator, not a writeup tool, and not a
hill-climbing optimizer; those belong to the outer loop. SMAI's job is
narrower and, by construction, harder to fool: take a comparative claim,
verify it's even answerable, run the experiment, and return a verdict
the agent that did the work could not have nudged.

> _**Diagram — where SMAI fits.** The outer AI scientist loop (proposes
> ideas → decides what to test → writes things up) with SMAI slotted in
> as the "execute this comparative experiment" stage: a definition flows
> in, a boolean verdict plus a content-hashed artifact set flow out._

> **Status — private pre-release.** APIs may break without notice. PyPI
> distribution and a public docs site are gated on an explicit decision;
> until that gate is flipped, install from the local workspace with
> `uv sync`. Apache 2.0 licensed.

---

## Why an AI scientist needs this

A long-horizon research agent is a stack of LLM turns, and every turn is
cheap to nudge toward a result that looks better than it is. Left
unstructured, a handful of failure modes show up reliably:

- the agent quietly reframes what it's measuring once early numbers
  disappoint, so "we passed" ends up true under a metric nobody set out
  to test;
- two things change between baseline and treatment, and the delta gets
  pinned on whichever change the agent was rooting for;
- hyperparameters get ratcheted until a threshold clears — hill-climbing
  in the costume of a controlled comparison, with the implementer as its
  own judge;
- "we ran X, then Y, then Z" gets written up as if a comparison had
  happened, when no comparison ever did;
- six hours into the loop, the experiment actually running is no longer
  the one defined at hour one — intermediate turns rewrote it, and
  nothing kept the receipts.

These are exactly the failures that pre-defined RL-style environments
guarded against, and exactly the ones that purely agent-driven research
systems have been documented giving back up. SMAI is the structure that
closes them off *without* surrendering the open-endedness: the outer
agent keeps doing all the creative work; SMAI owns only the stretch
where a claim becomes a verdict.

What that buys you is a short list of guarantees, not a feature list:

**Confounded experiments fail to compile.** Factor exclusivity, control
completeness, metric well-formedness, comparability of the entries being
compared, and "this looks suspiciously like a sequential pipeline
pretending to be an experiment" are all mechanically checked before any
code runs.

**The verdict path has no LLM in it.** The metric is fixed at compile
time; the aggregation rule is fixed; the agent that wrote the
implementation never sees the verdict criterion. Self-grading isn't
discouraged — it's structurally impossible.

**The contracts are the locked surface.** Compiling a definition emits
four immutable JSON artifacts, each content-hashed and parented to its
predecessor. Agents read them; nothing rewrites them mid-flight. The
audit trail is reproducible by construction.

---

## How it works

SMAI is two layers, and the seam between them is deliberate.

**The methodology layer is one library: `smai-core`.** No agents, no
network, no orchestrator — `pydantic` + `jsonschema` + stdlib is the
entire dependency list. It does four things, in order: take an
experiment definition, verify the claim is answerable, emit the
immutable contracts, and — once raw multi-seed metrics exist, from
wherever — produce the verdict. That's the whole API surface:
`compile_experiment(...)`, then `evaluate(...)`.

**The pipeline layer is everything that produces those raw metrics for
you.** Agents draft the definition, write the harness, fill in each
technique implementation, and review the diffs; an orchestrator drives
the state machines and the worker loop; four plugin interfaces wire it
to an LLM, a metadata store, an artifact store, and compute. None of
this is load-bearing for the *guarantees* — those live in `smai-core`,
enforced at the package boundary — which is precisely why you can drop
the pipeline and run `smai-core` against your own infrastructure.

The two split this way because they have opposite shapes. `smai-core` is
*atomic*: pull out a piece and the others' guarantees break, so it ships
as a single tightly-scoped package. The pipeline is *composable*: every
piece is independently useful, and the two canonical integration
patterns — "deep delegation" and "library only" — fall straight out of
that asymmetry.

> _**Diagram — the two layers + the contract chain.** `smai-core` as a
> small sealed box (`compile` → 4 contracts; `evaluate` → verdict), with
> the pipeline layer (planner / harness builder / technique implementer /
> code reviewer, orchestrator, the 4 plugins) wrapped around it. Arrows
> show which agent reads which contract — and, crucially, that no arrow
> runs from `ValidationConfig` to the technique implementer._

### The contract chain

A definition pins a comparative claim down to a verifiable shape: *one
factor* that varies across entries, *every other variable* held fixed,
*the metric* that decides it, and *which entry is the baseline*.
`compile_experiment` checks that shape against several dozen mechanical
rules and, if it holds, emits four contracts:

- **`ExperimentPlan`** — the input definition, made canonical, plus an
  envelope. The faithful design-time record.
- **`HarnessContract`** — what the harness substrate must expose: the
  factor is the one variable it *doesn't* fix; everything else from
  `controlled_conditions` is flattened into `fixed_variables`, with the
  required metrics and the no-go zones spelled out.
- **`TechniqueContract[]`** — one per entry, each parented to the
  harness it must conform to, carrying only that entry's technique id
  and params.
- **`ValidationConfig`** — the metric, direction, aggregation rule, and
  comparison rule + threshold. The single thing `evaluate` consumes
  besides the raw metrics — and the one contract the implementer never
  sees.

Every artifact's envelope carries its own content hash and its parent's,
so any downstream consumer — the code-reviewer agent included — can
prove what it's reading against. "What is this experiment really
comparing, and how will it be judged?" is answered by these artifacts,
never by re-reading the YAML.

Once technique implementations pass code review, the rest is mechanical:
the orchestrator runs the seeds, collects the raw metrics, and calls
`evaluate(validation_config, raw_metrics)` — a pure function, no LLM, no
agent, byte-equal-deterministic across runs — which returns `pass`,
`fail`, or `inconclusive` with the intermediates attached. That boolean
is the output of the whole thing: did the experiment validate the
hypothesis?

### What ships in this repo

The OSS surface, Apache 2.0: the `smai-core` methodology layer, the
agent loops, the orchestrator engine, the runtime, the `smai` CLI, four
reference plugin interfaces with both a local and a production
implementation behind each, and — new in v2 — a JSON HTTP API plus a
React SPA that both consume the same in-process `Runtime`. The
local↔production swap (SQLite→Postgres, local filesystem→S3, local
GPU→Modal/RunPod) is a config change, not a code change.

---

## Quick start

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
git clone <repo-url> smai && cd smai
uv sync                 # install the workspace (editable) + dev deps
uv run pytest           # ~2100 tests, ~100s warm
```

Boot the laptop deployment with the web UI:

```bash
# Dev defaults: SQLite + local filesystem + local-GPU Docker + Bedrock
# (us.anthropic.claude-opus-4-6-v1 in us-east-1). Have AWS creds in the
# default chain and model access granted in the Bedrock console for that
# model/region, or point at Anthropic / OpenAI instead (see Configuration).
$ uv run smai ui
smai ui: starting in-process worker (sqlite metadata store detected).
         Listening on http://127.0.0.1:8000/
```

Open `http://127.0.0.1:8000/`. The dashboard shows live state and a
recent-activity feed; updates arrive over SSE, no polling:

```
+------------------------------------------------------------+
|  SMAI                                       [v 0.1.0]      |
+------------------------------------------------------------+
|  Comparison groups        Proposals        Papers          |
|  cg_01J7PA8K... [running]    prop_xyz       arxiv:2103.x   |
|  cg_01J7PA8L... [complete]   [designed]     [registered]   |
|                                                            |
|  Recent activity (live)                                    |
|   17:23:11  cg_01J7PA8K... implementing -> implemented     |
|   17:23:14  prop_xyz       designing    -> designed        |
+------------------------------------------------------------+
```

In another terminal, submit an experiment and watch it run:

```bash
$ uv run smai run tests/fixtures/experiments/cutout_on_cifar10.yaml --watch
cg_01J7PA8K2X9F4ZB6QH4N0P1234: complete
```

It shows up in the UI the instant the write commits. Two adjacent verbs
are worth knowing now: `smai dev` is the headless equivalent of
`smai ui` — same plugin set, no API, no SPA — for a browser-less
session; and `smai compile experiment.yaml` runs *only* the methodology
layer, emitting the four contracts to stdout without touching the
metadata store or compute.

### Before you run

`smai compile` needs no plugins or credentials, but it is pure
methodology and does not read the metadata store, so any technique your
experiment references must be supplied inline: `smai compile
experiment.yaml --techniques techniques.json` (a JSON list of
`TechniqueRef` objects, or an object keyed by id; repeatable).
`smai run` reads registered techniques from the store, so it needs no
such flag once a proposal or paper-ingestion run has populated them.

Everything else dispatches real agent calls and (eventually) real
compute, so it needs whatever the selected plugins need. The dev
defaults (`smai dev` / `smai ui`) need:

- **AWS credentials** in the default chain (`~/.aws/...`, env vars, or
  an instance role), **and Bedrock model access granted** in the AWS
  console for the configured model + region. A bare credential chain
  is not enough; an ungranted model fails with
  `AccessDeniedException: ... is not available for this account`.
- **Docker running locally**, with the reference images built. SMAI
  does not publish these; build them yourself from
  `plugins/smai-compute-localgpu/dockerfiles/` (the `docker build`
  command is in each Dockerfile's header comment):
  `smai-agent:dev` (agent-side code exec, lean, no ML stack),
  `smai-runtime:dev` (GPU experiment runs, `nvidia/cuda` base, ~5-8 GB,
  amd64), and `smai-runtime-cpu:dev` (CPU-only experiment runs, lean
  multi-arch base + ML stack). On macOS / Apple Silicon, Docker Desktop
  has no GPU passthrough, so the dev-default LocalGpu Compute refuses
  GPU jobs; add `controlled_conditions: { compute: { gpu: false } }` to
  an experiment YAML for CPU-only runs (methodology smoke runs, kNN
  comparisons, small models). That makes the dispatcher hand the run
  `smai-runtime-cpu:dev` — lean and multi-arch, so it runs natively on
  Apple Silicon rather than emulating the amd64 CUDA image. The setting
  also flows into the `HarnessContract`, so a Modal / RunPod deployment
  honors it too. Override the image names per deployment via
  `engine.runtime_image` / `engine.runtime_cpu_image` in `smai.yaml`.

Pointing at a different plugin set changes the requirements: Anthropic
needs `ANTHROPIC_API_KEY`; OpenAI needs `OPENAI_API_KEY` (and the SDK
requires it *at construction*, so even `smai verify` fails without it);
Modal needs `~/.modal.toml` or `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET`;
RunPod needs `RUNPOD_API_KEY`; S3 needs the AWS chain plus a
pre-existing bucket. The per-plugin "credentials" column under
[Configuration](#configuration) is the full table.

Run `smai verify` once your config is in place: it pings every
configured plugin (a real 1-token LLM completion, a read-only store
query, a HEAD on the artifact store, a no-op compute `status`) and
reports PASS/FAIL per interface, so a misconfigured `smai.yaml` fails
fast with a clear diagnostic instead of mid-run.

---

## Configuration

`smai dev` boots with no config file and no flags. Past that, settings
layer — in-code defaults, then `smai.yaml`, then `SMAI_*` env vars
(double-underscore for nested keys, e.g.
`SMAI_PLUGINS__METADATA_STORE_CONFIG__URI=...`), then CLI flags — each
overriding the last. `smai start` (production) is the same machinery
with the plugin selections made required instead of defaulted; it
refuses to boot on an incomplete config or a stale schema.

```yaml
engine:
  poll_interval_seconds: 30        # 10 under dev defaults
  worker_count: 1                  # >1 turns on leasing; production only
  # runtime_image / runtime_cpu_image: GPU vs CPU container images for
  # experiment seed runs (defaults: smai-runtime:dev / smai-runtime-cpu:dev).
  # The run-record dispatcher picks per the CG's compute.gpu flag.

plugins:
  llm_provider:   bedrock          # bedrock | anthropic | openai
  metadata_store: sqlite           # sqlite | postgres
  artifact_store: localfs          # localfs | s3
  compute:        localgpu         # localgpu | modal | runpod

  # Each *_config dict is passed verbatim as keyword arguments to the
  # selected plugin's constructor; see the key table below. A wrong
  # key fails fast at boot ("got an unexpected keyword argument ...").
  llm_provider_config:   { region: us-east-1, model_id: us.anthropic.claude-opus-4-6-v1 }
  metadata_store_config: {}                              # sqlite default: in-memory (see note below)
  artifact_store_config: {}                              # localfs default: ~/.smai/artifacts
  compute_config:        {}                              # localgpu default: smai-agent:dev / smai-runtime:dev / smai-runtime-cpu:dev

# `smai ui` only — the whole block is optional.
api:
  host: 127.0.0.1
  port: 8000
  with_worker: auto                # auto: on for sqlite+localfs, off otherwise
  auth: { enabled: false }         # opt-in bearer-token mode
```

**Per-plugin `*_config` keys.** The value of each `*_config` block is splatted as `**kwargs` into the plugin constructor, so the keys are exactly the constructor parameters:

| plugin | `*_config` keys (defaults in parens) | credentials |
|---|---|---|
| `bedrock` | `region` (`us-east-1`), `model_id` | AWS default credential chain + Bedrock model access granted in the console for that model/region |
| `anthropic` | `model_id` (`claude-opus-4-7`) | `ANTHROPIC_API_KEY` |
| `openai` | `model_id` | `OPENAI_API_KEY` (**required at construction**, even for `smai verify`) |
| `sqlite` | `uri` (`sqlite+aiosqlite:///:memory:`); a SQLAlchemy URL, absolute path = four slashes: `sqlite+aiosqlite:////home/me/.smai/state.db` | — |
| `postgres` | `uri`, `use_advisory_locks` (`True`), `tenant_aware` (`False`), `fair_scheduling` (`off`), `fair_scheduling_weights`, `engine_kwargs` (e.g. `{pool_size: 10}`) | URL embeds credentials |
| `localfs` | `root` (`$SMAI_ARTIFACTS_ROOT` or `~/.smai/artifacts`) | — |
| `s3` | `bucket` (**required**), `region`, `prefix` (`""`), `presigned_url_expiry_seconds`, `max_object_size_bytes` | AWS default credential chain; the bucket must already exist |
| `localgpu` | `agent_image` (`smai-agent:dev`), `runtime_image` (`smai-runtime:dev`), `runtime_cpu_image` (`smai-runtime-cpu:dev`), `workspace` — these only tune the "image not found" build hint; the experiment-run image is `engine.runtime_image` / `runtime_cpu_image` | Docker running locally; build all three reference images yourself (SMAI does not publish them). On Linux the workspace bind-mount must be writable by uid 1000 (the image's `smai` user). |
| `modal` | `app_name` (`smai`), `default_gpu_type` (`T4`), `max_timeout_seconds` (`86400`) | `~/.modal.toml` or `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` |
| `runpod` | `api_base`, `default_gpu_type` (`NVIDIA RTX A4000`), `default_timeout_seconds` (`3600`), `max_timeout_seconds`, `default_container_disk_gb` (`10`) | `RUNPOD_API_KEY` |

Per-submit knobs like a Compute job's `gpu_type` / `cpu` / `memory_mb` (Modal) or `workspace` (localgpu) are *job* options the engine passes per dispatch, not constructor keys, so they don't go in `compute_config`.

The same file works for `smai dev` and `smai start`. Swapping
`sqlite`→`postgres`, `localfs`→`s3`, `localgpu`→`modal`/`runpod` is the
entire local→production migration: the engine, the agent loops, the
pipeline state machines, and the contract artifacts are untouched.
`smai init` scaffolds an annotated `smai.yaml`; `smai verify` pings
every configured plugin and reports PASS/FAIL per interface before you
commit to a run.

**Where state lives.** `smai dev` and `smai ui` provision `~/.smai/{state.db,artifacts,workspaces}` for you (creating the directories and injecting the paths), so a bare `metadata_store_config: {}` works under those verbs. `smai migrate`, `smai verify`, and `smai start` use the `smai.yaml` value verbatim, and an empty `metadata_store_config: {}` resolves to the sqlite plugin's *in-memory* default, which makes `smai migrate` a silent no-op (it reports success, then the database evaporates on exit). For those verbs, set an explicit file path:

```yaml
plugins:
  metadata_store_config: { uri: "sqlite+aiosqlite:////absolute/path/state.db" }   # 4 slashes
  # ...or for production:
  # metadata_store_config: { uri: "postgresql+asyncpg://user:pw@host:5432/smai" }
```

(The sqlite plugin expands a leading `~` and creates the parent directory, so `sqlite+aiosqlite:///~/.smai/state.db` works too.)

---

## Two ways to use it

| | What you get | What you import |
|---|---|---|
| **Tier A — deep delegation** | Hand SMAI a definition (or a plain-English technique description and let the planner draft one). Read back a verdict and the artifact set. The full pipeline runs: agents, orchestrator, runtime, plugins. | the `smai` CLI, or `Runtime` programmatically |
| **Tier B — methodology as library** | Compile your own definitions, run the experiments on your own infrastructure, call the evaluator for the verdict. The pipeline layer never enters the picture. | `smai_core` alone |

A "mixed" mode — some pipeline pieces, not others — works (the soundness
guarantees are enforced at the `smai-core` boundary regardless) but
isn't a documented happy path; the pipeline's between-turn coordination
is tuned for the two canonical shapes.

### Tier A — programmatic

```python
import asyncio
from smai_cli import Runtime, dev_defaults, load_runtime_config

async def main() -> None:
    config = load_runtime_config(defaults=dev_defaults())
    yaml_text = open("tests/fixtures/experiments/cutout_on_cifar10.yaml").read()

    async with Runtime.start_in_band(config) as runtime:
        # A multi-factor experiment registers N CGs; a single-factor one returns one.
        cg_ids = await runtime.experiments.submit_text(yaml_text)
        for cg_id in cg_ids:
            snap = await runtime.status.wait_for_terminal(cg_id, timeout=None)
            print(f"{cg_id}: {snap.state}")

asyncio.run(main())
```

`Runtime.start_in_band(...)` is async-context-managed — plugins open on
enter, drain and close on exit. The *primary* input path is proposals,
not raw definitions: `runtime.proposals.submit(...)` hands the planner a
technique description, it drafts an `ExperimentDefinition` into the
`designed` state, a human gate approves it, and approval registers the
CGs. For production (out-of-band worker, Postgres + S3 + remote compute)
use `Runtime.start_worker(...)`; deployment recipes live in
`packages/smai-cli/OPERATIONS.md`. The `smai ui` HTTP process wraps this
*same* `Runtime` in FastAPI routes plus SSE plus an optional SPA mount.

### Tier B — methodology as library

```python
import yaml
from smai_core import compile_experiment, evaluate, load_default_registries
from smai_core import EntryMetrics, RawMetrics, SeedRunOutcome
from smai_core.dsl import DslDocumentAdapter

# 1. Compile a YAML definition into the four contracts.
doc = DslDocumentAdapter.validate_python(
    yaml.safe_load(open("tests/fixtures/experiments/cutout_on_cifar10.yaml"))
)
artifact_set = compile_experiment(doc, load_default_registries())

# 2. Run the experiment however you like; produce raw multi-seed metrics
#    in SMAI's shape (the harness contract pins the per-entry metric keys).
raw = RawMetrics(by_entry={
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
})

# 3. Evaluate — the locked verdict path. No LLM, no agent, deterministic.
result = evaluate(artifact_set.validation_config, raw)
print(result.verdict.result)   # "pass" | "fail" | "inconclusive"
```

`smai_core` exposes an entry-point-discoverable factor-type plugin
contract (`smai.factor_types`), so a Tier-B integrator can register new
factor types without forking; `additive` and `substitutive` ship
built-in.

---

# Reference

The sections below are the detail behind the overview — read them when
you need the specifics, skip them when you don't.

## The experiment definition

A definition pins the comparative claim to a verifiable shape: one
varying factor, everything else fixed, the deciding metric, and the
baseline entry. In YAML, roughly:

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
comparisons under one research question; each compiles, runs, and is
verified independently.

## What the compiler checks

`compile_experiment(...)` runs the definition past several dozen rules in
eight categories; a confounded experiment fails here, before any code
runs. Each rule produces a structured violation with a stable code
(e.g. `metric.parametric_required_parameters_present`,
`validation.threshold_sign_matches_direction`). The categories:

- **Factor structure** — exactly one factor varies; its name is unique;
  its type is registered; entries cover at least two levels.
- **Entry ↔ factor compatibility** — every entry references the declared
  factor at a level it admits; no two entries reduce to the same
  technique + params.
- **Controlled-conditions completeness** — required fields present; seed
  count matches `seed_count_required`; seeds unique.
- **Metric well-formedness** — the metric family is registered; required
  parameters present (`k` for `top_k_accuracy`); the comparison
  threshold's sign agrees with `direction`; cost metrics live in
  `optional_telemetry`, not `required`.
- **Validation soundness** — the baseline entry resolves; the comparison
  rule is well-formed; `seed_count_required` is positive; any trend
  check is applicable.
- **Technique compatibility** — every referenced `technique_id` is
  registered; its factor-type matches the factor; technique params
  validate against the technique's schema; numeric level values lie in
  declared ranges.
- **Pipeline-encoding heuristics** — flags definitions shaped like a
  *sequence of steps* masquerading as a comparative experiment.
- **Cross-CG factor-model rules** — for grouped experiments, shared
  conditions are consistent and factor names don't collide across CGs.

## The four contracts

A successful compile produces an `ArtifactSet` of four contracts. Each
artifact's envelope carries a content hash and its parent's hash, so a
downstream consumer can prove what it's reading against.

**`ExperimentPlan`** — substantively the input definition plus the
envelope; the faithful design-time record.

```json
{ "envelope": { "hash": "...", "parent_hash": null, "type": "ExperimentPlan" },
  "body": { "hypothesis": "...", "factors": [...], "controlled_conditions": {...},
            "entries": [...], "validation": {...} } }
```

**`HarnessContract`** — what the harness substrate must expose. The
factor is the lone variable the harness *doesn't* fix; everything else
in `controlled_conditions` is flattened into `fixed_variables`.

```json
{ "envelope": { "hash": "...", "parent_hash": "<ExperimentPlan hash>" },
  "body": {
    "factor": { "name": "cutout_augmentation", "type": "additive" },
    "seeds": [42, 1337, 2024, 9999, 55],
    "fixed_variables": [
      { "path": "dataset.name",               "value": "cifar10" },
      { "path": "architecture.name",          "value": "resnet50" },
      { "path": "optimization.learning_rate", "value": 0.001 },
      "..."
    ],
    "required_metrics": [ { "kind": "parametric", "family": "top_k_accuracy", "parameters": { "k": 1 } } ],
    "optional_telemetry": [],
    "no_go_zones": [ "..." ] } }
```

**`TechniqueContract[]`** — one per entry, parented to the harness it
must conform to; carries the entry's technique id, params, and baseline
flag — nothing about how the result is judged.

**`ValidationConfig`** — the single input the verdict path takes besides
the raw metrics. Locked at compile time; never rewritten downstream; the
technique implementer never sees it.

```json
{ "envelope": { "hash": "...", "parent_hash": "<ExperimentPlan hash>" },
  "body": {
    "metric":      { "kind": "parametric", "family": "top_k_accuracy", "parameters": { "k": 1 } },
    "direction":   "higher_is_better",
    "aggregation": { "method": "mean" },
    "comparison":  { "rule": "compare_to_baseline", "threshold": 0.003,
                     "baseline_entry_id": "no_aug_baseline" },
    "seed_count_required": 5 } }
```

These are JSON: the pipeline serializes them to the `ArtifactStore` and
agents read them as JSON; a Tier-B integrator can dump them with
`artifact_set.to_json()`.

## `evaluate(...)`

Once raw multi-seed metrics exist — whoever produced them — the verdict
is a pure function of two inputs:

```
evaluate(
  validation_config: ValidationConfig,   # locked at compile time
  raw: RawMetrics,                       # { entry_id -> [ {seed, required, optional}, ... ] }
) -> EvaluationResult                     # verdict: pass | fail | inconclusive
```

No LLM, no agent, no metric substitution. It (1) aggregates each entry's
seed runs by the configured rule (`mean`, …), (2) compares each
non-baseline entry against the baseline by the configured rule
(`compare_to_baseline`) and threshold, and (3) emits a verdict with the
structured intermediates attached. Two evaluations of the same
`ValidationConfig` against the same `RawMetrics` produce byte-equal
results — which is what makes "the implementer never sees the verdict
criterion" mean something: the criterion *is* the `ValidationConfig`,
and the implementation only ever sees the `HarnessContract` and its own
`TechniqueContract`.

## The pipeline layer

`smai-core` defines *what* a sound comparative experiment looks like; the
pipeline layer is *how* you get from contracts to raw metrics without
writing the harness or technique code yourself. Four pieces:

- **Agents** (`smai-agents`) — six fleet roles over a custom
  Bedrock-Converse-style loop, each with its own per-task model
  selection. The **planner** drafts an `ExperimentDefinition` from a
  technique description; the **harness builder** writes the harness
  substrate from the `HarnessContract`; the **technique implementer**
  fills in each `TechniqueContract`; the **code reviewer** reads the
  contracts and the diffs and flags drift; the **contextual evaluator**
  annotates the numeric verdict with discussion; the **supervisor** sits
  over multi-CG orchestration.
- **Orchestrator** (`smai-orchestrator`) — a generic pipeline-spec
  engine: state machines + a worker loop with leasing + a checkpointer.
  The same engine drives four specs: CG execution (the inner loop where
  harness and technique code are produced and reviewed), the proposal
  pipeline (primary input, with a human gate at `designed`), paper
  ingestion (a supporting utility), and a `RunRecord` sub-spec the CG
  loop inlines per seed. Multi-worker deployments lease + fair-schedule;
  single-worker dev mode skips the leasing fast-path.
- **Runtime** (`smai-runtime`) — the fixed templates and
  manifest-driven type checks the harness builder emits against; it
  sits between the agents and `Compute`.
- **Plugins** — the boundary between the pipeline and the outside world.
  Four Python `Protocol`s: `LlmProvider` (where agent calls go),
  `MetadataStore` (where pipeline-tracking records live), `ArtifactStore`
  (where the four contracts and the harness/technique code live),
  `Compute` (where seed runs execute). Reference implementations for
  local-dev and production ship for each — see the matrix below.

## CLI verbs

```
smai dev               Boot the laptop deployment (SQLite + LocalFs + LocalGpu + Bedrock;
                       in-band worker; tighter poll; headless).
smai start             Boot the production deployment (out-of-band worker; explicit plugin
                       selections required; refuses incomplete config or stale schema).
smai ui                Boot the API + SPA process. Auto-detects --with-worker from the plugin
                       shape (sqlite+localfs → on; anything else → off).
smai run <yaml>        Compile + register a CG; optional --watch polls until terminal.
smai submit-proposal   Primary input verb. Submit a novel-technique description; the planner
  <description>        drafts the ExperimentDefinition.
smai approve-proposal  Human gate at `designed`. Approval atomically registers 1–N CGs.
  <id> / reject-proposal
smai ingest <arxiv-id> Supporting input verb. Fetch + parse + screen + plan + register
                       paper-derived TechniqueRefs. Rarely needed in default workflows.
smai status [<id>]     Read pipeline-tracking state from MetadataStore; optional --watch.
smai compile <yaml>    Methodology-only: emit the four contracts to disk or stdout. Never
                       touches MetadataStore or Compute.
smai migrate           Apply schema migrations (Alembic-backed per MetadataStore plugin).
                       --check / --dry-run / --prune.
smai verify            Plugin-ping pre-flight: structured PASS/FAIL per plugin.
smai init              Scaffold a smai.yaml with sensible defaults and inline comments.
smai plugins           List discovered plugins per interface and the selected one.
smai version           Print versions of smai, smai-core, and loaded plugin packages.
smai serve             Deprecated in v2 — use `smai ui`. Read-only dashboard; source-tree
                       removal scheduled for v2.1.
```

## Plugin matrix

Four pluggable boundaries; every reference implementation ships in this
repo, so the OSS package is self-contained. Discovery is entry-point
based (the dbt-adapter pattern): namespaces `smai.llm_providers`,
`smai.metadata_stores`, `smai.artifact_stores`, `smai.computes`. `smai
plugins` walks all four.

| Interface | Plugin | What it is |
|---|---|---|
| `LlmProvider` | `smai-llm-bedrock` | AWS Bedrock Converse; AWS credential chain + `model_id`; caching via `cachePoint`. (`smai dev` default.) |
| | `smai-llm-anthropic` | Anthropic SDK; `ANTHROPIC_API_KEY`; caching via `cache_control: ephemeral`. |
| | `smai-llm-openai` | OpenAI SDK; `OPENAI_API_KEY`; `supports_caching=False` (server-side caching is opaque to callers). |
| `MetadataStore` | `smai-store-sqlite` | Single-file, zero-config; SQLAlchemy 2.0 async Core + aiosqlite. (`smai dev` default.) |
| | `smai-store-postgres` | Production self-host; same Core schema as SQLite; advisory-lock fast path on lease acquisition; opt-in `tenant_aware=True`. |
| `ArtifactStore` | `smai-artifacts-localfs` | Root-rooted local filesystem; no presigned URLs. (`smai dev` default.) |
| | `smai-artifacts-s3` | BYO bucket; SigV4 presigned URLs; `boto3` + `asyncio.to_thread`. |
| `Compute` | `smai-compute-localgpu` | Local Docker subprocess on the host GPU; ships agent + runtime Dockerfiles. (`smai dev` default.) |
| | `smai-compute-modal` | Modal Sandboxes; sync SDK + `asyncio.to_thread`; GPU type via `**plugin_options`. |
| | `smai-compute-runpod` | RunPod Pods API over raw `httpx`; six-tier GPU dispatch table. |

A new plugin lives in its own subdirectory under `plugins/`, declares
its entry point in `pyproject.toml`, and subclasses the conformance test
base from `smai-core` (`smai_core.plugins.conformance`) — override one
`make_<interface>()` factory and inherit the contract suite. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the plugin-author guide.

## Repo layout

`uv` workspace; every package and plugin is its own `pyproject.toml` /
`src/` / `tests/` tree.

```
smai/
├── packages/
│   ├── smai-core/            # methodology layer: data model, DSL + compiler, contract
│   │                         # artifacts, mechanical evaluator, four plugin Protocols.
│   │                         # Allowlisted deps: pydantic, jsonschema, stdlib.
│   ├── smai-runtime/         # harness/technique runtime + fixed templates + manifest checks.
│   ├── smai-agents/          # custom Bedrock-Converse-style agent loop; six fleet roles.
│   ├── smai-orchestrator/    # engine + pipeline-spec format + worker loop + checkpointer;
│   │                         # the four PipelineSpec instances; Alembic migrations.
│   ├── smai-cli/             # CLI verbs, config layering, RuntimeConfig, plugin
│   │                         # instantiation, in-band Runtime.
│   ├── smai-api-spec/        # shared HTTP contract: Pydantic models + URL constants +
│   │                         # error taxonomy. Imported by smai-api and Yaarp's hosted API.
│   ├── smai-api-conformance/ # parameterizable pytest suite asserting wire-shape parity.
│   ├── smai-events/          # EventChannel Protocol + in-process pub/sub; drives SSE.
│   ├── smai-api/             # FastAPI implementation of the smai-api-spec contract + SSE.
│   ├── smai-ui/              # Python wrapper for the React SPA bundle (Hatch build hook).
│   └── smai/                 # umbrella package — currently a stub; eventual
│                             # `from smai import Runtime` re-export surface.
├── apps/
│   └── ui/                   # React 19 + Vite 6 + TanStack Router/Query + Tailwind 4 SPA;
│                             # OpenAPI-codegen'd typed client; SSE-as-cache-invalidator.
├── plugins/                  # the four Protocol implementations (see Plugin matrix).
├── tests/
│   ├── integration/          # cross-package integration + smoke E2E tests.
│   └── fixtures/             # shared YAML experiment fixtures.
├── tools/check_deps.py       # methodology-atomicity dependency lint (DEC-029 enforcement).
├── pyproject.toml            # uv workspace root + dev deps + pytest config.
└── LICENSE                   # Apache 2.0
```

`smai-core` depends on nothing inside the workspace; pipeline packages
depend on `smai-core`; plugins depend on `smai-core` (and `MetadataStore`
plugins additionally on `smai-orchestrator`'s record types — the one
plugin→pipeline edge `tools/check_deps.py` allows); the CLI depends on
everything. Full layout, dependency graph, dev setup, the five CI gates,
and project conventions are in [`CONTRIBUTING.md`](CONTRIBUTING.md);
production deployment recipes are in
[`packages/smai-cli/OPERATIONS.md`](packages/smai-cli/OPERATIONS.md).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for repo layout, local-dev
setup, the project conventions, and the plugin-author guide. The four
plugin Protocols ship parameterizable conformance test bases under
`smai_core.plugins.conformance` — a new plugin subclasses one, overrides
a factory, and inherits the contract suite.

## License

Apache 2.0 — see [`LICENSE`](LICENSE). The SDK, DSL, agent loops,
contracts, runtime, mechanical evaluator, HTTP API + SPA, and reference
plugin implementations all ship under Apache 2.0.
