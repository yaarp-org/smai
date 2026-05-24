# SMAI

> Methodology-as-infrastructure for comparative-claim ML research.
> Definition in, verdict out, with the metric, the factor structure, and
> the verdict path locked at compile time.

SMAI is the experiment-execution stage that an outer auto-research
pipeline (an "AI scientist" loop) hands off to when its question turns
**comparative**: *does technique X beat technique Y under matched
conditions?* It is not an idea generator, not a writeup tool, and not a
hill-climbing optimizer; those belong to the outer loop. SMAI takes a
comparative claim, verifies it is even answerable, runs the experiment,
and returns a verdict the agent that did the work could not have nudged.

It does that by compiling the experiment before any code runs. A
confounded comparison fails to compile. The metric, the aggregation
rule, and the verdict path are fixed at compile time, and no LLM sits
anywhere between the raw numbers and the verdict, so an implementing
agent cannot grade its own work.

Two short companion posts lay out the reasoning behind this design,
where mechanical checking ends and agent reasoning must begin:

- [Where agents belong in automated research](https://henry-eigen.github.io/2026/05/18/where-agents-belong.html)
- [How SMAI works](https://henry-eigen.github.io/2026/05/13/how-yaarp-works.html)

> **Status: private pre-release.** APIs may break without notice. PyPI
> distribution and a public docs site are gated on an explicit decision;
> until then, install from the workspace with `uv sync`. Apache 2.0
> licensed.

---

## Table of Contents

- [Why SMAI](#why-smai)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Using SMAI](#using-smai)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Plugins](#plugins)
- [CLI reference](#cli-reference)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Why SMAI

A long-horizon research agent is a stack of LLM turns, and any turn is
cheap to nudge toward a result that looks better than it is. SMAI closes
that off for the comparative stretch of the work without taking the
open-endedness away: the outer agent still does all the creative work,
and SMAI owns only the stretch where a claim becomes a verdict. What
that buys is a short list of structural guarantees, not a feature list:

- **Confounded experiments fail to compile.** Factor exclusivity,
  control completeness, metric well-formedness, comparability of the
  entries, and "this looks like a sequential pipeline pretending to be
  an experiment" are all checked mechanically, before any code runs.
- **The verdict path has no LLM in it.** The metric is fixed at compile
  time, the aggregation rule is fixed, and the agent that wrote an
  implementation never sees the verdict criterion. Self-grading is not
  discouraged, it is structurally impossible.
- **The contracts are the locked surface.** Compiling a definition emits
  four immutable, content-hashed JSON artifacts. Agents read them;
  nothing rewrites them mid-flight. The audit trail is reproducible by
  construction.

The companion posts linked above explain why these are the right
guarantees, and where they sit relative to the parts that must stay in
an agent's hands.

## Installation

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
git clone https://github.com/yaarp-org/smai.git && cd smai
uv sync                 # install the workspace (editable) + dev deps
uv run pytest           # run the test suite
```

There is no PyPI release yet (see the status note above); the `uv`
workspace installs every package and plugin editable.

## Quick start

Boot the laptop deployment with the web UI:

```bash
# Dev defaults: SQLite + local filesystem + local-GPU Docker + Bedrock.
$ uv run smai ui
smai ui: starting in-process worker (sqlite metadata store detected).
         Listening on http://127.0.0.1:8000/
```

Open `http://127.0.0.1:8000/`. In another terminal, submit an experiment
and watch it run:

```bash
$ uv run smai run tests/fixtures/experiments/cutout_on_cifar10.yaml --watch
cg_01J7PA8K2X9F4ZB6QH4N0P1234: complete
```

`smai dev` is the headless equivalent of `smai ui` (same plugins, no API
or SPA). `smai compile experiment.yaml` runs only the methodology layer,
emitting the four contracts to stdout without touching the metadata
store or compute. The dev defaults dispatch real agent and compute
calls, so they need credentials and a local Docker setup; the
prerequisites are in [Configuration](#configuration) and
[`packages/smai-cli/README.md`](packages/smai-cli/README.md).

## Using SMAI

SMAI has two integration patterns, and the seam between them is
deliberate:

| | What you get | What you import |
|---|---|---|
| **Tier A, deep delegation** | Hand SMAI a definition (or a plain-English technique description, and let the planner draft one). Read back a verdict and the artifact set. The full pipeline runs: agents, orchestrator, runtime, plugins. | the `smai` CLI, or `Runtime` programmatically |
| **Tier B, methodology as library** | Compile your own definitions, run the experiments on your own infrastructure, call the evaluator for the verdict. The pipeline layer never enters the picture. | `smai_core` alone |

### Tier A: programmatic

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

The primary input path is proposals, not raw definitions:
`runtime.proposals.submit(...)` hands the planner a technique
description, it drafts a definition into the `designed` state, and a
human gate approves it, which registers the comparison groups.
Production deployment (out-of-band worker, Postgres + S3 + remote
compute) uses `Runtime.start_worker(...)`; recipes are in
[`packages/smai-cli/OPERATIONS.md`](packages/smai-cli/OPERATIONS.md).

### Tier B: methodology as library

`smai_core` is the methodology layer on its own: `compile_experiment(...)`
turns a definition into the four contracts, `evaluate(...)` turns raw
multi-seed metrics into a verdict. Its entire dependency list is
`pydantic` + `jsonschema` + stdlib, so it drops into any infrastructure.
The DSL, the compiler checks, the contract shapes, and a worked Tier-B
example are documented in
[`packages/smai-core/README.md`](packages/smai-core/README.md).

## Configuration

`smai dev` and `smai ui` boot with no config file. Past that, settings
layer: in-code defaults, then `smai.yaml`, then `SMAI_*` env vars, then
CLI flags, each overriding the last. A minimal `smai.yaml`:

```yaml
plugins:
  llm_provider:   bedrock          # bedrock | anthropic | openai
  metadata_store: sqlite           # sqlite | postgres
  artifact_store: localfs          # localfs | s3
  compute:        localgpu         # localgpu | modal | runpod

  llm_provider_config:
    region: us-east-1
    model_id: us.anthropic.claude-opus-4-6-v1
```

The same file works for `smai dev` and `smai start`; the only difference
is which fields are defaulted versus required. Swapping
`sqlite`->`postgres`, `localfs`->`s3`, and `localgpu`->`modal`/`runpod`
is the entire local-to-production migration, a config change rather than
a code change. `smai init` scaffolds an annotated `smai.yaml`;
`smai verify` pings every configured plugin before a run.

The full reference (the config search order, the `SMAI_*` env
conventions, the per-plugin `*_config` keys, per-role agent models, and
where state lives) is in
[`packages/smai-cli/README.md`](packages/smai-cli/README.md). Each
plugin's own README documents its credentials; see [Plugins](#plugins).

## Architecture

SMAI is two layers, and they have opposite shapes.

**The methodology layer is one library, `smai-core`.** No agents, no
network, no orchestrator. It takes an experiment definition, verifies
the claim is answerable, emits the four immutable contracts, and (once
raw multi-seed metrics exist, from wherever) produces the verdict. It is
*atomic*: pull a piece out and the others' guarantees break, so it ships
as one tightly-scoped package, and it is where every guarantee in
[Why SMAI](#why-smai) is enforced.

**The pipeline layer is everything that produces those raw metrics for
you.** Agents draft the definition, write the harness, fill in each
technique implementation, and review the diffs; an orchestrator drives
the state machines and the worker loop; four plugin interfaces wire it
to an LLM, a metadata store, an artifact store, and compute. It is
*composable*: every piece is independently useful, which is exactly why
you can drop it and run `smai-core` against your own infrastructure.

```
   experiment definition          one factor varies, everything else is
   (the SMAI DSL)                 fixed, the metric and baseline named
          |
          v
   +-- smai-core ----------------------------------------------+
   |  compile_experiment()   a confounded comparison fails to   |
   |          |              compile, before any code is run    |
   |          v                                                 |
   |  four content-hashed contracts:                            |
   |  ExperimentPlan . HarnessContract . TechniqueContract[]    |
   |  . ValidationConfig (the verdict path; never shown to the  |
   |    technique implementer)                                  |
   +----------+-------------------------------------------------+
              v
   +-- pipeline layer -----------------------------------------+
   |  agents implement the harness and each technique against  |
   |  the contracts; the orchestrator runs every seed on       |
   |  compute and collects raw multi-seed metrics              |
   +----------+------------------------------------------------+
              v
   +-- smai-core ----------------------------------------------+
   |  evaluate()   a pure function (no LLM, deterministic)     |
   |  applies the locked ValidationConfig to the raw metrics   |
   +----------+------------------------------------------------+
              v
   verdict   (pass / fail / inconclusive)
```

The repo is a `uv` workspace; each package is its own `pyproject.toml` /
`src/` / `tests/` tree:

| Package | Role |
|---|---|
| [`smai-core`](packages/smai-core/README.md) | Methodology layer: data model, DSL + compiler, the four contracts, the mechanical evaluator, the four plugin Protocols. |
| `smai-runtime` | Harness/technique runtime, fixed templates, the `HarnessAPIManifest`, the no-go-zone hash check. |
| `smai-agents` | Custom agent loop and the six fleet roles (planner, harness builder, technique implementer, code reviewer, contextual evaluator, supervisor). |
| `smai-orchestrator` | The pipeline engine: state machines, worker loop, checkpointer, the four `PipelineSpec` instances, Alembic migrations. |
| [`smai-cli`](packages/smai-cli/README.md) | The 16 `smai` verbs, config layering, `RuntimeConfig`, the in-band `Runtime`. |
| `smai-api`, `smai-api-spec`, `smai-api-conformance` | The JSON HTTP API, its shared Pydantic contract, and the wire-shape conformance suite. |
| `smai-events` | The `EventChannel` Protocol and in-process pub/sub that drive the SSE channel. |
| `smai-ui` | Python wrapper that bundles the React SPA (under `apps/ui/`) into the API wheel. |

`smai-core` depends on nothing else in the workspace; the pipeline
packages depend on `smai-core`; the CLI depends on everything. The full
dependency graph and the repo layout are in
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Plugins

SMAI has four pluggable boundaries. Every reference implementation ships
in this repo, so the OSS package is self-contained. Discovery is
entry-point based (the dbt-adapter pattern). Each plugin's README
documents its config keys and credentials.

| Interface | Plugin | What it is |
|---|---|---|
| `LlmProvider` | [`smai-llm-bedrock`](plugins/smai-llm-bedrock/README.md) | AWS Bedrock Converse. `smai dev` default. |
| | [`smai-llm-anthropic`](plugins/smai-llm-anthropic/README.md) | Anthropic SDK. |
| | [`smai-llm-openai`](plugins/smai-llm-openai/README.md) | OpenAI SDK. |
| `MetadataStore` | [`smai-store-sqlite`](plugins/smai-store-sqlite/README.md) | Single-file SQLite. `smai dev` default. |
| | [`smai-store-postgres`](plugins/smai-store-postgres/README.md) | Postgres for production self-host. |
| `ArtifactStore` | [`smai-artifacts-localfs`](plugins/smai-artifacts-localfs/README.md) | Local filesystem. `smai dev` default. |
| | [`smai-artifacts-s3`](plugins/smai-artifacts-s3/README.md) | S3, bring your own bucket. |
| `Compute` | [`smai-compute-localgpu`](plugins/smai-compute-localgpu/README.md) | Local Docker on the host GPU. `smai dev` default. |
| | [`smai-compute-modal`](plugins/smai-compute-modal/README.md) | Modal Sandboxes. |
| | [`smai-compute-runpod`](plugins/smai-compute-runpod/README.md) | RunPod Pods API. |

Writing a new plugin: subclass the conformance test base from
`smai_core.plugins.conformance`, override one factory, and declare the
entry point. The plugin-author guide is in
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## CLI reference

The `smai` CLI has 16 verbs. The ones you reach for first:

```
smai dev               Boot the headless laptop deployment.
smai ui                Boot the API + SPA process (dashboard + live updates).
smai run <yaml>        Compile + register a comparison group; --watch polls to terminal.
smai submit-proposal   Submit a novel-technique description; the planner drafts the design.
smai compile <yaml>    Methodology only: emit the four contracts. No store, no compute.
smai status [<id>]     Read pipeline-tracking state for a CG, proposal, or paper.
smai verify            Ping every configured plugin; PASS/FAIL per interface.
```

The full verb table, the configuration reference, and the
`techniques.json` format are in
[`packages/smai-cli/README.md`](packages/smai-cli/README.md).

## Documentation

| | |
|---|---|
| [Where agents belong in automated research](https://henry-eigen.github.io/2026/05/18/where-agents-belong.html) | The argument: where mechanical checking ends and agent reasoning begins. |
| [How SMAI works](https://henry-eigen.github.io/2026/05/13/how-yaarp-works.html) | That argument realized as SMAI's pipeline. |
| [`packages/smai-core/README.md`](packages/smai-core/README.md) | The experiment DSL, the compiler checks, the four contracts, `evaluate()`. |
| [`packages/smai-cli/README.md`](packages/smai-cli/README.md) | The 16 CLI verbs and the full configuration reference. |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Repo layout, dependency graph, dev setup, the CI gates, the plugin-author guide. |
| [`packages/smai-cli/OPERATIONS.md`](packages/smai-cli/OPERATIONS.md) | Production deployment recipes: systemd / launchd, pool sizing, bearer-token mode. |

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for repo layout, local-dev
setup, the project conventions, and the plugin-author guide. The four
plugin Protocols ship parameterizable conformance test bases under
`smai_core.plugins.conformance`: a new plugin subclasses one, overrides a
factory, and inherits the contract suite.

## License

Apache 2.0, see [`LICENSE`](LICENSE). The SDK, DSL, agent loops,
contracts, runtime, mechanical evaluator, HTTP API, SPA, and reference
plugin implementations all ship under Apache 2.0.
