# smai-core

The methodology layer of [SMAI](../../README.md): the experiment DSL, the
compiler that proves a comparative claim is even answerable, the four
immutable contract artifacts, and the deterministic mechanical evaluator.

`smai-core` is *atomic*. It holds every soundness guarantee SMAI makes,
and it holds them at the package boundary: a confounded experiment fails
to compile here, the verdict path has no LLM in it here, and the
contracts are content-hashed here. Its entire runtime dependency list is
`pydantic` + `jsonschema` + stdlib (no agents, no network, no
orchestrator), enforced by `tools/check_deps.py`. That is what lets you
drop the pipeline layer and run `smai-core` standalone against your own
infrastructure, the integration pattern SMAI calls **Tier B**.

For the conceptual "why", read two short posts:
[Where agents belong in automated research](https://henry-eigen.github.io/2026/05/18/where-agents-belong.html)
and [How SMAI works](https://henry-eigen.github.io/2026/05/13/how-yaarp-works.html).
The repo-root [`README.md`](../../README.md) is the project overview;
this file is the reference for the DSL, the compiler, the contracts, and
`evaluate()`.

> **Status: private pre-release, Apache 2.0.** APIs may break without
> notice until the publish gate is flipped.

---

## The experiment definition

An experiment definition pins a comparative claim down to a verifiable
shape: *one factor* that varies across entries, *every other variable*
held fixed, *the metric* that decides the question, and *which entry is
the baseline*. In YAML:

```yaml
kind: experiment
experiment:
  id: aug_cutout_vs_none_cifar10
  hypothesis: |
    Cutout augmentation improves CIFAR-10 top-1 accuracy over an
    identical no-augmentation pipeline, holding architecture and
    optimizer fixed.
  factor_model_id: null

  # The ONE thing that varies across entries.
  factors:
    - name: cutout_augmentation
      type: additive            # additive | substitutive

  # Everything else, pinned.
  controlled_conditions:
    dataset:      { name: cifar10, split: standard_50k_train_10k_test }
    architecture: { name: resnet50 }
    optimization: { optimizer: adamw, learning_rate: 0.001, weight_decay: 0.0001,
                    batch_size: 128, epochs: 100 }
    seeds:        [42, 1337, 2024, 9999, 55]

  # The entries being compared. Exactly one is the baseline.
  entries:
    - id: no_aug_baseline
      is_baseline: true
      level: { factor: cutout_augmentation, name: absent, technique_id: null }
    - id: cutout_treatment
      is_baseline: false
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

A document with `kind: experiment` is a single-factor comparison: one
factor varies, and `compile_experiment` returns one `ContractArtifactSet`.
A document with `kind: factor_model` is a multi-CG **factor model**: it
groups several single-factor comparisons under one research question.
Each child comparison compiles, runs, and is verified independently;
`compile_experiment` returns a `dict[experiment_id, ContractArtifactSet]`
(the factor model itself emits no artifact). The DSL discriminates on the
top-level `kind` field.

`smai-core` does not import a YAML library (it would break the dependency
allowlist). The caller parses YAML and feeds the resulting dict to
`DslDocumentAdapter.validate_python`, passing `context={"smai_mode":
"dsl"}` so the authoring-stage validation gates fire correctly.

---

## `compile_experiment(...)`

```python
compile_experiment(document, registries) -> ContractArtifactSet
compile_experiment(factor_model_document, registries) -> dict[str, ContractArtifactSet]
```

`compile_experiment` verifies the definition and, if it holds, emits the
contract artifacts. Verification runs the definition past several dozen
mechanical rules; a confounded experiment fails here, before any code
runs. Each rule produces a structured violation with a **stable code**
(for example `metric.parametric_required_parameters_present` or
`validation.threshold_sign_matches_direction`), so a downstream consumer
can branch on the code rather than parse message strings. Any
error-severity finding raises `VerificationError`.

The rules fall into eight categories:

| Category | What it checks |
|---|---|
| **Factor structure** | Exactly one factor varies; its name is unique; its type is a registered factor-type plugin; entries cover at least two levels. |
| **Entry / factor compatibility** | Every entry references the declared factor at a level it admits; no two entries reduce to the same technique + params. |
| **Controlled-conditions completeness** | Required condition fields present; the seed count matches `seed_count_required`; seeds are unique. |
| **Metric well-formedness** | The metric family is registered; required parameters are present (`k` for `top_k_accuracy`); the comparison threshold's sign agrees with `direction`; cost metrics live in optional telemetry, not the required set. |
| **Validation soundness** | The baseline entry resolves; the comparison rule is well-formed; `seed_count_required` is positive; any trend check is applicable. |
| **Technique compatibility** | Every referenced `technique_id` is registered; its factor-type matches the factor; technique params validate against the technique's schema; numeric level values lie in declared ranges. |
| **Pipeline-encoding heuristics** | Flags a definition shaped like a *sequence of steps* masquerading as a comparative experiment. |
| **Cross-CG factor-model rules** | For a `factor_model` document: shared conditions stay consistent and factor names do not collide across child CGs. |

The lower-level `verify` / `verify_to_report` entry points return a
structured `ValidationReport` (errors plus warnings) without raising;
`compile_experiment` is the convenience wrapper that returns just the
artifact set(s).

---

## The four contracts

A successful compile produces a `ContractArtifactSet` with four contract
artifacts. Every artifact carries an `ArtifactEnvelope`: a `content_hash`
(a sha256 over the artifact's canonical form) plus `parent_experiment_id`
and `registry_hashes` provenance, so any consumer (the code-reviewer
agent included) can prove what it is reading against rather than
re-reading the YAML.

| Contract | What it is |
|---|---|
| **`ExperimentPlan`** | The input definition made canonical, plus the envelope. The faithful design-time record. |
| **`HarnessContract`** | What the shared harness substrate must expose. The factor is the one variable the harness does *not* fix; everything else from `controlled_conditions` is flattened into a `fixed_variables` list of `{path, value}` pairs (`dataset.name` to `cifar10`, and so on), alongside the required metrics and the no-go zones. |
| **`TechniqueContract[]`** | One contract per entry, each parented to the harness it must conform to, carrying that entry's technique id, params, and baseline flag, and nothing about how the result is judged. |
| **`ValidationConfig`** | The metric, direction, aggregation rule, and comparison rule + threshold: the single thing `evaluate` consumes besides the raw metrics. |

`ValidationConfig` is locked at compile time, never rewritten
downstream, and (this is the load-bearing point) **the technique
implementer never sees it**. An implementation only ever sees the
`HarnessContract` and its own `TechniqueContract`, so self-grading is not
discouraged, it is structurally impossible. The `ValidationConfig` body
is the exact input to `evaluate()`:

```json
{ "envelope": { "artifact_kind": "validation_config", "content_hash": "...",
                "parent_experiment_id": "aug_cutout_vs_none_cifar10" },
  "body": {
    "metric":      { "kind": "parametric", "family": "top_k_accuracy",
                     "parameters": { "k": 1 } },
    "direction":   "higher_is_better",
    "aggregation": { "method": "mean" },
    "comparison":  { "rule": "compare_to_baseline", "threshold": 0.003,
                     "baseline_entry_id": "no_aug_baseline" },
    "seed_count_required": 5 } }
```

The artifacts are Pydantic models; serialize one with `.model_dump_json()`.

---

## `evaluate(...)`

Once raw multi-seed metrics exist, from wherever, the verdict is a pure
function of two inputs:

```python
evaluate(
    config: ValidationConfig,        # locked at compile time
    raw_metrics: RawMetrics,         # { entry_id -> { seed -> SeedRunOutcome } }
    *,
    entries: list[Entry] | None = None,   # optional, only enables trend observation
) -> EvaluationResult
```

`evaluate` has no LLM, no agent, no I/O, and no global state: the same
inputs always produce byte-equal outputs. It runs three steps:

1. **Aggregate** each entry's completed seed runs by the configured rule
   (`mean` or `median`).
2. **Compare** each non-baseline entry against the baseline by the
   configured rule (`compare_to_baseline` or `compare_to_target`) and
   threshold, respecting `direction`.
3. **Verdict.** `EvaluationResult.verdict.result` is `pass`, `fail`, or
   `inconclusive`, with structured intermediates (per-entry statistics,
   per-treatment deltas, anomalies, cost telemetry) attached.

For a multi-treatment experiment the verdict is `pass` iff at least one
treatment beats baseline by the threshold. `inconclusive` is the answer
when too few seeds completed for any entry, or when the baseline's seeds
all failed. A structurally malformed `RawMetrics` (unknown entry id,
missing required metric key, non-numeric value) raises
`RawMetricsShapeError`, deliberately distinct from `inconclusive`: that
is an integrator wiring bug, not an experiment that went badly.

---

## Tier B: using smai-core as a library

Install the workspace with `uv sync`, then compile and evaluate
yourself; the pipeline layer never enters the picture.

```python
import yaml
from smai_core import compile_experiment, evaluate, load_default_registries
from smai_core import EntryMetrics, RawMetrics, SeedRunOutcome
from smai_core.dsl import DslDocumentAdapter

# 1. Parse + validate a YAML definition, then compile to the four contracts.
doc = DslDocumentAdapter.validate_python(
    yaml.safe_load(open("tests/fixtures/experiments/cutout_on_cifar10.yaml")),
    context={"smai_mode": "dsl"},
)
artifact_set = compile_experiment(doc, load_default_registries(
    technique_registry={"tech_cutout": ...},   # supply referenced TechniqueRefs
))

# 2. Run the experiment however you like; produce raw multi-seed metrics in
#    SMAI's shape. The HarnessContract pins the per-entry runtime metric keys.
raw = RawMetrics(by_entry={
    "no_aug_baseline": EntryMetrics(
        entry_id="no_aug_baseline",
        seed_outcomes={
            42:   SeedRunOutcome(completed=True, required={"top_k_accuracy_k_1": 0.802}),
            1337: SeedRunOutcome(completed=True, required={"top_k_accuracy_k_1": 0.798}),
            # ... one SeedRunOutcome per seed
        },
    ),
    "cutout_treatment": EntryMetrics(
        entry_id="cutout_treatment",
        seed_outcomes={
            42:   SeedRunOutcome(completed=True, required={"top_k_accuracy_k_1": 0.823}),
            1337: SeedRunOutcome(completed=True, required={"top_k_accuracy_k_1": 0.819}),
        },
    ),
})

# 3. Evaluate. The locked verdict path: no LLM, no agent, deterministic.
result = evaluate(artifact_set.validation_config, raw)
print(result.verdict.result)   # "pass" | "fail" | "inconclusive"
```

`load_default_registries()` bundles the closed v1 metric registry, the
`mean` / `median` aggregation rules, the `compare_to_baseline` /
`compare_to_target` comparison rules, and the built-in factor-type
plugins. The technique registry is *input* to the methodology layer, not
state it owns: pass referenced `TechniqueRef`s in via the
`technique_registry` keyword.

`SeedRunOutcome` carries `completed: bool` to disambiguate "this run
failed" from "this seed has not reported yet". A failed seed sets
`completed=False` and a `failure_reason` string; `required` is then left
unset.

---

## Factor-type plugins

A *factor type* (`additive`, `substitutive`) drives the entry / factor
compatibility checks for a factor. `smai-core` ships both built-in, and
exposes the `FactorTypePlugin` Protocol as an extension seam: a Tier-B
integrator can register a new factor type without forking `smai-core`.

Discovery is entry-point based, under the `smai.factor_types` group:

```toml
[project.entry-points."smai.factor_types"]
my_factor_type = "my_package.factor_types:plugin"
```

A `FactorTypePlugin` declares a unique `name`, a human-readable
`description`, and a single `validate(experiment, registries) ->
list[ValidationError]` method that emits the plugin's full check list.
`load_builtin_factor_type_plugins()` discovers every registered plugin at
startup; a name collision is a startup error.

---

## License

Apache 2.0. See [`LICENSE`](../../LICENSE) at the repo root.
