# smai-runtime

Harness/technique Python runtime, fixed-template integration layer, and
`HarnessAPIManifest` per `designs/smai/10-runtime-and-templates.md`.

## What ships

- **Workspace** (`smai_runtime.workspace`) — layout (`harness/`, `techniques/`,
  `contracts/`), materializer, contract-loader.
- **Fixed templates** (`smai_runtime.templates._files`) — byte-stable
  `experiment.py` and `techniques/__init__.py` shipped as resources, copied
  into the workspace and hash-checked at run start.
- **`HarnessAPIManifest`** (`smai_runtime.manifest`) — pipeline-layer
  post-build artifact with canonicalization-and-hash logic mirroring
  `smai_core.canonical_json`.
- **Manifest-driven type check** (`smai_runtime.type_check`) — structural
  check on `apply()` output against the manifest's `type_signature`s.
- **No-go-zone hash check** (`smai_runtime.no_go_zone`) — fails closed if a
  fixed-template file is modified or missing.
- **Integrator** (`smai_runtime.integrator`) — splices technique outputs
  into `HarnessComponents` via the closed v1 integration patterns.
- **Metric-emission contract** (`smai_runtime.metrics`) — produces
  `RawMetrics`-compatible output that `smai_core.evaluate` consumes.
- **Factor-aware helpers** (`smai_runtime.factor_aware`) — additive baseline
  detection, substitutive-entry sanity check.
- **Runner** (`smai_runtime.runner.run`) — the substance behind the fixed
  `experiment.py` template; orchestrates the §3.4 startup sequence.

## Dependencies

- `smai-core` (methodology layer; provides `HarnessContract`,
  `TechniqueContract`, `RawMetrics`, canonical hashing).
- `pydantic>=2`.

The full ML stack (`torch`, `numpy`, etc. per §8.5) is a substrate guarantee,
not a hard install requirement: `seed_everything` lazily imports torch /
numpy so smai-runtime is importable in test environments without them.
