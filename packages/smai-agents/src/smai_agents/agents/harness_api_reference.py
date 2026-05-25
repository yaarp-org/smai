"""Harness-API reference generator for the harness-builder agent.

The harness builder must implement a small, closed ABI: the entry-point
functions :mod:`smai_runtime.runner` imports from ``harness/__init__.py``
(``10-runtime-and-templates.md`` §8.2) and the
:class:`smai_runtime.HarnessComponents` bundle ``build_harness`` returns.

The framework source lives in the installed ``smai_runtime`` package,
*outside* the agent's workspace sandbox — the file tools
(:mod:`smai_agents.std_tools.files`) reject every out-of-workspace path.
So :func:`run_harness_builder_session` stages the output of
:func:`build_harness_api_reference` into the workspace as
``contracts/harness_api_reference.md`` for the agent to ``read_file``.

The reference is GENERATED, not hand-copied: the
:class:`HarnessComponents` field set and the manifest extension-point
mapping tables are introspected from :mod:`smai_runtime` at call time, so
they cannot drift from the framework. The three ABI function signatures
are documented inline against the byte-stable ``smai_runtime.runner``
dispatch sequence (§8.2 steps 5 / 9 / 10) — that call sequence is itself
no-go-zone-hashed, so it is the stable surface to pin against.
"""

from __future__ import annotations

from smai_core import HarnessContract
from smai_core.artifacts.harness_contract import HarnessContractBody
from smai_runtime import (
    ADMISSIBLE_PATTERNS_FOR_KEY,
    COMPONENT_FIELD_FOR_KEY,
    HARNESS_CONTRACT_FILENAME,
    RUNTIME_TEMPLATE_VERSION,
    HarnessComponents,
)

# Local-relative path under the workspace; the agent reads it via
# ``read_file("contracts/harness_api_reference.md")``.
HARNESS_API_REFERENCE_FILENAME = "harness_api_reference.md"
WORKSPACE_HARNESS_API_REFERENCE_PATH = "contracts/" + HARNESS_API_REFERENCE_FILENAME

# Workspace-relative path to the locked HarnessContract — derived from
# the runtime's canonical filename constant so a rename in
# smai_runtime.runtime_files cascades here without manual sync.
WORKSPACE_HARNESS_CONTRACT_PATH = "contracts/" + HARNESS_CONTRACT_FILENAME

# Dotted accessor on the contract for the locked per-experiment config.
# Derived from the actual HarnessContract model attributes via
# introspection so renames surface at test time, not at agent time.
_LOCKED_CONFIG_FIELD = "fixed_variables"
assert _LOCKED_CONFIG_FIELD in HarnessContractBody.model_fields, (
    f"HarnessContractBody no longer has a {_LOCKED_CONFIG_FIELD!r} field — "
    "harness_api_reference.py needs to track the rename."
)
assert "body" in HarnessContract.model_fields, (
    "HarnessContract no longer exposes the body envelope — "
    "harness_api_reference.py needs to track the rename."
)
HARNESS_CONTRACT_LOCKED_CONFIG_PATH = f"body.{_LOCKED_CONFIG_FIELD}"


def _clean_type(annotation: object) -> str:
    """Render a type annotation without noisy module prefixes."""
    return str(annotation).replace("collections.abc.", "").replace("typing.", "")


def _render_components_fields() -> str:
    lines: list[str] = []
    for name, field in HarnessComponents.model_fields.items():
        type_repr = _clean_type(field.annotation)
        default = "[]" if field.default == [] else repr(field.default)
        lines.append(f"- `{name}: {type_repr}` (default `{default}`)")
    return "\n".join(lines)


def _render_extension_points() -> str:
    lines: list[str] = []
    for key in COMPONENT_FIELD_FOR_KEY:
        field = COMPONENT_FIELD_FOR_KEY[key]
        patterns = ", ".join(sorted(ADMISSIBLE_PATTERNS_FOR_KEY[key]))
        lines.append(f"- `{key}` → `HarnessComponents.{field}` (patterns: {patterns})")
    return "\n".join(lines)


def build_harness_api_reference() -> str:
    """Generate the harness-API reference markdown for the agent workspace.

    Introspects :class:`smai_runtime.HarnessComponents` and the
    extension-point mapping tables so the field set and key→field mapping
    track the framework automatically. Returns a markdown document.
    """
    return f"""# Harness API reference (generated from smai_runtime)

This file is generated from the installed `smai_runtime` framework. The
framework source itself is OUTSIDE this workspace and cannot be read with
the file tools — use this reference, do not attempt to open
`smai_runtime/*.py`.

## What you must produce

Build `harness/__init__.py` exposing exactly three module-level
functions (the ABI `smai_runtime.runner` imports by name, §8.2):

```python
def build_harness(config: dict) -> HarnessComponents:
    \"\"\"Construct the experiment harness from `config` and return a
    HarnessComponents bundle. Called once per run (runner step 5).\"\"\"

def run_training_loop(components: HarnessComponents, config: dict, seed: int):
    \"\"\"Run the training procedure and return the trained model. The
    `components` passed in have already had the technique's output
    spliced in by the integrator (runner step 9).\"\"\"

def evaluate(trained_model, components: HarnessComponents, config: dict) -> dict:
    \"\"\"Evaluate `trained_model` and return a metrics dict — plain
    JSON-serializable values keyed by metric name (runner step 10).\"\"\"
```

`config` is a plain `dict[str, Any]` the runtime assembles from the
contracts (it carries `epochs`, `subset`, seed-related keys, etc.).

## HarnessComponents

`build_harness` must return an instance of `smai_runtime.HarnessComponents`
— a Pydantic model with a CLOSED v1 field set (no extra fields allowed):

{_render_components_fields()}

Import it with `from smai_runtime import HarnessComponents`.

## Manifest extension points

The HarnessAPIManifest declares extension points; the integrator splices
technique outputs into the matching `HarnessComponents` field. The closed
v1 key → field mapping (and the integration patterns each key admits):

{_render_extension_points()}

For an **additive** factor every extension point must default to a
working no-op so the baseline (`technique_id=null`) runs the harness
as-is; declare those manifest extension points `optional=true`. For a
**substitutive** factor the slot is mandatory — declare them
`optional=false`.

## Workflow

### 1. Read the locked config from the harness contract

The locked per-experiment config (dataset, architecture, optimization,
seeds, compute) lives in `{WORKSPACE_HARNESS_CONTRACT_PATH}` at
`HarnessContract.{HARNESS_CONTRACT_LOCKED_CONFIG_PATH}`, a list of
`{{ "path": <jsonpath>, "value": <value>, "type_hint": ... }}` entries.
Read THIS file (not `experiment_plan.json` or any other path); it is
the locked source of truth. The compiler may also have emitted an
`experiment_plan.json` for human review somewhere up the pipeline, but
that file is OUTSIDE this workspace and unreadable; the harness contract
is the in-workspace authoritative artifact.

### 2. Write the harness

Write `harness/__init__.py` (the three ABI functions above) plus any
helper modules under `harness/`, `techniques/baseline.py`, and
`config.yaml`. Use the values from
`HarnessContract.{HARNESS_CONTRACT_LOCKED_CONFIG_PATH}` to populate
`config.yaml` (or wherever your `build_harness` reads its defaults
from); substituting in a guess will be caught when the runtime later
re-validates the contract.

### 3. Validate, THEN emit the manifest (in that order)

The `emit_harness_manifest` tool refuses to write the manifest unless
`validation_results.json` (which the runtime writes at the workspace
root on the success path of a `--mode validation` run — the
`run_experiment` tool always passes that flag) exists and reports
`passed: true`. Run `run_experiment` FIRST; only after that returns
`completed: true` should you call `emit_harness_manifest`. Calling
`emit_harness_manifest` before a passing validation produces a tool
error of the form *"the agent must run a passing validation via
run_experiment before emitting the manifest"*.

### 4. Manifest field values

When you do call `emit_harness_manifest` these three fields are easy to
get wrong:

- **`runtime_template_version`** must equal exactly
  `"{RUNTIME_TEMPLATE_VERSION}"` (the current runtime's template
  version, generated from the installed `smai_runtime` package).
  Anything else is rejected by the tool BEFORE any artifact write.
- **`harness_version_hash`** must be recomputed from the CURRENT
  on-disk `harness/` directory contents on every emission attempt. Do
  NOT reuse a hash from a previous turn; between turns you may have
  edited a file. The integrator's `compute_harness_version_hash`
  function (over the workspace `harness/` directory) is the only way to
  get the right value, and the tool re-derives the expected hash itself
  every call, so a stale hash always fails. The validation message tells
  you the expected value when the input is wrong.
- **`parent_harness_contract_hash`** must equal
  `envelope.content_hash` of the harness contract you read in step 1.

### 5. Tool quick reference

The standard file-operation tools take structured Pydantic inputs (NOT
shell commands). Common confusions:

- `list_files` takes `path` (a directory; optional, defaults to the
  workspace root) and `glob` (a filename glob, optional). It does NOT
  take a `command` parameter; do not pass shell argv like
  `command="find . -name ..."`.
- `search` takes `pattern` (regex) plus an optional `path` that must be
  a DIRECTORY (it walks the directory recursively). If you want to grep
  a single file, use `read_file` followed by your own filtering.
- `edit_file` takes three REQUIRED fields: `path` (the file to edit),
  `old_string` (exact text to find — must occur exactly once;
  whitespace and indentation count), and `new_string` (the
  replacement). Omitting any of them — most often `path` — surfaces as
  a Pydantic validation error and the same call shape will keep
  failing until corrected. To create a new file use `write_file`, not
  `edit_file`.
"""


__all__ = [
    "HARNESS_API_REFERENCE_FILENAME",
    "HARNESS_CONTRACT_LOCKED_CONFIG_PATH",
    "WORKSPACE_HARNESS_API_REFERENCE_PATH",
    "WORKSPACE_HARNESS_CONTRACT_PATH",
    "build_harness_api_reference",
]
