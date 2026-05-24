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

from smai_runtime import (
    ADMISSIBLE_PATTERNS_FOR_KEY,
    COMPONENT_FIELD_FOR_KEY,
    HarnessComponents,
)

# Local-relative path under the workspace; the agent reads it via
# ``read_file("contracts/harness_api_reference.md")``.
HARNESS_API_REFERENCE_FILENAME = "harness_api_reference.md"
WORKSPACE_HARNESS_API_REFERENCE_PATH = "contracts/" + HARNESS_API_REFERENCE_FILENAME


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

1. Read the contracts in `contracts/` (`harness_contract.json`).
2. Write `harness/__init__.py` (the three ABI functions above) plus any
   helper modules under `harness/`, `techniques/baseline.py`, and
   `config.yaml`.
3. Validate with a short capped run via the `run_experiment` tool.
4. Once validation passes, emit the manifest with `emit_harness_manifest`.
"""


__all__ = [
    "HARNESS_API_REFERENCE_FILENAME",
    "WORKSPACE_HARNESS_API_REFERENCE_PATH",
    "build_harness_api_reference",
]
