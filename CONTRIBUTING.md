# Contributing

This is a uv-managed Python workspace. The notes below cover local development tooling. Plugin-author and contribution-workflow guidance lands in a later task.

## Python version

3.11 or newer. Every workspace member pins `requires-python = ">=3.11"`.

## Local dev setup

```bash
uv sync
```

This resolves the workspace, creates `.venv/`, and installs every package in editable mode along with the dev tools (`pytest`, `pytest-asyncio`, `ruff`, `pyright`).

## Running tests

From the workspace root:

```bash
pytest
```

Pytest is configured to discover tests under `packages/` and `plugins/`. To run a single package:

```bash
pytest packages/smai-core/tests
```

`pytest-asyncio` runs in `auto` mode, so `async def` test functions are picked up without per-test markers.

## Linting

Ruff handles lint and formatting at the workspace level. The selected rule set is `E`, `F`, `I`, `W`, `B`, `UP` and the line length is 100 characters — long enough to keep type annotations and Pydantic field declarations on one line, short enough to stay readable in side-by-side diffs.

```bash
ruff check .              # lint
ruff format --check .     # formatting check
ruff format .             # auto-fix formatting
```

## Type-checking

Pyright runs over `packages/` and `plugins/` from the workspace root:

```bash
pyright
```

`smai-core` runs in `strict` mode; the rest of the workspace runs in `standard` mode. `smai-core` is the user-visible methodology API (per DEC-029), so its type surface is held to a higher bar than the pipeline-layer packages.

## Dependency-allowlist lint

`tools/check_deps.py` mechanically enforces the methodology-atomic / pipeline-composable boundary at the package level. Three rules:

1. `smai-core`'s declared runtime dependencies stay on the allowlist (Pydantic + a JSON Schema validator + standard library).
2. No `.py` file under `packages/smai-core/src/` imports a pipeline package (`smai_agents`, `smai_orchestrator`, `smai_runtime`, `smai_cli`) or any plugin package (`smai_llm_*`, `smai_store_*`, `smai_artifacts_*`, `smai_compute_*`).
3. No `plugins/smai-*` package depends on or imports a pipeline package.

The lint exists because `00-vision.md` §4 principle #2 — "methodology atomicity must be enforced at the package boundary, not assumed" — needs a load-bearing mechanical check, not a code-review convention. DEC-029 makes the methodology layer (`smai-core`) a hard package boundary so `pip install smai-core` cannot silently drag in pipeline dependencies; this lint is what makes that boundary real.

```bash
python tools/check_deps.py             # exit 0 on pass, 1 on fail
python tools/check_deps.py --verbose   # also log what was checked
```

The lint runs in CI between Pyright and Pytest. It uses only the standard library so it can run before any workspace dependency is installed.
