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
