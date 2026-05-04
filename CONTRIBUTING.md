# Contributing to SMAI

This guide covers local-dev setup, project conventions, the plugin-author
workflow, and the PR / commit conventions used in this repo.

The repo is a [`uv`](https://docs.astral.sh/uv/) workspace. Every package
under `packages/` and every plugin under `plugins/` is its own
`pyproject.toml` / `src/` / `tests/` tree, installed editably via the
workspace root.

---

## Repo layout

```
smai/
├── packages/
│   ├── smai-core/              # methodology layer (DEC-029)
│   ├── smai-runtime/           # harness/technique runtime
│   ├── smai-agents/            # agent loop + six fleet roles
│   ├── smai-orchestrator/      # engine + pipeline-specs + migrations
│   ├── smai-cli/               # CLI verbs + Runtime
│   └── smai/                   # umbrella re-export package
├── plugins/                    # 9 plugins across 4 interfaces (see README)
├── tests/                      # cross-package integration tests
├── tools/check_deps.py         # methodology-atomicity dep lint
├── pyproject.toml              # uv workspace root + dev deps
├── pyrightconfig.json          # Pyright execution environments per package
└── LICENSE                     # Apache 2.0
```

The cross-package coupling is one-directional: `smai-core` depends on
nothing inside the workspace; pipeline packages depend on `smai-core`;
plugins depend on `smai-core` (and, for `MetadataStore` plugins,
`smai-orchestrator` for the pipeline-tracking record types). The CLI
depends on everything. `tools/check_deps.py` enforces this mechanically
(see "Dependency-allowlist lint" below).

---

## Local-dev setup

Requires Python 3.11+ and `uv`.

```bash
git clone <repo-url> smai
cd smai
uv sync                   # workspace + dev deps + every package editable
```

`uv sync` resolves the workspace, creates `.venv/`, and installs every
package (and every plugin) editably along with the dev tools (`pytest`,
`pytest-asyncio`, `ruff`, `pyright`, `pyyaml`, `hypothesis`, `httpx`,
`modal`).

### Running the gates locally

The five gates each run in seconds-to-tens-of-seconds against the
workspace root:

```bash
uv run pytest                            # ~1700 tests; ~30s warm
uv run pyright                           # type-check (strict on packages above + all plugins)
uv run ruff check .                      # lint
uv run ruff format --check .             # formatting check (no rewrite)
uv run python tools/check_deps.py        # dep-allowlist lint
```

To run a single package's tests:

```bash
uv run pytest packages/smai-core/tests
uv run pytest plugins/smai-llm-bedrock/tests
uv run pytest tests/integration -v
```

`pytest-asyncio` is in `auto` mode; `async def` test functions are
discovered without per-test markers.

To auto-fix ruff issues:

```bash
uv run ruff format .                     # auto-format
uv run ruff check . --fix                # auto-fix safe lints
```

### SPA + API dev workflow

The canonical local-dev pattern for the API + SPA is two terminals
(`designs/smai/12-ui-process.md` §8 / `13-frontend.md` §12.1):

```bash
# Terminal 1: API + (auto-detected) in-process worker on a sqlite/localfs config.
uv run smai ui --reload

# Terminal 2: Vite dev server with HMR; proxies /api/* to the API on :8000.
cd apps/ui && pnpm install && pnpm dev
```

`smai ui --reload` enables uvicorn auto-reload on Python source
changes; touching a `.tsx` updates the browser via Vite HMR
independently. Editing `smai.yaml` requires restarting `smai ui` (no
hot-reload; per `09-cli.md` §3).

For a UI against a remote backend (Postgres + S3), pass
`--no-worker` — workers run separately as `smai start` against the
shared backend per `12-ui-process.md` §10.3.

---

## Project conventions

### No credentials in CI

Per Task 3.G3: real-substrate credentialed tests are **local-manual
only**. They carry **two** decorators:

```python
import os
import pytest


@pytest.mark.credentialed
@pytest.mark.skipif("AWS_TEST_BUCKET" not in os.environ, reason="needs AWS bucket")
async def test_real_aws_round_trip() -> None:
    ...
```

- `@pytest.mark.credentialed` — registered in the root `pyproject.toml`.
  Marks the test as substrate-touching; CI lanes filter it out.
- `@pytest.mark.skipif("ENV" not in os.environ, ...)` — skips cleanly
  when the substrate-specific env var is absent. CI runs without
  credentials and the test is collected-but-skipped (silent green).

When you add a new credentialed test, follow this two-decorator pattern.
The env var name is plugin-specific by convention:
`AWS_TEST_BUCKET` / `MODAL_TOKEN_ID` / `ANTHROPIC_API_KEY` /
`OPENAI_API_KEY` / `RUNPOD_API_KEY` / `SMAI_POSTGRES_TEST_URL`.

### Per-task test-fixture filename hygiene

Cross-package or cross-plugin pytest fixtures live in files named
`_<task>_<purpose>.py` (e.g., `_e1_fakes.py`, `_g4_fakes.py`,
`_h1_dashboard_fakes.py`, `_modal_fakes.py`, `_runpod_fakes.py`,
`_f5_anthropic_fakes.py`). The pattern avoids `sys.path` collisions
with the per-plugin `tests/conftest.py` that adds `tests/` to the
import path — generic names like `_fakes.py` collide across sibling
plugins.

### Type-checking (Pyright)

`pyright` is configured in `pyrightconfig.json`. `smai-core` runs in
**strict mode** as a hard policy because the methodology layer is the
user-visible Tier B API surface (per DEC-029). Pipeline packages and
plugins also run under strict mode in their `executionEnvironments`
blocks; see `pyrightconfig.json` for the active list.

When you add a new package or plugin, register it in
`pyrightconfig.json` — both as a strict execution environment for its
own `src/` and on the `extraPaths` of any package that imports it
under `TYPE_CHECKING`.

### Linting and formatting (ruff)

A single `ruff` config lives in the root `pyproject.toml`. The selected
rule set is `E`, `F`, `I`, `W`, `B`, `UP`. Line length is 100 — long
enough for type annotations and Pydantic field declarations on one
line; short enough to stay readable in side-by-side diffs.

```bash
uv run ruff check .                      # lint
uv run ruff format --check .             # formatting check
uv run ruff format .                     # auto-format
```

### Dependency-allowlist lint

`tools/check_deps.py` enforces three rules:

1. `smai-core/pyproject.toml` declares only `pydantic`, `jsonschema`,
   plus stdlib (no other runtime deps). Optional dep groups (e.g.,
   `[project.optional-dependencies] conformance = ["pytest>=8.0"]`)
   don't count.
2. No file under `packages/smai-core/src/` imports a pipeline package
   (`smai_agents`, `smai_orchestrator`, `smai_runtime`, `smai_cli`) or
   any plugin package — except under `if TYPE_CHECKING:` and except for
   a narrow `smai_core/plugins/conformance/` exemption (the conformance
   suite needs the pipeline-tracking record types from
   `smai-orchestrator` at runtime; the `[conformance]` extra gates the
   import).
3. No `plugins/smai-*` package depends on or imports a pipeline
   package. `smai-orchestrator` is allowed (every `MetadataStore`
   plugin returns pipeline-tracking records from
   `smai_orchestrator.entities.tracking`); `smai-agents`,
   `smai-runtime`, `smai-cli`, and the `smai` umbrella are forbidden.

These rules are the mechanical enforcement of `00-vision.md` §4
principle #2 ("methodology atomicity must be enforced at the package
boundary, not assumed"). Run the lint before submitting:

```bash
uv run python tools/check_deps.py
uv run python tools/check_deps.py --verbose       # log what was checked
```

The lint runs in CI between Pyright and Pytest. It uses only the
standard library so it can run before any workspace dependency is
installed.

### License

Apache 2.0 — see `LICENSE` and `oss_strategy.md` §6. New code added to
this repo inherits the license; you don't need to add per-file SPDX
headers (the repo doesn't currently use them, though doing so would be
the conventional shape if the policy is ever tightened).

---

## Plugin-author guide

SMAI's four pluggable boundaries are `LlmProvider`, `MetadataStore`,
`ArtifactStore`, and `Compute`. Each is a `runtime_checkable`
[`typing.Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)
defined in `smai_core.plugins`; each ships a parameterizable
conformance test base class under `smai_core.plugins.conformance`.

The plugin-author workflow is identical across all four interfaces:

1. **Implement the Protocol** in your plugin's `src/<package>/`.
2. **Subclass the conformance base** in `tests/test_conformance.py`,
   override the single `make_<interface>()` factory, and inherit the
   contract suite.
3. **Register the entry point** in `pyproject.toml`.
4. **Add capability flags honestly** — the conformance suite asserts
   that capability flags reflect actual behavior. Lying about
   `supports_caching` / `supports_presigned_urls` / etc. is a
   conformance failure.

### `LlmProvider`

Defined in `smai_core.plugins.llm_provider`. Send normalized messages,
receive `ModelResponse`. Tool use is supported via a normalized
content-block shape; provider-specific shapes (Bedrock Converse content
blocks, Anthropic `tool_use` blocks, OpenAI `tool_calls`) are translated
in the plugin.

```python
from smai_core import (
    CacheConfig,
    LlmCapabilities,
    LlmProvider,
    ModelResponse,
    NormalizedMessage,
    ToolDefinition,
)


class MyProvider:  # implements LlmProvider Protocol structurally
    def __init__(self, *, model_id: str, api_key: str | None = None) -> None:
        self.name = "my_provider"
        self.capabilities = LlmCapabilities(
            model_id=model_id,
            supports_caching=False,        # be honest
            supports_tool_use=True,
            max_input_tokens=128_000,
            max_output_tokens=4_096,
        )
        # Construct your client here; honor the env var convention if
        # api_key is None.

    async def call(
        self,
        *,
        system: str,
        messages: list[NormalizedMessage],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        cache_config: CacheConfig | None = None,
    ) -> ModelResponse:
        # 1. Translate (system, messages, tools) into your SDK's shape.
        # 2. Call the SDK; classify HTTP-status / SDK-exception kinds.
        # 3. Translate the SDK response back into ModelResponse.
        ...
```

Conformance — `tests/test_conformance.py`:

```python
from smai_core.plugins.conformance import LlmProviderConformance
from my_provider import MyProvider


class TestMyProviderConformance(LlmProviderConformance):
    def make_provider(self):
        return MyProvider(model_id="my-model-1", openai_client=fake_client())
```

Two patterns are common across the LLM plugins shipping in-tree:

- **Sync SDK + `asyncio.to_thread`** (`smai-llm-bedrock`,
  `smai-artifacts-s3`, `smai-compute-modal`): the SDK is sync; wrap each
  call in `await asyncio.to_thread(...)` to keep the agent loop's
  async semantics.
- **Async-native SDK** (`smai-llm-anthropic`, `smai-llm-openai`):
  pass `await` through directly.

Both patterns expose a fake-client constructor seam (`anthropic_client=`
/ `openai_client=` / `bedrock_client=`) so conformance tests can run
in-process against a queue-of-canned-responses fake. The
`_conformance_inject_fault(kind: str)` hook on the provider is how the
six-kind error-class contract surfaces in the suite (rate-limit, 5xx,
auth, invalid-request, transient-then-succeed, tool-use-response).

### `MetadataStore`

Defined in `smai_core.plugins.metadata_store`. CRUD for the six
pipeline-tracking record types (`ComparisonGroupRecord`, `EntryRecord`,
`RunRecord`, `ProposalRecord`, `PaperRecord`, `FactorModelRecord`),
plus 19 scheduling queries × 5 entity kinds, plus
`acquire_lease` / `release_lease` / `extend_lease` per DEC-035 #2,
plus `count_with_in_flight_jobs` per DEC-035 #3.

Per DEC-030 / DEC-036 the substrate is **SQL-shaped only** —
`SqliteStore` and `PostgresStore` share a SQLAlchemy 2.0 async Core
schema and an Alembic env, lifted to a shared package under
`smai_orchestrator.migrations`. A new `MetadataStore` plugin should
either:

- **Reuse the shared Core schema** (the recommended path for any SQL
  dialect — see how `smai-store-postgres` imports
  `smai-store-sqlite`'s `_schema.py` / `_serde.py`), or
- **Wrap an external substrate** (e.g., a managed multi-tenant store)
  while still returning the six record types and honoring the 47
  Protocol method signatures.

Capability flags (`MetadataStoreCapabilities`) include
`is_tenant_aware`, `supports_transactions`,
`supports_advisory_locks`, `supports_leasing`, etc. The conformance
suite parameterizes some tests on these flags (e.g., the tenant-
fairness contract skip-collects when `is_tenant_aware=False`;
`test_supports_leasing_matches_behavior` gates the lease-primitive
contract). **Describe what you do honestly** — capability-flag
dishonesty is a conformance failure. Note that `supports_leasing`
is load-bearing for multi-worker deployments: per `09-cli.md` §6.2
/ DEC-035 #2, `smai start` hard-exits when
`engine.worker_count > 1` against a store reporting
`supports_leasing=False`.

The migration framework is in `smai_orchestrator.migrations`. See
`packages/smai-orchestrator/src/smai_orchestrator/migrations/MIGRATIONS.md`
for the rollback story, retention policies, and the runbook for adding
a new revision.

Conformance — note `make_store` is `async`:

```python
from smai_core.plugins.conformance import MetadataStoreConformance
from my_store import MyStore


class TestMyStoreConformance(MetadataStoreConformance):
    async def make_store(self):
        store = MyStore(uri="sqlite+aiosqlite:///:memory:")
        await store.upgrade_to_head()
        return store
```

### `ArtifactStore`

Defined in `smai_core.plugins.artifact_store`. Six methods —
`put` / `get` / `exists` / `list` / `delete` / `url_for`. `url_for` is
capability-gated: `supports_presigned_urls` controls whether a plugin
returns a real presigned URL or raises `PresignedUrlsUnsupported`.

Two reference patterns:

- **Local-rooted** (`smai-artifacts-localfs`): a `root: Path`
  constructor knob; keys map to relative paths under `root`. Presigned
  URLs unsupported.
- **Cloud BYO** (`smai-artifacts-s3`): caller supplies a bucket + region;
  the plugin doesn't auto-create. Presigned URLs supported via the
  cloud provider's signing mechanism (SigV4 for S3).

Conformance:

```python
from smai_core.plugins.conformance import ArtifactStoreConformance
from my_artifacts import MyStore


class TestMyArtifactsConformance(ArtifactStoreConformance):
    def make_store(self):
        return MyStore(...)
```

### `Compute`

Defined in `smai_core.plugins.compute`. Four methods — `submit` /
`status` / `cancel` / `logs`. `JobHandle` is the typed primitive
returned from `submit` and consumed by everything else.

`JobHandle.metadata: dict[str, Any]` is the per-substrate plumbing for
state-mapping. Convention from the in-tree plugins:

- `submitted_at: float` — wall-clock for caller-side timeout enforcement.
- `cancel_requested: bool` — distinguishes user-cancel from
  substrate-kill in the state-mapping table.

Image validation is per-substrate. Some substrates (`localgpu`) can
eagerly `docker pull` and raise `JobImageInvalid`; some (`runpod`)
defer until job-launch time. The conformance suite's
`test_invalid_image_raises` accepts either shape via the
`_skip_if_image_validation_deferred` opt-out.

Conformance:

```python
from smai_core.plugins.conformance import ComputeConformance
from my_compute import MyCompute


class TestMyComputeConformance(ComputeConformance):
    def make_compute(self):
        return MyCompute(...)

    def make_fresh_compute(self):
        # Override only if your plugin holds in-process state (e.g., a
        # background poller cache); the default returns make_compute().
        return MyCompute(...)
```

### `pyproject.toml` template for a new plugin

A new plugin under `plugins/smai-foo-bar/`:

```toml
[project]
name = "smai-foo-bar"
version = "0.1.0"
description = "FooBar <Interface> plugin: <one-line>."
requires-python = ">=3.11"
dependencies = [
    "smai-core",
    # Add your substrate SDK + any other runtime deps. NEVER add a
    # pipeline package (smai-agents / smai-orchestrator-only-allowed-
    # for-MetadataStore / smai-runtime / smai-cli / smai umbrella) —
    # tools/check_deps.py rule 3 will flag it.
]

[project.entry-points."smai.<interface_namespace>"]
# One of: smai.llm_providers, smai.metadata_stores, smai.artifact_stores,
# smai.computes. The entry name is what users put in plugins.<role>:
foo_bar = "smai_foo_bar:FooBarPlugin"

[tool.uv.sources]
smai-core = { workspace = true }

[dependency-groups]
dev = [
    "smai-core[conformance]",     # pulls pytest in for the bases
    # Add fakes / mocks (vcrpy, moto, httpx-mock, ...) here.
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/smai_foo_bar"]
```

Then register the package in the workspace root `pyproject.toml`:

```toml
[tool.uv.workspace]
members = [
    # ...
    "plugins/smai-foo-bar",
]
```

And in `pyrightconfig.json`, add `plugins/smai-foo-bar/src` to the
strict-mode `executionEnvironments` block, and to the `extraPaths` of
any execution environment that imports your plugin under
`TYPE_CHECKING`.

### Testing your plugin

```bash
# Subset to just your plugin:
uv run pytest plugins/smai-foo-bar/tests -v

# Plus the cross-package smoke tests (ensure nothing else broke):
uv run pytest packages plugins tests -q

# All five gates locally before opening a PR:
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run python tools/check_deps.py
uv run pytest -q
```

Credentialed tests (the `@pytest.mark.credentialed` lane) run
**locally only** when the relevant env var is set:

```bash
export AWS_TEST_BUCKET=my-test-bucket AWS_TEST_REGION=us-east-1
uv run pytest plugins/smai-artifacts-s3/tests/test_real_aws.py -v
```

CI runs without these env vars and the tests skip cleanly.

---

## Submitting a change

### Branch and PR

`main` is the default branch. Open a feature branch, push, and open
a PR. CI runs the five gates on every push.

### Commit message style

The repo uses **single-line commit summaries** that lead with the verb
of the change and a parenthetical detail block explaining the
non-obvious decisions. Skim `git log --oneline` for the established
voice. Avoid noise commits — a feature lands in one commit by
preference, with the parenthetical doing the heavy lifting on intent.

For AI-pair-programmed commits, append a `Co-Authored-By:` trailer
when relevant.

### Before opening a PR

- [ ] All five gates pass locally (`pytest`, `pyright`, `ruff check`,
      `ruff format --check`, `tools/check_deps.py`).
- [ ] If you added a credentialed test, it's marked
      `@pytest.mark.credentialed` AND `@pytest.mark.skipif` on its
      env var.
- [ ] If you added a new package or plugin, it's registered in the
      root `pyproject.toml`'s `[tool.uv.workspace] members`, in
      `pyrightconfig.json`, and (for plugins) declares its
      `[project.entry-points]`.
- [ ] If your change touches a Protocol shape on the four plugin
      interfaces, the conformance test base under
      `smai_core.plugins.conformance` is updated AND every in-tree
      plugin is verified to still pass.
- [ ] If your change touches the design corpus assumption set, surface
      the discrepancy in the PR description rather than silently
      diverging — the canonical specs live in `designs/smai/`.

---

## Reference docs

- **Architecture** — `designs/smai/00-vision.md` (anchor),
  `designs/smai/01-data-model.md` through
  `designs/smai/10-runtime-and-templates.md` (component specs).
- **OSS / closed split** — `designs/smai/oss_strategy.md`.
- **Plugin Protocols** — `designs/smai/07-plugin-interfaces.md`,
  especially §3 (discovery), §4–§7 (per-interface shape), §8
  (cross-cutting conformance discipline).
- **CLI surface** — `designs/smai/09-cli.md`.
- **Runtime + templates** — `designs/smai/10-runtime-and-templates.md`.
- **Operations** — `packages/smai-cli/OPERATIONS.md` (systemd /
  supervisord / launchd recipes; connection-pool sizing).
- **Migrations** —
  `packages/smai-orchestrator/src/smai_orchestrator/migrations/MIGRATIONS.md`.

The `designs/smai/` corpus is the authoritative spec; this repo's
README + CONTRIBUTING are entry points and orientation. When the two
diverge, the design corpus wins.
