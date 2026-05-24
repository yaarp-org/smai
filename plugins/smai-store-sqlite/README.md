# smai-store-sqlite

`MetadataStore` plugin: SQLite reference local implementation (`SqliteStore`).
The `smai dev` default metadata store. Backed by SQLAlchemy 2.0 async Core
with the `aiosqlite` driver; shares its declarative schema (tables, cursor
pagination, lease CAS semantics) with `smai-store-postgres`, which swaps the
driver to `asyncpg` and adds Postgres-specific fast paths.

## Configuration

```yaml
plugins:
  metadata_store: sqlite
  metadata_store_config:
    uri: "sqlite+aiosqlite:////home/me/.smai/state.db"   # optional; see below
```

The `uri` key is a SQLAlchemy async URL. Defaults to
`sqlite+aiosqlite:///:memory:` (in-memory, data lost on exit).

**Four-slash rule.** SQLite file paths require four slashes in the URL:
`sqlite+aiosqlite:////absolute/path/state.db`. Three slashes (`///`) means a
path relative to the current directory; four slashes (`////`) is an absolute
path.

**Tilde expansion.** The plugin expands a leading `~` and creates the parent
directory if absent, so `sqlite+aiosqlite:///~/.smai/state.db` works:

```yaml
metadata_store_config:
  uri: "sqlite+aiosqlite:///~/.smai/state.db"
```

**In-memory default gotcha.** `smai dev` and `smai ui` inject the correct
file path under `~/.smai/state.db` automatically, so an empty
`metadata_store_config: {}` is safe under those verbs. But `smai migrate`,
`smai verify`, and `smai start` use the `smai.yaml` value verbatim: an empty
`metadata_store_config: {}` resolves to the in-memory default, which makes
`smai migrate` a silent no-op (schema applied in memory, then lost). For those
verbs, always set an explicit URI.

## Discovery

Registered via the `smai.metadata_stores` entry-point group:

```toml
[project.entry-points."smai.metadata_stores"]
sqlite = "smai_store_sqlite:SqliteStore"
```

Tier A integrators (the `smai` CLI) instantiate the plugin through entry-point
discovery; Tier B integrators import `SqliteStore` directly.

## Tests

The conformance suite and URI-resolution tests run with no external
dependencies:

```bash
uv run pytest plugins/smai-store-sqlite/
```

The conformance suite (`tests/test_conformance.py`) runs the full
`MetadataStoreConformance` contract from `smai-core` against an in-memory
SQLite instance. There is no credentialed lane; all SQLite tests are
always-on.
