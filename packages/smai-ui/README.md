# smai-ui

SPA bundle wrapper for SMAI v2.

Ships the React SPA at `apps/ui/` as Python package data so
`smai-api`'s `make_api_app(...)` can mount it at `/` and serve the
dashboard alongside the JSON API. Production installs need no Node
toolchain — the wheel arrives with `static_spa/` already populated.

Per `designs/smai/12-ui-process.md` §8.2 and `13-frontend.md` §12.3,
the chosen ownership shape is "Hatch build hook on this package's
`pyproject.toml`" (RESOLVED 2026-05-03 / `13` §14 OQ3): the hook in
`build_hook.py` runs `pnpm build` and stages `apps/ui/dist/*` into
`src/smai_ui/static_spa/` at wheel/editable build time.

## Surface

```python
from smai_ui import get_static_bundle_path

bundle = get_static_bundle_path()   # -> Path to a directory holding index.html + assets/
```

Raises `FileNotFoundError` if the bundle was not staged (typically: a
developer install on a machine without pnpm and without a pre-built
`apps/ui/dist/`). `smai-api`'s SPA mount catches this and falls back to
API-only mode.

## Local development

The build hook is invoked automatically by `pip install -e
packages/smai-ui` and by `uv sync` when `smai-ui` is in the workspace.
On a machine with pnpm installed (and `apps/ui/node_modules/` populated
via `pnpm install` from `apps/ui/`), the hook runs `pnpm build`
end-to-end and stages the result.

When pnpm is not on `PATH`, the hook degrades gracefully: it will use
an existing `apps/ui/dist/` if a developer pre-built it, otherwise it
logs a warning and skips staging. This is also the path CI takes — the
Python gates do not depend on the SPA bundle being present.

The fallback Approach B from the task brief (run `pnpm build`
explicitly before `pip install`) is therefore always available; the
hook simply automates it when pnpm is on `PATH`.

## Production install

```
pip install smai-api[ui]
```

`smai-ui` is an optional extra of `smai-api` — `pip install smai-api`
alone gives you the JSON API; the `[ui]` extra adds the SPA mount.
